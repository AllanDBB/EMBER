"""exp05 · La ley del umbral sobre embeddings perceptuales reales.

La sección de limitaciones del paper concede que todos los experimentos corren
sobre vectores gaussianos aleatorios en R³², que son casi ortogonales entre sí,
y que eso reduce la interferencia cruzada y probablemente sobreestima la
precisión de recuperación. La pregunta que deja abierta es si el umbral en r = 1
se mueve bajo estructura correlacionada, densidad de clúster desigual y
desplazamiento distribucional.

Este experimento la responde sin depender del robot: corre la misma grilla sobre
flujos construidos desde embeddings de CIFAR-100 extraídos con un ResNet-18
preentrenado.

El problema nuevo que eso abre, y que se resuelve explícitamente
---------------------------------------------------------------
Sobre datos reales `K_proto` deja de conocerse. El flujo se construye tomando
`n` clases, pero eso es la etiqueta del generador, no la estructura que la
memoria ve: dos clases de CIFAR-100 pueden colapsar en un solo prototipo
perceptual, y una clase puede partirse en varios. La ley se enuncia sobre
prototipos, así que hay que estimarlos y reportar la ley contra `K̂` con su
intervalo — que es exactamente lo que un robot tendría que hacer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ember.core.policies import Merge
from ember.data.embeddings import embedding_stream, extract_cifar100_embeddings
from ember.data.prototypes import estimate_n_prototypes
from ember.data.synthetic import clustered_stream
from ember.experiment import ExperimentRun
from ember.nas.engine import run_search
from ember.nas.space import enumerate_space
from ember.nas.stats import bootstrap_ci, eta_squared
from ember.tasks.battery import t1_rare_retention
from experiments._common import READ_EVERY, make_factory

UMBRAL_DE_FUSION = Merge().threshold
"""El umbral por encima del cual dos experiencias se consideran la misma."""

CACHE = Path("data/cache")
CAPACIDADES = (20,)
RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
SEMILLAS = (0, 1, 2)
VISITAS_POR_PROTOTIPO = 20


class StreamEvaluator:
    """Evalúa un genotipo sobre flujos ya construidos. Serializable para los workers."""

    def __init__(self, streams: list, read_every: int = READ_EVERY) -> None:
        self.streams = streams
        self.read_every = read_every

    def __call__(self, genotype) -> dict[str, float]:
        dim = self.streams[0].spec.dim
        factory = make_factory(genotype, dim=dim)
        puntajes = [
            t1_rare_retention(factory, s, seed=i, read_every=self.read_every, dim=dim).score
            for i, s in enumerate(self.streams)
        ]
        return {"rare_retention": float(np.mean(puntajes))}


def _celda(streams: list, k_hat: int, k_ci: tuple[int, int], capacity: int) -> dict:
    resultados = run_search(StreamEvaluator(streams), n_jobs=-1, progress=False)
    return {
        "capacity": capacity,
        "k_declarado": streams[0].spec.n_prototypes,
        "k_hat": k_hat,
        "k_hat_ci": list(k_ci),
        "r": k_hat / capacity,
        "r_declarado": streams[0].spec.n_prototypes / capacity,
        "eta2_write": eta_squared(resultados.records, "write", metric="rare_retention"),
        "eta2_evict": eta_squared(resultados.records, "evict", metric="rare_retention"),
    }


def _estimar(stream) -> tuple[int, tuple[int, int], bool]:
    est = estimate_n_prototypes(stream.keys(), k_max=min(96, len(stream) - 1), seed=0)
    return est.k_hat, est.ci, est.has_structure


def _genotipo_de_fusion():
    from ember.core.genotype import Genotype
    from ember.core.policies import (
        Merge,
        MinStrength,
        NearestNeighbour,
        NoDecay,
        PredErrorGated,
    )

    return Genotype(
        read=NearestNeighbour(),
        write=Merge(threshold=UMBRAL_DE_FUSION),
        strength=PredErrorGated(),
        decay=NoDecay(),
        evict=MinStrength(),
        reinforce=0.0,
    )


def prototipos_efectivos(stream, *, seed: int = 0) -> tuple[int, float]:
    """Cuántas ranuras ocupa la experiencia rutinaria después de consolidar.

    Devuelve `(K_efectivo, tasa_de_fusion)`.

    **Esta es la precondición que la ley del umbral no enuncia.** La ley dice que
    fusionar paga cuando los `K` prototipos recurrentes caben en las `C` ranuras.
    Pero `K` es el número de prototipos del generador, no el número de trazas que
    la memoria termina teniendo: si la consolidación es parcial —porque la
    dispersión intra-prototipo deja parte de las visitas por debajo del umbral de
    fusión— cada prototipo ocupa varias ranuras, y la rutina llena la memoria
    igual que si no se hubiera fusionado nada.

    Medido sin presión de capacidad, para aislar la consolidación del desalojo:
    lo que se cuenta es cuántas trazas distintas sobreviven al flujo de
    experiencia común cuando hay lugar para todas.

    Una tasa de fusión alta **no** implica compresión suficiente: con el 50 % de
    las escrituras fusionadas quedan ~2 trazas por prototipo, y en una memoria de
    capacidad 20 con 10 prototipos eso ocupa las 20 ranuras y no deja ninguna
    para los eventos raros.
    """
    comunes = [it for it in stream if not it.is_rare]
    mem = make_factory(_genotipo_de_fusion(), dim=stream.spec.dim)(len(comunes) + 1, seed)
    for it in comunes:
        mem.write(it.key, it.value, it.pred_error)
    return len(mem), mem.n_merges / max(mem.n_writes, 1)


def comparar_dominios(
    *,
    capacities: tuple[int, ...] = CAPACIDADES,
    ratios: tuple[float, ...] = RATIOS,
    seeds: tuple[int, ...] = SEMILLAS,
    verbose: bool = True,
) -> dict:
    banco = extract_cifar100_embeddings(CACHE)
    salida: dict[str, dict] = {}

    for dominio in ("synthetic", "cifar100"):
        celdas = []
        for capacity in capacities:
            for ratio in ratios:
                k = max(1, int(round(ratio * capacity)))
                n_common = VISITAS_POR_PROTOTIPO * k
                n_rare = max(5, capacity // 2)

                streams = []
                for s in seeds:
                    if dominio == "synthetic":
                        streams.append(
                            clustered_stream(
                                n_prototypes=k,
                                capacity=capacity,
                                n_common=n_common,
                                n_rare=n_rare,
                                dim=banco.dim,
                                seed=s,
                            )
                        )
                    else:
                        streams.append(
                            embedding_stream(
                                banco,
                                n_prototypes=k,
                                capacity=capacity,
                                n_common=n_common,
                                n_rare=n_rare,
                                seed=s,
                            )
                        )

                k_hat, k_ci, tiene_estructura = _estimar(streams[0])
                k_efectivo, tasa = prototipos_efectivos(streams[0])

                celda = _celda(streams, k_hat, k_ci, capacity)
                celda["tiene_estructura"] = tiene_estructura
                celda["tasa_de_fusion"] = tasa
                celda["k_efectivo"] = k_efectivo
                celda["r_efectivo"] = k_efectivo / capacity
                # La compresión sirve solo si deja ranuras libres para lo raro.
                celda["compresion_util"] = bool(k_efectivo < capacity)
                celdas.append(celda)

                if verbose:
                    avisos = []
                    if not tiene_estructura:
                        avisos.append("sin estructura")
                    if not celda["compresion_util"]:
                        avisos.append(
                            f"consolidación insuficiente: K_ef={k_efectivo} >= C={capacity}"
                        )
                    print(
                        f"  [{dominio:<9}] C={capacity} K={k:<4} K̂={k_hat:<4}"
                        f" r̂={celda['r']:<6.2f}"
                        f" η²w={celda['eta2_write']:.3f} η²e={celda['eta2_evict']:.3f}"
                        + (f"  ({'; '.join(avisos)})" if avisos else ""),
                        flush=True,
                    )

        salida[dominio] = {"celdas": celdas}

    # ── dónde cae el cruce en cada dominio ──────────────────────────────────
    # El cruce se localiza contra dos variables distintas a propósito. El ratio
    # nominal es el que enuncia el borrador; el efectivo cuenta las trazas que la
    # consolidación deja de verdad. Si la ley se sostiene sobre el efectivo pero
    # no sobre el nominal, la variable de control del régimen es el efectivo — y
    # sobre datos sintéticos los dos coinciden, que es por qué nadie lo notó.
    for datos in salida.values():
        for campo, destino in (("r", "r_umbral"), ("r_efectivo", "r_umbral_efectivo")):
            ordenadas = sorted(datos["celdas"], key=lambda c: c[campo])
            cruce = None
            for a, b in zip(ordenadas, ordenadas[1:], strict=False):
                if a["eta2_write"] > a["eta2_evict"] and b["eta2_write"] <= b["eta2_evict"]:
                    cruce = float(np.sqrt(a[campo] * b[campo]))
                    break
            datos[destino] = cruce
            datos[f"{destino}_ci"] = (
                list(
                    bootstrap_ci([c[campo] for c in ordenadas if c["eta2_write"] > c["eta2_evict"]])
                )
                if cruce
                else None
            )

    return salida


def main() -> int:
    with ExperimentRun("exp05_real_embeddings") as run:
        run.set_seeds(SEMILLAS)
        run.note(
            "Sobre datos reales K_proto no se conoce: la ley se enuncia contra K "
            "estimado con su intervalo, no contra el número de clases del generador."
        )
        run.log(f"{len(enumerate_space())} genotipos por celda")

        resultado = comparar_dominios()
        run.record("dominios", resultado)

        print(f"\n{'dominio':<14}{'cruce r nominal':>18}{'cruce r efectivo':>19}")
        print("-" * 52)
        for dominio, datos in resultado.items():
            nom, efe = datos["r_umbral"], datos["r_umbral_efectivo"]
            print(
                f"{dominio:<14}{nom if nom else float('nan'):>18.2f}"
                f"{efe if efe else float('nan'):>19.2f}"
            )

        sint = resultado["synthetic"]["r_umbral"]
        real = resultado["cifar100"]["r_umbral"]
        if sint and real:
            desplazamiento = real / sint
            run.record("desplazamiento_del_umbral", desplazamiento)
            print(f"\nDesplazamiento real/sintético: {desplazamiento:.2f}×")
            if not 0.5 <= desplazamiento <= 2.0:
                run.note(
                    f"El umbral se desplaza {desplazamiento:.2f}× al pasar a datos "
                    "reales. La ley depende del dominio y hay que enunciarla con esa "
                    "calificación."
                )

        # ── la precondición de la ley ───────────────────────────────────────
        print(f"\n{'dominio':<14}{'sim. intra-prototipo':>22}{'fusión alcanzable':>20}")
        print("-" * 58)
        for dominio, datos in resultado.items():
            intra = float(np.median([c["similitud_intra_prototipo"] for c in datos["celdas"]]))
            alcanzable = all(c["fusion_alcanzable"] for c in datos["celdas"])
            print(f"{dominio:<14}{intra:>22.3f}{'sí' if alcanzable else 'no':>20}")
            run.record(
                f"precondicion_{dominio}",
                {
                    "similitud_intra_prototipo": intra,
                    "umbral_de_fusion": UMBRAL_DE_FUSION,
                    "fusion_alcanzable": alcanzable,
                },
            )
            if not alcanzable:
                run.note(
                    f"{dominio}: la similitud intra-prototipo ({intra:.3f}) queda por "
                    f"debajo del umbral de fusión ({UMBRAL_DE_FUSION}). El eje de "
                    "escritura es inoperante en este dominio y NO EXISTE régimen de "
                    "compresión, por bajo que sea r. La ley del umbral tiene una "
                    "precondición que el borrador no enuncia."
                )

        for dominio, datos in resultado.items():
            sin_estructura = [
                c["k_declarado"] for c in datos["celdas"] if not c["tiene_estructura"]
            ]
            if sin_estructura:
                run.note(
                    f"{dominio}: sin estructura de prototipos detectable en K="
                    f"{sin_estructura}; r no está definido en esas celdas."
                )

        try:
            from ember.figures import figura_comparacion_dominios

            destino = figura_comparacion_dominios(
                resultado["synthetic"]["celdas"],
                resultado["cifar100"]["celdas"],
                "paper/figures/fig2_dominios.pdf",
            )
            run.record("figura", str(destino))
            print(f"\nFigura: {destino}")
        except ImportError:
            run.note("matplotlib no disponible; figura no generada")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
