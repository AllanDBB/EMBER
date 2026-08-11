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

from ember.data.embeddings import embedding_stream, extract_cifar100_embeddings
from ember.data.prototypes import estimate_n_prototypes
from ember.data.synthetic import clustered_stream
from ember.experiment import ExperimentRun
from ember.nas.engine import run_search
from ember.nas.space import enumerate_space
from ember.nas.stats import bootstrap_ci, eta_squared
from ember.tasks.battery import t1_rare_retention
from experiments._common import READ_EVERY, make_factory

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
                celda = _celda(streams, k_hat, k_ci, capacity)
                celda["tiene_estructura"] = tiene_estructura
                celdas.append(celda)

                if verbose:
                    print(
                        f"  [{dominio:<9}] C={capacity} K={k:<4} K̂={k_hat:<4}"
                        f" r̂={celda['r']:<6.2f}"
                        f" η²w={celda['eta2_write']:.3f} η²e={celda['eta2_evict']:.3f}"
                        f"{'' if tiene_estructura else '  (sin estructura detectable)'}",
                        flush=True,
                    )

        salida[dominio] = {"celdas": celdas}

    # ── dónde cae el cruce en cada dominio ──────────────────────────────────
    for datos in salida.values():
        ordenadas = sorted(datos["celdas"], key=lambda c: c["r"])
        cruce = None
        for a, b in zip(ordenadas, ordenadas[1:], strict=False):
            if a["eta2_write"] > a["eta2_evict"] and b["eta2_write"] <= b["eta2_evict"]:
                cruce = float(np.sqrt(a["r"] * b["r"]))
                break
        datos["r_umbral"] = cruce
        datos["r_umbral_ci"] = (
            list(bootstrap_ci([c["r"] for c in ordenadas if c["eta2_write"] > c["eta2_evict"]]))
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

        print(f"\n{'dominio':<14}{'cruce r̂':>12}")
        print("-" * 28)
        for dominio, datos in resultado.items():
            cruce = datos["r_umbral"]
            print(f"{dominio:<14}{cruce if cruce else float('nan'):>12.2f}")

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

        # ── cuánta estructura tiene realmente cada dominio ──────────────────
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
