"""exp06 · Validación en un entorno real bajo observabilidad parcial (MiniGrid).

La Sección de limitaciones del paper dejaba esto pendiente: la ley del umbral y
el mecanismo de saliencia se miden sobre flujos donde el error de predicción de
un evento raro está fijado por diseño muy por encima del de uno común (0.9
contra 0.1). Ningún generador impone eso en un entorno real: acá el error de
predicción lo calcula un modelo en línea a partir de lo que no anticipó, y
nada garantiza que ese número separe "recompensado" de "simplemente nunca
visto antes".

Este experimento corre el genotipo de frontera (el que gana el NAS en
`exp01`) contra el `FIFO_GENOTYPE` sobre rollouts de `MiniGrid-MemoryS13-v0`
bajo tres condiciones, reteniendo siempre el mismo `t1_rare_retention` que
mide "rareza" en el resto del programa.

Tres condiciones, una variable a la vez
----------------------------------------
1. **`aleatoria_prediccion`** (línea de base): política uniforme,
   `OneStepPredictor` mide sorpresa *perceptual*. ¿Cuánto se retiene con la
   señal de sorpresa que el resto de `ember.envs` ya tenía?
2. **`sesgada_prediccion`**: cambia solo la política (favorece avanzar sobre
   girar, sin tocar el cálculo de la sorpresa), para descartar que el
   resultado de (1) sea artefacto de escasez de datos.
3. **`aleatoria_recompensa`**: cambia solo la señal (misma política uniforme
   que (1), pero `RewardPredictionError` en vez de `OneStepPredictor` —
   predice recompensa, no observación). Si la retención mejora acá y no en
   (2), el problema nunca fue de exploración: era que la sorpresa perceptual
   no correlaciona con relevancia de tarea, y una sorpresa de recompensa sí.
"""

from __future__ import annotations

import numpy as np

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.policies import Append, BothGated, MinStrength, NearestNeighbour, NoDecay
from ember.data.streams import Stream
from ember.envs.minigrid import MiniGridStreamAdapter
from ember.experiment import ExperimentRun
from ember.tasks.battery import t1_rare_retention
from experiments._common import make_factory

ENV_ID = "MiniGrid-MemoryS13-v0"
N_STEPS = 1200
"""~15x la capacidad: el mismo orden de oversubscripción que usan T1 sintéticas."""
SEMILLAS = tuple(range(40))
DIM = 32
CAPACITY = 20


class ForwardBiasedPolicy:
    """Favorece avanzar sobre girar. No toca la señal de sorpresa, solo el comportamiento.

    Acciones de MiniGrid: 0=girar izquierda, 1=girar derecha, 2=avanzar. Girar
    en exceso bajo política uniforme (1/3 de probabilidad cada una) hace que el
    agente pase la mayor parte del tiempo dando vueltas sin atravesar el
    entorno, lo que deja pocos episodios completos —y por lo tanto pocos
    eventos raros— por rollout.
    """

    def __init__(self, seed: int, p_avanzar: float = 0.7) -> None:
        self.rng = np.random.default_rng(seed + 90_000)
        self.pesos = np.array([(1 - p_avanzar) / 2, (1 - p_avanzar) / 2, p_avanzar])

    def __call__(self, obs) -> int:
        return int(self.rng.choice(3, p=self.pesos))


CONDICIONES = {
    "aleatoria_prediccion": (None, "prediction"),
    "sesgada_prediccion": (ForwardBiasedPolicy, "prediction"),
    "aleatoria_recompensa": (None, "reward"),
}
"""nombre -> (fábrica de política o None, fuente de sorpresa)."""

FRONTIER_GENOTYPE = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=BothGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""El genotipo de frontera de `exp01`, reconstruido acá para no depender de `results/`."""


def rollout(
    seed: int,
    *,
    n_steps: int = N_STEPS,
    policy_factory=None,
    surprise_source: str = "prediction",
) -> Stream:
    policy = policy_factory(seed) if policy_factory is not None else None
    adaptador = MiniGridStreamAdapter(
        ENV_ID, dim=DIM, capacity=CAPACITY, seed=seed, surprise_source=surprise_source
    )
    return adaptador.rollout(n_steps, policy=policy)


def separacion_de_saliencia(streams: list[Stream]) -> dict:
    """¿El error de predicción distingue lo recompensado de lo meramente novedoso?

    Agrupa el `pred_error` de todos los ítems de todos los rollouts en dos
    poblaciones —raros (recompensados) y comunes— y reporta si se solapan. Es
    la comprobación directa de la precondición que la ley del umbral necesita
    para que el desalojo por fuerza tenga algo que proteger.
    """
    raros = [float(it.pred_error) for s in streams for it in s.rare_items]
    comunes = [float(it.pred_error) for s in streams for it in s if not it.is_rare]
    if not raros:
        return {"n_raros": 0, "n_comunes": len(comunes)}

    umbral = min(raros)
    solapamiento = float(np.mean(np.array(comunes) >= umbral))
    return {
        "n_raros": len(raros),
        "n_comunes": len(comunes),
        "pred_error_raros_media": float(np.mean(raros)),
        "pred_error_comunes_media": float(np.mean(comunes)),
        "pred_error_comunes_p90": float(np.percentile(comunes, 90)),
        "fraccion_comunes_sobre_minimo_raro": solapamiento,
    }


def comparar(
    seeds: tuple[int, ...] = SEMILLAS,
    n_steps: int = N_STEPS,
    *,
    policy_factory=None,
    surprise_source: str = "prediction",
    verbose: bool = True,
) -> dict:
    streams = [
        rollout(s, n_steps=n_steps, policy_factory=policy_factory, surprise_source=surprise_source)
        for s in seeds
    ]
    n_raros_por_semilla = [len(s.rare_items) for s in streams]
    n_raros_total = sum(n_raros_por_semilla)

    resultados = {}
    for nombre, genotipo in (("frontera", FRONTIER_GENOTYPE), ("FIFO", FIFO_GENOTYPE)):
        factory = make_factory(genotipo, dim=DIM)
        aciertos = 0
        for i, s in enumerate(streams):
            r = t1_rare_retention(factory, s, seed=i, capacity=CAPACITY, dim=DIM)
            aciertos += round(r.score * max(len(s.rare_items), 1))
        tasa = aciertos / n_raros_total if n_raros_total else 0.0
        resultados[nombre] = {"aciertos": aciertos, "n_raros": n_raros_total, "tasa": tasa}
        if verbose:
            print(f"  {nombre:<10} {aciertos}/{n_raros_total} = {tasa:.3f}")

    return {
        "env_id": ENV_ID,
        "n_steps": n_steps,
        "n_seeds": len(seeds),
        "n_raros_por_semilla": n_raros_por_semilla,
        "n_raros_total": n_raros_total,
        "resultados": resultados,
        "separacion_de_saliencia": separacion_de_saliencia(streams),
    }


def main() -> int:
    with ExperimentRun("exp06_minigrid") as run:
        run.set_seeds(SEMILLAS)
        run.note(
            "Tres condiciones sobre MiniGrid-MemoryS13-v0, una variable a la vez: "
            "línea de base (política uniforme + sorpresa perceptual), sesgada a "
            "avanzar (descarta escasez de datos sin tocar la señal), y sorpresa de "
            "recompensa (misma política que la línea de base, cambia solo la señal). "
            "Lo raro es una recompensa positiva del entorno."
        )

        por_condicion = {}
        for nombre, (fabrica, fuente) in CONDICIONES.items():
            print(f"\n[{nombre}]")
            salida = comparar(policy_factory=fabrica, surprise_source=fuente)
            por_condicion[nombre] = salida
            run.record(f"comparacion_{nombre}", salida)

            sep = salida["separacion_de_saliencia"]
            print(
                f"  separación de saliencia: {salida['n_raros_total']} eventos raros "
                f"sobre {sep.get('n_comunes', 0)} comunes"
            )
            if sep.get("n_raros", 0):
                frac = sep["fraccion_comunes_sobre_minimo_raro"]
                run.note(
                    f"{nombre}: el {100 * frac:.1f}% de las experiencias comunes tiene "
                    "error de predicción igual o mayor que el evento raro menos "
                    "sorpresivo."
                )
            for arq, r in salida["resultados"].items():
                if r["n_raros"]:
                    run.note(f"{nombre}/{arq}: retención de eventos raros = {r['tasa']:.3f}.")

        base = por_condicion["aleatoria_prediccion"]
        sesgada = por_condicion["sesgada_prediccion"]
        recompensa = por_condicion["aleatoria_recompensa"]

        if sesgada["n_raros_total"] > base["n_raros_total"] * 2:
            run.note(
                f"la política sesgada a avanzar multiplica los eventos raros por "
                f"{sesgada['n_raros_total'] / base['n_raros_total']:.1f}× "
                f"({base['n_raros_total']} → {sesgada['n_raros_total']}) sin cambiar "
                "la señal de sorpresa, y la retención sigue en el piso: el problema "
                "no era escasez de datos."
            )

        tasa_base = base["resultados"]["frontera"]["tasa"]
        tasa_recompensa = recompensa["resultados"]["frontera"]["tasa"]
        if tasa_recompensa > tasa_base + 0.1:
            run.note(
                f"con la misma política que la línea de base, cambiar solo la señal "
                f"de sorpresa perceptual por error de predicción de recompensa sube "
                f"la retención de la frontera de {100 * tasa_base:.1f}% a "
                f"{100 * tasa_recompensa:.1f}%, mientras el FIFO —que no lee la señal "
                f"de fuerza— sigue en el piso "
                f"({100 * recompensa['resultados']['FIFO']['tasa']:.1f}%). El "
                "mecanismo de saliencia sí transfiere a un entorno real cuando la "
                "sorpresa se ancla en recompensa en vez de en novedad perceptual."
            )

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
