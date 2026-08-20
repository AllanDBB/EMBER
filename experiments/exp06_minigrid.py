"""exp06 · Validación en un entorno real bajo observabilidad parcial (MiniGrid).

La Sección de limitaciones del paper dejaba esto pendiente: la ley del umbral y
el mecanismo de saliencia se miden sobre flujos donde el error de predicción de
un evento raro está fijado por diseño muy por encima del de uno común (0.9
contra 0.1). Ningún generador impone eso en un entorno real: acá el error de
predicción lo calcula `OneStepPredictor` a partir de lo que el modelo de
transición no anticipó, y nada garantiza que ese número separe "recompensado"
de "simplemente nunca visto antes".

Este experimento corre el genotipo de frontera (el que gana el NAS en
`exp01`) contra el `FIFO_GENOTYPE` sobre rollouts de `MiniGrid-MemoryS13-v0`
con política aleatoria, retiene el mismo `t1_rare_retention` que mide "rareza"
en el resto del programa, y mide directamente si el error de predicción
separa lo recompensado de lo meramente novedoso.
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

FRONTIER_GENOTYPE = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=BothGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""El genotipo de frontera de `exp01`, reconstruido acá para no depender de `results/`."""


def rollout(seed: int, *, n_steps: int = N_STEPS) -> Stream:
    return MiniGridStreamAdapter(ENV_ID, dim=DIM, capacity=CAPACITY, seed=seed).rollout(n_steps)


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
    seeds: tuple[int, ...] = SEMILLAS, n_steps: int = N_STEPS, *, verbose: bool = True
) -> dict:
    streams = [rollout(s, n_steps=n_steps) for s in seeds]
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
            "Política aleatoria uniforme sobre MiniGrid-MemoryS13-v0. Lo raro es una "
            "recompensa positiva del entorno; a diferencia de los flujos sintéticos, "
            "nada garantiza que el error de predicción de OneStepPredictor separe eso "
            "de la novedad perceptual ordinaria."
        )

        salida = comparar()
        run.record("comparacion", salida)

        sep = salida["separacion_de_saliencia"]
        print(
            f"\nseparación de saliencia: {salida['n_raros_total']} eventos raros sobre "
            f"{sep.get('n_comunes', 0)} comunes"
        )
        if sep.get("n_raros", 0) and sep["fraccion_comunes_sobre_minimo_raro"] > 0.05:
            frac = sep["fraccion_comunes_sobre_minimo_raro"]
            run.note(
                f"el {100 * frac:.1f}% de las experiencias comunes tiene error de "
                "predicción igual o mayor que el evento raro menos sorpresivo. El "
                "error de predicción de un paso no separa lo recompensado de lo "
                "meramente novedoso en este dominio, así que el desalojo por fuerza "
                "no tiene una señal confiable que proteger — precondición para el "
                "mecanismo de saliencia que ningún flujo sintético expone, análoga a "
                "la de exp05 para el eje de escritura."
            )

        for nombre, r in salida["resultados"].items():
            if r["n_raros"] and r["tasa"] < 0.05:
                run.note(f"{nombre}: retención de eventos raros en el piso ({r['tasa']:.3f}).")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
