"""exp03 · Auditoría de observabilidad de los ejes del espacio de diseño.

Un eje que no cambia el resultado en ningún punto del espacio no está midiendo
nada, y cualquier conclusión sobre "el mecanismo no importa" es en realidad una
conclusión sobre el instrumento.

El piloto tenía dos ejes muertos: el modo de lectura, porque las tres opciones
devolvían el vecino más cercano, y el refuerzo por lectura, porque todas las
tareas emitían sus lecturas después de todas las escrituras. La diferencia
máxima entre genotipos hermanos era exactamente 0.000000 en ambos casos. El
espacio tenía 96 arquitecturas funcionalmente distintas, no 576 — seis veces
menos que lo que la ventaja metodológica del paper afirmaba.

Este experimento vuelve a hacer esa auditoría y **sale con código distinto de
cero si algún eje vuelve a morir**. Corre en CI. Es la única forma de que
"576 arquitecturas" sea una afirmación verificada y no una esperanza.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ember.core.genotype import Genotype
from ember.experiment import ExperimentRun
from ember.nas.engine import run_search
from ember.nas.space import AXES, BIOLOGICAL_ROOT, enumerate_space, siblings
from ember.nas.stats import axis_liveness
from experiments._common import SEEDS, EvalConfig, GenotypeEvaluator

TOLERANCIA = 1e-9
"""Por debajo de esto la diferencia es ruido de punto flotante, no un efecto."""


def genotipos_a_auditar(n_bases: int | None, seed: int = 0) -> list[Genotype]:
    """Genotipos a evaluar para poder medir observabilidad.

    Con `n_bases=None` devuelve el espacio completo. Con un número, muestrea
    `n_bases` genotipos base y agrega **todos sus hermanos en cada eje**.

    El muestreo tiene que ser por conjuntos de hermanos, no por prefijo ni por
    salto. Un prefijo de la enumeración comparte el valor de los ejes que varían
    más lento —los primeros 96 genotipos tienen todos el mismo modo de lectura y
    el mismo modo de escritura— así que la auditoría reportaría esos ejes como
    muertos aunque estén vivos. Un salto tiene el problema simétrico: nunca
    incluye dos genotipos que difieran solo en el eje que varía más rápido.
    """
    espacio = list(enumerate_space())
    if n_bases is None:
        return espacio

    rng = np.random.default_rng(seed)
    bases = [espacio[i] for i in rng.choice(len(espacio), size=n_bases, replace=False)]

    seleccion: dict[str, Genotype] = {}
    for base in bases:
        seleccion[base.label()] = base
        for eje in AXES:
            for hermano in siblings(base, eje):
                seleccion[hermano.label()] = hermano
    return list(seleccion.values())


def medir_liveness(
    *,
    seeds: tuple[int, ...] = SEEDS,
    n_bases: int | None = None,
    read_every: int = 5,
    n_jobs: int = -1,
) -> dict[str, float]:
    """Máxima diferencia de puntaje entre genotipos hermanos, por eje.

    La observabilidad de un eje se puede **refutar** con una muestra: basta un
    par de hermanos que difieran. Confirmar que un eje está muerto, en cambio,
    requiere el espacio completo — por eso el modo rápido solo sirve para
    detectar regresiones, y el reporte final se corre sin `n_bases`.
    """
    genotipos = genotipos_a_auditar(n_bases)
    evaluador = GenotypeEvaluator(EvalConfig(seeds=seeds, read_every=read_every))
    resultados = run_search(evaluador, genotypes=genotipos, n_jobs=n_jobs, progress=False)
    return {eje: axis_liveness(resultados.records, eje) for eje in AXES}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="muestrea conjuntos de hermanos en vez del espacio completo, para CI",
    )
    args = parser.parse_args()

    # 8 bases no alcanzan: `reinforce` solo es observable bajo desalojo por
    # mínima fuerza, que es un cuarto del espacio. El modo rápido puede dar
    # falsos positivos; el reporte que vale es el del espacio completo, que
    # tarda ~15 s.
    n_bases = 24 if args.rapido else None
    seeds = (0,) if args.rapido else SEEDS

    with ExperimentRun("exp03_axis_liveness") as run:
        run.set_seeds(seeds)
        run.note(
            "El piloto tenía los ejes 'read' y 'reinforce' en 0.000000: "
            "el espacio efectivo era de 96 arquitecturas, no 576."
        )

        liveness = medir_liveness(seeds=seeds, n_bases=n_bases)
        run.record("liveness", liveness)
        run.record("alcance", f"{n_bases} bases + hermanos" if n_bases else "espacio completo")
        run.record("n_genotipos", len(genotipos_a_auditar(n_bases)))
        run.record("tolerancia", TOLERANCIA)

        print(f"\n{'eje':<12}{'liveness':>12}   raíz biológica")
        print("-" * 78)
        for eje, valor in sorted(liveness.items(), key=lambda kv: -kv[1]):
            marca = "MUERTO" if valor < TOLERANCIA else "      "
            print(f"{eje:<12}{valor:>12.6f} {marca} {BIOLOGICAL_ROOT[eje]}")

        muertos = [e for e, v in liveness.items() if v < TOLERANCIA]
        run.record("ejes_muertos", muertos)

        if muertos:
            print(
                f"\nFALLO: {len(muertos)} eje(s) inobservable(s): {', '.join(muertos)}\n"
                "No se puede reportar el tamaño nominal del espacio de diseño hasta "
                "arreglarlos.",
                file=sys.stderr,
            )
            run.note(f"ejes inobservables: {muertos}")
            return 1

        print(f"\nOK: los {len(AXES)} ejes son observables.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
