"""exp09 · Baselines externos de gestión de memoria, y el `EpisodicBuffer` real.

Responde a dos pedidos de la revisión de BIP2026:

- (R1, R2) Comparar contra gestión de memoria **no bioinspirada**, fuera del
  espacio de 576: muestreo de reservorio (Vitter), LRU, LFU, repetición
  priorizada por sorpresa (greedy y estocástica), una caché de utilidad
  (frecuencia × recencia) y selección por cobertura del espacio de claves
  (Isele y Cosgun 2018). Ver `ember.baselines` y los docstrings de
  `ember.core.policies`.
- (R1) El "e-MDB FIFO" del paper es un proxy optimista (lectura por vecino más
  cercano). Se evalúa además el buffer real —`deque` FIFO con recuperación
  secuencial— tal como está en el código público de GII (ver
  `SequentialScan`).

Todos los métodos son genotipos de `PolicyMemory` que difieren del proxy FIFO
solo en el desalojo (o, el buffer real, solo en la lectura), así que se evalúan
con exactamente las mismas tareas, semillas y capacidad.

Hipótesis y qué las refutaría
-----------------------------
H1. La ventaja de la frontera sobre el FIFO **no** es exclusiva de los
    mecanismos bioinspirados: una política externa que lea la misma señal de
    sorpresa (repetición priorizada greedy, `per_min`) debería quedar cerca de la
    frontera en retención de raros, porque el desalojo por mínima fuerza con
    fuerza `1 + 2·sorpresa` es, en esencia, esa misma política. Se refuta si
    `per_min` queda claramente por debajo de la frontera en T1 (IC de la
    diferencia pareada por semilla excluye cero a favor de la frontera por más
    de 0.1).
H2. Las políticas que no leen la sorpresa (reservorio, LRU, LFU, utilidad)
    no retienen raros mejor que el FIFO por más que un margen chico, porque en
    el flujo de T1 lo raro no se distingue por frecuencia ni recencia de uso.
    Se refuta si alguna supera a la frontera en el puntaje de batería.
H3. El buffer real de e-MDB retiene exactamente lo mismo que el proxy (mismo
    desalojo) pero **reconstruye peor**: sin búsqueda por contenido, una clave
    degradada no encuentra su episodio. Se refuta si su gate de reconstrucción
    no cae respecto del proxy.

Los parámetros de las políticas externas (α, ε, vida media, umbral del barrido)
se fijaron antes de correr a valores de la literatura o relativos a la
capacidad, y no se ajustaron mirando resultados.

Protocolo
---------
1. **Celda exp01** (C=20, K=5, 300 comunes, 20 raros, lectura cada 5): batería
   T1–T3 y gate R1–R4 con las semillas 0–4 (las 0–2 son las de exp01). El
   espacio de 576 se re-evalúa con las mismas cinco semillas para poder ubicar
   a cada método como intervalo de rango (invariante 5). También se lo ubica
   contra la distribución publicada de exp01 (semillas 0–2).
2. **Barrido de r** para T1 con el protocolo de exp02 (C=20, 20 visitas por
   prototipo, `n_rare = C/2`): r de compresión, transición y selección. En cada
   celda se re-evalúa el espacio para el rango.
3. IC bootstrap sobre semillas para cada métrica, y para la diferencia pareada
   frontera − método (la semilla es la unidad independiente: fija el flujo).
4. Sensibilidad del buffer real al umbral de coincidencia del barrido.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from ember.baselines import EXTERNAL_BASELINES, IN_SPACE_REFERENCES, METHODS
from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.policies import SequentialScan
from ember.experiment import ExperimentRun, load_results
from ember.nas.engine import run_search
from ember.nas.space import enumerate_space
from ember.nas.stats import bootstrap_ci
from ember.tasks.reconstruction import RECON_THRESHOLD, reconstruction_gate
from experiments._common import (
    CAPACITY,
    EvalConfig,
    GenotypeEvaluator,
    RareRetentionEvaluator,
    make_factory,
)

SEMILLAS = (0, 1, 2, 3, 4)
SEMILLAS_EXP01 = (0, 1, 2)
TAREAS = ("rare_retention", "noise_under_pressure", "sequential_interference")
TAREAS_GATE = ("pattern_completion", "noise_robustness", "ab_interference", "capacity_profile")

RATIOS = (0.25, 0.75, 1.0, 2.0)
"""Compresión (0.25), transición (0.75), borde (1.0) y selección (2.0)."""

VISITAS_POR_PROTOTIPO = 20
"""Igual que exp02: fija la recurrencia al barrer r (ver skill ember-experiment)."""

UMBRALES_SECUENCIAL = (0.5, 0.85, 0.95)
"""Sensibilidad del buffer real al criterio de coincidencia del barrido."""

TOL = 1e-9


# ═══════════════════════════════════════════════════════════════ estadística


def resumen(valores: list[float], *, seed: int = 0) -> dict[str, object]:
    """Media, desvío y IC bootstrap al 95 % sobre la unidad independiente (semilla)."""
    v = [float(x) for x in valores]
    lo, hi = bootstrap_ci(v, seed=seed)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "ci": [lo, hi],
        "per_seed": v,
    }


def rango_en_espacio(
    score: float, espacio: list[float], *, en_el_espacio: bool, tol: float = TOL
) -> dict[str, object]:
    """Intervalo de rango `(optimista, pesimista)` de un puntaje dentro del espacio.

    Si el método ya es un punto del espacio, su puntaje está en `espacio` y el
    intervalo es el de `SearchResults.rank_of` sobre 576. Si es externo, se lo
    inserta: el intervalo es sobre 577 y cuenta los empates como puestos que
    podría ocupar. Nunca se reporta un puesto puntual (invariante 5).
    """
    mejores = sum(1 for s in espacio if s > score + tol)
    empates = sum(1 for s in espacio if abs(s - score) <= tol)
    if not en_el_espacio:
        empates += 1
    total = len(espacio) + (0 if en_el_espacio else 1)
    peores = sum(1 for s in espacio if s < score - tol)
    return {
        "rank_optimistic": mejores + 1,
        "rank_pessimistic": mejores + empates,
        "of": total,
        "n_strictly_better": mejores,
        "n_strictly_worse": peores,
        "percentile_beaten": peores / len(espacio),
    }


# ═════════════════════════════════════════════════════════════ evaluadores


@dataclass(frozen=True, slots=True)
class _TrabajoMetodo:
    """Una evaluación (método, semilla) serializable para un worker."""

    name: str
    genotype: Genotype
    seed: int
    config: EvalConfig
    with_gate: bool


def _evaluar_metodo(t: _TrabajoMetodo) -> tuple[str, int, dict[str, float]]:
    config = replace(t.config, seeds=(t.seed,))
    out = dict(GenotypeEvaluator(config)(t.genotype))
    if t.with_gate:
        gate = reconstruction_gate(make_factory(t.genotype, dim=config.dim), seeds=(t.seed,))
        out.update({f"gate_{k}": v for k, v in gate.per_task.items()})
    return t.name, t.seed, out


@dataclass(frozen=True, slots=True)
class _TrabajoT1:
    name: str
    genotype: Genotype
    seed: int
    config: EvalConfig


def _evaluar_t1(t: _TrabajoT1) -> tuple[str, int, float]:
    config = replace(t.config, seeds=(t.seed,))
    return t.name, t.seed, RareRetentionEvaluator(config)(t.genotype)["rare_retention"]


def _mapear(fn, trabajos: list, n_jobs: int) -> list:
    if n_jobs == 1:
        return [fn(t) for t in trabajos]
    workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, trabajos, chunksize=2))


def evaluar_metodos(
    metodos: dict[str, Genotype],
    *,
    seeds: tuple[int, ...] = SEMILLAS,
    config: EvalConfig | None = None,
    with_gate: bool = True,
    n_jobs: int = -1,
) -> dict[str, dict[str, list[float]]]:
    """Batería (y gate) por método y por semilla: `{método: {tarea: [por semilla]}}`."""
    config = config or EvalConfig()
    trabajos = [
        _TrabajoMetodo(n, g, s, config, with_gate) for n, g in metodos.items() for s in seeds
    ]
    crudos = _mapear(_evaluar_metodo, trabajos, n_jobs)
    por_metodo: dict[str, dict[int, dict[str, float]]] = {}
    for nombre, s, out in crudos:
        por_metodo.setdefault(nombre, {})[s] = out
    resultado: dict[str, dict[str, list[float]]] = {}
    for nombre in metodos:
        filas = [por_metodo[nombre][s] for s in seeds]
        claves = list(filas[0])
        resultado[nombre] = {k: [f[k] for f in filas] for k in claves}
        resultado[nombre]["battery_mean"] = [float(np.mean([f[k] for k in TAREAS])) for f in filas]
        if with_gate:
            resultado[nombre]["gate_mean"] = [
                float(np.mean([f[f"gate_{k}"] for k in TAREAS_GATE])) for f in filas
            ]
    return resultado


def config_de_ratio(ratio: float, capacity: int = CAPACITY) -> EvalConfig:
    """Celda del barrido de r con el protocolo de exp02 (visitas por prototipo fijas)."""
    k = max(1, int(round(ratio * capacity)))
    return EvalConfig(
        capacity=capacity,
        n_prototypes=k,
        n_common=VISITAS_POR_PROTOTIPO * k,
        n_rare=max(5, capacity // 2),
    )


def evaluar_t1_ratio(
    metodos: dict[str, Genotype],
    ratio: float,
    *,
    seeds: tuple[int, ...] = SEMILLAS,
    n_jobs: int = -1,
) -> dict[str, list[float]]:
    """T1 por método y semilla en una celda de r."""
    config = config_de_ratio(ratio)
    trabajos = [_TrabajoT1(n, g, s, config) for n, g in metodos.items() for s in seeds]
    crudos = _mapear(_evaluar_t1, trabajos, n_jobs)
    tabla: dict[str, dict[int, float]] = {}
    for nombre, s, v in crudos:
        tabla.setdefault(nombre, {})[s] = v
    return {n: [tabla[n][s] for s in seeds] for n in metodos}


def diferencia_pareada(a: list[float], b: list[float], *, seed: int = 0) -> dict[str, object]:
    """`a - b` pareado por semilla, con IC bootstrap sobre semillas."""
    d = [float(x - y) for x, y in zip(a, b, strict=True)]
    return resumen(d, seed=seed)


# ═══════════════════════════════════════════════════════════════════ main


def _en_el_espacio(g: Genotype, espacio: set[Genotype]) -> bool:
    return g in espacio


def comparar(
    *,
    seeds: tuple[int, ...] = SEMILLAS,
    ratios: tuple[float, ...] = RATIOS,
    metodos: dict[str, Genotype] = METHODS,
    evaluar_espacio: bool = True,
    umbrales: tuple[float, ...] = UMBRALES_SECUENCIAL,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, object]:
    """Toda la comparación. `main()` solo la envuelve en un `ExperimentRun`."""
    espacio_set = set(enumerate_space())
    en_espacio = {n: _en_el_espacio(g, espacio_set) for n, g in metodos.items()}
    out: dict[str, object] = {
        "seeds": list(seeds),
        "capacity": CAPACITY,
        "in_space": en_espacio,
        "genotypes": {n: g.label() for n, g in metodos.items()},
    }

    # ── 1. celda exp01: batería + gate ─────────────────────────────────────
    t0 = time.time()
    crudo = evaluar_metodos(metodos, seeds=seeds, n_jobs=n_jobs)
    celda: dict[str, dict[str, object]] = {}
    for nombre, por_tarea in crudo.items():
        celda[nombre] = {k: resumen(v) for k, v in por_tarea.items()}
        celda[nombre]["gate_passes"] = bool(np.mean(por_tarea["gate_mean"]) >= RECON_THRESHOLD)
    out["exp01_cell"] = celda

    frontera = crudo["frontier"]
    out["advantage_of_frontier"] = {
        nombre: {
            m: diferencia_pareada(frontera[m], crudo[nombre][m])
            for m in ("battery_mean", "rare_retention", *TAREAS[1:])
        }
        for nombre in metodos
        if nombre != "frontier"
    }
    if verbose:
        print(
            f"celda exp01: {len(metodos)} métodos × {len(seeds)} semillas ({time.time() - t0:.0f} s)"
        )

    # ── 2. rango dentro del espacio de 576 ─────────────────────────────────
    if evaluar_espacio:
        t0 = time.time()
        esp = run_search(GenotypeEvaluator(EvalConfig(seeds=seeds)), n_jobs=n_jobs, progress=False)
        puntajes = [r.mean for r in esp.records]
        out["space_battery"] = {
            "n": len(puntajes),
            "max": max(puntajes),
            "median": float(np.median(puntajes)),
            "min": min(puntajes),
        }
        out["rank_in_space"] = {
            n: rango_en_espacio(
                float(np.mean(crudo[n]["battery_mean"])), puntajes, en_el_espacio=en_espacio[n]
            )
            for n in metodos
        }
        # Contra la distribución publicada de exp01 (semillas 0–2).
        try:
            publicados = [r["mean"] for r in load_results("exp01_nas_full")["all_records"]]
        except FileNotFoundError:
            publicados = []
        if publicados and set(SEMILLAS_EXP01) <= set(seeds):
            idx = [seeds.index(s) for s in SEMILLAS_EXP01]
            out["rank_in_exp01_published"] = {
                n: rango_en_espacio(
                    float(np.mean([crudo[n]["battery_mean"][i] for i in idx])),
                    publicados,
                    en_el_espacio=en_espacio[n],
                )
                for n in metodos
            }
        if verbose:
            print(f"espacio de 576 × {len(seeds)} semillas ({time.time() - t0:.0f} s)")

    # ── 3. barrido de r sobre T1 ───────────────────────────────────────────
    barrido = []
    for ratio in ratios:
        t0 = time.time()
        config = config_de_ratio(ratio)
        t1 = evaluar_t1_ratio(metodos, ratio, seeds=seeds, n_jobs=n_jobs)
        fila: dict[str, object] = {
            "r": config.n_prototypes / config.capacity,
            "n_prototypes": config.n_prototypes,
            "n_common": config.n_common,
            "n_rare": config.n_rare,
            "rare_retention": {n: resumen(v) for n, v in t1.items()},
            "advantage_of_frontier": {
                n: diferencia_pareada(t1["frontier"], v) for n, v in t1.items() if n != "frontier"
            },
        }
        if evaluar_espacio:
            esp = run_search(
                RareRetentionEvaluator(replace(config, seeds=seeds)), n_jobs=n_jobs, progress=False
            )
            puntajes = [r.mean for r in esp.records]
            fila["space_max"] = max(puntajes)
            fila["space_median"] = float(np.median(puntajes))
            fila["rank_in_space"] = {
                n: rango_en_espacio(float(np.mean(v)), puntajes, en_el_espacio=en_espacio[n])
                for n, v in t1.items()
            }
        barrido.append(fila)
        if verbose:
            print(f"r = {fila['r']:.2f} ({time.time() - t0:.0f} s)")
    out["ratio_sweep"] = barrido

    # ── 4. sensibilidad del buffer real al umbral de coincidencia ──────────
    variantes = {
        f"emdb_sequential_tau{tau}": FIFO_GENOTYPE.with_axis("read", SequentialScan(threshold=tau))
        for tau in umbrales
    }
    sens = evaluar_metodos(variantes, seeds=seeds, n_jobs=n_jobs)
    out["emdb_threshold_sensitivity"] = {
        n: {k: resumen(v) for k, v in d.items() if k in ("gate_mean", "battery_mean", *TAREAS)}
        for n, d in sens.items()
    }
    return out


def _imprimir(res: dict) -> None:
    celda = res["exp01_cell"]
    rango = res.get("rank_in_space", {})
    print(f"\n{'método':<26}{'raros':>16}{'batería':>16}{'gate':>16}   rango en el espacio")
    print("-" * 100)
    for n, d in celda.items():

        def f(k, d=d):
            m = d[k]
            return f"{m['mean']:.3f} [{m['ci'][0]:.2f},{m['ci'][1]:.2f}]"

        rr = rango.get(n)
        rs = f"#{rr['rank_optimistic']}–#{rr['rank_pessimistic']} de {rr['of']}" if rr else ""
        print(f"{n:<26}{f('rare_retention'):>16}{f('battery_mean'):>16}{f('gate_mean'):>16}   {rs}")
    print("\nT1 por r (media sobre semillas):")
    nombres = list(celda)
    print(
        f"{'método':<26}"
        + "".join(f"{'r=' + format(c['r'], '.2f'):>10}" for c in res["ratio_sweep"])
    )
    for n in nombres:
        print(
            f"{n:<26}"
            + "".join(f"{c['rare_retention'][n]['mean']:>10.3f}" for c in res["ratio_sweep"])
        )


def main() -> int:
    with ExperimentRun("exp09_external_baselines") as run:
        run.set_seeds(SEMILLAS)
        run.note(
            "Baselines externos implementados en ember.core.policies (sección "
            "'políticas externas'), fuera de SEARCH_SPACE. Las 576 y exp01–exp06 no cambian."
        )
        run.note(
            "Buffer real de e-MDB: GII/emdb_cognitive_nodes_gii, "
            "cognitive_nodes/cognitive_nodes/episodic_buffer.py (commit d15f96a). Es un "
            "deque(maxlen) sin consulta por contenido; la traducción de una consulta a "
            "'primer episodio en orden de inserción con coseno >= 0.85' es un supuesto."
        )
        run.note(
            "Rango de un método externo: intervalo sobre 577 (el espacio más el método). "
            "Para un método que ya es punto del espacio, intervalo sobre 576."
        )
        res = comparar()
        for k, v in res.items():
            run.record(k, v)
        run.record("methods_external", list(EXTERNAL_BASELINES))
        run.record("methods_in_space_references", list(IN_SPACE_REFERENCES))
        _imprimir(res)

        # Pregunta clave: ¿empata la repetición priorizada con la frontera?
        adv = res["advantage_of_frontier"]["per_min"]
        for m in ("battery_mean", "rare_retention"):
            lo, hi = adv[m]["ci"]
            if lo <= 0.0 <= hi:
                run.note(
                    f"per_min empata con la frontera en {m}: diferencia "
                    f"{adv[m]['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] (IC incluye 0)."
                )
            elif hi < 0.0:
                run.note(f"per_min SUPERA a la frontera en {m}: {adv[m]['mean']:+.3f}.")
        for n, a in res["advantage_of_frontier"].items():
            if a["battery_mean"]["ci"][1] < 0.0:
                run.note(f"{n} supera a la frontera en batería: {a['battery_mean']['mean']:+.3f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
