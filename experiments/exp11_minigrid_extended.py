"""exp11 · MiniGrid ampliado: estadística por rollout, más entornos, políticas y relevancias.

Responde al revisor 1 sobre `exp06`: un solo entorno, 40 rollouts, dos
políticas sin objetivo, intervalos de Wilson que tratan como independientes
eventos del mismo rollout, y —lo más serio— una importancia ("recompensa
inmediata") casi idéntica a la señal que funcionó (error de predicción de
recompensa). El resultado de `exp06` podía estar definido por la tarea.

Tres partes
-----------
1. **Estadística correcta.** Se re-analizan las tres condiciones de `exp06`
   con el rollout como unidad: tasa agregada (aciertos / eventos) con IC por
   bootstrap de clúster sobre rollouts, tasa media por rollout, IC entre 5
   bloques de semillas independientes, y el efecto de diseño respecto de
   Wilson. Primero sobre los 40 rollouts originales (debe reproducir `exp06`
   bit a bit), después sobre 200 (5 bloques de 40; el bloque 0 es `exp06`).
2. **Matriz entorno × política × importancia × señal.** Cuatro entornos
   (MemoryS13, DoorKey-6x6, FourRooms, KeyCorridorS3R2), cuatro
   políticas (uniforme, sesgada a avanzar, planificador BFS con ruido,
   Q-learning tabular), cinco definiciones de importancia y seis señales de
   saliencia (ver `ember.envs.salience`). Cada rollout se corre una sola vez y
   todas las celdas se calculan sobre la misma experiencia.
3. **Tres memorias** con la capacidad de `exp06` (20): la frontera de `exp01`,
   el proxy FIFO-NN, y la mejor configuración de `exp01` que no lee la señal de
   saliencia (fuerza `constant` o `novelty`).

Hipótesis y qué la refutaría
----------------------------
H1 (estadística). La conclusión de `exp06` —la frontera retiene más que el FIFO
con `reward_pe`— sobrevive a la unidad correcta: el IC por clúster de la
diferencia frontera − FIFO excluye 0 con 200 rollouts. Se refuta si lo incluye.
Se espera que el efecto de diseño sea > 1 (eventos del mismo rollout
correlacionados); si es ~1, Wilson no estaba equivocado en la práctica.

H2 (transferencia). La saliencia transfiere cuando la señal está alineada con
la importancia, y solo entonces: la ganancia de la frontera sobre el FIFO en
una celda (importancia, señal) crece con la separación AUC de la señal para esa
importancia, y es nula o negativa donde AUC ≈ 0.5. Se refuta si hay celdas con
ganancia clara y AUC ≈ 0.5 (la ganancia vendría de otra cosa), o si una señal
alineada (AUC alta) no produce ganancia (el mecanismo no la aprovecharía).

H3 (importancia retrasada e intrínseca). Ninguna señal causal separa bien la
importancia retrasada (`previa_k`, `evento_clave`), porque en el momento de la
escritura nada distingue esos pasos; `td_error` sería la excepción parcial. Se
refuta si `reward_pe` o `perceptual` retienen `previa_k` claramente por encima
del FIFO. Para la importancia intrínseca, se espera que `count_novelty` separe
`novedad_estado`; se refuta si no lo hace.

Lo que este experimento no mide: el efecto de la memoria sobre el aprendizaje o
el desempeño del agente (eso es `exp13`).

Unidad estadística
------------------
El rollout (un stream de 1200 pasos, con su propia memoria). Nunca el evento.
Las semillas de rollout 0..199 se agrupan en 5 bloques de 40
(bloque = semilla // 40) para el IC entre semillas.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.memory import PolicyMemory
from ember.core.policies import ExponentialDecay, MinStrength, NearestNeighbour, NoveltyGated
from ember.core.policies import Merge as MergeWrite
from ember.experiment import ExperimentRun
from ember.tasks.battery import HIT_SIMILARITY, t1_rare_retention
from experiments._common import make_factory, wilson_ci
from experiments.exp06_minigrid import CONDICIONES as CONDICIONES_EXP06
from experiments.exp06_minigrid import FRONTIER_GENOTYPE
from experiments.exp06_minigrid import rollout as rollout_exp06

DIM = 32
CAPACITY = 20
N_STEPS = 1200
N_BLOQUES = 5
ROLLOUTS_POR_BLOQUE = 40
SEMILLAS = tuple(range(N_BLOQUES * ROLLOUTS_POR_BLOQUE))
SEMILLAS_EXP06 = tuple(range(40))
N_BOOT = 2000

ENTORNOS = {
    "MemoryS13": "MiniGrid-MemoryS13-v0",
    "DoorKey6x6": "MiniGrid-DoorKey-6x6-v0",
    "FourRooms": "MiniGrid-FourRooms-v0",
    "KeyCorridorS3R2": "MiniGrid-KeyCorridorS3R2-v0",
}
"""Clave corta (sin puntos, para `\\result{}`) → id de gymnasium."""

POLITICAS = ("aleatoria", "sesgada", "planificador", "qlearning")

BEST_NO_SALIENCE_GENOTYPE = Genotype(
    read=NearestNeighbour(),
    write=MergeWrite(threshold=0.85),
    strength=NoveltyGated(),
    decay=ExponentialDecay(rate=0.995),
    evict=MinStrength(),
    reinforce=0.0,
)
"""La mejor configuración de `exp01` cuya fuerza no lee `pred_error`.

Entre las 288 con `strength ∈ {constant, novelty}`, la de mayor puntaje medio
(0.685) es `nn|merge|novelty|0.995|min_strength|0.0` (empatada con su gemela de
`reinforce=0.5`, que en este protocolo sin lecturas intercaladas es idéntica).
Es la respuesta a "¿hace falta la señal, o basta con una buena configuración
que no la use?". `tests/test_exp11_minigrid_extended.py` verifica contra
`results/exp01_nas_full` que siga siendo la mejor.
"""

MEMORIAS = {
    "frontera": FRONTIER_GENOTYPE,
    "FIFO": FIFO_GENOTYPE,
    "sin_saliencia": BEST_NO_SALIENCE_GENOTYPE,
}

MEMORIAS_SIN_SENAL = ("FIFO", "sin_saliencia")
"""Memorias cuya fuerza no lee `pred_error`: su resultado es el mismo con cualquier señal.

Se evalúan una sola vez por rollout (ahorra 10 de 18 corridas de memoria) y se
reportan a nivel de importancia, no de señal. `evaluar_rollout` verifica la
premisa con la etiqueta de fuerza del genotipo.
"""

_T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


# ════════════════════════════════════════════════════════ estadística por rollout


def _remuestreos(n_rollouts: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_rollouts, size=(n_boot, n_rollouts))


def cluster_bootstrap_ratio(
    hits: np.ndarray, n: np.ndarray, *, n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, float | None]:
    """IC95 de la tasa agregada Σh/Σn remuestreando rollouts enteros (bootstrap de clúster).

    Es la corrección que pide el revisor: los eventos de un mismo rollout
    comparten memoria, política y mapa, así que se remuestrea el rollout y no
    el evento. Los remuestreos sin ningún evento se descartan.
    """
    hits = np.asarray(hits, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if n.sum() == 0:
        return {"ci_low": None, "ci_high": None, "se": None}
    idx = _remuestreos(len(n), n_boot, seed)
    num, den = hits[idx].sum(axis=1), n[idx].sum(axis=1)
    tasas = num[den > 0] / den[den > 0]
    lo, hi = np.percentile(tasas, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "se": float(tasas.std(ddof=1))}


def cluster_bootstrap_diff(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    n: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict[str, float | None]:
    """IC95 pareado de (Σh_a − Σh_b)/Σn con los mismos rollouts remuestreados para ambos."""
    a = np.asarray(hits_a, dtype=np.float64)
    b = np.asarray(hits_b, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if n.sum() == 0:
        return {"dif": None, "ci_low": None, "ci_high": None}
    idx = _remuestreos(len(n), n_boot, seed)
    den = n[idx].sum(axis=1)
    ok = den > 0
    difs = (a[idx].sum(axis=1)[ok] - b[idx].sum(axis=1)[ok]) / den[ok]
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return {"dif": float((a.sum() - b.sum()) / n.sum()), "ci_low": float(lo), "ci_high": float(hi)}


def block_ci(hits: np.ndarray, n: np.ndarray, bloques: np.ndarray) -> dict[str, Any]:
    """IC95 t entre bloques de semillas independientes (tasa agregada por bloque)."""
    por_bloque = []
    for b in np.unique(bloques):
        m = bloques == b
        if n[m].sum() > 0:
            por_bloque.append(float(hits[m].sum() / n[m].sum()))
    k = len(por_bloque)
    if k < 2:
        return {
            "por_bloque": por_bloque,
            "media": None,
            "sd": None,
            "ci_low": None,
            "ci_high": None,
        }
    media = float(np.mean(por_bloque))
    sd = float(np.std(por_bloque, ddof=1))
    margen = _T_975.get(k - 1, 1.96) * sd / np.sqrt(k)
    return {
        "por_bloque": por_bloque,
        "media": media,
        "sd": sd,
        "ci_low": media - margen,
        "ci_high": media + margen,
    }


def per_rollout_mean(
    hits: np.ndarray, n: np.ndarray, *, n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, float | None]:
    """Media de la tasa por rollout (cada rollout con eventos pesa lo mismo), con IC bootstrap."""
    m = n > 0
    if not m.any():
        return {"media": None, "ci_low": None, "ci_high": None}
    tasas = hits[m] / n[m]
    idx = _remuestreos(len(tasas), n_boot, seed)
    medias = tasas[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return {"media": float(tasas.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def rate_summary(
    hits: np.ndarray, n: np.ndarray, bloques: np.ndarray, *, seed: int = 0
) -> dict[str, Any]:
    """Tasa agregada con Wilson (como `exp06`), IC por clúster, IC por bloques y efecto de diseño.

    El efecto de diseño es `var_cluster / var_binomial`: cuántas veces más
    varianza tiene la tasa que la que supone Wilson. `n_efectivo = n / deff`.
    """
    hits = np.asarray(hits, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    total_n, total_h = int(n.sum()), int(hits.sum())
    tasa = total_h / total_n if total_n else None
    w_lo, w_hi = wilson_ci(total_h, total_n)
    cl = cluster_bootstrap_ratio(hits, n, seed=seed)
    deff = None
    if tasa is not None and 0 < tasa < 1 and cl["se"] is not None:
        deff = float(cl["se"] ** 2 / (tasa * (1 - tasa) / total_n))
    ancho_w = w_hi - w_lo
    ancho_cl = (cl["ci_high"] - cl["ci_low"]) if cl["ci_low"] is not None else None
    return {
        "aciertos": total_h,
        "n_eventos": total_n,
        "n_rollouts": len(n),
        "n_rollouts_con_eventos": int((n > 0).sum()),
        "tasa": tasa,
        "wilson_low": w_lo if total_n else None,
        "wilson_high": w_hi if total_n else None,
        "cluster_low": cl["ci_low"],
        "cluster_high": cl["ci_high"],
        "ensanchamiento": (ancho_cl / ancho_w) if (ancho_cl is not None and ancho_w > 0) else None,
        "deff": deff,
        "n_efectivo": (total_n / deff) if deff else None,
        "por_rollout": per_rollout_mean(hits, n, seed=seed),
        "semillas": block_ci(hits, n, bloques),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Correlación de Spearman (rangos medios en empates), sin scipy."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return None

    def rangos(v: np.ndarray) -> np.ndarray:
        orden = v.argsort(kind="mergesort")
        r = np.empty(len(v))
        r[orden] = np.arange(1, len(v) + 1)
        for val in np.unique(v):
            m = v == val
            r[m] = r[m].mean()
        return r

    rx, ry = rangos(x), rangos(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ══════════════════════════════════════════════════════════ parte 1: exp06


def _rollout_exp06(args: tuple[str, int, int]) -> dict[str, Any]:
    """Un rollout de `exp06`, con su protocolo exacto, y los aciertos por memoria."""
    condicion, seed, n_steps = args
    fabrica, fuente = CONDICIONES_EXP06[condicion]
    s = rollout_exp06(seed, n_steps=n_steps, policy_factory=fabrica, surprise_source=fuente)
    n = len(s.rare_items)
    fila: dict[str, Any] = {"seed": seed, "n": n}
    for nombre in ("frontera", "FIFO"):
        f = make_factory(MEMORIAS[nombre], dim=DIM)
        r = t1_rare_retention(f, s, seed=seed, capacity=CAPACITY, dim=DIM)
        fila[nombre] = round(r.score * max(n, 1))
    return fila


def reanalizar_exp06(
    seeds: tuple[int, ...],
    n_steps: int = N_STEPS,
    *,
    n_jobs: int | None = None,
    condiciones: tuple[str, ...] = tuple(CONDICIONES_EXP06),
) -> dict[str, Any]:
    """Las condiciones de `exp06`, con el rollout como unidad estadística."""
    trabajos = [(c, s, n_steps) for c in condiciones for s in seeds]
    filas = _map(_rollout_exp06, trabajos, n_jobs)
    salida: dict[str, Any] = {}
    for c in condiciones:
        fc = [f for f, (cc, _, _) in zip(filas, trabajos, strict=True) if cc == c]
        n = np.array([f["n"] for f in fc])
        bloques = np.array([f["seed"] // ROLLOUTS_POR_BLOQUE for f in fc])
        por_mem = {
            m: rate_summary(np.array([f[m] for f in fc]), n, bloques, seed=1)
            for m in ("frontera", "FIFO")
        }
        dif = cluster_bootstrap_diff(
            np.array([f["frontera"] for f in fc]), np.array([f["FIFO"] for f in fc]), n, seed=2
        )
        salida[c] = {
            "n_rollouts": len(fc),
            "n_eventos_por_rollout": n.tolist(),
            "aciertos_por_rollout": {m: [f[m] for f in fc] for m in ("frontera", "FIFO")},
            "resultados": por_mem,
            "frontera_menos_fifo": dif,
        }
    return salida


# ════════════════════════════════════════════════════════ parte 2: la matriz


def make_policy(nombre: str, seed: int, n_actions: int) -> Any:
    """Política por nombre. `aleatoria` es None: uniforme con el generador del adaptador."""
    from ember.envs.agents import ForwardBiasedPolicy, NoisyPlanner, TabularQAgent

    if nombre == "aleatoria":
        return None
    if nombre == "sesgada":
        return ForwardBiasedPolicy(seed, n_actions)
    if nombre == "planificador":
        return NoisyPlanner(seed, n_actions)
    if nombre == "qlearning":
        return TabularQAgent(seed, n_actions)
    raise ValueError(f"política desconocida: {nombre!r}")


def retention_per_step(
    trace: Any,
    signal: np.ndarray,
    genotype: Genotype,
    consultar: np.ndarray,
    *,
    seed: int,
    capacity: int = CAPACITY,
) -> np.ndarray:
    """Escribe el rollout entero con `signal` como `pred_error` y consulta los pasos pedidos.

    Devuelve un booleano por paso (False donde no se consultó). Es
    `t1_rare_retention` con `read_every=0` (el protocolo de `exp06`) evaluado
    una sola vez para todas las definiciones de importancia a la vez: las
    escrituras no dependen de qué se etiquete como importante, y las lecturas
    finales no alteran la memoria para las siguientes (refuerzo 0 en las tres
    memorias, sin escrituras posteriores).
    """
    mem = PolicyMemory(
        dim=int(trace.keys.shape[1]), capacity=capacity, genotype=genotype, seed=seed
    )
    for t in range(len(trace)):
        mem.write(trace.keys[t], t, float(signal[t]))
    aciertos = np.zeros(len(trace), dtype=bool)
    for t in np.flatnonzero(consultar):
        r = mem.read(trace.keys[t])
        aciertos[t] = r.value == t and r.similarity >= HIT_SIMILARITY
    return aciertos


def evaluar_rollout(args: tuple[str, str, int, int]) -> dict[str, Any]:
    """Un rollout de la matriz: todas las (importancia, señal, memoria) sobre la misma experiencia."""
    from ember.envs.minigrid import MiniGridStreamAdapter
    from ember.envs.salience import (
        IMPORTANCE,
        SIGNALS,
        compute_signal,
        importance_mask,
        separation_auc,
    )

    env_key, politica, seed, n_steps = args
    adaptador = MiniGridStreamAdapter(ENTORNOS[env_key], dim=DIM, capacity=CAPACITY, seed=seed)
    # El número de acciones de MiniGrid es 7 en todos estos entornos.
    trace = adaptador.rollout_trace(n_steps, policy=make_policy(politica, seed, 7))

    mascaras = {imp: importance_mask(trace, imp) for imp in IMPORTANCE}
    union = np.zeros(len(trace), dtype=bool)
    for m in mascaras.values():
        union |= m

    salida: dict[str, Any] = {
        "seed": seed,
        "n": {imp: int(m.sum()) for imp, m in mascaras.items()},
        "n_episodios": int(trace.episode[-1]) + 1,
        "hits": {},
        "auc": {},
    }
    for sig in SIGNALS:
        valores = compute_signal(trace, sig, seed=seed)
        salida["auc"][sig] = {imp: separation_auc(valores, m) for imp, m in mascaras.items()}
        ac = retention_per_step(trace, valores, MEMORIAS["frontera"], union, seed=seed)
        salida["hits"][sig] = {imp: int(ac[m].sum()) for imp, m in mascaras.items()}
        if sig == SIGNALS[0]:
            for mem_nombre in MEMORIAS_SIN_SENAL:
                genotipo = MEMORIAS[mem_nombre]
                assert genotipo.strength.label in ("constant", "novelty"), mem_nombre
                ac = retention_per_step(trace, valores, genotipo, union, seed=seed)
                salida["hits"][mem_nombre] = {imp: int(ac[m].sum()) for imp, m in mascaras.items()}
    return salida


def _map(fn: Any, trabajos: list, n_jobs: int | None) -> list:
    if n_jobs == 1:
        return [fn(t) for t in trabajos]
    workers = n_jobs if (n_jobs and n_jobs > 0) else (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, trabajos, chunksize=4))


def resumir_celda(filas: list[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    """Resume los rollouts de un (entorno, política) en la matriz importancia × señal × memoria."""
    from ember.envs.salience import IMPORTANCE, SIGNALS

    bloques = np.array([f["seed"] // ROLLOUTS_POR_BLOQUE for f in filas])
    out: dict[str, Any] = {"n_rollouts": len(filas)}
    out["episodios_por_rollout"] = float(np.mean([f["n_episodios"] for f in filas]))
    for imp in IMPORTANCE:
        n = np.array([f["n"][imp] for f in filas])
        techo = float(np.minimum(n, CAPACITY).sum() / n.sum()) if n.sum() else None
        bloque_imp: dict[str, Any] = {
            "n_eventos": int(n.sum()),
            "eventos_por_rollout": float(n.mean()),
            "n_rollouts_con_eventos": int((n > 0).sum()),
            "techo": techo,
            "senales": {},
        }
        if n.sum() == 0:
            out[imp] = bloque_imp
            continue
        h = {m: np.array([f["hits"][m][imp] for f in filas]) for m in MEMORIAS_SIN_SENAL}
        for m in MEMORIAS_SIN_SENAL:
            bloque_imp[m] = rate_summary(h[m], n, bloques, seed=seed)
        for sig in SIGNALS:
            h["frontera"] = np.array([f["hits"][sig][imp] for f in filas])
            aucs = np.array(
                [f["auc"][sig][imp] for f in filas if f["auc"][sig][imp] is not None],
                dtype=np.float64,
            )
            auc_ci = per_rollout_mean(aucs, np.ones_like(aucs), seed=seed)
            celda: dict[str, Any] = {"frontera": rate_summary(h["frontera"], n, bloques, seed=seed)}
            celda["auc"] = {
                "media": float(aucs.mean()) if len(aucs) else None,
                "ci_low": auc_ci["ci_low"],
                "ci_high": auc_ci["ci_high"],
            }
            celda["frontera_menos_fifo"] = cluster_bootstrap_diff(
                h["frontera"], h["FIFO"], n, seed=seed + 1
            )
            celda["frontera_menos_sin_saliencia"] = cluster_bootstrap_diff(
                h["frontera"], h["sin_saliencia"], n, seed=seed + 2
            )
            bloque_imp["senales"][sig] = celda
        out[imp] = bloque_imp
    return out


def correr_matriz(
    entornos: tuple[str, ...] = tuple(ENTORNOS),
    politicas: tuple[str, ...] = POLITICAS,
    seeds: tuple[int, ...] = SEMILLAS,
    n_steps: int = N_STEPS,
    *,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Corre todos los rollouts y resume por (entorno, política)."""
    trabajos = [(e, p, s, n_steps) for e in entornos for p in politicas for s in seeds]
    filas = _map(evaluar_rollout, trabajos, n_jobs)
    salida: dict[str, Any] = {}
    for e in entornos:
        salida[e] = {}
        for p in politicas:
            fp = [f for f, t in zip(filas, trabajos, strict=True) if t[0] == e and t[1] == p]
            salida[e][p] = resumir_celda(fp)
    return salida


def resumen_matriz(matriz: dict[str, Any]) -> dict[str, Any]:
    """El resultado central: para cada (importancia, señal), agregado sobre entornos × políticas.

    Una celda (entorno, política) cuenta como "gana" si el IC por clúster de
    frontera − FIFO está entero por encima de 0, "pierde" si está entero por
    debajo, y "empata" si no. La correlación de Spearman entre la AUC de la
    señal y la ganancia, sobre todas las celdas, es la prueba directa de H2.
    """
    from ember.envs.salience import IMPORTANCE, SIGNALS

    resumen: dict[str, Any] = {}
    aucs_todas, ganancias_todas = [], []
    for imp in IMPORTANCE:
        resumen[imp] = {}
        for sig in SIGNALS:
            gana = pierde = empata = 0
            tasas_f, tasas_fifo, tasas_ss, difs, aucs = [], [], [], [], []
            for e in matriz:
                for p in matriz[e]:
                    bloque = matriz[e][p][imp]
                    celda = bloque["senales"].get(sig)
                    if celda is None:
                        continue
                    d = celda["frontera_menos_fifo"]
                    if d["ci_low"] > 0:
                        gana += 1
                    elif d["ci_high"] < 0:
                        pierde += 1
                    else:
                        empata += 1
                    tasas_f.append(celda["frontera"]["tasa"])
                    tasas_fifo.append(bloque["FIFO"]["tasa"])
                    tasas_ss.append(bloque["sin_saliencia"]["tasa"])
                    difs.append(d["dif"])
                    if celda["auc"]["media"] is not None:
                        aucs.append(celda["auc"]["media"])
                        aucs_todas.append(celda["auc"]["media"])
                        ganancias_todas.append(d["dif"])
            if not difs:
                resumen[imp][sig] = {"n_celdas": 0}
                continue
            resumen[imp][sig] = {
                "n_celdas": len(difs),
                "gana": gana,
                "empata": empata,
                "pierde": pierde,
                "tasa_frontera_media": float(np.mean(tasas_f)),
                "tasa_fifo_media": float(np.mean(tasas_fifo)),
                "tasa_sin_saliencia_media": float(np.mean(tasas_ss)),
                "ganancia_media": float(np.mean(difs)),
                "ganancia_min": float(np.min(difs)),
                "ganancia_max": float(np.max(difs)),
                "auc_media": float(np.mean(aucs)) if aucs else None,
            }
    resumen["spearman_auc_ganancia"] = spearman(np.array(aucs_todas), np.array(ganancias_todas))
    resumen["n_celdas_total"] = len(aucs_todas)
    return resumen


# ══════════════════════════════════════════════════════════════════ main


def main() -> int:
    t0 = time.time()
    with ExperimentRun("exp11_minigrid_extended") as run:
        run.set_seeds(SEMILLAS)
        run.note(
            "Unidad estadística: el rollout. IC por bootstrap de clúster sobre rollouts "
            f"({N_BOOT} remuestreos) y t entre {N_BLOQUES} bloques de "
            f"{ROLLOUTS_POR_BLOQUE} semillas (bloque = semilla // {ROLLOUTS_POR_BLOQUE}). "
            "Wilson se reporta solo para medir el efecto de diseño."
        )
        run.record(
            "config",
            {
                "dim": DIM,
                "capacity": CAPACITY,
                "n_steps": N_STEPS,
                "n_rollouts": len(SEMILLAS),
                "n_bloques": N_BLOQUES,
                "n_boot": N_BOOT,
                "entornos": ENTORNOS,
                "politicas": list(POLITICAS),
                "memorias": {k: g.label() for k, g in MEMORIAS.items()},
            },
        )

        print("[parte 1] exp06 con el rollout como unidad")
        original = reanalizar_exp06(SEMILLAS_EXP06)
        run.record("exp06_original", original)
        ampliado = reanalizar_exp06(SEMILLAS)
        run.record("exp06_ampliado", ampliado)
        for c, datos in ampliado.items():
            for m, r in datos["resultados"].items():
                print(
                    f"  {c:<22}{m:<9} {r['tasa']:.3f}  Wilson [{r['wilson_low']:.3f}, "
                    f"{r['wilson_high']:.3f}]  clúster [{r['cluster_low']:.3f}, "
                    f"{r['cluster_high']:.3f}]  deff={r['deff']}"
                )

        rec = original["aleatoria_recompensa"]["resultados"]["frontera"]
        run.note(
            f"exp06 (40 rollouts, aleatoria_recompensa, frontera): {rec['aciertos']}/"
            f"{rec['n_eventos']}; Wilson [{rec['wilson_low']:.3f}, {rec['wilson_high']:.3f}] "
            f"vs clúster [{rec['cluster_low']:.3f}, {rec['cluster_high']:.3f}]."
        )

        print("[parte 2] matriz entorno × política × importancia × señal")
        matriz = correr_matriz()
        run.record("matriz", matriz)
        resumen = resumen_matriz(matriz)
        run.record("resumen", resumen)
        run.note(
            f"Spearman(AUC, ganancia frontera−FIFO) sobre {resumen['n_celdas_total']} celdas: "
            f"{resumen['spearman_auc_ganancia']}."
        )
        run.note(f"duración total: {time.time() - t0:.0f} s")
        print(f"listo en {time.time() - t0:.0f} s")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
