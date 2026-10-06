"""exp08 · Saliencia imperfecta: ¿cuánto de la retención sale de la construcción?

La objeción (revisor 1 de BIP2026)
----------------------------------
En el flujo sintético de T1 la sorpresa de lo rutinario vive en [0, 0.35] y la
de lo importante en [0.65, 1]: la sorpresa es un clasificador perfecto de
importancia. Con esa señal, que desalojar la traza más débil conserve lo raro
es casi una consecuencia de cómo se armó el benchmark. El 14.41× de `exp01`
mide sobre todo que la señal no hace nada hasta que un desalojo la lee.

Qué hace este experimento
-------------------------
Degrada la señal de sorpresa con `ember.data.salience` a lo largo de cinco
ejes, cada uno en una grilla que arranca en la señal original:

- `overlap`: solapamiento de las distribuciones, con AUC nominal 0.999 → 0.5;
- `noise`: ruido gaussiano aditivo, σ = 0.1 → 2;
- `delay_partial` y `delay_lag`: la sorpresa llega `k` pasos tarde, a otra
  experiencia, con probabilidad `p` (k = 1, p = 0.25 → 1) o siempre (k = 1 → 20);
- `misleading`: la señal se invierte en una fracción 0.05 → 0.5 de los eventos;
- `overlap_distractors`: el eje `overlap` sobre un flujo donde un 25 % de lo
  rutinario se reemplaza por experiencias nuevas sin importancia. Existe
  porque la frontera usa `strength=both`, que suma la *novedad* a la sorpresa,
  y en el flujo original la novedad también es un clasificador perfecto por
  construcción (lo raro es lo único que no se repite). Sin este eje, una
  frontera robusta a AUC = 0.5 no diría nada sobre la sorpresa.

En cada condición mide el AUC empírico de la señal (para comparar ejes en la
misma escala) y evalúa, sobre T1 con el protocolo exacto de `exp01`:

(a) la frontera de `exp01` (fija, elegida sobre la señal limpia), sus hermanas
    que cambian solo la fuerza (`pred_error`, `novelty`, `constant`), el proxy
    FIFO, la mejor configuración sin saliencia (`strength=constant`) elegida
    *en cada condición* —sesgo a favor del baseline, conservador para EMBER—,
    y un oráculo que conoce las etiquetas (la arquitectura de la frontera con
    `strength=pred_error` y la sorpresa reemplazada por la etiqueta), como techo;
(b) las 576 configuraciones, para ver si la frontera cambia y si la interacción
    saliencia × desalojo (el 14.41×) sobrevive;
(c) el benchmark de arquitecturas de `exp04` (SDM, ENN, Spiking-SDM, FIFO).

Dos regímenes de flujo (`SETTINGS`): `main`, el flujo exacto de T1 en `exp01`
(r = 0.25), sobre todos los ejes; y `selection` (r = 2, diseño de `exp02`),
sobre los dos ejes de solapamiento. Las 576 corren en los ejes de solapamiento
de `main` (`BUSQUEDA_COMPLETA`); en el resto de las condiciones corren las 144
configuraciones sin saliencia y los genotipos fijos.

Métricas por corrida: recall de retención (la métrica T1 del paper: fracción de
los importantes que se recuperan con acierto al final), precisión de retención
(fracción de las trazas guardadas al final que son importantes), F1 sobre el
conjunto retenido, y la prevalencia como referencia de la precisión. Todo con
IC bootstrap al 95 % sobre semillas (la unidad independiente: cada semilla es
un flujo distinto), y diferencias pareadas por semilla.

Hipótesis, escrita antes de correr
----------------------------------
H1. La ventaja de la frontera es función de la discriminabilidad de la señal
    (AUC), no de que los rangos no se solapen: su recall decae suavemente con
    el AUC empírico, y supera al FIFO y a la mejor configuración sin saliencia
    mientras el AUC esté claramente por encima de 0.5.
H2. La interacción saliencia × desalojo decae con el AUC y tiende a 1× en
    AUC = 0.5 para `strength=pred_error`.
H3. A igual AUC empírico, los ejes dan recall parecido: el AUC resume la
    calidad de la señal.

Qué la refutaría
----------------
- Que la ventaja se derrumbe ya en AUC alto (≥ 0.9): el resultado dependía de
  la separación perfecta y la objeción es correcta.
- Que en AUC = 0.5 la frontera siga superando con holgura a la configuración
  sin saliencia sin distractores pero no con ellos: la ventaja viene del canal
  de novedad, no de la sorpresa, y la atribución del paper es incorrecta.
- Que ejes con el mismo AUC den recall muy distinto (por ejemplo, el retraso
  peor de lo que su AUC implica): el AUC no gobierna, y hay que reportar por eje.

Nada de lo anterior se ajustó después de ver los datos: las grillas, las
semillas y la fracción de distractores están fijadas en este archivo.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.policies import (
    Append,
    BothGated,
    Constant,
    MinStrength,
    NearestNeighbour,
    NoDecay,
    NoveltyGated,
    PredErrorGated,
)
from ember.data.salience import SignalCondition, degrade_stream, stream_auc
from ember.data.streams import Stream, StreamItem
from ember.data.synthetic import clustered_stream
from ember.experiment import ExperimentRun
from ember.memories import ARCHITECTURES
from ember.nas.engine import SearchRecord, SearchResults
from ember.nas.space import enumerate_space
from ember.nas.stats import bootstrap_ci, conditional_effect, variance_share
from ember.tasks.battery import t1_rare_retention
from experiments._common import CAPACITY, DIM, READ_EVERY, make_factory

NOMBRE = "exp08_imperfect_salience"
SEMILLAS = tuple(range(5))
"""Cinco flujos independientes por condición: el mínimo pedido. La máquina se
comparte con otros siete experimentos; más semillas no entraban en ~30 min."""


@dataclass(frozen=True, slots=True)
class StreamSetting:
    """Parámetros del flujo de T1 sobre el que se degrada la señal."""

    n_prototypes: int
    n_common: int
    n_rare: int

    @property
    def r(self) -> float:
        return self.n_prototypes / CAPACITY


SETTINGS: dict[str, StreamSetting] = {
    "main": StreamSetting(n_prototypes=5, n_common=300, n_rare=20),
    "selection": StreamSetting(n_prototypes=40, n_common=800, n_rare=10),
}
"""`main` es el flujo exacto de T1 en `exp01`/`exp04` (r = 0.25, 20 raros = C).

`selection` es el régimen r = 2 con el diseño de `exp02` (20 visitas por
prototipo, `n_rare = C // 2`). Se agregó después de una corrida reducida de
prueba, por dos razones que esa corrida hizo visibles y que se declaran acá en
vez de esconderlas: (1) en `main` hay 5 prototipos para 20 ranuras, y una
configuración con fusión y desalojo aleatorio, sin ninguna saliencia, ya
retiene ~0.76 de lo raro —la compresión resuelve la tarea sin señal—; (2) con
20 raros en 20 ranuras y la memoria llena, la precisión de retención es
idéntica al recall de conjunto por aritmética, así que no informa nada aparte.
En `selection` la fusión no puede liberar espacio y la precisión se separa
del recall.
"""

FRACCION_DISTRACTORES = 0.25
N_BOOT = 2000

METRICAS = ("recall", "recall_set", "precision", "f1", "n_retained")

FRONTIER = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=BothGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""La frontera de `exp01`: `nn + append + both + decay 1.0 + min_strength`."""

FRONTIER_PE = FRONTIER.with_axis("strength", PredErrorGated())
FRONTIER_NOVELTY = FRONTIER.with_axis("strength", NoveltyGated())
FRONTIER_CONSTANT = FRONTIER.with_axis("strength", Constant())
ORACLE = FRONTIER_PE
"""El oráculo es esta arquitectura con la sorpresa reemplazada por la etiqueta."""

GENOTIPOS_FIJOS: dict[str, Genotype] = {
    "frontier": FRONTIER,
    "frontier_pe": FRONTIER_PE,
    "frontier_novelty": FRONTIER_NOVELTY,
    "frontier_constant": FRONTIER_CONSTANT,
    "fifo": FIFO_GENOTYPE,
}

ARQUITECTURAS = ("SDM", "ENN", "SpikingSDM", "FIFO")
"""Las de `exp04` que pasan el gate de reconstrucción (el Spiking puro no lo pasa)."""

_CLEAN = SignalCondition()
_AUCS = (0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)

EJES: dict[str, tuple[SignalCondition, ...]] = {
    "overlap": (_CLEAN, *(SignalCondition(auc=a) for a in _AUCS)),
    "noise": (_CLEAN, *(SignalCondition(noise=s) for s in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0))),
    "delay_partial": (
        _CLEAN,
        *(SignalCondition(delay=1, delay_prob=p) for p in (0.25, 0.5, 0.75, 1.0)),
    ),
    "delay_lag": (_CLEAN, *(SignalCondition(delay=k) for k in (1, 2, 5, 20))),
    "misleading": (
        _CLEAN,
        *(SignalCondition(misleading=f) for f in (0.05, 0.1, 0.2, 0.3, 0.5)),
    ),
    "overlap_distractors": (
        SignalCondition(distractors=FRACCION_DISTRACTORES),
        *(
            SignalCondition(auc=a, distractors=FRACCION_DISTRACTORES)
            for a in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5)
        ),
    ),
}
"""Cada eje en orden de degradación creciente. Todos arrancan en su condición limpia."""

COMPARADORES = ("fifo", "no_salience_best", "no_salience_fixed")


def condiciones_unicas(ejes: dict[str, Sequence[SignalCondition]]) -> list[SignalCondition]:
    """Las condiciones de todos los ejes, sin repetir, en orden de aparición."""
    vistas: dict[str, SignalCondition] = {}
    for conds in ejes.values():
        for c in conds:
            vistas.setdefault(c.label(), c)
    return list(vistas.values())


# ══════════════════════════════════════════════════════════════ los flujos


@lru_cache(maxsize=1024)
def flujo(
    setting: StreamSetting, condition: SignalCondition, seed: int, oracle: bool = False
) -> Stream:
    """El flujo de T1 de `exp01` con la señal degradada. Cacheado por proceso.

    Con `oracle=True` la sorpresa se reemplaza por la etiqueta de importancia
    (1 lo importante, 0 lo rutinario): es la señal perfecta que define el techo.
    """
    base = clustered_stream(
        n_prototypes=setting.n_prototypes,
        capacity=CAPACITY,
        n_common=setting.n_common,
        n_rare=setting.n_rare,
        dim=DIM,
        seed=seed,
    )
    s = degrade_stream(base, condition, seed=seed)
    if not oracle:
        return s
    items = [
        StreamItem(key=it.key, value=it.value, pred_error=float(it.is_rare), is_rare=it.is_rare)
        for it in s.items
    ]
    return Stream(items=items, spec=s.spec, rare_items=[it for it in items if it.is_rare])


def prevalencia(stream: Stream) -> float:
    """Fracción de importantes en el flujo: la precisión de una retención al azar."""
    return len(stream.rare_items) / max(len(stream), 1)


# ═════════════════════════════════════════════════════════════ las métricas


def medir_retencion(factory, stream: Stream, seed: int, read_every: int = READ_EVERY) -> np.ndarray:
    """Recall, recall de conjunto, precisión, F1 y trazas retenidas de una corrida de T1.

    Corre `t1_rare_retention` tal cual —mismo protocolo que `exp01` y `exp04`,
    incluidas las lecturas intercaladas— y captura la memoria que construyó
    para leer qué quedó guardado al final.

    - `recall` es el puntaje de T1: fracción de los importantes que se recuperan
      con acierto (valor correcto y similitud ≥ 0.85).
    - `recall_set` y `precision` miran el conjunto de trazas al final: qué
      fracción de lo importante sigue guardado, y qué fracción de lo guardado
      es importante. `f1` combina esas dos.
    """
    capturadas: list[Any] = []

    def capturar(capacity: int, s: int):
        mem = factory(capacity, s)
        capturadas.append(mem)
        return mem

    res = t1_rare_retention(capturar, stream, seed=seed, read_every=read_every)
    mem = capturadas[0]
    importantes = {it.value for it in stream.rare_items}
    valores = list(mem.store.values)
    acertados = sum(v in importantes for v in valores)
    precision = acertados / len(valores) if valores else 0.0
    recall_set = acertados / max(len(importantes), 1)
    f1 = 2 * precision * recall_set / (precision + recall_set) if precision + recall_set else 0.0
    return np.array([res.score, recall_set, precision, f1, len(valores)], dtype=np.float64)


class _FabricaArquitectura:
    """`(capacity, seed) -> Memory` para una arquitectura registrada. Serializable."""

    def __init__(self, nombre: str) -> None:
        self.nombre = nombre

    def __call__(self, capacity: int, seed: int):
        return ARCHITECTURES[self.nombre](dim=DIM, capacity=capacity, seed=seed)


def _tarea(
    args: tuple[str, Any, StreamSetting, SignalCondition, tuple[int, ...]],
) -> tuple[str, str, str, list]:
    """Una unidad de trabajo del pool: un sistema de memoria en una condición."""
    tipo, sistema, setting, cond, seeds = args
    if tipo == "arch":
        factory = _FabricaArquitectura(sistema)
        clave = sistema
    else:
        factory = make_factory(sistema, dim=DIM)
        clave = sistema.label()
    oracle = tipo == "oracle"
    filas = [medir_retencion(factory, flujo(setting, cond, s, oracle), s).tolist() for s in seeds]
    return (tipo, clave, cond.label(), filas)


def evaluar_condiciones(
    condiciones: Sequence[SignalCondition],
    *,
    setting: StreamSetting = SETTINGS["main"],
    seeds: Sequence[int] = SEMILLAS,
    genotipos: Iterable[Genotype] | None = None,
    arquitecturas: Sequence[str] = ARQUITECTURAS,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Evalúa genotipos, arquitecturas y el oráculo en cada condición.

    Devuelve las matrices crudas (semilla × métrica) por sistema y condición,
    más el AUC empírico y la prevalencia por semilla. `genotipos=None` es el
    espacio completo de 576.
    """
    seeds = tuple(seeds)
    genos = list(enumerate_space()) if genotipos is None else list(genotipos)
    # Los genotipos fijos tienen que estar siempre, aunque se pase un subconjunto.
    etiquetas = {g.label() for g in genos}
    genos += [g for g in GENOTIPOS_FIJOS.values() if g.label() not in etiquetas]

    tareas: list[tuple[str, Any, StreamSetting, SignalCondition, tuple[int, ...]]] = []
    for c in condiciones:
        tareas += [("geno", g, setting, c, seeds) for g in genos]
        tareas += [("arch", a, setting, c, seeds) for a in arquitecturas]
        tareas.append(("oracle", ORACLE, setting, c, seeds))

    if n_jobs == 1:
        salidas = [_tarea(t) for t in tareas]
    else:
        trabajadores = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
        with ProcessPoolExecutor(max_workers=trabajadores) as pool:
            salidas = list(pool.map(_tarea, tareas, chunksize=16))

    crudo: dict[str, Any] = {"genotypes": {}, "architectures": {}, "oracle": {}}
    destino = {"geno": "genotypes", "arch": "architectures", "oracle": "oracle"}
    for tipo, clave, cond, filas in salidas:
        crudo[destino[tipo]].setdefault(clave, {})[cond] = np.array(filas)

    crudo["auc"] = {
        c.label(): [stream_auc(flujo(setting, c, s)) for s in seeds] for c in condiciones
    }
    crudo["prevalence"] = {
        c.label(): [prevalencia(flujo(setting, c, s)) for s in seeds] for c in condiciones
    }
    crudo["setting"] = setting
    crudo["genotype_objects"] = {g.label(): g for g in genos}
    crudo["seeds"] = seeds
    return crudo


# ═══════════════════════════════════════════════════════════════ el resumen


def _ic(valores: Sequence[float], seed: int = 0) -> dict[str, Any]:
    v = np.asarray(valores, dtype=np.float64)
    lo, hi = bootstrap_ci(v, n_boot=N_BOOT, seed=seed)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "ci": [lo, hi],
    }


def _resumen_metricas(m: np.ndarray) -> dict[str, Any]:
    return {nombre: _ic(m[:, j], seed=j) for j, nombre in enumerate(METRICAS)}


def _diferencia(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Diferencia pareada por semilla (a − b) en recall y F1, con IC bootstrap."""
    salida = {}
    for nombre in ("recall", "precision", "f1"):
        j = METRICAS.index(nombre)
        d = _ic(a[:, j] - b[:, j], seed=100 + j)
        d["better"] = bool(d["ci"][0] > 0.0)
        salida[nombre] = d
    return salida


def _mejor(labels: Sequence[str], matrices: dict[str, np.ndarray]) -> str:
    """El de mayor recall medio; desempata por etiqueta, como `run_search`."""
    return min(labels, key=lambda g: (-float(matrices[g][:, 0].mean()), g))


def _cociente_saliencia(
    matrices: dict[str, np.ndarray], genos: dict[str, Genotype], seeds: int
) -> dict[str, Any]:
    """El 14.41× de `exp01`: recall por compuerta de fuerza con `evict=min_strength, write=append`.

    Devuelve la media por opción y los cocientes contra `constant`, con IC
    bootstrap remuestreando semillas (todas las configuraciones a la vez, para
    conservar la correlación entre ellas dentro de cada flujo).
    """
    sub = {
        g: m
        for g, m in matrices.items()
        if genos[g].axis("evict") == "min_strength" and genos[g].axis("write") == "append"
    }
    por_opcion: dict[str, np.ndarray] = {}
    for g, m in sub.items():
        por_opcion.setdefault(genos[g].axis("strength"), []).append(m[:, 0])  # type: ignore[arg-type]
    if "constant" not in por_opcion:
        return {}
    por_opcion = {k: np.array(v) for k, v in por_opcion.items()}  # (n_genos, n_seeds)

    rng = np.random.default_rng(7)
    idx = rng.integers(0, seeds, size=(N_BOOT, seeds))
    # v[:, idx] tiene forma (configuraciones, remuestreos, semillas).
    boot = {k: v[:, idx].mean(axis=(0, 2)) for k, v in por_opcion.items()}

    medias = {k: float(v.mean()) for k, v in por_opcion.items()}
    base = medias["constant"]
    salida: dict[str, Any] = {"mean_by_strength": medias, "ratio_vs_constant": {}}
    for k in por_opcion:
        if k == "constant":
            continue
        punto = medias[k] / base if base > 0 else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            b = boot[k] / boot["constant"]
        b = b[np.isfinite(b)]
        ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b.size else None
        salida["ratio_vs_constant"][k] = {"mean": punto, "ci": ci}
    otras = [k for k in medias if k != "constant"]
    mejor = max(otras, key=lambda k: medias[k])
    salida["best_strength"] = mejor
    salida["ratio_best_vs_constant"] = salida["ratio_vs_constant"][mejor]["mean"]
    return salida


def _busqueda(matrices: dict[str, np.ndarray], genos: dict[str, Genotype], seeds: int) -> dict:
    """Frontera, rango de la frontera fija y descomposición de varianza, sobre recall."""
    registros = sorted(
        (
            SearchRecord(
                genotype=genos[g],
                scores={"recall": float(m[:, 0].mean())},
                mean=float(m[:, 0].mean()),
            )
            for g, m in matrices.items()
        ),
        key=lambda r: (-r.mean, r.genotype.label()),
    )
    res = SearchResults(records=registros)
    mejor = registros[0]
    opt, pes = res.rank_of(FRONTIER)
    salida: dict[str, Any] = {
        "n_genotypes": len(registros),
        "best": {
            "genotype": mejor.genotype.as_dict(),
            "label": mejor.genotype.label(),
            "recall": mejor.mean,
        },
        "frontier_rank": [opt, pes],
        "frontier_is_best": mejor.genotype == FRONTIER
        or abs(mejor.mean - res.score_of(FRONTIER)) <= 1e-9,
        "top5": [r.genotype.label() for r in registros[:5]],
    }
    share = variance_share(registros, metric="recall")
    salida["variance_share"] = {
        k: share[k] for k in ("strength", "evict", "write", "strength×evict") if k in share
    }
    salida["salience_interaction"] = _cociente_saliencia(matrices, genos, seeds)
    try:
        salida["salience_marginal"] = conditional_effect(
            registros, "strength", given={}, metric="recall"
        )
    except ValueError:
        salida["salience_marginal"] = {}
    return salida


def resumir(crudo: dict[str, Any], ejes: dict[str, Sequence[SignalCondition]]) -> dict[str, Any]:
    """Arma el JSON del experimento a partir de las matrices crudas."""
    seeds = crudo["seeds"]
    n = len(seeds)
    genos: dict[str, Genotype] = crudo["genotype_objects"]
    por_geno = crudo["genotypes"]
    conds = condiciones_unicas(ejes)

    constantes = [g for g, obj in genos.items() if obj.axis("strength") == "constant"]
    limpia = conds[0].label()
    fija_sin_saliencia = _mejor(constantes, {g: por_geno[g][limpia] for g in constantes})

    data: dict[str, Any] = {
        "conditions": {},
        "axes": {eje: [c.label() for c in cs] for eje, cs in ejes.items()},
        "genotypes": {k: g.label() for k, g in GENOTIPOS_FIJOS.items()}
        | {"oracle": ORACLE.label() + " (pe = etiqueta)", "no_salience_fixed": fija_sin_saliencia},
        "stream": asdict(crudo["setting"]) | {"r": crudo["setting"].r, "capacity": CAPACITY},
        "seeds": list(seeds),
        "n_seeds": n,
        "metrics": list(METRICAS),
    }

    for c in conds:
        lab = c.label()
        # No todas las condiciones corren el espacio completo: las que no, corren
        # las 144 configuraciones sin saliencia más los genotipos fijos.
        mats = {g: por_geno[g][lab] for g in genos if lab in por_geno[g]}
        completa = len(mats) >= 576
        sin_sal = _mejor(constantes, mats)
        ciegas = [g for g in mats if genos[g].axis("strength") in ("constant", "novelty")]
        ciega = _mejor(ciegas, mats)

        sistemas: dict[str, np.ndarray] = {k: mats[g.label()] for k, g in GENOTIPOS_FIJOS.items()}
        sistemas["no_salience_best"] = mats[sin_sal]
        sistemas["no_salience_fixed"] = mats[fija_sin_saliencia]
        if completa:
            sistemas["pe_blind_best"] = mats[ciega]
        sistemas["oracle"] = crudo["oracle"][ORACLE.label()][lab]

        entrada: dict[str, Any] = {
            "condition": asdict(c),
            "auc": _ic(crudo["auc"][lab], seed=50),
            "prevalence": float(np.mean(crudo["prevalence"][lab])),
            "policies": {k: _resumen_metricas(m) for k, m in sistemas.items()},
            "selected": {"no_salience_best": sin_sal}
            | ({"pe_blind_best": ciega} if completa else {}),
            "diffs": {
                f"{a}_vs_{b}": _diferencia(sistemas[a], sistemas[b])
                for a in ("frontier", "frontier_pe")
                for b in COMPARADORES
            },
            "search": _busqueda(mats, genos, n) if completa else None,
            "architectures": {
                a: _resumen_metricas(m[lab]) for a, m in crudo["architectures"].items()
            },
        }
        if "SDM" in crudo["architectures"] and "FIFO" in crudo["architectures"]:
            entrada["diffs"]["SDM_vs_FIFO"] = _diferencia(
                crudo["architectures"]["SDM"][lab], crudo["architectures"]["FIFO"][lab]
            )
        data["conditions"][lab] = entrada

    data["breakdown"] = quiebres(data, ejes)
    return data


def quiebres(data: dict[str, Any], ejes: dict[str, Sequence[SignalCondition]]) -> dict[str, Any]:
    """Por eje y comparador: hasta qué calidad de señal la frontera sigue ganando.

    `last_better` es la condición más degradada tal que ella *y todas las
    anteriores* del eje muestran una ventaja con IC pareado que excluye el 0;
    `first_not_better` es la siguiente. Se reportan con su AUC empírico, que es
    la escala común entre ejes.
    """
    salida: dict[str, Any] = {}
    for eje, conds in ejes.items():
        salida[eje] = {}
        for diff in data["conditions"][conds[0].label()]["diffs"]:
            ultimo, primero = None, None
            for c in conds:
                d = data["conditions"][c.label()]["diffs"][diff]["recall"]
                if d["better"]:
                    ultimo = c.label()
                else:
                    primero = c.label()
                    break
            salida[eje][diff] = {
                "last_better": ultimo,
                "auc_last_better": (
                    data["conditions"][ultimo]["auc"]["mean"] if ultimo is not None else None
                ),
                "first_not_better": primero,
                "auc_first_not_better": (
                    data["conditions"][primero]["auc"]["mean"] if primero is not None else None
                ),
            }
    return salida


# ════════════════════════════════════════════════════════════════════ main


def _imprimir(data: dict[str, Any]) -> None:
    cols = ("frontier", "frontier_pe", "no_salience_best", "fifo", "oracle")
    for eje, labels in data["axes"].items():
        print(f"\n── eje {eje} ── recall (precisión)")
        print(f"{'condición':<22}{'AUC':>6}" + "".join(f"{c:>18}" for c in cols) + f"{'SDM':>14}")
        for lab in labels:
            e = data["conditions"][lab]
            fila = f"{lab:<22}{e['auc']['mean']:>6.3f}"
            for c in cols:
                p = e["policies"][c]
                fila += f"{p['recall']['mean']:>11.3f} ({p['precision']['mean']:.2f})"
            sdm = e["architectures"].get("SDM")
            if sdm:
                fila += f"{sdm['recall']['mean']:>7.3f} ({sdm['precision']['mean']:.2f})"
            print(fila)


BUSQUEDA_COMPLETA: dict[str, tuple[str, ...]] = {"main": ("overlap", "overlap_distractors")}
"""Dónde se corren las 576 (parte b). En el resto de las condiciones se corren las
144 sin saliencia (para elegir la mejor) más los genotipos fijos: la curva
principal no necesita más y el presupuesto de cómputo no daba para todo."""

EJES_POR_SETTING: dict[str, dict[str, tuple[SignalCondition, ...]]] = {
    "main": EJES,
    "selection": {k: EJES[k] for k in ("overlap", "overlap_distractors")},
}
"""En `selection` cada corrida cuesta ~3× más; se corren los ejes de la curva
principal (solapamiento, con y sin distractores)."""


def correr(
    ejes_por_setting: dict[str, dict[str, tuple[SignalCondition, ...]]] = EJES_POR_SETTING,
    *,
    seeds: Sequence[int] = SEMILLAS,
    genotipos: Iterable[Genotype] | None = None,
    arquitecturas: Sequence[str] = ARQUITECTURAS,
    n_jobs: int = -1,
    log=print,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evalúa y resume cada régimen de flujo. Devuelve `(datos, crudos)`."""
    todos = list(enumerate_space()) if genotipos is None else list(genotipos)
    sin_saliencia = [g for g in todos if g.axis("strength") == "constant"]
    datos: dict[str, Any] = {}
    crudos: dict[str, Any] = {}
    for nombre, ejes in ejes_por_setting.items():
        conds = condiciones_unicas(ejes)
        completas = condiciones_unicas(
            {e: cs for e, cs in ejes.items() if e in BUSQUEDA_COMPLETA.get(nombre, ())}
        )
        etiquetas = {c.label() for c in completas}
        reducidas = [c for c in conds if c.label() not in etiquetas]
        t0 = time.time()
        crudo: dict[str, Any] = {}
        for grupo, genos in ((completas, todos), (reducidas, sin_saliencia)):
            if not grupo:
                continue
            parcial = evaluar_condiciones(
                grupo,
                setting=SETTINGS[nombre],
                seeds=seeds,
                genotipos=genos,
                arquitecturas=arquitecturas,
                n_jobs=n_jobs,
            )
            crudo = _fusionar(crudo, parcial)
        log(f"{nombre}: {len(conds)} condiciones evaluadas en {time.time() - t0:.0f} s")
        datos[nombre] = resumir(crudo, ejes)
        crudos[nombre] = crudo
    return datos, crudos


def _fusionar(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Une dos salidas crudas de `evaluar_condiciones` sobre condiciones disjuntas."""
    if not a:
        return b
    for clave in ("genotypes", "architectures", "oracle"):
        for sistema, por_cond in b[clave].items():
            a[clave].setdefault(sistema, {}).update(por_cond)
    for clave in ("auc", "prevalence", "genotype_objects"):
        a[clave].update(b[clave])
    return a


def main() -> int:
    from pathlib import Path

    from ember.figures import figura_saliencia_imperfecta

    with ExperimentRun(NOMBRE) as run:
        run.set_seeds(SEMILLAS)
        run.log(
            f"576 genotipos + {len(ARQUITECTURAS)} arquitecturas + oráculo, "
            f"{len(SEMILLAS)} semillas, regímenes {list(EJES_POR_SETTING)}"
        )
        datos, crudos = correr(log=run.log)
        run.record("settings", datos)
        for nombre, data in datos.items():
            print(f"\n════ régimen {nombre} · {data['stream']} ════")
            _imprimir(data)

        # Chequeo de continuidad con exp01: la condición limpia con sus semillas.
        limpia = crudos["main"]["genotypes"][FRONTIER.label()]["clean"]
        rec_exp01 = float(limpia[:3, 0].mean())
        run.record("check_exp01_frontier_recall_seeds012", rec_exp01)
        if abs(rec_exp01 - 0.9833333333333334) > 1e-9:
            run.note(f"ATENCIÓN: la frontera limpia no reproduce exp01 ({rec_exp01:.4f})")
        run.note(
            f"Recall de la frontera en la condición limpia de main con semillas 0-2: "
            f"{rec_exp01:.4f} (exp01 reporta 0.9833 de rare_retention)."
        )
        run.note(
            "no_salience_best y pe_blind_best se eligen en cada condición sobre las mismas "
            "semillas que se reportan: sesgo optimista a favor del baseline, conservador "
            "para EMBER. no_salience_fixed se elige una vez, en la condición limpia."
        )
        run.note(
            "precision/recall_set/f1 miran las trazas guardadas al final del flujo; recall es "
            "el puntaje de T1 (recuperación con acierto). En main (20 raros, C = 20, memoria "
            "llena) la precisión es idéntica a recall_set por aritmética."
        )
        run.note(
            "El régimen selection se agregó tras una corrida reducida de prueba; ver el "
            "docstring de SETTINGS."
        )

        ruta = figura_saliencia_imperfecta(
            datos, Path("paper/figures/fig_exp08_imperfect_salience.pdf")
        )
        run.log(f"figura → {ruta}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
