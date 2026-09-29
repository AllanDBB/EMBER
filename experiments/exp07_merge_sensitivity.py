"""exp07 · Sensibilidad al umbral de fusión y a los parámetros del camino de escritura.

Qué piden los revisores
-----------------------
El umbral de fusión por coseno está fijo en 0.85 (`Merge.threshold`) y decide el
resultado de CIFAR-100: ahí la similitud intra-prototipo mediana es ~0.40 y la
fusión casi nunca se dispara, así que el eje de escritura queda inerte y no hay
régimen de compresión (R1, R2). Piden un análisis de sensibilidad a ese umbral y
a los demás umbrales del camino de escritura.

Qué hay en el camino de escritura (`PolicyMemory.write`)
--------------------------------------------------------
No existe un umbral de admisión: toda experiencia se escribe (el "admission
gate" del paper es el gate de reconstrucción de `exp04`, un veredicto de
evaluación, no un parámetro de la memoria). Los únicos parámetros del camino de
escritura son:

- `Merge.threshold` (0.85): decide si la escritura consolida o crea traza.
- `gain` de las compuertas de fuerza (`NoveltyGated`, `PredErrorGated`,
  `BothGated`; 2.0): escala la fuerza inicial, que es lo que suma una fusión y
  lo que lee el desalojo por mínima fuerza. Se barre en 1D.

`HIT_SIMILARITY` (0.85) es un umbral del *lector* de la tarea, no de la memoria,
y no se toca: sobre una traza rara guardada tal cual la similitud es 1.

Hipótesis y qué las refutaría (escrito antes de correr)
-------------------------------------------------------
H1 (sintético). El cruce de régimen escritura→desalojo cerca de r ≈ 1 es robusto
al umbral mientras el umbral quede entre la similitud entre prototipos (coseno
de vectores aleatorios en R^32, ~0 ± 0.18) y la intra-prototipo (~0.93). Se
predice: cruce en r ∈ [0.5, 1] para umbral ∈ [~0.5, 0.90]; el cruce desaparece
para umbral ≥ 0.95 (la fusión deja de dispararse) y se degrada para umbral
≤ 0.4 (la fusión empieza a absorber eventos raros en trazas comunes). Lo
refutaría un cruce que se mueva de forma monótona con el umbral dentro de la
banda "segura", o que no exista para umbrales intermedios.

H2 (CIFAR-100). Las distribuciones de similitud intra-clase e inter-clase se
solapan mucho (mediana intra ~0.38 vs. percentil 95 inter ~0.42), así que se
predice que **no existe** un umbral que haga alcanzable la fusión sin colapsar
clases distintas: todo umbral que fusione una fracción sustancial de las
visitas tendrá pureza de fusión baja y absorberá eventos raros, y la escritura
no dominará en ningún r. Lo refutaría un umbral con (a) tasa de fusión
sustancial, (b) pureza alta (≥ 0.9) y (c) η²_write > η²_evict en r bajo —
es decir, que el régimen de compresión reaparezca en CIFAR-100 solo con elegir
el umbral. Si eso pasa, la "precondición" que enuncia el paper (la
consolidación requiere observaciones repetidas similares) se reduce a una
cuestión de calibración y hay que decirlo.

Diseño
------
Mismo protocolo que `exp05` (T1, C = 20, 20 visitas por prototipo, n_rare =
C/2, lectura cada 5 escrituras) con dos cambios: más ratios para localizar el
cruce, y η² por semilla (como `exp02`) para tener IC bootstrap sobre semillas.
El umbral se varía construyendo los genotipos de fusión con `Merge(threshold=t)`;
el espacio de búsqueda (`SEARCH_SPACE`) no se toca. Los 288 genotipos `append`
no dependen del umbral y se evalúan una sola vez por (dominio, r, semilla).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.memory import PolicyMemory
from ember.core.policies import (
    Append,
    BothGated,
    Merge,
    MinStrength,
    NearestNeighbour,
    NoDecay,
    NoveltyGated,
    PredErrorGated,
)
from ember.core.types import unit
from ember.data.embeddings import EmbeddingBank, embedding_stream
from ember.data.streams import Stream
from ember.data.synthetic import clustered_stream
from ember.experiment import ExperimentRun
from ember.nas.engine import SearchRecord
from ember.nas.space import enumerate_space
from ember.nas.stats import bootstrap_ci, eta_squared
from ember.tasks.battery import t1_rare_retention
from experiments._common import READ_EVERY

CACHE = Path("data/cache")
BANCO = CACHE / "cifar100_resnet18_d32.npz"
DOMINIOS = ("synthetic", "cifar100")
CAPACIDAD = 20
RATIOS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SEMILLAS = (0, 1, 2, 3, 4)
UMBRALES = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97)
"""Grilla del umbral de fusión.

Cubre desde la cola alta de la similitud inter-clase de CIFAR-100 (p95 ≈ 0.42)
hasta por encima de la similitud intra-prototipo sintética (~0.93), donde la
fusión sintética se apaga. 0.85 es el valor por defecto del espacio.
"""
UMBRAL_POR_DEFECTO = Merge().threshold
VISITAS_POR_PROTOTIPO = 20

GANANCIAS = (0.5, 1.0, 2.0, 4.0, 8.0)
"""Barrido 1D de la ganancia de las compuertas de fuerza (por defecto 2.0)."""
RATIOS_GANANCIA = (0.5, 1.0, 2.0)

FRONTIER_GENOTYPE = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=BothGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""El genotipo de frontera de `exp01` (el mismo que reconstruye `exp06`)."""


# ══════════════════════════════════════════════════════════════ construcción


def clave_umbral(t: float) -> str:
    """Clave sin puntos para `\\result{}`: 0.85 → `t085`."""
    return f"t{int(round(t * 100)):03d}"


def clave_ratio(r: float) -> str:
    """Clave sin puntos para `\\result{}`: 0.25 → `r0p25`."""
    return "r" + f"{r:g}".replace(".", "p")


def construir_stream(
    dominio: str, ratio: float, seed: int, *, capacity: int, banco: EmbeddingBank | None
) -> Stream:
    """El mismo flujo que construye `exp05` para una celda y una semilla."""
    k = max(1, int(round(ratio * capacity)))
    n_common = VISITAS_POR_PROTOTIPO * k
    n_rare = max(5, capacity // 2)
    if dominio == "synthetic":
        dim = banco.dim if banco is not None else 32
        return clustered_stream(
            n_prototypes=k,
            capacity=capacity,
            n_common=n_common,
            n_rare=n_rare,
            dim=dim,
            seed=seed,
        )
    if banco is None:
        raise ValueError("el dominio cifar100 necesita el banco de embeddings")
    return embedding_stream(
        banco, n_prototypes=k, capacity=capacity, n_common=n_common, n_rare=n_rare, seed=seed
    )


def genotipos_append() -> list[Genotype]:
    """Los 288 genotipos que crean traza siempre. No dependen del umbral."""
    return [g for g in enumerate_space() if isinstance(g.write, Append)]


def genotipos_merge(umbral: float) -> list[Genotype]:
    """Los 288 genotipos de fusión del espacio, con el umbral reemplazado.

    La etiqueta del eje sigue siendo `merge`, así que η² agrupa igual que en el
    espacio original; con `umbral = 0.85` los genotipos son idénticos a los de
    `SEARCH_SPACE`.
    """
    return [
        g.with_axis("write", Merge(threshold=umbral))
        for g in enumerate_space()
        if isinstance(g.write, Merge)
    ]


def con_ganancia(g: Genotype, gain: float) -> Genotype:
    """Reemplaza la ganancia de la compuerta de fuerza; `Constant` no tiene."""
    s = g.strength
    if isinstance(s, (NoveltyGated, PredErrorGated, BothGated)):
        return g.with_axis("strength", type(s)(gain=gain))
    return g


# ══════════════════════════════════════════════════════════════ evaluación


def puntaje_t1(genotype: Genotype, stream: Stream, seed: int) -> float:
    """Retención de eventos raros (T1) con el protocolo de `exp05`."""
    dim = stream.spec.dim

    def factory(capacity: int, s: int) -> PolicyMemory:
        return PolicyMemory(dim=dim, capacity=capacity, genotype=genotype, seed=s)

    return t1_rare_retention(factory, stream, seed=seed, read_every=READ_EVERY, dim=dim).score


def _evaluar_bloque(args: tuple[Any, Stream, int, list[Genotype]]) -> tuple[Any, list[float]]:
    """Worker: evalúa una lista de genotipos sobre un flujo. Nivel de módulo por pickle."""
    etiqueta, stream, seed, genotipos = args
    return etiqueta, [puntaje_t1(g, stream, seed) for g in genotipos]


def _correr_bloques(tareas: list, n_jobs: int) -> dict[Any, list[float]]:
    """Corre los bloques en paralelo, los más largos primero para balancear carga."""
    tareas = sorted(tareas, key=lambda t: -len(t[1]) * len(t[3]))
    if n_jobs == 1:
        return dict(_evaluar_bloque(t) for t in tareas)
    trabajadores = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    with ProcessPoolExecutor(max_workers=trabajadores) as pool:
        return dict(pool.map(_evaluar_bloque, tareas, chunksize=1))


def _registros(genotipos: list[Genotype], puntajes: list[float]) -> list[SearchRecord]:
    return [
        SearchRecord(genotype=g, scores={"rare_retention": p}, mean=p)
        for g, p in zip(genotipos, puntajes, strict=True)
    ]


# ═══════════════════════════════════════════ auditoría del camino de escritura


class AuditedMemory(PolicyMemory):
    """`PolicyMemory` que registra con quién se fusiona cada escritura.

    No cambia el comportamiento: consulta la misma política de ruteo sobre el
    mismo estado de claves que va a consultar `PolicyMemory.write` (el reloj y
    el decaimiento que corren antes no tocan las claves), y después delega.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fusiones: list[tuple[Any, Any]] = []
        """Pares (valor entrante, valor de la traza que lo absorbe)."""
        self.n_raros_escritos = 0

    def write(self, key: Any, value: Any, pred_error: float = 0.5) -> None:
        if es_raro(value):
            self.n_raros_escritos += 1
        objetivo = self.genotype.write.route(self.store, unit(key))
        if objetivo is not None:
            self.fusiones.append((value, self.store.values[objetivo]))
        super().write(key, value, pred_error)


def es_raro(value: Any) -> bool:
    """Los eventos raros llevan valores `rareN`; los comunes, la etiqueta de su prototipo."""
    return isinstance(value, str) and value.startswith("rare")


def resumen_fusiones(mem: AuditedMemory) -> dict[str, float | int | None]:
    """Tasa de fusión, pureza y absorción de raros de una memoria auditada.

    Una fusión es **pura** si une una experiencia con una traza del mismo
    prototipo (o clase). Un evento raro nunca se fusiona puramente: su valor es
    único, así que fusionarlo es perderlo en el acto.
    """
    n = len(mem.fusiones)
    puras = sum(1 for v, w in mem.fusiones if v == w)
    raros = sum(1 for v, _ in mem.fusiones if es_raro(v))
    return {
        "n_escrituras": mem.n_writes,
        "n_fusiones": n,
        "tasa_fusion": n / max(mem.n_writes, 1),
        "pureza": (puras / n) if n else None,
        "raros_absorbidos": raros / max(mem.n_raros_escritos, 1),
    }


def auditar_sin_presion(stream: Stream, umbral: float, seed: int) -> dict[str, Any]:
    """Consolidación del flujo común sin presión de capacidad (como `exp05`).

    Devuelve `K_efectivo` (trazas que deja la rutina cuando hay lugar para
    todas), la tasa de fusión y la pureza de las fusiones. El ruteo solo mira
    las claves, así que el resultado no depende de los otros ejes.
    """
    comunes = [it for it in stream if not it.is_rare]
    g = FRONTIER_GENOTYPE.with_axis("write", Merge(threshold=umbral))
    mem = AuditedMemory(dim=stream.spec.dim, capacity=len(comunes) + 1, genotype=g, seed=seed)
    for it in comunes:
        mem.write(it.key, it.value, it.pred_error)
    r = resumen_fusiones(mem)
    r["k_efectivo"] = len(mem)
    return r


def auditar_bajo_presion(stream: Stream, umbral: float, seed: int) -> dict[str, Any]:
    """El protocolo T1 completo con la frontera en modo fusión, auditado."""
    g = FRONTIER_GENOTYPE.with_axis("write", Merge(threshold=umbral))
    creadas: list[AuditedMemory] = []

    def factory(capacity: int, s: int) -> AuditedMemory:
        m = AuditedMemory(dim=stream.spec.dim, capacity=capacity, genotype=g, seed=s)
        creadas.append(m)
        return m

    res = t1_rare_retention(factory, stream, seed=seed, read_every=READ_EVERY, dim=stream.spec.dim)
    out = resumen_fusiones(creadas[0])
    out["puntaje"] = res.score
    return out


def distribucion_similitudes(streams: list[Stream]) -> dict[str, dict[str, float]]:
    """Cuantiles del coseno intra-prototipo, inter-prototipo y raro→común más cercano.

    Es lo que fija la grilla: un umbral útil tiene que caer por encima de lo
    inter y por debajo de lo intra, y el raro→común mide cuándo empieza a
    absorber eventos raros.
    """
    intra: list[float] = []
    inter: list[float] = []
    raro: list[float] = []
    for st in streams:
        comunes = [it for it in st if not it.is_rare]
        k = np.stack([it.key for it in comunes])
        v = np.array([it.value for it in comunes])
        s = k @ k.T
        i, j = np.triu_indices(len(comunes), k=1)
        mismo = v[i] == v[j]
        intra.extend(s[i, j][mismo].tolist())
        inter.extend(s[i, j][~mismo].tolist())
        for it in st.rare_items:
            raro.append(float((k @ it.key).max()))

    def q(x: list[float]) -> dict[str, float]:
        p = np.percentile(np.asarray(x), [5, 25, 50, 75, 95])
        return dict(zip(("p05", "p25", "p50", "p75", "p95"), map(float, p), strict=True))

    return {"intra": q(intra), "inter": q(inter), "raro_max_comun": q(raro)}


# ═════════════════════════════════════════════════════════════════ análisis


def _media_ci(x: list[float]) -> dict[str, Any]:
    return {
        "media": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "ci": list(bootstrap_ci(x, seed=0)),
    }


def localizar_cruce(rs: list[float], w: list[float], e: list[float]) -> float | None:
    """Primer r donde la escritura deja de dominar; media geométrica del par que cruza.

    `None` si la escritura nunca domina antes de ceder (o nunca cede).
    """
    orden = np.argsort(rs)
    for a, b in zip(orden, orden[1:], strict=False):
        if w[a] > e[a] and w[b] <= e[b]:
            return float(np.sqrt(rs[a] * rs[b]))
    return None


def resumir_cruce(celdas: dict[str, dict], campo_r: str) -> dict[str, Any]:
    """Cruce sobre la media de semillas y cruce por semilla con su IC bootstrap."""
    cs = list(celdas.values())
    rs = [c[campo_r] for c in cs]
    medio = localizar_cruce(rs, [c["eta2_write"] for c in cs], [c["eta2_evict"] for c in cs])
    n_sem = len(cs[0]["eta2_write_por_semilla"])
    por_semilla = []
    for s in range(n_sem):
        r_s = [c[f"{campo_r}_por_semilla"][s] for c in cs] if campo_r != "r" else rs
        x = localizar_cruce(
            r_s,
            [c["eta2_write_por_semilla"][s] for c in cs],
            [c["eta2_evict_por_semilla"][s] for c in cs],
        )
        por_semilla.append(x)
    validos = [x for x in por_semilla if x is not None]
    return {
        "cruce": medio,
        "cruce_por_semilla": por_semilla,
        "n_semillas_con_cruce": len(validos),
        "cruce_ci": list(bootstrap_ci(validos, seed=0)) if validos else None,
    }


def barrer_umbral(
    *,
    dominios: tuple[str, ...] = DOMINIOS,
    umbrales: tuple[float, ...] = UMBRALES,
    ratios: tuple[float, ...] = RATIOS,
    seeds: tuple[int, ...] = SEMILLAS,
    capacity: int = CAPACIDAD,
    banco: EmbeddingBank | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, Any]:
    """Barrido principal: una celda por (dominio, umbral, r), η² por semilla."""
    streams = {
        (d, r, s): construir_stream(d, r, s, capacity=capacity, banco=banco)
        for d in dominios
        for r in ratios
        for s in seeds
    }

    g_app = genotipos_append()
    g_mer = {t: genotipos_merge(t) for t in umbrales}
    tareas: list = []
    for (d, r, s), st in streams.items():
        tareas.append(((d, r, s, None), st, s, g_app))
        for t in umbrales:
            tareas.append(((d, r, s, t), st, s, g_mer[t]))
    puntajes = _correr_bloques(tareas, n_jobs)

    i_front = g_app.index(FRONTIER_GENOTYPE)
    i_fifo = g_app.index(FIFO_GENOTYPE)
    frontera_merge = FRONTIER_GENOTYPE.with_axis("write", Merge())

    salida: dict[str, Any] = {}
    for d in dominios:
        muestra = [streams[(d, 1.0 if 1.0 in ratios else ratios[-1], s)] for s in seeds]
        dom: dict[str, Any] = {"similitudes": distribucion_similitudes(muestra), "umbrales": {}}
        for t in umbrales:
            i_fm = [g.with_axis("write", Merge()) for g in g_mer[t]].index(frontera_merge)
            celdas: dict[str, dict] = {}
            for r in ratios:
                ew, ee, front, fifo, fm = [], [], [], [], []
                por_genotipo = []
                sin, con = [], []
                for s in seeds:
                    pa = puntajes[(d, r, s, None)]
                    pm = puntajes[(d, r, s, t)]
                    recs = _registros(g_app, pa) + _registros(g_mer[t], pm)
                    ew.append(eta_squared(recs, "write", metric="rare_retention"))
                    ee.append(eta_squared(recs, "evict", metric="rare_retention"))
                    front.append(pa[i_front])
                    fifo.append(pa[i_fifo])
                    fm.append(pm[i_fm])
                    por_genotipo.append(pm)
                    st = streams[(d, r, s)]
                    sin.append(auditar_sin_presion(st, t, s))
                    con.append(auditar_bajo_presion(st, t, s))

                # La mejor configuración con fusión se elige sobre la media de
                # semillas, y su IC sale de sus puntajes por semilla: elegir el
                # máximo por semilla sesgaría hacia arriba.
                medias = np.mean(np.array(por_genotipo), axis=0)
                i_best = int(np.argmax(medias))
                best = [float(p[i_best]) for p in por_genotipo]

                k_ef = [a["k_efectivo"] for a in sin]
                purezas = [a["pureza"] for a in con if a["pureza"] is not None]
                n_fus = sum(a["n_fusiones"] for a in con)
                n_pur = sum(round(a["pureza"] * a["n_fusiones"]) for a in con if a["n_fusiones"])
                purezas_sin = [a["pureza"] for a in sin if a["pureza"] is not None]

                celda = {
                    "umbral": t,
                    "ratio": r,
                    "k": max(1, int(round(r * capacity))),
                    "capacity": capacity,
                    "r": r,
                    "eta2_write": float(np.mean(ew)),
                    "eta2_write_ci": list(bootstrap_ci(ew, seed=0)),
                    "eta2_write_sd": float(np.std(ew, ddof=1)) if len(ew) > 1 else 0.0,
                    "eta2_write_por_semilla": ew,
                    "eta2_evict": float(np.mean(ee)),
                    "eta2_evict_ci": list(bootstrap_ci(ee, seed=0)),
                    "eta2_evict_sd": float(np.std(ee, ddof=1)) if len(ee) > 1 else 0.0,
                    "eta2_evict_por_semilla": ee,
                    "dominante": "write" if np.mean(ew) > np.mean(ee) else "evict",
                    "tasa_fusion": _media_ci([a["tasa_fusion"] for a in sin]),
                    "pureza_sin_presion": float(np.mean(purezas_sin)) if purezas_sin else None,
                    "k_efectivo": _media_ci(k_ef),
                    "r_efectivo": float(np.mean(k_ef)) / capacity,
                    "r_efectivo_por_semilla": [k / capacity for k in k_ef],
                    "tasa_fusion_presion": _media_ci([a["tasa_fusion"] for a in con]),
                    "pureza_fusion": float(np.mean(purezas)) if purezas else None,
                    "pureza_fusion_agregada": (n_pur / n_fus) if n_fus else None,
                    "raros_absorbidos": _media_ci([a["raros_absorbidos"] for a in con]),
                    "puntaje_frontera": _media_ci(front),
                    "puntaje_fifo": _media_ci(fifo),
                    "puntaje_frontera_merge": _media_ci(fm),
                    "puntaje_mejor_merge": _media_ci(best),
                    "mejor_merge_genotipo": g_mer[t][i_best].label(),
                }
                celdas[clave_ratio(r)] = celda
                if verbose:
                    print(
                        f"  [{d:<9}] t={t:.2f} r={r:<5} η²w={celda['eta2_write']:.3f}"
                        f" η²e={celda['eta2_evict']:.3f} fus={celda['tasa_fusion']['media']:.2f}"
                        f" pur={celda['pureza_fusion'] if celda['pureza_fusion'] is None else round(celda['pureza_fusion'], 2)}"
                        f" rar_abs={celda['raros_absorbidos']['media']:.2f}"
                        f" r_ef={celda['r_efectivo']:.2f}"
                        f" front={celda['puntaje_frontera']['media']:.2f}"
                        f" best_m={celda['puntaje_mejor_merge']['media']:.2f}",
                        flush=True,
                    )

            nominal = resumir_cruce(celdas, "r")
            efectivo = resumir_cruce(celdas, "r_efectivo")
            r_min = clave_ratio(min(ratios))
            dom["umbrales"][clave_umbral(t)] = {
                "umbral": t,
                "celdas": celdas,
                "cruce_r_nominal": nominal["cruce"],
                "cruce_r_nominal_ci": nominal["cruce_ci"],
                "cruce_r_nominal_por_semilla": nominal["cruce_por_semilla"],
                "n_semillas_con_cruce": nominal["n_semillas_con_cruce"],
                "cruce_r_efectivo": efectivo["cruce"],
                "cruce_r_efectivo_ci": efectivo["cruce_ci"],
                "escritura_domina_en_r_min": celdas[r_min]["dominante"] == "write",
                "eta2_write_r_min": celdas[r_min]["eta2_write"],
                "eta2_evict_r_min": celdas[r_min]["eta2_evict"],
                "tasa_fusion_r_min": celdas[r_min]["tasa_fusion"]["media"],
                "pureza_fusion_r_min": celdas[r_min]["pureza_fusion"],
                "raros_absorbidos_r_min": celdas[r_min]["raros_absorbidos"]["media"],
            }
        salida[d] = dom
    return salida


def barrer_ganancia(
    *,
    dominios: tuple[str, ...] = DOMINIOS,
    ganancias: tuple[float, ...] = GANANCIAS,
    ratios: tuple[float, ...] = RATIOS_GANANCIA,
    seeds: tuple[int, ...] = SEMILLAS,
    capacity: int = CAPACIDAD,
    banco: EmbeddingBank | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Barrido 1D de la ganancia de las compuertas de fuerza, umbral por defecto."""
    streams = {
        (d, r, s): construir_stream(d, r, s, capacity=capacity, banco=banco)
        for d in dominios
        for r in ratios
        for s in seeds
    }
    espacio = list(enumerate_space())
    por_g = {x: [con_ganancia(g, x) for g in espacio] for x in ganancias}
    tareas = [
        ((d, r, s, x), st, s, por_g[x]) for (d, r, s), st in streams.items() for x in ganancias
    ]
    puntajes = _correr_bloques(tareas, n_jobs)
    i_front = espacio.index(FRONTIER_GENOTYPE)

    salida: dict[str, Any] = {}
    for d in dominios:
        salida[d] = {}
        for x in ganancias:
            celdas = {}
            for r in ratios:
                ew, ee, es, fr = [], [], [], []
                for s in seeds:
                    p = puntajes[(d, r, s, x)]
                    recs = _registros(por_g[x], p)
                    ew.append(eta_squared(recs, "write", metric="rare_retention"))
                    ee.append(eta_squared(recs, "evict", metric="rare_retention"))
                    es.append(eta_squared(recs, "strength", metric="rare_retention"))
                    fr.append(p[i_front])
                celdas[clave_ratio(r)] = {
                    "ratio": r,
                    "eta2_write": _media_ci(ew),
                    "eta2_evict": _media_ci(ee),
                    "eta2_strength": _media_ci(es),
                    "dominante": "write" if np.mean(ew) > np.mean(ee) else "evict",
                    "puntaje_frontera": _media_ci(fr),
                }
            salida[d][f"g{x:g}".replace(".", "p")] = {"gain": x, "celdas": celdas}
    return salida


# ══════════════════════════════════════════════════════════════ orquestación


def main() -> int:
    with ExperimentRun("exp07_merge_sensitivity") as run:
        run.set_seeds(SEMILLAS)
        run.note(
            "No hay umbral de admisión en PolicyMemory: toda experiencia se escribe. "
            "Los parámetros del camino de escritura son Merge.threshold (barrido "
            "principal) y la ganancia de las compuertas de fuerza (barrido 1D)."
        )
        run.note(
            "Merge conserva la clave de la primera experiencia (no promedia hacia "
            "un centroide): el umbral se compara contra un ejemplar, no contra el "
            "prototipo."
        )
        banco = EmbeddingBank.load(BANCO)
        run.record(
            "config",
            {
                "capacity": CAPACIDAD,
                "ratios": list(RATIOS),
                "umbrales": list(UMBRALES),
                "umbral_por_defecto": UMBRAL_POR_DEFECTO,
                "visitas_por_prototipo": VISITAS_POR_PROTOTIPO,
                "n_rare": max(5, CAPACIDAD // 2),
                "read_every": READ_EVERY,
                "semillas": list(SEMILLAS),
                "ganancias": list(GANANCIAS),
                "ratios_ganancia": list(RATIOS_GANANCIA),
                "frontera": FRONTIER_GENOTYPE.as_dict(),
            },
        )

        run.log(
            f"{len(DOMINIOS)} dominios × {len(UMBRALES)} umbrales × {len(RATIOS)} r × "
            f"{len(SEMILLAS)} semillas × 576 genotipos"
        )
        umbral = barrer_umbral(banco=banco)
        run.record("umbral_fusion", umbral)

        run.log("barrido 1D de la ganancia de fuerza")
        ganancia = barrer_ganancia(banco=banco)
        run.record("ganancia_fuerza", ganancia)

        # ── resumen legible ────────────────────────────────────────────────
        print(
            f"\n{'dominio':<10}{'umbral':>7}{'cruce r':>9}{'cruce r_ef':>11}"
            f"{'fus(rmin)':>10}{'pureza':>8}{'raros abs':>10}{'w domina':>9}"
        )
        for d, datos in umbral.items():
            for u in datos["umbrales"].values():

                def f(x: float | None) -> str:
                    return "   —" if x is None else f"{x:.2f}"

                print(
                    f"{d:<10}{u['umbral']:>7.2f}{f(u['cruce_r_nominal']):>9}"
                    f"{f(u['cruce_r_efectivo']):>11}{u['tasa_fusion_r_min']:>10.2f}"
                    f"{f(u['pureza_fusion_r_min']):>8}{u['raros_absorbidos_r_min']:>10.2f}"
                    f"{'sí' if u['escritura_domina_en_r_min'] else 'no':>9}"
                )

        # ── la pregunta clave para CIFAR-100 ──────────────────────────────
        # ¿Algún umbral hace alcanzable la fusión sin colapsar clases? Criterio
        # fijado antes de correr: tasa de fusión ≥ 0.25, pureza ≥ 0.9 y
        # escritura dominante en el r más bajo.
        viables = [
            u["umbral"]
            for u in umbral["cifar100"]["umbrales"].values()
            if u["tasa_fusion_r_min"] >= 0.25
            and (u["pureza_fusion_r_min"] or 0.0) >= 0.9
            and u["escritura_domina_en_r_min"]
        ]
        con_cruce = [
            {
                "umbral": u["umbral"],
                "r_nominal": u["cruce_r_nominal"],
                "r_efectivo": u["cruce_r_efectivo"],
            }
            for u in umbral["cifar100"]["umbrales"].values()
            if u["cruce_r_nominal"] is not None
        ]
        run.record(
            "cifar100_pregunta_clave",
            {
                "criterio": "tasa_fusion>=0.25 & pureza>=0.9 & write domina en r_min",
                "umbrales_viables": viables,
                "existe_umbral_viable": bool(viables),
                "umbrales_con_cruce": con_cruce,
            },
        )
        print(f"\nCIFAR-100: umbrales viables = {viables or 'ninguno'}")
        print(f"CIFAR-100: umbrales con cruce = {con_cruce or 'ninguno'}")
        if not viables:
            run.note(
                "CIFAR-100: ningún umbral de la grilla hace alcanzable la fusión con "
                "pureza ≥ 0.9 y escritura dominante en r bajo."
            )

        try:
            from ember.figures import figura_sensibilidad_fusion

            destino = figura_sensibilidad_fusion(umbral, "paper/figures/fig_merge_sensitivity.pdf")
            run.record("figura", str(destino))
            print(f"\nFigura: {destino}")
        except ImportError:
            run.note("matplotlib no disponible; figura no generada")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
