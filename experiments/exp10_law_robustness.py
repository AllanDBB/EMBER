"""exp10 · ¿Gobierna r = K_proto / C la transición, fuera del generador en que se vio?

Responde a la objeción del revisor 1: la transición en r ≈ 1 se mostró para tres
capacidades y un único generador estacionario, y está muy ligada a la
definición misma de capacidad. Llamarla "ley" es prematuro mientras no se
sepa si `r` es la variable que gobierna o una parte de una relación de presión
de capacidad más general. Hay dos partes.

Parte A · robustez de la transición
-----------------------------------
Se barre `r` en familias de flujos que mueven una perilla a la vez respecto del
generador de `exp02` (ver `ember.data.families`): frecuencias de Zipf,
deriva por prototipo, dispersión intra-prototipo, similitud entre prototipos,
prevalencia de raros, recurrencia (= largo del flujo con este diseño) y
capacidad en un rango más ancho (C = 5 … 160). Como en `exp02`, **las visitas
por prototipo se mantienen constantes dentro de cada familia**: el largo del
flujo crece con `K`. El umbral de fusión queda en su valor por defecto (0.85);
su barrido es `exp07`.

En cada celda se mide η² de escritura y de desalojo sobre las 576
configuraciones (retención de raros, T1), con IC bootstrap sobre semillas, y
se localiza el cruce de dominancia interpolando `d = η²_write − η²_evict` en
escala logarítmica. El cruce se expresa en seis variables de presión,
declaradas acá antes de correr:

- `r_nom`       = K / C (la del paper);
- `r_eff`       = K_efectivo / C, con K_efectivo las trazas que deja la
  consolidación sin presión de capacidad (`exp05.prototipos_efectivos`);
- `r_hill`      = exp(H) / C, el número efectivo de prototipos (número de Hill
  de orden 1 de las frecuencias de visita);
- `r_masa`      = exp(H) · (K_efectivo / K_vistos) / C: la masa de frecuencia
  corregida por fragmentación;
- `r_carga_nom` = (K + n_raros) / C;
- `r_carga_eff` = (K_efectivo + n_raros) / C: todas las trazas que el flujo
  pide después de consolidar.

La variable que gobierna es la que **colapsa** los cruces de todas las familias
a un mismo valor. Se mide con el desvío estándar de log(cruce) entre familias,
con un bootstrap conjunto sobre semillas.

Hipótesis A y qué la refuta. Si `r_nom` gobierna, (i) en toda familia hay un
cruce y cae en `r_nom ∈ [0.5, 1.5]`, y (ii) ninguna alternativa colapsa los
cruces con una dispersión menor que la de `r_nom` en más del 95 % de las
réplicas bootstrap. Una familia sin cruce, un cruce fuera de ese intervalo, o
una alternativa sistemáticamente más colapsada refutan la forma fuerte y
obligan a enunciar la transición como hipótesis acotada.

Parte B · generalización held-out de la frontera
------------------------------------------------
Se rankean las 576 configuraciones en la familia estándar (exactamente la
batería de `exp01`) y se re-evalúan en familias held-out que no se usaron para
rankear, con semillas nuevas (10–14). En cada familia T1 corre sobre el flujo
held-out; T2 y T3 no dependen del flujo y corren con las semillas nuevas a
capacidad 20. Como esas dos tareas son comunes a todas las familias, se
reporta también el ranking de T1 sola, que es el que realmente cambia.

Hipótesis B y qué la refuta. La frontera de `exp01` sigue arriba fuera de la
distribución en que se eligió: su intervalo de rango cae en el 10 % superior
(pesimista ≤ 58) en toda familia held-out, y la correlación de Spearman entre
rankings es ≥ 0.5. Un intervalo pesimista > 58 en alguna familia, o una
correlación < 0.5, la refutan para esa familia.

Presupuesto: 5 semillas en la familia estándar y en las familias principales de
cada perilla; 3 en las secundarias; 3 en C = 40 y C = 80 y 2 en C = 160, que son
las celdas caras (flujos de hasta 4800 escrituras). En C = 80 y C = 160 la
grilla de `r` se recorta a la zona del cruce. Todo eso queda en `run.note`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.data.families import FamilySpec, family_stream, hill_number, visit_counts
from ember.data.streams import Stream
from ember.experiment import ExperimentRun, load_results
from ember.nas.engine import SearchRecord, SearchResults, run_search
from ember.nas.stats import bootstrap_ci, eta_squared
from ember.tasks.battery import (
    t1_rare_retention,
    t2_noise_under_pressure,
    t3_sequential_interference,
)
from experiments._common import (
    CAPACITY,
    READ_EVERY,
    SEEDS,
    EvalConfig,
    GenotypeEvaluator,
    make_factory,
)
from experiments.exp05_real_embeddings import prototipos_efectivos

# ══════════════════════════════════════════════════════════════ la grilla

RATIOS = (0.1, 0.25, 0.5, 0.65, 0.8, 1.0, 1.5, 2.0)
"""Ratios objetivo de la Parte A. Más densos que `exp02` en la zona del cruce."""

RATIOS_ZIPF = (*RATIOS, 3.0, 4.0)
"""Con Zipf fuerte la masa efectiva cae, y el cruce podría irse por encima de 2."""

RATIOS_CAROS = (0.5, 0.65, 0.8, 1.0, 1.5)
"""Para C = 80 y C = 160: solo la zona del cruce, por costo."""

S5 = (0, 1, 2, 3, 4)
S3 = (0, 1, 2)
S2 = (0, 1)


@dataclass(frozen=True, slots=True)
class FamiliaA:
    """Una familia de la Parte A: generador, capacidad, ratios y semillas."""

    nombre: str
    grupo: str
    """La perilla que mueve: `estandar`, `zipf`, `deriva`, `ruido`, …"""
    spec: FamilySpec
    capacity: int = CAPACITY
    ratios: tuple[float, ...] = RATIOS
    seeds: tuple[int, ...] = S3

    def prototipos(self) -> list[int]:
        """Los K de la familia, sin repetir (en C chicas dos ratios pueden dar el mismo K)."""
        return sorted({max(1, int(round(r * self.capacity))) for r in self.ratios})


def familias_parte_a() -> list[FamiliaA]:
    """Las familias de la Parte A. Los nombres no llevan puntos: son rutas de `\\result`."""
    std = FamilySpec()
    fs = [
        FamiliaA("estandar", "estandar", std, seeds=S5),
        FamiliaA("zipf_05", "zipf", FamilySpec("zipf_05", zipf=0.5), ratios=(*RATIOS, 3.0)),
        FamiliaA(
            "zipf_10", "zipf", FamilySpec("zipf_10", zipf=1.0), ratios=(*RATIOS, 3.0), seeds=S5
        ),
        FamiliaA("zipf_15", "zipf", FamilySpec("zipf_15", zipf=1.5), ratios=RATIOS_ZIPF, seeds=S5),
        FamiliaA("deriva_15", "deriva", FamilySpec("deriva_15", drift_deg=15.0)),
        FamiliaA("deriva_30", "deriva", FamilySpec("deriva_30", drift_deg=30.0), seeds=S5),
        FamiliaA("deriva_60", "deriva", FamilySpec("deriva_60", drift_deg=60.0)),
        FamiliaA("ruido_065", "ruido", FamilySpec("ruido_065", noise=0.065)),
        FamiliaA("ruido_075", "ruido", FamilySpec("ruido_075", noise=0.075), seeds=S5),
        FamiliaA("ruido_085", "ruido", FamilySpec("ruido_085", noise=0.085)),
        FamiliaA("similitud_05", "similitud", FamilySpec("similitud_05", correlation=0.5)),
        FamiliaA("raros_01pct", "raros", FamilySpec("raros_01pct", rare_prevalence=0.01)),
        FamiliaA("raros_05pct", "raros", FamilySpec("raros_05pct", rare_prevalence=0.05)),
        FamiliaA("raros_10pct", "raros", FamilySpec("raros_10pct", rare_prevalence=0.10)),
        FamiliaA("raros_20pct", "raros", FamilySpec("raros_20pct", rare_prevalence=0.20), seeds=S5),
        FamiliaA("visitas_05", "recurrencia", FamilySpec("visitas_05", visits=5)),
        FamiliaA("visitas_10", "recurrencia", FamilySpec("visitas_10", visits=10)),
        FamiliaA("visitas_40", "recurrencia", FamilySpec("visitas_40", visits=40)),
        FamiliaA("C005", "capacidad", std, capacity=5, seeds=S5),
        FamiliaA("C010", "capacidad", std, capacity=10, seeds=S5),
        FamiliaA("C040", "capacidad", std, capacity=40),
        FamiliaA("C080", "capacidad", std, capacity=80, ratios=RATIOS_CAROS),
        FamiliaA("C160", "capacidad", std, capacity=160, ratios=RATIOS_CAROS, seeds=S2),
    ]
    return fs


VARIABLES = ("r_nom", "r_eff", "r_hill", "r_masa", "r_carga_nom", "r_carga_eff")
"""Las medidas de presión candidatas, en el orden en que se declararon."""


# ═════════════════════════════════════════════════ evaluación en paralelo

_CACHE_FLUJOS: dict[tuple, Stream] = {}
"""Caché por proceso worker. Cada familia crea su propio pool, así que no crece."""


def _flujo(spec: FamilySpec, k: int, c: int, seed: int, n_common: int | None) -> Stream:
    clave = (spec, k, c, seed, n_common)
    if clave not in _CACHE_FLUJOS:
        _CACHE_FLUJOS[clave] = family_stream(spec, k, c, seed=seed, n_common=n_common)
    return _CACHE_FLUJOS[clave]


def _clave(k: int, seed: int) -> str:
    return f"K{k}|s{seed}"


class FamilyT1Evaluator:
    """T1 sobre todas las celdas (K, semilla) de una familia. Serializable."""

    def __init__(self, familia: FamiliaSerializable) -> None:
        self.f = familia

    def __call__(self, genotype: Genotype) -> dict[str, float]:
        f = self.f
        factory = make_factory(genotype)
        out = {}
        for k in f.ks:
            for s in f.seeds:
                stream = _flujo(f.spec, k, f.capacity, s, None)
                out[_clave(k, s)] = t1_rare_retention(
                    factory, stream, seed=s, read_every=READ_EVERY
                ).score
        return out


@dataclass(frozen=True, slots=True)
class FamiliaSerializable:
    """Lo mínimo que viaja al worker."""

    spec: FamilySpec
    capacity: int
    ks: tuple[int, ...]
    seeds: tuple[int, ...]


# ═══════════════════════════════════════════════════ medidas de presión


def medir_presion(stream: Stream, capacity: int, seed: int) -> dict[str, float]:
    """Las seis variables de presión de un flujo, más sus ingredientes."""
    k = stream.spec.n_prototypes
    k_eff, tasa = prototipos_efectivos(stream, seed=seed)
    k_vistos = int(visit_counts(stream).size)
    hill = hill_number(stream, order=1.0)
    n_raros = len(stream.rare_items)
    fragmentacion = k_eff / max(k_vistos, 1)
    return {
        "k_eff": float(k_eff),
        "k_vistos": float(k_vistos),
        "hill": hill,
        "fragmentacion": fragmentacion,
        "tasa_de_fusion": tasa,
        "n_raros": float(n_raros),
        "r_nom": k / capacity,
        "r_eff": k_eff / capacity,
        "r_hill": hill / capacity,
        "r_masa": hill * fragmentacion / capacity,
        "r_carga_nom": (k + n_raros) / capacity,
        "r_carga_eff": (k_eff + n_raros) / capacity,
    }


# ═════════════════════════════════════════════════ localizar el cruce


def localizar_cruce(x: np.ndarray, d: np.ndarray) -> dict:
    """Dónde pasa `d = η²_w − η²_e` de positivo a no positivo, en escala log de `x`.

    Se toma el primer cambio de signo de + a − con las celdas ordenadas por `x`
    e interpola linealmente en log x. Si `d` nunca es positivo la escritura no
    domina en ninguna celda (`estado = "siempre_desalojo"`); si nunca deja de
    serlo, el cruce queda por encima de la grilla (`"siempre_escritura"`).
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    orden = np.argsort(x, kind="stable")
    x, d = x[orden], d[orden]
    signo = d > 0
    cambios = int(np.sum(signo[:-1] != signo[1:]))
    for i in range(len(x) - 1):
        if d[i] > 0 >= d[i + 1] and x[i] > 0 and x[i + 1] > 0:
            lx0, lx1 = np.log(x[i]), np.log(x[i + 1])
            t = d[i] / (d[i] - d[i + 1])
            return {
                "estado": "cruza",
                "x_cruce": float(np.exp(lx0 + t * (lx1 - lx0))),
                "x_below": float(x[i]),
                "x_above": float(x[i + 1]),
                "n_cambios_de_signo": cambios,
            }
    estado = "siempre_escritura" if signo.all() else "siempre_desalojo"
    if signo.any() and not signo.all():
        estado = "sin_cruce_descendente"
    return {
        "estado": estado,
        "x_cruce": None,
        "x_below": None,
        "x_above": None,
        "n_cambios_de_signo": cambios,
    }


def _cruces_bootstrap(
    d_por_semilla: np.ndarray,
    x_por_semilla: dict[str, np.ndarray],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cruce de cada variable en réplicas bootstrap sobre semillas (NaN si no cruza).

    La unidad independiente es la semilla: todas las celdas de una semilla se
    remuestrean juntas, porque comparten generador de ruido y de raros.
    `d_por_semilla` es (celdas, semillas); devuelve (n_boot, variables).
    """
    n_sem = d_por_semilla.shape[1]
    salida = np.full((n_boot, len(VARIABLES)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n_sem, size=n_sem)
        d = d_por_semilla[:, idx].mean(axis=1)
        for j, v in enumerate(VARIABLES):
            c = localizar_cruce(x_por_semilla[v][:, idx].mean(axis=1), d)
            if c["x_cruce"] is not None:
                salida[b, j] = c["x_cruce"]
    return salida


# ═══════════════════════════════════════════════════════════ Parte A


def barrer_familia(
    familia: FamiliaA,
    *,
    genotypes: list[Genotype] | None = None,
    n_boot: int = 1000,
    verbose: bool = True,
) -> tuple[dict, np.ndarray]:
    """Barre `r` en una familia. Devuelve el resumen y el bootstrap de cruces."""
    ks = tuple(familia.prototipos())
    serial = FamiliaSerializable(familia.spec, familia.capacity, ks, familia.seeds)
    t0 = time.time()
    res = run_search(FamilyT1Evaluator(serial), genotypes=genotypes, n_jobs=-1, progress=False)

    celdas = []
    d_sem = np.zeros((len(ks), len(familia.seeds)))
    x_sem = {v: np.zeros((len(ks), len(familia.seeds))) for v in VARIABLES}
    for i, k in enumerate(ks):
        w, e, medidas = [], [], []
        for j, s in enumerate(familia.seeds):
            clave = _clave(k, s)
            w.append(eta_squared(res.records, "write", metric=clave))
            e.append(eta_squared(res.records, "evict", metric=clave))
            stream = family_stream(familia.spec, k, familia.capacity, seed=s)
            m = medir_presion(stream, familia.capacity, s)
            medidas.append(m)
            d_sem[i, j] = w[-1] - e[-1]
            for v in VARIABLES:
                x_sem[v][i, j] = m[v]
        celda = {
            "K": k,
            "capacity": familia.capacity,
            "n_common": familia.spec.n_common(k),
            "n_rare": int(medidas[0]["n_raros"]),
            "eta2_write": float(np.mean(w)),
            "eta2_write_ci": list(bootstrap_ci(w, seed=0)),
            "eta2_write_por_semilla": [float(v) for v in w],
            "eta2_evict": float(np.mean(e)),
            "eta2_evict_ci": list(bootstrap_ci(e, seed=0)),
            "eta2_evict_por_semilla": [float(v) for v in e],
            "d": float(np.mean(d_sem[i])),
            "dominante": "write" if np.mean(w) > np.mean(e) else "evict",
        }
        for clave_m in medidas[0]:
            celda[clave_m] = float(np.mean([m[clave_m] for m in medidas]))
        celdas.append(celda)

    d = np.array([c["d"] for c in celdas])
    cruces = {v: localizar_cruce(np.array([c[v] for c in celdas]), d) for v in VARIABLES}
    boot = _cruces_bootstrap(
        d_sem, x_sem, n_boot=n_boot, rng=np.random.default_rng(len(familia.nombre))
    )
    for j, v in enumerate(VARIABLES):
        validos = boot[:, j][~np.isnan(boot[:, j])]
        cruces[v]["frac_bootstrap_con_cruce"] = float(len(validos) / n_boot)
        cruces[v]["ci"] = (
            [float(np.percentile(validos, 2.5)), float(np.percentile(validos, 97.5))]
            if len(validos) >= 20
            else None
        )

    resumen = {
        "nombre": familia.nombre,
        "grupo": familia.grupo,
        "spec": familia.spec.as_dict(),
        "capacity": familia.capacity,
        "seeds": list(familia.seeds),
        "n_seeds": len(familia.seeds),
        "celdas": celdas,
        "cruce": cruces,
        "segundos": round(time.time() - t0, 1),
    }
    if verbose:
        c = cruces["r_nom"]
        txt = f"r*={c['x_cruce']:.2f}" if c["x_cruce"] else c["estado"]
        alt = cruces["r_carga_eff"]
        txt_alt = f"{alt['x_cruce']:.2f}" if alt["x_cruce"] else alt["estado"]
        print(
            f"  [{familia.nombre:<13}] C={familia.capacity:<4} semillas={len(familia.seeds)}"
            f"  {txt:<22} r_carga_eff*={txt_alt:<18} ({resumen['segundos']:.0f} s)",
            flush=True,
        )
        for cel in celdas:
            print(
                f"      K={cel['K']:<4} r={cel['r_nom']:<5.2f} r_eff={cel['r_eff']:<5.2f}"
                f" hill/C={cel['r_hill']:<5.2f} η²w={cel['eta2_write']:.3f}"
                f" η²e={cel['eta2_evict']:.3f}",
                flush=True,
            )
    return resumen, boot


def colapso_de_cruces(resumenes: list[dict], boots: list[np.ndarray]) -> dict:
    """Qué variable alinea mejor los cruces entre familias.

    Para cada variable: los cruces de las familias que cruzan, el desvío
    estándar de su logaritmo (0 = colapso perfecto), el cociente máx/mín y la
    distancia típica a 1. El bootstrap combina, en cada réplica, un remuestreo
    de semillas **por familia** —las familias son independientes entre sí— y
    mide en cuántas réplicas cada variable es la más colapsada.
    """
    salida: dict[str, dict] = {}
    n_boot = boots[0].shape[0]
    pila = np.stack(boots)  # (familias, n_boot, variables)

    sd_boot = np.full((n_boot, len(VARIABLES)), np.nan)
    for j in range(len(VARIABLES)):
        logs = np.log(pila[:, :, j])  # NaN si no cruza
        for b in range(n_boot):
            v = logs[:, b][~np.isnan(logs[:, b])]
            if len(v) >= 3:
                sd_boot[b, j] = float(np.std(v, ddof=1))

    completas = ~np.isnan(sd_boot).any(axis=1)
    mejor = np.argmin(np.where(np.isnan(sd_boot), np.inf, sd_boot), axis=1)

    for j, v in enumerate(VARIABLES):
        cruces = {
            r["nombre"]: r["cruce"][v]["x_cruce"]
            for r in resumenes
            if r["cruce"][v]["x_cruce"] is not None
        }
        sin_cruce = [r["nombre"] for r in resumenes if r["cruce"][v]["x_cruce"] is None]
        logs = np.log(np.array(list(cruces.values())))
        sd_b = sd_boot[:, j][~np.isnan(sd_boot[:, j])]
        salida[v] = {
            "n_familias_con_cruce": len(cruces),
            "familias_sin_cruce": sin_cruce,
            "cruce_mediano": float(np.exp(np.median(logs))) if len(logs) else None,
            "cruce_min": float(np.exp(logs.min())) if len(logs) else None,
            "cruce_max": float(np.exp(logs.max())) if len(logs) else None,
            "cociente_max_min": float(np.exp(logs.max() - logs.min())) if len(logs) else None,
            "sd_log": float(np.std(logs, ddof=1)) if len(logs) >= 2 else None,
            "sd_log_ci": (
                [float(np.percentile(sd_b, 2.5)), float(np.percentile(sd_b, 97.5))]
                if len(sd_b) >= 20
                else None
            ),
            "mad_log_a_1": float(np.median(np.abs(logs))) if len(logs) else None,
            "frac_bootstrap_mas_colapsada": float(np.mean(mejor[completas] == j))
            if completas.any()
            else None,
            "cruces": cruces,
        }
    # Contraste pareado contra la variable del paper.
    j0 = VARIABLES.index("r_nom")
    for j, v in enumerate(VARIABLES):
        ok = ~np.isnan(sd_boot[:, j]) & ~np.isnan(sd_boot[:, j0])
        salida[v]["frac_bootstrap_menor_que_r_nom"] = (
            float(np.mean(sd_boot[ok, j] < sd_boot[ok, j0])) if ok.any() else None
        )
    return salida


def parte_a(
    familias: list[FamiliaA] | None = None,
    *,
    genotypes: list[Genotype] | None = None,
    n_boot: int = 1000,
    verbose: bool = True,
) -> dict:
    familias = familias if familias is not None else familias_parte_a()
    resumenes, boots = [], []
    for f in familias:
        r, b = barrer_familia(f, genotypes=genotypes, n_boot=n_boot, verbose=verbose)
        resumenes.append(r)
        boots.append(b)

    colapso = colapso_de_cruces(resumenes, boots)
    ganadora = min(
        (v for v in VARIABLES if colapso[v]["sd_log"] is not None),
        key=lambda v: colapso[v]["sd_log"],
    )
    en_banda = {
        r["nombre"]: (
            r["cruce"]["r_nom"]["x_cruce"] is not None
            and 0.5 <= r["cruce"]["r_nom"]["x_cruce"] <= 1.5
        )
        for r in resumenes
    }
    return {
        "familias": {r["nombre"]: r for r in resumenes},
        "orden": [r["nombre"] for r in resumenes],
        "variables": list(VARIABLES),
        "colapso": colapso,
        "variable_mas_colapsada": ganadora,
        "r_nom_cruce_en_05_15": en_banda,
        "r_nom_cruce_en_05_15_todas": bool(all(en_banda.values())),
    }


# ═══════════════════════════════════════════════════════════ Parte B

SEMILLAS_HELDOUT = (10, 11, 12, 13, 14)
"""Semillas que no se usaron para rankear (el ranking estándar usa 0, 1, 2)."""


@dataclass(frozen=True, slots=True)
class FamiliaB:
    """Una familia held-out: el flujo de T1 que reemplaza al de `exp01`."""

    nombre: str
    spec: FamilySpec
    n_prototypes: int = 5
    capacity: int = CAPACITY
    n_common: int = 300
    nota: str = ""


def familias_parte_b() -> list[FamiliaB]:
    """Held-out respecto de la geometría de `exp01` (K=5, C=20, 300 comunes, 20 raros)."""
    b = {"n_rare": 20}
    return [
        FamiliaB("semillas_nuevas", FamilySpec("semillas_nuevas", **b), nota="solo semillas"),
        FamiliaB("zipf_10", FamilySpec("zipf_10", zipf=1.0, **b)),
        FamiliaB("zipf_15", FamilySpec("zipf_15", zipf=1.5, **b)),
        FamiliaB("deriva_30", FamilySpec("deriva_30", drift_deg=30.0, **b)),
        FamiliaB("deriva_60", FamilySpec("deriva_60", drift_deg=60.0, **b)),
        FamiliaB("ruido_075", FamilySpec("ruido_075", noise=0.075, **b)),
        FamiliaB("similitud_05", FamilySpec("similitud_05", correlation=0.5, **b)),
        FamiliaB("raros_01pct", FamilySpec("raros_01pct", rare_prevalence=0.01)),
        FamiliaB("raros_20pct", FamilySpec("raros_20pct", rare_prevalence=0.20)),
        FamiliaB("transicion_r08", FamilySpec("transicion_r08", **b), 16, n_common=320),
        FamiliaB("seleccion_r15", FamilySpec("seleccion_r15", **b), 30, n_common=600),
        FamiliaB("seleccion_r40", FamilySpec("seleccion_r40", **b), 80, n_common=1600),
        FamiliaB(
            "capacidad_80",
            FamilySpec("capacidad_80", n_rare=80),
            20,
            capacity=80,
            n_common=1200,
            nota="T1 en C=80; T2 y T3 quedan en C=20",
        ),
        FamiliaB("largo_x4", FamilySpec("largo_x4", **b), n_common=1200),
        FamiliaB(
            "compuesta",
            FamilySpec("compuesta", zipf=1.0, drift_deg=30.0, noise=0.065, rare_prevalence=0.10),
            16,
            n_common=320,
            nota="Zipf 1 + deriva 30° + ruido 0.065 + 10 % raros, r = 0.8",
        ),
    ]


class HeldOutEvaluator:
    """La batería de `exp01` con el flujo de T1 reemplazado. Serializable."""

    def __init__(self, familia: FamiliaB, seeds: tuple[int, ...] = SEMILLAS_HELDOUT) -> None:
        self.f = familia
        self.seeds = seeds

    def __call__(self, genotype: Genotype) -> dict[str, float]:
        f = self.f
        factory = make_factory(genotype)
        out: dict[str, float] = {}
        for s in self.seeds:
            stream = _flujo(f.spec, f.n_prototypes, f.capacity, s, f.n_common)
            out[f"t1|{s}"] = t1_rare_retention(factory, stream, seed=s, read_every=READ_EVERY).score
            out[f"t2|{s}"] = t2_noise_under_pressure(factory, seed=s, capacity=CAPACITY).score
            out[f"t3|{s}"] = t3_sequential_interference(factory, seed=s, capacity=CAPACITY).score
        return out


def _rangos_promedio(v: np.ndarray) -> np.ndarray:
    """Rangos con empates promediados (base 1)."""
    orden = np.argsort(v, kind="stable")
    rangos = np.empty(len(v), dtype=np.float64)
    vs = v[orden]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and vs[j + 1] == vs[i]:
            j += 1
        rangos[orden[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return rangos


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Correlación de Spearman con empates promediados."""
    ra, rb = _rangos_promedio(np.asarray(a, float)), _rangos_promedio(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """τ_b de Kendall (corrige por empates, que en este espacio son masivos)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(len(a), k=1)
    sa, sb = da[iu], db[iu]
    num = float((sa * sb).sum())
    den = np.sqrt(float((sa != 0).sum()) * float((sb != 0).sum()))
    return num / den if den > 0 else float("nan")


def _resultados(genotipos: list[Genotype], valores: np.ndarray) -> SearchResults:
    registros = [
        SearchRecord(genotype=g, scores={"score": float(v)}, mean=float(v))
        for g, v in zip(genotipos, valores, strict=True)
    ]
    registros.sort(key=lambda r: (-r.mean, r.genotype.label()))
    return SearchResults(records=registros)


def comparar_rankings(
    genotipos: list[Genotype],
    estandar: np.ndarray,
    heldout: np.ndarray,
    frontera: Genotype,
    *,
    heldout_por_semilla: np.ndarray | None = None,
    n_boot: int = 500,
) -> dict:
    """Correlaciones de rango, posición de la frontera y del FIFO en el held-out."""
    res = _resultados(genotipos, heldout)
    n = len(genotipos)
    fo, fp = res.rank_of(frontera)
    io, ip = res.rank_of(FIFO_GENOTYPE)
    fifo = res.score_of(FIFO_GENOTYPE)
    peores = int(np.sum(heldout < fifo - 1e-9))
    mejor = res.records[0].mean
    empatados_arriba = [r.genotype.label() for r in res.records if abs(r.mean - mejor) <= 1e-9]

    salida = {
        "spearman": spearman(estandar, heldout),
        "kendall_tau_b": kendall_tau_b(estandar, heldout),
        "frontera_score": res.score_of(frontera),
        "frontera_rango_optimista": fo,
        "frontera_rango_pesimista": fp,
        "frontera_en_top10pct": bool(fp <= int(np.ceil(0.10 * n))),
        "frontera_gap_al_mejor": float(mejor - res.score_of(frontera)),
        "fifo_score": fifo,
        "fifo_rango_optimista": io,
        "fifo_rango_pesimista": ip,
        "fifo_percentil": float(100.0 * peores / n),
        "mejor_score": float(mejor),
        "mejor_genotipo": res.records[0].genotype.as_dict(),
        "n_empatados_en_el_mejor": len(empatados_arriba),
    }
    if heldout_por_semilla is not None and heldout_por_semilla.shape[1] > 1:
        rng = np.random.default_rng(0)
        rhos, taus = [], []
        k = heldout_por_semilla.shape[1]
        for _ in range(n_boot):
            idx = rng.integers(0, k, size=k)
            v = heldout_por_semilla[:, idx].mean(axis=1)
            rhos.append(spearman(estandar, v))
            taus.append(kendall_tau_b(estandar, v))
        salida["spearman_ci"] = [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))]
        salida["kendall_ci"] = [float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))]
    return salida


def parte_b(
    familias: list[FamiliaB] | None = None,
    *,
    seeds: tuple[int, ...] = SEMILLAS_HELDOUT,
    genotypes: list[Genotype] | None = None,
    n_boot: int = 500,
    verbose: bool = True,
) -> dict:
    familias = familias if familias is not None else familias_parte_b()

    # ── el ranking estándar: exactamente la batería de exp01 ────────────────
    std = run_search(
        GenotypeEvaluator(EvalConfig(seeds=SEEDS)),
        genotypes=genotypes,
        n_jobs=-1,
        progress=False,
    )
    genotipos = sorted((r.genotype for r in std.records), key=lambda g: g.label())
    por_label = {r.genotype.label(): r for r in std.records}
    std_bat = np.array([por_label[g.label()].mean for g in genotipos])
    std_t1 = np.array([por_label[g.label()].scores["rare_retention"] for g in genotipos])
    frontera = std.records[0].genotype
    fo, fp = std.rank_of(frontera)
    io, ip = std.rank_of(FIFO_GENOTYPE)

    estandar = {
        "frontera": frontera.as_dict(),
        "frontera_label": frontera.label(),
        "frontera_score": std.records[0].mean,
        "frontera_rango": [fo, fp],
        "fifo_score": std.score_of(FIFO_GENOTYPE),
        "fifo_rango": [io, ip],
        "fifo_percentil": float(
            100.0 * np.sum(std_bat < std.score_of(FIFO_GENOTYPE) - 1e-9) / len(genotipos)
        ),
    }
    if verbose:
        print(f"  estándar: frontera {frontera.label()}  {std.records[0].mean:.3f}")

    salida_fam: dict[str, dict] = {}
    for f in familias:
        t0 = time.time()
        res = run_search(HeldOutEvaluator(f, seeds), genotypes=genotypes, n_jobs=-1, progress=False)
        por = {r.genotype.label(): r.scores for r in res.records}
        t1s = np.array([[por[g.label()][f"t1|{s}"] for s in seeds] for g in genotipos])
        bat = np.array(
            [
                [np.mean([por[g.label()][f"{t}|{s}"] for t in ("t1", "t2", "t3")]) for s in seeds]
                for g in genotipos
            ]
        )
        fila = {
            "spec": f.spec.as_dict(),
            "n_prototypes": f.n_prototypes,
            "capacity_t1": f.capacity,
            "n_common": f.n_common,
            "r_nom": f.n_prototypes / f.capacity,
            "nota": f.nota,
            "bateria": comparar_rankings(
                genotipos,
                std_bat,
                bat.mean(axis=1),
                frontera,
                heldout_por_semilla=bat,
                n_boot=n_boot,
            ),
            "t1": comparar_rankings(
                genotipos,
                std_t1,
                t1s.mean(axis=1),
                frontera,
                heldout_por_semilla=t1s,
                n_boot=n_boot,
            ),
            "segundos": round(time.time() - t0, 1),
        }
        salida_fam[f.nombre] = fila
        if verbose:
            bb, tt = fila["bateria"], fila["t1"]
            print(
                f"  [{f.nombre:<16}] batería ρ={bb['spearman']:.2f} τ={bb['kendall_tau_b']:.2f}"
                f" frontera #{bb['frontera_rango_optimista']}–{bb['frontera_rango_pesimista']}"
                f" FIFO p{bb['fifo_percentil']:.0f} | T1 ρ={tt['spearman']:.2f}"
                f" frontera #{tt['frontera_rango_optimista']}–{tt['frontera_rango_pesimista']}"
                f" ({fila['segundos']:.0f} s)",
                flush=True,
            )

    peor = max(salida_fam.values(), key=lambda x: x["bateria"]["frontera_rango_pesimista"])
    return {
        "estandar": estandar,
        "semillas_heldout": list(seeds),
        "familias": salida_fam,
        "orden": [f.nombre for f in familias],
        "frontera_top10pct_en_todas_bateria": bool(
            all(v["bateria"]["frontera_en_top10pct"] for v in salida_fam.values())
        ),
        "frontera_top10pct_en_todas_t1": bool(
            all(v["t1"]["frontera_en_top10pct"] for v in salida_fam.values())
        ),
        "frontera_peor_rango_pesimista_bateria": peor["bateria"]["frontera_rango_pesimista"],
        "spearman_min_bateria": float(min(v["bateria"]["spearman"] for v in salida_fam.values())),
        "spearman_min_t1": float(min(v["t1"]["spearman"] for v in salida_fam.values())),
    }


# ═══════════════════════════════════════════════════════════════ main


def main() -> int:
    with ExperimentRun("exp10_law_robustness") as run:
        run.set_seeds(S5 + SEMILLAS_HELDOUT)
        run.note(
            "Parte A: 5 semillas en estandar, zipf_10, zipf_15, deriva_30, ruido_075, "
            "raros_20pct, C005 y C010; 3 en las demás familias y en C040/C080; 2 en C160. "
            "C080 y C160 solo barren r en {0.5, 0.65, 0.8, 1.0, 1.5}, por costo."
        )
        run.note("El umbral de fusión queda en 0.85 (valor por defecto): su barrido es exp07.")
        run.note(
            "Parte B: T2 y T3 no dependen del flujo; en cada familia held-out corren con "
            "semillas nuevas (10-14) a C=20. Por eso se reporta también T1 sola."
        )

        t0 = time.time()
        run.log("Parte A · robustez de la transición")
        a = parte_a()
        a["segundos"] = round(time.time() - t0, 1)
        run.record("parte_a", a)

        print(
            f"\n{'variable':<14}{'familias':>9}{'mediana':>9}{'máx/mín':>9}{'sd log':>8}"
            f"{'P(mejor)':>10}{'P(<r_nom)':>11}"
        )
        for v, c in a["colapso"].items():
            if c["sd_log"] is None:
                continue
            print(
                f"{v:<14}{c['n_familias_con_cruce']:>9}{c['cruce_mediano']:>9.2f}"
                f"{c['cociente_max_min']:>9.2f}{c['sd_log']:>8.3f}"
                f"{c['frac_bootstrap_mas_colapsada'] or 0:>10.2f}"
                f"{c['frac_bootstrap_menor_que_r_nom'] or 0:>11.2f}"
            )
        sin = a["colapso"]["r_nom"]["familias_sin_cruce"]
        if sin:
            run.note(f"Familias sin cruce de dominancia en r nominal: {', '.join(sin)}.")
        fuera = [k for k, ok in a["r_nom_cruce_en_05_15"].items() if not ok]
        if fuera:
            run.note(
                "El cruce en r nominal cae fuera de [0.5, 1.5] o no existe en: "
                + ", ".join(fuera)
                + ". La forma fuerte de la hipótesis A queda refutada para esas familias."
            )

        t1 = time.time()
        run.log("Parte B · generalización held-out")
        b = parte_b()
        b["segundos"] = round(time.time() - t1, 1)
        try:
            exp01 = load_results("exp01_nas_full")
            b["frontera_coincide_con_exp01"] = bool(
                exp01["frontier"]["genotype"] == b["estandar"]["frontera"]
            )
        except FileNotFoundError:
            b["frontera_coincide_con_exp01"] = None
        run.record("parte_b", b)
        if not b["frontera_top10pct_en_todas_bateria"]:
            run.note(
                "La frontera de exp01 sale del 10 % superior en al menos una familia held-out."
            )

        try:
            from ember.figures import figura_cruces_por_familia

            destino = figura_cruces_por_familia(a, "paper/figures/fig_exp10_crossings.pdf")
            run.record("figura", str(destino))
            print(f"\nFigura: {destino}")
        except ImportError:
            run.note("matplotlib no disponible; figura no generada")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
