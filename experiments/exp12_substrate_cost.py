"""exp12 · Costo de los sustratos: bytes, tiempo y rendimiento a igual presupuesto.

Responde a la crítica del revisor 1 de BIP2026: la comparación entre sustratos
de `exp04` fija la capacidad en **número de trazas** (C = 20), pero una traza no
cuesta lo mismo en todas las arquitecturas. La SDM, además de su lista de
trazas, materializa 512 direcciones y 512 contadores de dimensión d que existen
aunque la memoria esté casi vacía. Comparar a igual C puede esconder órdenes de
magnitud de diferencia de huella y de costo de cómputo.

Qué se mide
-----------
1. **Contabilidad de memoria.** Bytes de estado por componente, por dos vías
   independientes que tienen que coincidir: `nbytes` real de los arreglos
   (`ember.footprint.inventario_real`) y la fórmula cerrada en C, d y los
   hiperparámetros (`ember.footprint.bytes_analiticos`).
2. **Complejidad y tiempo.** Tiempo de pared de `write` y `read` en régimen
   estacionario (memoria llena, con desalojos), barriendo C y d; pendiente
   log-log contra la complejidad analítica.
3. **Igual presupuesto de bytes.** Para cada presupuesto B se deriva la
   capacidad (y, en un segundo modo, el número de hard locations) de cada
   arquitectura, y se re-corre el protocolo de `exp04` (gate de reconstrucción
   más batería) con 10 semillas e IC bootstrap.
4. **Ablación de la SDM.** (a) solo contadores, (a+) contadores con borrado
   exacto, (b) solo lista de trazas con vecino más cercano, (c) completa.
5. **Sensibilidad de la SDM.** Número de hard locations, fracción de
   activación, grilla 2D de ambas, y umbral de admisión (la SDM no tiene uno:
   se agrega como compuerta externa para poder barrerlo). El umbral de fusión
   lo barre `exp07` y no se repite.

Hipótesis y qué las refutaría (escrito antes de correr)
-------------------------------------------------------
H1. La ventaja de la SDM sobre el FIFO en `exp04` (retención de raros 1.000 vs
    0.038 a C = 20) **no** se sostiene a igual presupuesto de bytes con
    hiperparámetros por defecto: con los ~131 KiB fijos de la SDM, un FIFO
    guarda cientos de trazas y el flujo de T1 (320 escrituras) entra casi
    entero. Se refuta si, a igual presupuesto, la SDM sigue por encima del FIFO
    en la media de la batería con IC que no cruza cero.
H2. Lo que rinde la SDM en la batería lo aporta la **lista de trazas** con su
    compuerta de saliencia y el desalojo por mínima fuerza, no los contadores:
    la variante (b) empata con la completa (c) dentro del IC, y (a) cae. Se
    refuta si (c) supera a (b) con IC pareado que excluye cero en la media de
    la batería, o si (a) empata con (c).
H3. Como la lectura de la SDM termina decidiendo contra las claves exactas, su
    puntaje es poco sensible al número de hard locations y a la fracción de
    activación dentro de un rango razonable. Se refuta si la media de la
    batería varía más de 0.10 entre M = 64 y M = 2048 a fracción 0.05.
H4. El costo por operación de la SDM está dominado por el término O(M·d) del
    conjunto activo a la C del paper, y el del FIFO por O(C·d); ambos quedan
    tapados por el overhead fijo del intérprete a C y d chicos. Se refuta si la
    pendiente log-log del `read` de la SDM contra d en la cola del barrido es
    menor que 0.5 o si la del FIFO contra C en la cola es menor que 0.5.

Advertencia sobre tiempos
-------------------------
La máquina se comparte con otros experimentos que corren en paralelo. Los
tiempos se miden en serie, antes del pool de procesos, **intercalando** las
configuraciones dentro de cada repetición para que la carga de fondo afecte a
todas por igual; la carga media del sistema se registra junto a cada barrido.
Las comparaciones relativas entre arquitecturas son más confiables que los
microsegundos absolutos.
"""

from __future__ import annotations

import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from ember.core.policies import PredErrorAdmission
from ember.experiment import ExperimentRun
from ember.footprint import (
    arreglos_no_inventariados,
    bytes_analiticos,
    capacidad_para_presupuesto,
    costo_fijo_y_por_traza,
    inventario_real,
)
from ember.memories import ARCHITECTURES
from ember.memories.sdm_ablation import AdmissionGated, SDMCountersOnly, sdm_traces_only
from ember.tasks.battery import run_battery
from ember.tasks.reconstruction import RECON_THRESHOLD, reconstruction_gate

DIM = 32
CAPACITY = 20
SEMILLAS = tuple(range(10))
SEMILLAS_LENTAS = tuple(range(5))
"""`Spiking` simula 30 presentaciones por escritura: se evalúa con 5 semillas."""

REGISTRADAS = ("FIFO", "ENN", "SDM", "SpikingSDM", "Spiking")
VARIANTES_SDM = ("SDM-a", "SDM-a+", "SDM-b", "SDM")
LENTAS = ("Spiking",)

# Barridos de tiempo.
C_GRID = (10, 20, 40, 80, 160, 320, 640, 1280)
D_GRID = (16, 32, 64, 128, 256, 512)
C_GRID_LENTA = (10, 20, 40, 80)
D_GRID_LENTA = (16, 32, 64, 128)
M_GRID = (64, 128, 256, 512, 1024, 2048)
N_MEDIDAS = 200
N_CALENTAMIENTO = 20
N_MEDIDAS_LENTA = 20
N_CALENTAMIENTO_LENTA = 3
REPETICIONES = 3

# Igual presupuesto.
PRESUPUESTOS = (4096, 16384, 65536, 147456, 524288)
"""4 KiB (~ FIFO a C = 20 más holgura), 16 KiB, 64 KiB, 144 KiB (~ SDM a C = 20), 512 KiB."""
M_MIN, M_MAX = 4, 2048

# Sensibilidad.
FRAC_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
GRID_2D_M = M_GRID
GRID_2D_FRAC = (0.01, 0.02, 0.05, 0.2)
ADMISION_GRID = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8)

N_BOOT = 10_000

COMPLEJIDAD_ANALITICA = {
    "FIFO": {
        "write": "O(C·d): novedad contra C claves + copia del store al agregar",
        "read": "O(C·d): similitud contra C claves",
    },
    "ENN": {
        "write": "O(C·d): similitud para decidir la fusión + copia del store",
        "read": "O(C·d)",
    },
    "SDM": {
        "write": "O(M·d + C·d): conjunto activo (M·d, dos veces si desaloja) + store",
        "read": "O(M·d + k·d + C·d)",
    },
    "SpikingSDM": {
        "write": "O(T·M·d + C·d): T pasos de integración LIF sobre M neuronas",
        "read": "O(T·M·d + C·d)",
    },
    "Spiking": {
        "write": "O(P·L·n² + n·d + C·d): P presentaciones de L pasos de STDP",
        "read": "O(t·n² + C·k + C·d): t pasos LIF y votación por ensamble",
    },
    "SDM-a": {"write": "O(M·(d+p))", "read": "O(M·d + k·(d+p) + C·p)"},
    "SDM-a+": {"write": "O(M·(d+p) + C·d)", "read": "O(M·d + k·(d+p) + C·p)"},
    "SDM-b": {"write": "O(C·d)", "read": "O(C·d)"},
}
"""Complejidad por operación. M hard locations, k activas, n neuronas, p = dim. del código."""

EXPONENTE_ASINTOTICO = {
    # (write_C, write_d, read_C, read_d) para C, d → ∞ con los demás fijos.
    "FIFO": (1, 1, 1, 1),
    "ENN": (1, 1, 1, 1),
    "SDM": (1, 1, 1, 1),
    "SpikingSDM": (1, 1, 1, 1),
    "Spiking": (1, 1, 1, 1),
    "SDM-a": (1, 1, 1, 1),
    "SDM-a+": (1, 1, 1, 1),
    "SDM-b": (1, 1, 1, 1),
}
"""Exponente del término dominante para C o d → ∞.

Es asintótico: en el rango medido, un término de costo fijo grande (M·d en la
SDM, P·L·n² en Spiking) puede dominar y aplanar la pendiente. Por eso se reporta
también la pendiente en la cola del barrido.
"""


# ══════════════════════════════════════════════════════════════ construcción


def construir(nombre: str, dim: int, capacity: int, seed: int, hp: dict | None = None) -> Any:
    """Instancia una arquitectura o variante por nombre, con hiperparámetros opcionales.

    `hp` puede incluir `admision` (umbral de error de predicción), que envuelve
    la memoria en `AdmissionGated`.
    """
    hp = dict(hp or {})
    admision = hp.pop("admision", None)
    if nombre == "SDM-a":
        mem = SDMCountersOnly(dim=dim, capacity=capacity, seed=seed, **hp)
    elif nombre == "SDM-a+":
        mem = SDMCountersOnly(dim=dim, capacity=capacity, seed=seed, exact_erase=True, **hp)
    elif nombre == "SDM-b":
        mem = sdm_traces_only(dim=dim, capacity=capacity, seed=seed)
    else:
        mem = ARCHITECTURES[nombre](dim=dim, capacity=capacity, seed=seed, **hp)
    if admision is not None:
        mem = AdmissionGated(mem, PredErrorAdmission(float(admision)))
    return mem


def _hp_contable(hp: dict | None) -> dict:
    """Hiperparámetros que afectan la huella (la admisión no la afecta)."""
    return {k: v for k, v in (hp or {}).items() if k != "admision"}


# ════════════════════════════════════════════════════ 1 · contabilidad de bytes


def _claves(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def verificar_contabilidad(
    nombre: str, capacity: int, dim: int, *, seed: int = 0, hp: dict | None = None
) -> dict[str, Any]:
    """Llena la memoria por encima de su capacidad y compara bytes reales con la fórmula.

    Para Spiking-SDM el conjunto activo por traza es estocástico: se le pasa a
    la fórmula el tamaño medio observado, y se reporta cuál fue.
    """
    mem = construir(nombre, dim, capacity, seed, hp)
    rng = np.random.default_rng(seed + 7000)
    for i, k in enumerate(_claves(rng, capacity + 5, dim)):
        mem.write(k, i, pred_error=float(rng.uniform(0.05, 0.95)))

    real = inventario_real(mem)
    hp_a = _hp_contable(hp)
    if nombre == "SpikingSDM":
        activas = [len(a) for a in mem._activas]
        hp_a["activos_por_traza"] = float(np.mean(activas)) if activas else 0.0
    analitico = bytes_analiticos(nombre, len(mem), dim, **hp_a)

    por_componente = {
        c: {"real": real["componentes"][c]["bytes"], "analitico": v["bytes"]}
        for c, v in analitico["componentes"].items()
    }
    return {
        "capacity": capacity,
        "dim": dim,
        "n_trazas": len(mem),
        "real": real,
        "analitico_total": analitico["total"],
        "por_componente": por_componente,
        "coincide": real["total"] == analitico["total"]
        and set(real["componentes"]) == set(analitico["componentes"])
        and all(v["real"] == v["analitico"] for v in por_componente.values()),
        "arreglos_no_inventariados": arreglos_no_inventariados(mem),
        **({"activos_por_traza": hp_a["activos_por_traza"]} if nombre == "SpikingSDM" else {}),
    }


def contabilidad(verbose: bool = True) -> dict[str, Any]:
    """Bytes por componente a C = 20 y verificación real vs. analítica en una grilla."""
    nombres = (*REGISTRADAS, "SDM-a", "SDM-a+", "SDM-b")
    grilla = [(20, 32), (5, 32), (80, 32), (20, 128)]
    out: dict[str, Any] = {"verificacion": {}, "C20": {}, "fijo": {}, "por_traza": {}}
    todo_coincide = True
    for n in nombres:
        out["verificacion"][n] = {}
        for C, d in grilla:
            if n in LENTAS and C > 20:
                continue
            v = verificar_contabilidad(n, C, d)
            out["verificacion"][n][f"C{C}_d{d}"] = {
                "coincide": v["coincide"],
                "real": v["real"]["total"],
                "analitico": v["analitico_total"],
                "no_inventariados": v["arreglos_no_inventariados"],
            }
            todo_coincide &= v["coincide"] and not v["arreglos_no_inventariados"]
            if (C, d) == (CAPACITY, DIM):
                out["C20"][n] = {
                    "total": v["real"]["total"],
                    "persistente_minimo": v["real"]["persistente_minimo"],
                    "por_clase": v["real"]["por_clase"],
                    "componentes": {c: x["bytes"] for c, x in v["real"]["componentes"].items()},
                    **(
                        {"activos_por_traza": v["activos_por_traza"]}
                        if "activos_por_traza" in v
                        else {}
                    ),
                }
        hp = {"activos_por_traza": out["C20"][n]["activos_por_traza"]} if n == "SpikingSDM" else {}
        fijo, por_traza = costo_fijo_y_por_traza(n, DIM, **hp)
        out["fijo"][n] = fijo
        out["por_traza"][n] = por_traza
    fifo = out["C20"]["FIFO"]["total"]
    out["razon_vs_FIFO_C20"] = {n: out["C20"][n]["total"] / fifo for n in nombres}
    out["todo_coincide"] = bool(todo_coincide)

    if verbose:
        print("\n── 1 · Bytes de estado a C = 20, d = 32 ──")
        print(
            f"{'arquitectura':<12}{'total':>10}{'mín. pers.':>12}{'fijo':>10}{'/traza':>9}{'×FIFO':>8}"
        )
        for n in nombres:
            c = out["C20"][n]
            print(
                f"{n:<12}{c['total']:>10}{c['persistente_minimo']:>12}{out['fijo'][n]:>10}"
                f"{out['por_traza'][n]:>9.0f}{out['razon_vs_FIFO_C20'][n]:>8.1f}"
            )
        print(f"real == analítico en toda la grilla: {todo_coincide}")
    return out


def diagnostico_spiking_sdm(dims: tuple[int, ...] = D_GRID, n: int = 200, seed: int = 0) -> dict:
    """Fracción de activaciones de Spiking-SDM que caen en el camino de respaldo.

    Replica la integración de `SpikingSDMMemory._activadas` para saber si alguna
    neurona LIF cruzó el umbral en la ventana. Si ninguna lo cruza, la memoria
    toma las `k_active` más integradas: la decisión de activación deja de ser un
    disparo y pasa a ser un top-k ruidoso.
    """
    out = {}
    for d in dims:
        mem = ARCHITECTURES["SpikingSDM"](dim=d, capacity=CAPACITY, seed=seed)
        rng = np.random.default_rng(seed + 11)
        respaldo, vmax = 0, []
        for k in _claves(rng, n, d):
            sp_pos, sp_neg = mem._codificar(k)
            vm = np.zeros(mem.n_hard, dtype=np.float32)
            disparo = np.zeros(mem.n_hard, dtype=bool)
            pico = 0.0
            for t in range(mem.T):
                vm = mem.leak * vm + (mem.W_pos @ sp_pos[t] + mem.W_neg @ sp_neg[t]) / mem.dim
                pico = max(pico, float(vm.max()))
                nuevas = (vm >= mem.v_th) & ~disparo
                disparo |= nuevas
                vm[nuevas] = 0.0
            respaldo += not disparo.any()
            vmax.append(pico)
        out[f"d{d}"] = {
            "fraccion_respaldo": respaldo / n,
            "potencial_max_medio": float(np.mean(vmax)),
            "potencial_max_max": float(np.max(vmax)),
            "v_th": float(mem.v_th),
        }
    return out


# ═════════════════════════════════════════════════════════ 2 · tiempo de pared


def _pctl(x: np.ndarray) -> dict[str, float]:
    return {
        "mediana_us": float(np.median(x)),
        "p10_us": float(np.percentile(x, 10)),
        "p90_us": float(np.percentile(x, 90)),
        "p99_us": float(np.percentile(x, 99)),
    }


def medir_una(
    nombre: str,
    capacity: int,
    dim: int,
    *,
    seed: int,
    n_medidas: int,
    n_calentamiento: int,
    hp: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tiempos individuales (µs) de `write` y `read` en régimen estacionario.

    La memoria se llena hasta su capacidad antes de medir, así que cada `write`
    medido incluye un desalojo. Las lecturas son versiones ruidosas de claves
    guardadas.
    """
    mem = construir(nombre, dim, capacity, seed, hp)
    rng = np.random.default_rng(seed + 9000)
    claves = _claves(rng, capacity + n_calentamiento + n_medidas, dim)
    pe = rng.uniform(0.05, 0.95, size=len(claves))
    for i in range(capacity + n_calentamiento):
        mem.write(claves[i], i, float(pe[i]))

    tw = np.empty(n_medidas)
    for j in range(n_medidas):
        i = capacity + n_calentamiento + j
        t0 = time.perf_counter_ns()
        mem.write(claves[i], i, float(pe[i]))
        tw[j] = (time.perf_counter_ns() - t0) / 1e3

    recientes = claves[-capacity:]
    consultas = recientes[rng.integers(len(recientes), size=n_calentamiento + n_medidas)]
    consultas = consultas + rng.standard_normal(consultas.shape).astype(np.float32) * 0.1
    for q in consultas[:n_calentamiento]:
        mem.read(q)
    tr = np.empty(n_medidas)
    for j, q in enumerate(consultas[n_calentamiento:]):
        t0 = time.perf_counter_ns()
        mem.read(q)
        tr[j] = (time.perf_counter_ns() - t0) / 1e3
    return tw, tr


def barrer_tiempos(
    configs: list[tuple[str, int, int, dict | None]],
    *,
    repeticiones: int = REPETICIONES,
    n_medidas: int = N_MEDIDAS,
    n_calentamiento: int = N_CALENTAMIENTO,
) -> dict[tuple, dict[str, Any]]:
    """Mide cada configuración `(nombre, C, d, hp)`, intercalándolas por repetición.

    Intercalar hace que la carga de fondo de la máquina afecte a todas las
    configuraciones por igual en vez de sesgar a las que se midieron en un mal
    momento.
    """
    muestras: dict[tuple, dict[str, list]] = {
        (n, C, d, tuple(sorted((hp or {}).items()))): {"w": [], "r": [], "med_w": [], "med_r": []}
        for n, C, d, hp in configs
    }
    carga = []
    for rep in range(repeticiones):
        carga.append(os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan"))
        for n, C, d, hp in configs:
            lenta = n in LENTAS
            tw, tr = medir_una(
                n,
                C,
                d,
                seed=rep,
                n_medidas=N_MEDIDAS_LENTA if lenta else n_medidas,
                n_calentamiento=N_CALENTAMIENTO_LENTA if lenta else n_calentamiento,
                hp=hp,
            )
            m = muestras[(n, C, d, tuple(sorted((hp or {}).items())))]
            m["w"].append(tw)
            m["r"].append(tr)
            m["med_w"].append(float(np.median(tw)))
            m["med_r"].append(float(np.median(tr)))

    out = {}
    for clave, m in muestras.items():
        w, r = np.concatenate(m["w"]), np.concatenate(m["r"])
        out[clave] = {
            "write": {**_pctl(w), "mediana_por_rep_us": m["med_w"]},
            "read": {**_pctl(r), "mediana_por_rep_us": m["med_r"]},
            "n_por_rep": len(m["w"][0]),
            "repeticiones": repeticiones,
        }
    out[("_carga",)] = {"loadavg_1min_por_rep": carga}
    return out


def pendiente_loglog(x: list[float], y: list[float], cola: int | None = None) -> float:
    """Pendiente de log(y) contra log(x); con `cola=n`, solo sobre los n puntos mayores."""
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    if cola is not None:
        xs, ys = xs[-cola:], ys[-cola:]
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def tiempos(verbose: bool = True) -> dict[str, Any]:
    """Barridos en C (d = 32), en d (C = 20 y C = 320) y en M para la SDM (C = 20, d = 32)."""
    nombres = (*REGISTRADAS, "SDM-a", "SDM-a+", "SDM-b")
    configs: list[tuple[str, int, int, dict | None]] = []
    for n in nombres:
        cs = C_GRID_LENTA if n in LENTAS else C_GRID
        ds = D_GRID_LENTA if n in LENTAS else D_GRID
        configs += [(n, C, DIM, None) for C in cs]
        configs += [(n, CAPACITY, d, None) for d in ds if d != DIM]
        if n not in LENTAS:
            configs += [(n, 320, d, None) for d in ds]
    configs += [("SDM", CAPACITY, DIM, {"n_hard": M}) for M in M_GRID]
    unicas: dict[tuple, tuple[str, int, int, dict | None]] = {}
    for n, C, d, hp in configs:
        unicas.setdefault((n, C, d, tuple(sorted((hp or {}).items()))), (n, C, d, hp))
    configs = list(unicas.values())

    t0 = time.time()
    crudo = barrer_tiempos(configs)
    duracion = time.time() - t0

    def get(n, C, d, hp=None):
        return crudo[(n, C, d, tuple(sorted((hp or {}).items())))]

    out: dict[str, Any] = {
        "barrido_C": {},
        "barrido_d_C20": {},
        "barrido_d_C320": {},
        "barrido_M_SDM": {},
        "pendientes": {},
        "C20_d32": {},
        "complejidad_analitica": COMPLEJIDAD_ANALITICA,
        "exponente_asintotico": {
            n: dict(zip(("write_C", "write_d", "read_C", "read_d"), e, strict=True))
            for n, e in EXPONENTE_ASINTOTICO.items()
        },
        "carga_loadavg": crudo[("_carga",)]["loadavg_1min_por_rep"],
        "duracion_s": round(duracion, 1),
    }
    for n in nombres:
        cs = C_GRID_LENTA if n in LENTAS else C_GRID
        ds = D_GRID_LENTA if n in LENTAS else D_GRID
        out["barrido_C"][n] = {f"C{C}": get(n, C, DIM) for C in cs}
        out["barrido_d_C20"][n] = {f"d{d}": get(n, CAPACITY, d) for d in ds}
        if n not in LENTAS:
            out["barrido_d_C320"][n] = {f"d{d}": get(n, 320, d) for d in ds}
        out["C20_d32"][n] = {
            "write_us": get(n, CAPACITY, DIM)["write"]["mediana_us"],
            "read_us": get(n, CAPACITY, DIM)["read"]["mediana_us"],
        }
        pend = {}
        for op in ("write", "read"):
            yc = [get(n, C, DIM)[op]["mediana_us"] for C in cs]
            pend[f"{op}_C"] = {
                "completa": pendiente_loglog(cs, yc),
                "cola3": pendiente_loglog(cs, yc, 3),
            }
            yd = [get(n, CAPACITY, d)[op]["mediana_us"] for d in ds]
            pend[f"{op}_d_C20"] = {
                "completa": pendiente_loglog(ds, yd),
                "cola3": pendiente_loglog(ds, yd, 3),
            }
            if n not in LENTAS:
                yd3 = [get(n, 320, d)[op]["mediana_us"] for d in ds]
                pend[f"{op}_d_C320"] = {
                    "completa": pendiente_loglog(ds, yd3),
                    "cola3": pendiente_loglog(ds, yd3, 3),
                }
        out["pendientes"][n] = pend
    out["barrido_M_SDM"] = {f"M{M}": get("SDM", CAPACITY, DIM, {"n_hard": M}) for M in M_GRID}
    for op in ("write", "read"):
        ym = [get("SDM", CAPACITY, DIM, {"n_hard": M})[op]["mediana_us"] for M in M_GRID]
        out["pendientes"]["SDM"][f"{op}_M"] = {
            "completa": pendiente_loglog(M_GRID, ym),
            "cola3": pendiente_loglog(M_GRID, ym, 3),
        }

    if verbose:
        print(f"\n── 2 · Tiempo por operación (mediana, µs) a C = 20, d = 32 · {duracion:.0f} s ──")
        print(
            f"{'arquitectura':<12}{'write':>10}{'read':>10}{'∂logW/∂logC':>13}{'∂logR/∂logd':>13}"
        )
        for n in nombres:
            c = out["C20_d32"][n]
            p = out["pendientes"][n]
            print(
                f"{n:<12}{c['write_us']:>10.1f}{c['read_us']:>10.1f}"
                f"{p['write_C']['cola3']:>13.2f}{p['read_d_C20']['cola3']:>13.2f}"
            )
    return out


# ══════════════════════════════════════════════════ 3-5 · evaluación por semilla


def evaluar(job: dict[str, Any]) -> dict[str, Any]:
    """Gate de reconstrucción y batería para una configuración y una semilla.

    Si `capacity` es `None`, la memoria recibe la capacidad que pide cada tarea
    (el protocolo de `exp04`). Si es un entero, esa capacidad fija se usa en
    toda tarea: es lo que significa darle a la arquitectura un presupuesto.
    Las tareas y sus flujos no cambian; solo cambia el tamaño de la memoria.
    """
    nombre, hp, cap_fija, s = job["nombre"], job.get("hp"), job.get("capacity"), job["seed"]

    def factory(capacity: int, seed: int):
        return construir(nombre, DIM, cap_fija if cap_fija is not None else capacity, seed, hp)

    t0 = time.time()
    gate = reconstruction_gate(factory, seeds=(s,), dim=DIM)
    out = {"gate": dict(gate.per_task), "gate_mean": gate.mean}
    if job.get("bateria", True):
        bat = run_battery(factory, seeds=(s,), dim=DIM, capacity=CAPACITY)
        out["battery"] = dict(bat.per_task)
        out["battery_mean"] = bat.mean
    out["segundos"] = time.time() - t0
    return out


def ic_bootstrap(valores: list[float], *, n_boot: int = N_BOOT, seed: int = 0) -> list[float]:
    """IC percentil 95 % de la media, remuestreando la unidad independiente (semilla)."""
    v = np.asarray(valores, float)
    if v.size < 2:
        return [float(v.mean()), float(v.mean())] if v.size else [float("nan")] * 2
    rng = np.random.default_rng(seed)
    medias = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return [float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))]


def resumir(por_semilla: list[dict[str, Any]]) -> dict[str, Any]:
    """Media, desvío e IC bootstrap sobre semillas, por tarea y agregados."""
    out: dict[str, Any] = {"n_semillas": len(por_semilla)}
    gm = [r["gate_mean"] for r in por_semilla]
    out["gate_mean"] = float(np.mean(gm))
    out["gate_std"] = float(np.std(gm))
    out["gate_ci"] = ic_bootstrap(gm)
    out["pasa_gate"] = out["gate_mean"] >= RECON_THRESHOLD
    out["gate"] = {
        t: float(np.mean([r["gate"][t] for r in por_semilla])) for t in por_semilla[0]["gate"]
    }
    if "battery" in por_semilla[0]:
        bm = [r["battery_mean"] for r in por_semilla]
        out["battery_mean"] = float(np.mean(bm))
        out["battery_std"] = float(np.std(bm))
        out["battery_ci"] = ic_bootstrap(bm)
        out["battery_por_semilla"] = [float(x) for x in bm]
        out["battery"] = {}
        for t in por_semilla[0]["battery"]:
            v = [r["battery"][t] for r in por_semilla]
            out["battery"][t] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "ci": ic_bootstrap(v),
            }
    return out


def diferencia_pareada(a: list[float], b: list[float], *, seed: int = 0) -> dict[str, Any]:
    """Diferencia media `a - b` sobre las mismas semillas, con IC bootstrap pareado."""
    d = np.asarray(a, float) - np.asarray(b, float)
    ci = ic_bootstrap(list(d), seed=seed)
    return {"diff": float(d.mean()), "ci": ci, "resuelta": bool(ci[0] > 0 or ci[1] < 0)}


def _correr_jobs(jobs: list[dict[str, Any]], n_jobs: int) -> list[dict[str, Any]]:
    # Las lentas primero, para que no queden solas al final del pool.
    orden = sorted(range(len(jobs)), key=lambda i: jobs[i]["nombre"] not in LENTAS)
    if n_jobs == 1:
        res = {i: evaluar(jobs[i]) for i in orden}
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futuros = {i: pool.submit(evaluar, jobs[i]) for i in orden}
            res = {i: f.result() for i, f in futuros.items()}
    return [res[i] for i in range(len(jobs))]


def evaluar_configs(
    configs: dict[str, dict[str, Any]],
    *,
    seeds: tuple[int, ...] = SEMILLAS,
    seeds_lentas: tuple[int, ...] = SEMILLAS_LENTAS,
    n_jobs: int = -1,
) -> dict[str, dict[str, Any]]:
    """Evalúa un conjunto de configuraciones `{etiqueta: {nombre, hp, capacity, bateria}}`."""
    n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    jobs, etiquetas = [], []
    for et, cfg in configs.items():
        ss = seeds_lentas if cfg["nombre"] in LENTAS else seeds
        for s in ss:
            jobs.append({**cfg, "seed": s})
            etiquetas.append(et)
    resultados = _correr_jobs(jobs, n_jobs)
    agrupados: dict[str, list] = {et: [] for et in configs}
    for et, r in zip(etiquetas, resultados, strict=True):
        agrupados[et].append(r)
    return {et: resumir(rs) for et, rs in agrupados.items()}


# ═════════════════════════════════════════════════════ 3 · igual presupuesto


def _m_escalado(nombre: str, presupuesto: int, dim: int) -> int | None:
    """Regla a priori: mayor M potencia de 2 cuyo costo fijo no pase de la mitad de B."""
    mejor = None
    M = M_MIN
    while M <= M_MAX:
        fijo, _ = costo_fijo_y_por_traza(nombre, dim, n_hard=M)
        if fijo <= presupuesto // 2:
            mejor = M
        M *= 2
    return mejor


def derivar_config(nombre: str, presupuesto: int, modo: str, dim: int = DIM) -> dict[str, Any]:
    """Capacidad e hiperparámetros de una arquitectura para un presupuesto en bytes.

    - `defaults`: hiperparámetros por defecto; C = lo que entra después del
      costo fijo del sustrato. Si el costo fijo ya supera B, es infactible.
    - `sustrato_escalado`: a las SDM se les elige M (potencia de 2, entre 4 y
      2048) para que su costo fijo no supere B/2, y C con lo que sobra. Regla
      fijada antes de ver resultados; no se optimiza M por puntaje. Spiking no
      entra en este modo (ver nota en `main`).
    """
    hp: dict[str, Any] = {}
    if modo == "sustrato_escalado" and nombre in ("SDM", "SpikingSDM"):
        M = _m_escalado(nombre, presupuesto, dim)
        if M is None:
            return {"factible": False, "capacity": 0, "hp": {}}
        hp["n_hard"] = M
    elif modo == "sustrato_escalado" and nombre in LENTAS:
        return {"factible": False, "capacity": 0, "hp": {}, "excluida": True}
    extra = (
        {"activos_por_traza": max(1, int(hp.get("n_hard", 256) * 0.05))}
        if nombre == "SpikingSDM"
        else {}
    )
    C = capacidad_para_presupuesto(nombre, presupuesto, dim, **hp, **extra)
    bytes_usados = bytes_analiticos(nombre, C, dim, **hp, **extra)["total"] if C else None
    return {
        "factible": C >= 1,
        "capacity": int(C),
        "hp": hp,
        "bytes": bytes_usados,
        "utilizacion": (bytes_usados / presupuesto) if bytes_usados else None,
    }


def igual_presupuesto(
    *,
    presupuestos: tuple[int, ...] = PRESUPUESTOS,
    nombres: tuple[str, ...] = (*REGISTRADAS, "SDM-b"),
    seeds: tuple[int, ...] = SEMILLAS,
    seeds_lentas: tuple[int, ...] = SEMILLAS_LENTAS,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, Any]:
    """Protocolo de `exp04` a igual presupuesto de bytes, en dos modos, más la referencia C = 20.

    `SDM-b` (la lista de trazas de la SDM, sin contadores) entra como referencia
    marcada: no es una arquitectura registrada, pero es la que dice qué se
    compraría con los bytes de los contadores.
    """
    configs: dict[str, dict[str, Any]] = {}
    derivadas: dict[str, dict[str, dict[str, Any]]] = {}
    for n in nombres:
        configs[f"C20|{n}"] = {
            "nombre": n,
            "hp": None,
            "capacity": None,
            "bateria": n not in LENTAS,
        }
    for modo in ("defaults", "sustrato_escalado"):
        for B in presupuestos:
            for n in nombres:
                d = derivar_config(n, B, modo)
                derivadas.setdefault(f"{modo}|B{B}", {})[n] = d
                if d["factible"]:
                    configs[f"{modo}|B{B}|{n}"] = {
                        "nombre": n,
                        "hp": d["hp"] or None,
                        "capacity": d["capacity"],
                        "bateria": n not in LENTAS,
                    }
    # Configuraciones idénticas (p. ej. FIFO en ambos modos) se evalúan una vez.
    unicas: dict[tuple, str] = {}
    alias: dict[str, str] = {}
    for et, c in configs.items():
        firma = (c["nombre"], tuple(sorted((c["hp"] or {}).items())), c["capacity"])
        alias[et] = unicas.setdefault(firma, et)
    evaluadas = evaluar_configs(
        {et: configs[et] for et in set(alias.values())},
        seeds=seeds,
        seeds_lentas=seeds_lentas,
        n_jobs=n_jobs,
    )
    res = {et: evaluadas[alias[et]] for et in configs}

    def bloque(prefijo: str, deriv: dict[str, dict] | None) -> dict[str, Any]:
        filas = {}
        for n in nombres:
            et = f"{prefijo}|{n}"
            fila: dict[str, Any] = {"referencia": n == "SDM-b"}
            if deriv is not None:
                fila.update(deriv[n])
            else:
                fila.update({"factible": True, "capacity": CAPACITY, "hp": {}})
            if et in res:
                fila.update(res[et])
            filas[n] = fila
        # Orden por media de la batería entre las registradas que pasan el gate
        # (protocolo de exp04), y también entre todas las factibles evaluadas.
        evaluadas_b = [n for n in nombres if "battery_mean" in filas[n] and n != "SDM-b"]
        admitidas = [n for n in evaluadas_b if filas[n]["pasa_gate"]]
        orden = sorted(admitidas, key=lambda n: -filas[n]["battery_mean"])
        pares = {}
        for a, b in zip(orden, orden[1:], strict=False):
            pares[f"{a}-{b}"] = diferencia_pareada(
                filas[a]["battery_por_semilla"], filas[b]["battery_por_semilla"]
            )
        vs_fifo = {}
        if "FIFO" in evaluadas_b:
            for n in [
                *evaluadas_b,
                *(["SDM-b"] if "battery_mean" in filas.get("SDM-b", {}) else []),
            ]:
                if n != "FIFO":
                    vs_fifo[n] = diferencia_pareada(
                        filas[n]["battery_por_semilla"], filas["FIFO"]["battery_por_semilla"]
                    )
        return {
            "filas": filas,
            "orden_admitidas": orden,
            "diferencias_adyacentes": pares,
            "diferencia_vs_FIFO": vs_fifo,
        }

    out: dict[str, Any] = {
        "referencia_C20": bloque("C20", None),
        "presupuestos": list(presupuestos),
    }
    orden_ref = out["referencia_C20"]["orden_admitidas"]
    for modo in ("defaults", "sustrato_escalado"):
        out[modo] = {}
        for B in presupuestos:
            b = bloque(f"{modo}|B{B}", derivadas[f"{modo}|B{B}"])
            comunes = [n for n in orden_ref if n in b["orden_admitidas"]]
            ref_restringido = [n for n in orden_ref if n in comunes]
            nuevo_restringido = [n for n in b["orden_admitidas"] if n in comunes]
            b["orden_cambia_vs_C20"] = ref_restringido != nuevo_restringido
            out[modo][f"B{B}"] = b

    if verbose:
        print("\n── 3 · Media de la batería a igual presupuesto (C derivada) ──")
        for modo in ("defaults", "sustrato_escalado"):
            print(f"\n  modo {modo}")
            print(f"  {'B':>8}  " + "".join(f"{n:>16}" for n in nombres))
            for B in presupuestos:
                filas = out[modo][f"B{B}"]["filas"]
                celdas = []
                for n in nombres:
                    f = filas[n]
                    if not f.get("factible"):
                        celdas.append(f"{'—':>16}")
                    elif "battery_mean" in f:
                        celdas.append(f"{f['battery_mean']:>9.3f} C={f['capacity']:<4}")
                    else:
                        celdas.append(f"{'gate ' + format(f['gate_mean'], '.2f'):>16}")
                print(f"  {B:>8}  " + "".join(celdas))
            print(f"  orden C=20: {orden_ref}")
            for B in presupuestos:
                print(f"  orden B={B}: {out[modo][f'B{B}']['orden_admitidas']}")
    return out


# ═══════════════════════════════════════════════════════════ 4 · ablación


def ablacion_sdm(
    *, seeds: tuple[int, ...] = SEMILLAS, n_jobs: int = -1, verbose: bool = True
) -> dict[str, Any]:
    """(a) solo contadores, (a+) con borrado exacto, (b) solo trazas, (c) completa, a C = 20."""
    configs = {
        v: {"nombre": v, "hp": None, "capacity": None, "bateria": True} for v in VARIANTES_SDM
    }
    res = evaluar_configs(configs, seeds=seeds, n_jobs=n_jobs)
    diffs = {}
    for a, b in (("SDM", "SDM-b"), ("SDM", "SDM-a"), ("SDM", "SDM-a+"), ("SDM-a+", "SDM-a")):
        diffs[f"{a}-{b}"] = {
            "battery_mean": diferencia_pareada(
                res[a]["battery_por_semilla"], res[b]["battery_por_semilla"]
            ),
        }
    out = {"variantes": res, "diferencias": diffs}
    if verbose:
        print("\n── 4 · Ablación de la SDM (C = 20) ──")
        print(f"{'variante':<9}{'gate':>7}{'raros':>8}{'ruido':>8}{'interf.':>9}{'media':>8}  IC")
        for v in VARIANTES_SDM:
            r = res[v]
            b = r["battery"]
            print(
                f"{v:<9}{r['gate_mean']:>7.3f}{b['rare_retention']['mean']:>8.3f}"
                f"{b['noise_under_pressure']['mean']:>8.3f}{b['sequential_interference']['mean']:>9.3f}"
                f"{r['battery_mean']:>8.3f}  [{r['battery_ci'][0]:.3f}, {r['battery_ci'][1]:.3f}]"
            )
        for k, d in diffs.items():
            x = d["battery_mean"]
            print(f"  {k}: {x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]")
    return out


# ═══════════════════════════════════════════════════════ 5 · sensibilidad


def _k(x: float) -> str:
    """Etiqueta de un real apta para claves de `\\result{}`: sin puntos ni barras."""
    return f"{x:g}".replace(".", "p")


def sensibilidad_sdm(
    *,
    m_grid: tuple[int, ...] = M_GRID,
    frac_grid: tuple[float, ...] = FRAC_GRID,
    grid_2d_m: tuple[int, ...] = GRID_2D_M,
    grid_2d_frac: tuple[float, ...] = GRID_2D_FRAC,
    admision_grid: tuple[float, ...] = ADMISION_GRID,
    seeds: tuple[int, ...] = SEMILLAS,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, Any]:
    """Barridos 1D de M, fracción y admisión alrededor del default, y 2D de M × fracción."""
    configs: dict[str, dict[str, Any]] = {}

    def cfg(hp: dict) -> dict[str, Any]:
        return {"nombre": "SDM", "hp": hp, "capacity": None, "bateria": True}

    for M in m_grid:
        configs[f"M{M}|f0.05"] = cfg({"n_hard": M, "activation_frac": 0.05})
    for f in frac_grid:
        configs[f"M512|f{f:g}"] = cfg({"n_hard": 512, "activation_frac": f})
    for M in grid_2d_m:
        for f in grid_2d_frac:
            configs[f"M{M}|f{f:g}"] = cfg({"n_hard": M, "activation_frac": f})
    for tau in admision_grid:
        configs[f"adm{tau:g}"] = cfg({"admision": tau})
    res = evaluar_configs(configs, seeds=seeds, n_jobs=n_jobs)

    def fila(et: str, hp: dict) -> dict[str, Any]:
        r = res[et]
        hp_c = _hp_contable(hp)
        M = hp_c.get("n_hard", 512)
        k = max(1, int(M * hp_c.get("activation_frac", 0.05)))
        return {
            **r,
            "n_hard": M,
            "k_active": k,
            "bytes_C20": bytes_analiticos("SDM", CAPACITY, DIM, **hp_c)["total"],
        }

    out: dict[str, Any] = {
        "n_hard": {f"M{M}": fila(f"M{M}|f0.05", {"n_hard": M}) for M in m_grid},
        "activation_frac": {
            f"f{_k(f)}": fila(f"M512|f{f:g}", {"activation_frac": f}) for f in frac_grid
        },
        "grilla_2d": {
            f"M{M}_f{_k(f)}": fila(f"M{M}|f{f:g}", {"n_hard": M, "activation_frac": f})
            for M in grid_2d_m
            for f in grid_2d_frac
        },
        "admision": {f"tau{_k(t)}": fila(f"adm{t:g}", {}) for t in admision_grid},
    }
    bm = [out["n_hard"][f"M{M}"]["battery_mean"] for M in m_grid]
    out["rango_battery_n_hard"] = float(max(bm) - min(bm))
    bf = [out["activation_frac"][f"f{_k(f)}"]["battery_mean"] for f in frac_grid]
    out["rango_battery_activation_frac"] = float(max(bf) - min(bf))
    b2 = [v["battery_mean"] for v in out["grilla_2d"].values()]
    out["rango_battery_grilla_2d"] = float(max(b2) - min(b2))

    if verbose:
        print("\n── 5 · Sensibilidad de la SDM (media batería [IC], gate) ──")
        for nombre_barrido in ("n_hard", "activation_frac", "admision"):
            print(f"  {nombre_barrido}")
            for et, r in out[nombre_barrido].items():
                print(
                    f"    {et:<10} k={r['k_active']:<4} bat={r['battery_mean']:.3f} "
                    f"[{r['battery_ci'][0]:.3f}, {r['battery_ci'][1]:.3f}]  "
                    f"raros={r['battery']['rare_retention']['mean']:.3f}  gate={r['gate_mean']:.3f}"
                )
        print("  grilla 2D (media batería)")
        print("    M\\f  " + "".join(f"{f:>8g}" for f in grid_2d_frac))
        for M in grid_2d_m:
            print(
                f"    {M:<6}"
                + "".join(
                    f"{out['grilla_2d'][f'M{M}_f{_k(f)}']['battery_mean']:>8.3f}"
                    for f in grid_2d_frac
                )
            )
    return out


# ═══════════════════════════════════════════════════════════════════ main


def hardware() -> dict[str, Any]:
    """Descripción de la máquina, para el manifiesto y los datos."""
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("model name"):
                    cpu = linea.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    blas = "desconocido"
    try:
        cfg = np.show_config(mode="dicts")
        b = cfg["Build Dependencies"]["blas"]
        blas = f"{b.get('name')} {b.get('version')}"
    except Exception:  # noqa: BLE001 - diagnóstico, no puede abortar el experimento
        pass
    return {
        "cpu": cpu,
        "n_cpu_logicas": os.cpu_count(),
        "plataforma": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": blas,
        "hilos_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "loadavg_al_inicio": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }


def main() -> int:
    with ExperimentRun("exp12_substrate_cost") as run:
        run.set_seeds(SEMILLAS)
        hw = hardware()
        run.record("hardware", hw)
        run.note(
            f"hardware: {hw['cpu']}, {hw['n_cpu_logicas']} CPU lógicas, numpy {hw['numpy']} "
            f"con {hw['blas']}; loadavg al inicio {hw['loadavg_al_inicio']}. La máquina "
            "se compartía con otros experimentos: los tiempos se midieron en serie e "
            "intercalados, pero los µs absolutos tienen ruido de carga."
        )
        run.record(
            "config",
            {
                "dim": DIM,
                "capacity": CAPACITY,
                "semillas": list(SEMILLAS),
                "semillas_lentas": list(SEMILLAS_LENTAS),
                "presupuestos": list(PRESUPUESTOS),
                "gate_threshold": RECON_THRESHOLD,
                "n_boot": N_BOOT,
            },
        )

        cont = contabilidad()
        run.record("contabilidad", cont)
        if not cont["todo_coincide"]:
            run.note("ATENCIÓN: la contabilidad real y la analítica no coinciden en algún punto")

        diag = diagnostico_spiking_sdm()
        run.record("spiking_sdm_respaldo", diag)
        if diag[f"d{DIM}"]["fraccion_respaldo"] > 0.5:
            run.note(
                "Spiking-SDM: a d = 32 ninguna neurona LIF cruza el umbral en la ventana "
                f"(fracción de respaldo {diag[f'd{DIM}']['fraccion_respaldo']:.2f}, potencial "
                f"máximo {diag[f'd{DIM}']['potencial_max_max']:.2f} < v_th "
                f"{diag[f'd{DIM}']['v_th']:.2f}): la activación es siempre el top-k de "
                "potencial integrado, no un disparo."
            )

        run.record("tiempos", tiempos())

        run.note(
            "Spiking se excluye del modo sustrato_escalado: no pasa el gate con sus "
            "valores por defecto, y escalar n_neurons eleva su write O(P·L·n²) fuera del "
            "presupuesto de cómputo. En modo defaults entra con 5 semillas y solo gate."
        )
        run.record("igual_presupuesto", igual_presupuesto())
        run.record("ablacion_sdm", ablacion_sdm())
        run.note(
            "La SDM no tiene umbral de admisión propio: se barre con una compuerta "
            "externa (AdmissionGated + PredErrorAdmission) que no entra al espacio de "
            "búsqueda. El umbral de fusión lo barre exp07."
        )
        run.record("sensibilidad_sdm", sensibilidad_sdm())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
