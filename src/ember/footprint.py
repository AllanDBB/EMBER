"""Huella de memoria de las arquitecturas: bytes de estado, por componente.

Comparar sustratos por "capacidad en número de trazas" esconde que una traza no
cuesta lo mismo en todos: la SDM, además de su lista de trazas, materializa 512
direcciones y 512 contadores de dimensión `d` que existen aunque la memoria
tenga una sola traza. Este módulo cuenta los bytes de dos maneras
independientes, para que una verifique a la otra:

- `inventario_real(mem)`: suma el `nbytes` de cada arreglo que la instancia
  tiene vivo, agrupado por componente.
- `bytes_analiticos(nombre, capacity, dim, **hp)`: la fórmula cerrada en función
  de `C`, `d` y los hiperparámetros.

`arreglos_no_inventariados(mem)` recorre los atributos por introspección y
devuelve los arreglos que el inventario no cubrió: si no está vacío, el
inventario está incompleto.

Convenciones de conteo
----------------------
- **Payload**: se cuenta una referencia de 8 bytes por traza (`payload_refs`),
  no el objeto apuntado. Todas las arquitecturas devuelven el mismo payload
  exacto, así que su tamaño es idéntico entre ellas y no discrimina.
- Cada componente lleva una **clase**:
  `trazas` (crece con C), `sustrato` (fijo, aprendido o escrito),
  `regenerable` (fijo y derivable de la semilla: direcciones de hard locations,
  vectores de decodificación, topología — se podría recomputar en vez de
  guardar, a costo de cómputo), `transitorio` (estado de la dinámica que se
  reinicia en cada lectura).
- `total` suma todo lo materializado. `persistente_minimo` excluye
  `regenerable` y `transitorio`: es la cota inferior de lo que habría que
  guardar para no perder nada.

Solo depende de numpy.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from ember.core.memory import PolicyMemory
from ember.core.store import TraceStore
from ember.memories.enn import ENNMemory
from ember.memories.sdm import SDMMemory
from ember.memories.sdm_ablation import AdmissionGated, SDMCountersOnly
from ember.memories.spiking import SpikingMemory
from ember.memories.spiking_sdm import SpikingSDMMemory

F32 = np.dtype(np.float32).itemsize
INTP = np.dtype(np.intp).itemsize
INT = np.dtype(int).itemsize
BOOL = np.dtype(np.bool_).itemsize
PTR = 8
"""Bytes de una referencia a payload (un puntero de 64 bits)."""

CLASES = ("trazas", "sustrato", "regenerable", "transitorio")

METADATOS_POR_TRAZA = ("strength", "age", "utility", "contribution", "last_use", "priority")
"""Escalares float32 por traza del `TraceStore`, además de la clave.

`last_use` y `priority` los agregó `exp09` para las políticas externas (LRU,
repetición priorizada). Existen en toda memoria aunque su política no los lea,
así que se cuentan: son estado materializado.
"""


def _comp(nbytes: int, clase: str) -> dict[str, Any]:
    assert clase in CLASES, clase
    return {"bytes": int(nbytes), "clase": clase}


def _resumen(componentes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = sum(c["bytes"] for c in componentes.values())
    minimo = sum(c["bytes"] for c in componentes.values() if c["clase"] in ("trazas", "sustrato"))
    por_clase = {
        k: sum(c["bytes"] for c in componentes.values() if c["clase"] == k) for k in CLASES
    }
    return {
        "componentes": componentes,
        "por_clase": por_clase,
        "total": int(total),
        "persistente_minimo": int(minimo),
    }


# ═══════════════════════════════════════════════════════════════ inventario real


def _store_real(store: TraceStore) -> dict[str, dict[str, Any]]:
    return {
        "claves_trazas": _comp(store.keys.nbytes, "trazas"),
        "metadatos": _comp(
            sum(getattr(store, nombre).nbytes for nombre in METADATOS_POR_TRAZA),
            "trazas",
        ),
        "payload_refs": _comp(PTR * len(store.values), "trazas"),
    }


def _nbytes_lista(arreglos: list[np.ndarray]) -> int:
    return int(sum(a.nbytes for a in arreglos))


def inventario_real(mem: Any) -> dict[str, Any]:
    """Bytes de cada componente que la instancia tiene materializado ahora."""
    if isinstance(mem, AdmissionGated):
        return inventario_real(mem.inner)

    if isinstance(mem, SDMCountersOnly):
        comp = _store_real(mem.store)
        # En esta variante las "claves" del store son los códigos de payload.
        comp["codigos_payload"] = comp.pop("claves_trazas")
        comp["direcciones"] = _comp(mem.H.nbytes, "regenerable")
        comp["contadores"] = _comp(mem.V.nbytes, "sustrato")
        if mem.erase_keys is not None:
            comp["claves_borrado"] = _comp(mem.erase_keys.nbytes, "trazas")
        return _resumen(comp)

    comp = _store_real(mem.store)
    if isinstance(mem, SDMMemory):
        comp["direcciones"] = _comp(mem.H.nbytes, "regenerable")
        comp["contadores"] = _comp(mem.V.nbytes, "sustrato")
    elif isinstance(mem, SpikingSDMMemory):
        comp["direcciones"] = _comp(mem.H.nbytes, "regenerable")
        comp["pesos_derivados"] = _comp(mem.W_pos.nbytes + mem.W_neg.nbytes, "regenerable")
        comp["contadores"] = _comp(mem.V.nbytes, "sustrato")
        comp["conjuntos_activos"] = _comp(_nbytes_lista(mem._activas), "trazas")
    elif isinstance(mem, SpikingMemory):
        comp["decodificadores"] = _comp(mem.D.nbytes, "regenerable")
        comp["mascara"] = _comp(mem.mask.nbytes, "regenerable")
        comp["pesos_sinapticos"] = _comp(mem.W.nbytes, "sustrato")
        comp["ensambles"] = _comp(_nbytes_lista(mem.store_patterns), "trazas")
        comp["estado_transitorio"] = _comp(
            mem.V.nbytes
            + mem.refrac.nbytes
            + mem.x_trace.nbytes
            + mem.y_trace.nbytes
            + mem.last_spikes.nbytes,
            "transitorio",
        )
    elif isinstance(mem, (ENNMemory, PolicyMemory)):
        pass
    else:
        raise TypeError(f"sin inventario para {type(mem).__name__}")
    return _resumen(comp)


def arreglos_no_inventariados(mem: Any) -> list[str]:
    """Atributos-arreglo de la instancia que `inventario_real` no contó.

    Recorre `__dict__` (y los slots del `TraceStore`) buscando `ndarray` o listas
    de `ndarray`. Los nombres cubiertos se declaran abajo; cualquier otro es un
    agujero en el inventario.
    """
    if isinstance(mem, AdmissionGated):
        return arreglos_no_inventariados(mem.inner)
    cubiertos = {
        "H",
        "V",
        "W",
        "W_pos",
        "W_neg",
        "D",
        "mask",
        "_activas",
        "store_patterns",
        "refrac",
        "x_trace",
        "y_trace",
        "last_spikes",
        "erase_keys",
    }
    faltan = []
    for nombre, valor in vars(mem).items():
        es_arreglo = isinstance(valor, np.ndarray) or (
            isinstance(valor, list) and valor and isinstance(valor[0], np.ndarray)
        )
        if es_arreglo and nombre not in cubiertos:
            faltan.append(nombre)
    store = mem.store
    for nombre in TraceStore.__slots__:
        if isinstance(getattr(store, nombre), np.ndarray) and nombre not in (
            "keys",
            *METADATOS_POR_TRAZA,
        ):
            faltan.append(f"store.{nombre}")
    return faltan


# ═══════════════════════════════════════════════════════════ fórmula analítica


def _defaults(cls: Any) -> dict[str, Any]:
    return {
        n: p.default
        for n, p in inspect.signature(cls.__init__).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def _store_analitico(C: int, dim: int) -> dict[str, dict[str, Any]]:
    return {
        "claves_trazas": _comp(C * dim * F32, "trazas"),
        "metadatos": _comp(len(METADATOS_POR_TRAZA) * C * F32, "trazas"),
        "payload_refs": _comp(C * PTR, "trazas"),
    }


ARQUITECTURAS_CONTABLES = (
    "FIFO",
    "ENN",
    "SDM",
    "SpikingSDM",
    "Spiking",
    "SDM-a",
    "SDM-a+",
    "SDM-b",
)


def bytes_analiticos(nombre: str, capacity: int, dim: int, **hp: Any) -> dict[str, Any]:
    """Bytes de estado con `capacity` trazas guardadas, por fórmula cerrada.

    Por componente, con `C = capacity`, `d = dim`, `M = n_hard`:

    | Arquitectura | Componente            | Bytes                     |
    |--------------|-----------------------|---------------------------|
    | toda         | claves + metadatos    | `4·C·d + 24·C + 8·C`      |
    | SDM          | direcciones, contad.  | `4·M·d` cada uno          |
    | Spiking-SDM  | H, W+, W−, contadores | `4·M·d` cada uno          |
    |              | conjuntos activos     | `8·C·a` (a = activas/traza)|
    | Spiking      | D; máscara; W         | `4·n·d`; `n²`; `4·n²`     |
    |              | ensambles             | `8·C·k`, k = max(2, ⌊n·s⌋)|
    |              | estado LIF            | `4·4·n + 8·n`             |
    | SDM-a        | contadores            | `4·M·(d + p)`, p = código |
    |              | códigos (en lugar de claves) | `4·C·p`            |
    | SDM-a+       | + claves de borrado   | `4·C·d`                   |

    Para Spiking-SDM el conjunto activo es estocástico; `activos_por_traza`
    permite pasar el tamaño medio observado (por defecto `k_active`, que es lo
    que devuelve el camino de respaldo cuando ninguna neurona cruza el umbral).
    """
    C, d = int(capacity), int(dim)

    if nombre in ("FIFO", "ENN", "SDM-b"):
        return _resumen(_store_analitico(C, d))

    if nombre == "SDM":
        p = {**_defaults(SDMMemory), **hp}
        M = int(p["n_hard"])
        comp = _store_analitico(C, d)
        comp["direcciones"] = _comp(M * d * F32, "regenerable")
        comp["contadores"] = _comp(M * d * F32, "sustrato")
        return _resumen(comp)

    if nombre == "SpikingSDM":
        p = {**_defaults(SpikingSDMMemory), **hp}
        M = int(p["n_hard"])
        k = max(1, int(M * p["activation_frac"]))
        a = float(p.get("activos_por_traza", k))
        comp = _store_analitico(C, d)
        comp["direcciones"] = _comp(M * d * F32, "regenerable")
        comp["pesos_derivados"] = _comp(2 * M * d * F32, "regenerable")
        comp["contadores"] = _comp(M * d * F32, "sustrato")
        comp["conjuntos_activos"] = _comp(round(C * a * INTP), "trazas")
        return _resumen(comp)

    if nombre == "Spiking":
        p = {**_defaults(SpikingMemory), **hp}
        n = int(p["n_neurons"])
        k = max(2, int(n * p["sparsity"]))
        comp = _store_analitico(C, d)
        comp["decodificadores"] = _comp(n * d * F32, "regenerable")
        comp["mascara"] = _comp(n * n * BOOL, "regenerable")
        comp["pesos_sinapticos"] = _comp(n * n * F32, "sustrato")
        comp["ensambles"] = _comp(C * k * INTP, "trazas")
        comp["estado_transitorio"] = _comp(4 * n * F32 + n * INT, "transitorio")
        return _resumen(comp)

    if nombre in ("SDM-a", "SDM-a+"):
        p = {**_defaults(SDMCountersOnly), **hp}
        M = int(p["n_hard"])
        cd = int(p["code_dim"] or d)
        comp = {
            "codigos_payload": _comp(C * cd * F32, "trazas"),
            "metadatos": _comp(len(METADATOS_POR_TRAZA) * C * F32, "trazas"),
            "payload_refs": _comp(C * PTR, "trazas"),
            "direcciones": _comp(M * d * F32, "regenerable"),
            "contadores": _comp(M * (d + cd) * F32, "sustrato"),
        }
        if nombre == "SDM-a+":
            comp["claves_borrado"] = _comp(C * d * F32, "trazas")
        return _resumen(comp)

    raise KeyError(f"arquitectura sin fórmula: {nombre}")


def costo_fijo_y_por_traza(nombre: str, dim: int, **hp: Any) -> tuple[int, float]:
    """Descompone la fórmula en `bytes = fijo + por_traza · C`.

    Todas las fórmulas son afines en C, así que dos evaluaciones alcanzan.
    """
    b0 = bytes_analiticos(nombre, 0, dim, **hp)["total"]
    b1 = bytes_analiticos(nombre, 1000, dim, **hp)["total"]
    return int(b0), (b1 - b0) / 1000.0


def capacidad_para_presupuesto(nombre: str, presupuesto: int, dim: int, **hp: Any) -> int:
    """Máximo C tal que `bytes_analiticos(nombre, C, dim) <= presupuesto`.

    Devuelve 0 si el costo fijo del sustrato ya excede el presupuesto.
    """
    fijo, por_traza = costo_fijo_y_por_traza(nombre, dim, **hp)
    if presupuesto < fijo + por_traza:
        return 0
    return int((presupuesto - fijo) // por_traza)
