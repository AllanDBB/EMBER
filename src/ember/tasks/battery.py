"""Fase 2: la batería bajo presión de capacidad.

La condición realista de un robot con memoria acotada operando de por vida:
llegan más experiencias de las que caben, y hay que decidir qué se descarta. Las
tres tareas estresan ejes distintos del espacio de diseño, que es lo que las
hace informativas en conjunto.

| Tarea | Qué estresa                                                  |
|-------|--------------------------------------------------------------|
| T1    | Compuerta de saliencia × política de desalojo                |
| T2    | Modo de lectura, con la memoria comprimida por la presión     |
| T3    | Resistencia a interferencia secuencial                        |

Los tres arreglos respecto del piloto
-------------------------------------
**T1 intercala lecturas.** El piloto emitía todas las lecturas después de todas
las escrituras, de modo que el refuerzo por lectura nunca llegaba a influir en
un desalojo y el eje era inobservable por diseño de tarea. Con `read_every > 0`
la memoria se consulta mientras se escribe — que además es lo que hace un robot:
consulta mientras opera, no al final de la sesión.

**T2 ejerce presión de verdad.** El piloto escribía 20 ítems en capacidad 20:
nunca desalojaba, y devolvía 0.767 para las 576 arquitecturas, con varianza
exactamente cero. Como entraba al promedio, aportaba una constante de 0.256 a
todo el mundo, y el 98 % del puntaje del FIFO era esa constante. Aquí se escriben
más ítems que la capacidad, y las consultas se ponderan por importancia: una
experiencia sorpresiva es la que el robot va a necesitar recordar después.

**T3 usa señal diferenciada.** Con error de predicción uniforme, el desalojo por
mínima fuerza degenera —todas las trazas pesan igual— y toda arquitectura puntúa
cero. Eso es un resultado informativo (sin una señal que distinga qué importa,
ninguna arquitectura resiste la interferencia) pero no es la comparación que la
tarea pretende hacer.
"""

from __future__ import annotations

import numpy as np

from ember.core.types import unit
from ember.data.streams import Stream
from ember.data.synthetic import block_stream, clustered_stream
from ember.tasks.protocol import MemoryFactory, SuiteResult, TaskResult

DIM = 32

HIT_SIMILARITY = 0.85
"""Similitud mínima para contar una recuperación como acierto.

Un único umbral para toda la evaluación. El piloto usaba 0.90 en el NAS y 0.75
en el benchmark de arquitecturas, lo que hacía que sus tablas de resultados no
fueran comparables entre sí aunque el paper las presentara como tales.
"""


def _acierto(resultado, esperado) -> bool:
    return resultado.value == esperado and resultado.similarity >= HIT_SIMILARITY


# ══════════════════════════════════════════════ T1 · retención de eventos raros


def t1_rare_retention(
    factory: MemoryFactory,
    stream: Stream | None = None,
    *,
    seed: int = 0,
    capacity: int | None = None,
    read_every: int = 0,
    dim: int = DIM,
) -> TaskResult:
    """¿Sobrevive lo raro-pero-importante al desplazamiento por lo común?

    `read_every=n` emite una consulta cada `n` escrituras sobre una clave ya
    escrita, para que el refuerzo por lectura pueda influir en desalojos
    posteriores. Con `read_every=0` se reproduce el protocolo del piloto, en el
    que ese eje es inobservable.
    """
    if stream is None:
        stream = clustered_stream(n_prototypes=5, capacity=capacity or 20, seed=seed)
    cap = capacity if capacity is not None else stream.spec.capacity

    mem = factory(cap, seed)
    rng = np.random.default_rng(seed + 5000)
    escritas: list[np.ndarray] = []

    for n, it in enumerate(stream, start=1):
        mem.write(it.key, it.value, it.pred_error)
        escritas.append(it.key)
        if read_every and n % read_every == 0:
            mem.read(escritas[int(rng.integers(len(escritas)))])

    aciertos = sum(_acierto(mem.read(it.key), it.value) for it in stream.rare_items)
    n_raros = max(len(stream.rare_items), 1)

    return TaskResult(
        name="rare_retention",
        score=aciertos / n_raros,
        detail={
            "n_rare": len(stream.rare_items),
            "r": stream.spec.r,
            "read_every": read_every,
            "evictions": getattr(mem, "n_evictions", None),
        },
    )


# ══════════════════════════════════════════════ T2 · ruido bajo presión real


def t2_noise_under_pressure(
    factory: MemoryFactory,
    *,
    seed: int = 0,
    dim: int = DIM,
    capacity: int = 20,
    n_items: int = 60,
    noise: float = 0.30,
    n_queries: int = 120,
) -> TaskResult:
    """Recuperación con ruido cuando la memoria no alcanza para todo lo escrito.

    Las consultas se ponderan por error de predicción: lo que el robot necesita
    recordar después es lo que fue sorpresivo cuando ocurrió. Eso hace que la
    política de desalojo importe, en vez de anularse como pasaba en el piloto.
    """
    rng = np.random.default_rng(seed + 2000)
    items = rng.standard_normal((n_items, dim)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    pred_errors = rng.uniform(0.05, 0.95, size=n_items).astype(np.float32)

    mem = factory(capacity, seed)
    for i, (k, pe) in enumerate(zip(items, pred_errors, strict=True)):
        mem.write(k, i, float(pe))

    pesos = pred_errors / pred_errors.sum()
    aciertos = 0
    for _ in range(n_queries):
        i = int(rng.choice(n_items, p=pesos))
        ruidosa = unit(items[i] + rng.standard_normal(dim).astype(np.float32) * noise)
        aciertos += mem.read(ruidosa).value == i

    return TaskResult(
        name="noise_under_pressure",
        score=aciertos / n_queries,
        detail={
            "n_items": n_items,
            "capacity": capacity,
            "ceiling": capacity / n_items,
            "evictions": getattr(mem, "n_evictions", None),
        },
    )


# ═══════════════════════════════════════════ T3 · interferencia secuencial


def t3_sequential_interference(
    factory: MemoryFactory,
    *,
    seed: int = 0,
    dim: int = DIM,
    capacity: int = 20,
    n_blocks: int = 4,
    per_block: int = 8,
    repeats: int = 3,
    pe_first: float = 0.9,
    pe_rest: float = 0.1,
) -> TaskResult:
    """¿Cuánto del primer bloque sobrevive al aprendizaje de los siguientes?"""
    stream = block_stream(
        n_blocks=n_blocks,
        per_block=per_block,
        repeats=repeats,
        dim=dim,
        capacity=capacity,
        pe_first=pe_first,
        pe_rest=pe_rest,
        seed=seed + 3000,
    )
    mem = factory(capacity, seed)
    for it in stream:
        mem.write(it.key, it.value, it.pred_error)

    del_primero = {it.value: it.key for it in stream.rare_items}
    aciertos = sum(_acierto(mem.read(k), v) for v, k in del_primero.items())

    return TaskResult(
        name="sequential_interference",
        score=aciertos / max(len(del_primero), 1),
        detail={
            "pe_first": pe_first,
            "pe_rest": pe_rest,
            "senal_diferenciada": pe_first != pe_rest,
            "evictions": getattr(mem, "n_evictions", None),
        },
    )


# ══════════════════════════════════════════════════════════════════ la batería

BATTERY_TASKS = (t1_rare_retention, t2_noise_under_pressure, t3_sequential_interference)


def run_battery(
    factory: MemoryFactory,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    dim: int = DIM,
    capacity: int = 20,
    n_prototypes: int = 5,
    read_every: int = 5,
    name: str = "",
) -> SuiteResult:
    """Corre las tres tareas promediando sobre semillas."""
    por_tarea: dict[str, float] = {}
    por_tarea_std: dict[str, float] = {}
    detalle: dict[str, object] = {}

    t1 = [
        t1_rare_retention(
            factory,
            clustered_stream(n_prototypes=n_prototypes, capacity=capacity, dim=dim, seed=s),
            seed=s,
            read_every=read_every,
            dim=dim,
        )
        for s in seeds
    ]
    t2 = [t2_noise_under_pressure(factory, seed=s, dim=dim, capacity=capacity) for s in seeds]
    t3 = [t3_sequential_interference(factory, seed=s, dim=dim, capacity=capacity) for s in seeds]

    for grupo in (t1, t2, t3):
        puntajes = [r.score for r in grupo]
        por_tarea[grupo[0].name] = float(np.mean(puntajes))
        por_tarea_std[grupo[0].name] = float(np.std(puntajes))
        detalle[grupo[0].name] = grupo[0].detail

    return SuiteResult(
        name=name or "battery",
        per_task=por_tarea,
        per_task_std=por_tarea_std,
        detail={"seeds": list(seeds), "capacity": capacity, **detalle},
    )
