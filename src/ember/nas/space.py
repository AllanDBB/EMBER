"""El espacio de diseño: 576 arquitecturas de memoria bioinspirada.

Seis ejes, cada uno motivado por un principio biológico o computacional
distinto, enumerables de forma exhaustiva. La ventaja metodológica de un espacio
completamente enumerable es que ningún resultado se puede atribuir al
comportamiento de un optimizador aproximado: se evalúan todos los puntos.

    3 lecturas × 2 escrituras × 4 fuerzas × 3 decaimientos × 4 desalojos × 2 refuerzos = 576

El `EpisodicBuffer` FIFO de e-MDB es uno de esos 576 puntos, no un baseline
externo. Eso convierte una comparación potencialmente circular en una pregunta
falsable: dado el espacio completo, ¿dónde cae el incumbente desplegado?

Sobre "arquitecturas funcionalmente distintas"
----------------------------------------------
El piloto reportaba 576 pero solo 96 eran funcionalmente distintas: los ejes de
lectura y de refuerzo no movían el resultado en absolutamente ningún par de
genotipos hermanos. `ember.nas.stats.axis_liveness` mide exactamente eso, y
`experiments/exp03_axis_liveness.py` falla si algún eje vuelve a morir. Antes de
escribir "576" en un paper hay que correr ese experimento.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.policies import (
    FIFO,
    Append,
    BothGated,
    Constant,
    ExponentialDecay,
    Merge,
    MinStrength,
    MinUtility,
    NearestNeighbour,
    NoDecay,
    NoveltyGated,
    PredErrorGated,
    Radius,
    Random,
    TopK,
)

SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    # Recuperación episódica vs. círculo de activación de Kanerva.
    "read": (NearestNeighbour(), TopK(k=3), Radius(fraction=0.70)),
    # Creación de traza vs. consolidación Hebbiana por fusión.
    "write": (Append(), Merge(threshold=0.85)),
    # Compuerta de LTP por saliencia y sorpresa (codificación dopaminérgica).
    "strength": (Constant(), NoveltyGated(), PredErrorGated(), BothGated()),
    # Debilitamiento sináptico sin refuerzo.
    "decay": (NoDecay(), ExponentialDecay(rate=0.995), ExponentialDecay(rate=0.98)),
    # Olvido por edad vs. por importancia.
    "evict": (FIFO(), MinStrength(), MinUtility(), Random()),
    # LTP por reactivación.
    "reinforce": (0.0, 0.5),
}

AXES: tuple[str, ...] = tuple(SEARCH_SPACE)

BIOLOGICAL_ROOT: dict[str, str] = {
    "read": "Recuperación episódica vs. círculo de activación (Kanerva)",
    "write": "Creación de traza vs. consolidación Hebbiana por fusión",
    "strength": "Compuerta de LTP por saliencia y sorpresa (dopamina)",
    "decay": "Debilitamiento sináptico sin refuerzo",
    "evict": "Olvido por edad vs. por importancia",
    "reinforce": "LTP por reactivación",
}


def space_size() -> int:
    """Cardinalidad del producto cartesiano."""
    n = 1
    for opciones in SEARCH_SPACE.values():
        n *= len(opciones)
    return n


def enumerate_space() -> Iterator[Genotype]:
    """Enumera los 576 genotipos en orden determinista."""
    for combo in product(*SEARCH_SPACE.values()):
        yield Genotype(**dict(zip(AXES, combo, strict=True)))


def axis_options(axis: str) -> list[str]:
    """Las etiquetas de un eje, en el orden en que se enumeran."""
    return [
        o.label if hasattr(o, "label") else str(o)  # `reinforce` son floats
        for o in SEARCH_SPACE[axis]
    ]


def siblings(genotype: Genotype, axis: str) -> list[Genotype]:
    """Genotipos que difieren de este únicamente en `axis`.

    Es la construcción que necesita el análisis de observabilidad: si el máximo
    de diferencias entre hermanos es exactamente cero, el eje no hace nada.
    """
    return [
        genotype.with_axis(axis, opcion)
        for opcion in SEARCH_SPACE[axis]
        if getattr(genotype, axis) != opcion
    ]


__all__ = [
    "AXES",
    "BIOLOGICAL_ROOT",
    "FIFO_GENOTYPE",
    "SEARCH_SPACE",
    "axis_options",
    "enumerate_space",
    "siblings",
    "space_size",
]
