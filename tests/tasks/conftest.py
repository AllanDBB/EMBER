"""Fábricas de memoria compartidas por las pruebas de tareas."""

import pytest

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.memory import PolicyMemory
from ember.core.policies import (
    Append,
    Constant,
    Merge,
    MinStrength,
    NearestNeighbour,
    NoDecay,
    PredErrorGated,
    Radius,
)

DIM = 32

FRONTERA = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=PredErrorGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""El genotipo que el NAS del piloto identificó como frontera en las 14 celdas."""

FUSION = Genotype(
    read=NearestNeighbour(),
    write=Merge(threshold=0.85),
    strength=PredErrorGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""Igual a la frontera pero consolidando por fusión. Paga solo bajo r < 1."""

SIN_SALIENCIA = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=Constant(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""Desalojo por mínima fuerza sin señal de saliencia: degenera en pseudoaleatorio."""

RADIO = Genotype(
    read=Radius(fraction=0.70),
    write=Append(),
    strength=PredErrorGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""La frontera leyendo por superposición en vez de por vecino más cercano."""


def fabrica(genotype: Genotype, dim: int = DIM):
    """Construye una `MemoryFactory` a partir de un genotipo."""

    def factory(capacity: int, seed: int):
        return PolicyMemory(dim=dim, capacity=capacity, genotype=genotype, seed=seed)

    return factory


@pytest.fixture
def fifo():
    return fabrica(FIFO_GENOTYPE)


@pytest.fixture
def frontera():
    return fabrica(FRONTERA)
