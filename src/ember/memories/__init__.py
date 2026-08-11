"""Arquitecturas de memoria bioinspirada con interfaz unificada.

Todas satisfacen el protocolo `ember.core.Memory`:

    mem = Arquitectura(dim=32, capacity=20, seed=0)
    mem.write(key, value, pred_error=0.5)
    resultado = mem.read(query)      # ReadResult(value, similarity, index)

`FIFO` no es una clase aparte: es `PolicyMemory` con `FIFO_GENOTYPE`. Eso es lo
que hace literal la afirmación de que el `EpisodicBuffer` de e-MDB es un punto
del espacio de diseño que se busca, y no un baseline externo con el que la
comparación sería circular.
"""

from __future__ import annotations

from ember.core.genotype import FIFO_GENOTYPE
from ember.core.memory import PolicyMemory
from ember.memories.enn import ENNMemory
from ember.memories.sdm import SDMMemory
from ember.memories.spiking import SpikingMemory
from ember.memories.spiking_sdm import SpikingSDMMemory


def FIFOMemory(dim: int, capacity: int, seed: int = 0, **kwargs: object) -> PolicyMemory:
    """El `EpisodicBuffer` de e-MDB, construido desde el genotipo."""
    return PolicyMemory(dim=dim, capacity=capacity, genotype=FIFO_GENOTYPE, seed=seed)


ARCHITECTURES: dict[str, type | object] = {
    "SDM": SDMMemory,
    "ENN": ENNMemory,
    "Spiking": SpikingMemory,
    "SpikingSDM": SpikingSDMMemory,
    "FIFO": FIFOMemory,
}
"""Registro de arquitecturas evaluables.

Agregar una entrada aquí la mete automáticamente en los tests de contrato de
`tests/memories/test_contrato.py` y en el benchmark de dos fases.
"""

__all__ = [
    "ARCHITECTURES",
    "ENNMemory",
    "FIFOMemory",
    "SDMMemory",
    "SpikingMemory",
    "SpikingSDMMemory",
]
