"""Baselines externos al espacio de diseño, expresados como genotipos del mismo motor.

Los revisores de BIP2026 (R1, R2) piden comparar contra gestión de memoria no
bioinspirada: muestreo de reservorio, LRU/LFU, repetición priorizada, cachés de
utilidad y selección por cobertura de aprendizaje continuo. Y R1 señala que el
"e-MDB FIFO" del paper es un proxy optimista: le regala una lectura por vecino
más cercano que el `EpisodicBuffer` real no tiene.

Cada baseline es un `Genotype` cuyo único eje distinto del proxy FIFO es el
desalojo (o, para el buffer real, la lectura). El resto queda como en el
`EpisodicBuffer`: lectura por vecino más cercano, escritura `append`, fuerza
constante, sin decaimiento ni refuerzo. Así la comparación contra la frontera
aísla la política de selección, y todos corren sobre `PolicyMemory`.

**Ninguno de estos genotipos pertenece a `SEARCH_SPACE`** salvo los marcados en
`IN_SPACE_REFERENCES`, que se incluyen como referencia porque ya son puntos del
espacio (el desalojo aleatorio y la compuerta de sorpresa con mínima fuerza).
"""

from __future__ import annotations

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.policies import (
    LFU,
    LRU,
    BothGated,
    CoverageMax,
    MinStrength,
    PredErrorGated,
    PrioritySurprise,
    Random,
    Reservoir,
    SequentialScan,
    StochasticPriority,
    UtilityCache,
)

FRONTIER_GENOTYPE = FIFO_GENOTYPE.with_axis("strength", BothGated()).with_axis(
    "evict", MinStrength()
)
"""La frontera de exp01: `nn + append + both + decay 1.0 + min_strength + reinforce 0`."""

EMDB_SEQUENTIAL_GENOTYPE = FIFO_GENOTYPE.with_axis("read", SequentialScan())
"""El `EpisodicBuffer` real de e-MDB: `deque` FIFO con recuperación secuencial.

Ver `SequentialScan` para la fuente (GII/emdb_cognitive_nodes_gii,
`cognitive_nodes/cognitive_nodes/episodic_buffer.py`) y el supuesto de cómo se
traduce una consulta a un buffer que no tiene consultas.
"""


def _con_desalojo(evict) -> Genotype:
    return FIFO_GENOTYPE.with_axis("evict", evict)


EXTERNAL_BASELINES: dict[str, Genotype] = {
    "reservoir": _con_desalojo(Reservoir()),
    "lru": _con_desalojo(LRU()),
    "lfu": _con_desalojo(LFU()),
    "per_min": _con_desalojo(PrioritySurprise()),
    "per_stoch": _con_desalojo(StochasticPriority()),
    "utility_cache": _con_desalojo(UtilityCache()),
    "coverage": _con_desalojo(CoverageMax()),
}
"""Baselines externos, uno por política publicada. Ver el docstring de cada política."""

IN_SPACE_REFERENCES: dict[str, Genotype] = {
    "random_evict": _con_desalojo(Random()),
    "pred_error_min_strength": FIFO_GENOTYPE.with_axis("strength", PredErrorGated()).with_axis(
        "evict", MinStrength()
    ),
}
"""Puntos del espacio que también son baselines conocidos.

`random_evict` es el desalojo aleatorio. `pred_error_min_strength` es la
repetición priorizada greedy por sorpresa **tal como la expresa el espacio**
(fuerza = 1 + 2·sorpresa, sin decaimiento ni refuerzo): difiere de `per_min`
solo en que la fuerza, a diferencia de la prioridad cruda, se refuerza al
fusionar — con `append` no hay fusión, así que deberían coincidir.
"""

METHODS: dict[str, Genotype] = {
    "frontier": FRONTIER_GENOTYPE,
    "fifo_nn_proxy": FIFO_GENOTYPE,
    "emdb_sequential": EMDB_SEQUENTIAL_GENOTYPE,
    **EXTERNAL_BASELINES,
    **IN_SPACE_REFERENCES,
}
"""Todo lo que exp09 evalúa, en el orden en que se reporta."""
