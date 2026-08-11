"""El flujo de experiencia: la unidad sobre la que se evalúa una memoria.

Un `Stream` es una secuencia de experiencias más los metadatos que la ley del
umbral necesita. El metadato que importa es `n_prototypes`: cuántas experiencias
recurrentes distintas contiene el flujo. Junto con la capacidad de la memoria da
el ratio

    r = K_proto / C

que es la variable de control del régimen. Por debajo de 1 los prototipos
recurrentes caben en memoria y consolidar por fusión paga; por encima de 1 no
hay nada que comprimir y la pregunta se reduce a qué política de olvido conserva
mejor lo importante.

Sobre datos sintéticos `n_prototypes` se conoce por construcción. Sobre datos
reales hay que estimarlo — ver `ember.data.prototypes`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray


class StreamItem(NamedTuple):
    """Una experiencia del flujo."""

    key: NDArray[np.float32]
    value: Any
    pred_error: float
    is_rare: bool = False


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """Metadatos del flujo, incluida la variable de control del régimen."""

    n_prototypes: int
    capacity: int
    dim: int
    source: str
    n_prototypes_ci: tuple[int, int] | None = None
    """Intervalo de confianza de `n_prototypes` cuando fue estimado, no conocido."""

    @property
    def r(self) -> float:
        """El ratio prototipos-a-capacidad que define el régimen."""
        return self.n_prototypes / self.capacity

    @property
    def regime(self) -> str:
        if self.r <= 0.5:
            return "compression"
        if self.r >= 1.0:
            return "selection"
        return "transition"

    @property
    def is_estimated(self) -> bool:
        """Si `n_prototypes` viene de un estimador y no de la construcción del flujo."""
        return self.n_prototypes_ci is not None


@dataclass(frozen=True, slots=True)
class Stream:
    """Un flujo de experiencia evaluable."""

    items: list[StreamItem]
    spec: StreamSpec
    rare_items: list[StreamItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[StreamItem]:
        return iter(self.items)

    def keys(self) -> NDArray[np.float32]:
        """Las claves como matriz (n, dim). Lo que consume el estimador de prototipos."""
        if not self.items:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        return np.stack([it.key for it in self.items])

    def with_capacity(self, capacity: int) -> Stream:
        """Copia con otra capacidad declarada. Mueve `r` sin regenerar el flujo."""
        from dataclasses import replace

        return Stream(
            items=self.items,
            spec=replace(self.spec, capacity=capacity),
            rare_items=self.rare_items,
        )
