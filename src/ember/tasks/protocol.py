"""Tipos comunes a todas las tareas de evaluación."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ember.core.protocol import Memory

MemoryFactory = Callable[[int, int], Memory]
"""`(capacity, seed) -> Memory`.

Una sola firma para toda arquitectura y todo genotipo. Es lo que permite que la
misma tarea evalúe una SDM, un circuito spiking y un punto del espacio del NAS
sin ramificar por tipo.
"""


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Resultado de una tarea sobre una arquitectura."""

    name: str
    score: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """Resultado agregado de un conjunto de tareas."""

    name: str
    per_task: dict[str, float]
    detail: dict[str, Any] = field(default_factory=dict)
    per_task_std: dict[str, float] = field(default_factory=dict)
    """Desvío estándar entre semillas de cada tarea de `per_task`.

    `per_task` promedia sobre semillas; sin esto un 1.000 o un 0.038 se leen
    como si no tuvieran dispersión, cuando son la media de solo un puñado de
    corridas.
    """

    @property
    def mean(self) -> float:
        if not self.per_task:
            return 0.0
        return float(sum(self.per_task.values()) / len(self.per_task))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "per_task": dict(self.per_task),
            "per_task_std": dict(self.per_task_std),
            "mean": self.mean,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class GateResult(SuiteResult):
    """Resultado del gate de reconstrucción, con el veredicto de admisión."""

    threshold: float = 0.50

    @property
    def passes(self) -> bool:
        """Si la arquitectura puede reconstruir lo suficiente como para ser evaluada.

        Una memoria que no recupera un episodio completo desde una clave
        degradada no sirve para un robot, donde toda consulta es ruidosa,
        parcial o desplazada del contexto de codificación. Medirle retención
        bajo presión de capacidad no diría nada.
        """
        return self.mean >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        # Llamada explícita a la base: `dataclass(slots=True)` recrea la clase y
        # deja sin celda `__class__` al `super()` de cero argumentos.
        return {
            **SuiteResult.to_dict(self),
            "threshold": self.threshold,
            "passes": self.passes,
        }
