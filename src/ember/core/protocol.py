"""El contrato que toda memoria de EMBER cumple.

Cualquier cosa que satisfaga este protocolo puede sustituir al `EpisodicBuffer`
de e-MDB, entrar al benchmark de arquitecturas y ser evaluada por las tareas sin
cambios. Los tests de `tests/memories/test_contrato.py` lo verifican de forma
parametrizada sobre todas las arquitecturas registradas.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from numpy.typing import NDArray

from ember.core.types import ReadResult


@runtime_checkable
class Memory(Protocol):
    """Memoria asociativa de capacidad acotada sobre vectores continuos."""

    @property
    def capacity(self) -> int:
        """Cuántas trazas pueden coexistir."""
        ...

    def write(self, key: NDArray, value: Any, pred_error: float = 0.5) -> None:
        """Guarda una experiencia.

        `key` se normaliza a la esfera unitaria. `pred_error` en [0, 1] es la
        señal de sorpresa que las compuertas de saliencia consumen.
        """
        ...

    def read(self, query: NDArray) -> ReadResult:
        """Recupera la experiencia que mejor corresponde a `query`.

        Devuelve `ReadResult(None, 0.0, None)` si la memoria está vacía.
        """
        ...

    def __len__(self) -> int:
        """Cuántas trazas hay guardadas ahora mismo."""
        ...
