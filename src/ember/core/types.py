"""Tipos básicos compartidos por todo el núcleo de EMBER."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

EPS = 1e-8
"""Constante de regularización para divisiones por norma."""


def unit(v: NDArray) -> NDArray[np.float32]:
    """Normaliza a norma 1 en float32.

    Todo vector clave entra al sistema por esta función: el núcleo asume en todas
    partes que las claves viven en la esfera unitaria, de modo que el producto
    interno es directamente la similitud coseno.
    """
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a) + EPS)


def unit_rows(m: NDArray) -> NDArray[np.float32]:
    """Normaliza cada fila de una matriz a norma 1 en float32."""
    a = np.asarray(m, dtype=np.float32)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)


class ReadResult(NamedTuple):
    """Lo que devuelve una lectura de memoria.

    `index` es la posición en el `TraceStore` de la traza que ganó, o `None` si
    la memoria estaba vacía o si el sustrato no expone índices de traza.
    """

    value: Any
    similarity: float
    index: int | None = None


EMPTY_READ = ReadResult(value=None, similarity=0.0, index=None)


@dataclass(frozen=True, slots=True)
class Episode:
    """Unidad de experiencia de e-MDB.

    Réplica del `Episode` de la arquitectura e-MDB, para que una memoria de EMBER
    pueda actuar como reemplazo directo del `EpisodicBuffer` sin traducir tipos:

        Episode = {old_perception, policy, action, perception, reward_list}
    """

    old_perception: Any = None
    policy: Any = None
    action: Any = None
    perception: Any = None
    reward_list: list[float] = field(default_factory=list)
