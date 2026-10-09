"""`TraceStore`: la contabilidad de trazas de memoria.

Es el único lugar del repositorio donde vive el estado de una memoria episódica.
Todas las políticas leen y escriben a través de él, y todas las arquitecturas lo
usan. Esa unicidad es deliberada: en el código piloto la contabilidad estaba
duplicada entre el motor del NAS y el del benchmark, con semánticas distintas, y
eso hacía que las tablas del paper no fueran comparables entre sí.

El campo `contribution` merece explicación. Un sustrato distribuido (SDM,
Spiking-SDM) no guarda la traza en un solo lugar: suma `strength * key` a un
conjunto de contenedores. Cuando esa traza se desaloja hay que restar
**exactamente lo mismo que se sumó**. El piloto restaba `key` en vez de
`strength * key`, dejando un residuo en cada desalojo. Guardar el escalar aquí
hace que el desalojo correcto sea el camino fácil.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.core.types import EPS, unit


class TraceStore:
    """Contenedor vectorizado de trazas de memoria.

    Los arreglos paralelos (`keys`, `strength`, `age`, `utility`,
    `contribution`, `last_use`, `priority`) y la lista `values` se mantienen
    siempre alineados por índice: `remove(i)` los compacta a todos.
    """

    __slots__ = (
        "dim",
        "capacity",
        "keys",
        "values",
        "strength",
        "age",
        "utility",
        "contribution",
        "last_use",
        "priority",
        "t",
    )

    def __init__(self, dim: int, capacity: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim debe ser positivo, se recibió {dim}")
        if capacity <= 0:
            raise ValueError(f"capacity debe ser positiva, se recibió {capacity}")

        self.dim = dim
        self.capacity = capacity
        self.keys: NDArray[np.float32] = np.zeros((0, dim), dtype=np.float32)
        self.values: list[Any] = []
        self.strength: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self.age: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self.utility: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self.contribution: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self.last_use: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        """Reloj (`t`) del último uso de cada traza: escritura, fusión o lectura.

        Lo consumen solo las políticas externas de caché (LRU, LFU con desempate
        LRU, caché de utilidad). Ninguna política del espacio de 576 lo lee.
        """
        self.priority: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        """Prioridad cruda de la traza: el error de predicción con que se escribió.

        A diferencia de `strength`, no la tocan el decaimiento, el refuerzo por
        lectura ni la compuerta de fuerza: es la señal de sorpresa tal cual, que
        es lo que usa la repetición priorizada. Solo la leen las políticas
        externas; ninguna del espacio de 576.
        """
        self.t = 0

    # ---------------------------------------------------------------- consulta

    def __len__(self) -> int:
        return len(self.values)

    @property
    def is_full(self) -> bool:
        return len(self.values) >= self.capacity

    def similarities(self, query: NDArray) -> NDArray[np.float32]:
        """Similitud coseno de `query` contra cada traza guardada."""
        if not self.values:
            return np.zeros(0, dtype=np.float32)
        q = unit(query)
        norms = np.linalg.norm(self.keys, axis=1) + EPS
        return ((self.keys @ q) / norms).astype(np.float32)

    def novelty(self, query: NDArray) -> float:
        """Cuán distinto es `query` de todo lo guardado: `1 - max(similitud)`.

        Vale 1.0 sobre un store vacío — nada es más novedoso que lo primero.
        """
        sims = self.similarities(query)
        if sims.size == 0:
            return 1.0
        return float(1.0 - sims.max())

    # ------------------------------------------------------------- mutaciones

    def append(
        self,
        key: NDArray,
        value: Any,
        strength: float,
        contribution: float = 0.0,
        priority: float = 0.0,
    ) -> int:
        """Agrega una traza nueva y devuelve su índice."""
        k = unit(key)
        if k.shape != (self.dim,):
            raise ValueError(f"la clave debe tener forma ({self.dim},), se recibió {k.shape}")

        self.keys = np.vstack([self.keys, k[None, :]]) if len(self.values) else k[None, :].copy()
        self.values.append(value)
        self.strength = np.append(self.strength, np.float32(strength))
        self.age = np.append(self.age, np.float32(0.0))
        self.utility = np.append(self.utility, np.float32(0.0))
        self.contribution = np.append(self.contribution, np.float32(contribution))
        self.last_use = np.append(self.last_use, np.float32(self.t))
        self.priority = np.append(self.priority, np.float32(priority))
        return len(self.values) - 1

    def remove(self, idx: int) -> None:
        """Elimina la traza `idx`, compactando todos los contenedores a la vez."""
        self.keys = np.delete(self.keys, idx, axis=0)
        self.values.pop(idx)
        self.strength = np.delete(self.strength, idx)
        self.age = np.delete(self.age, idx)
        self.utility = np.delete(self.utility, idx)
        self.contribution = np.delete(self.contribution, idx)
        self.last_use = np.delete(self.last_use, idx)
        self.priority = np.delete(self.priority, idx)

    def tick(self) -> None:
        """Avanza el reloj: envejece todas las trazas en un paso."""
        self.t += 1
        self.age += 1.0

    def reinforce(self, idx: NDArray | int, amount: float) -> None:
        """Suma fuerza a las trazas indicadas (LTP por reactivación)."""
        if amount:
            self.strength[idx] += np.float32(amount)

    def touch(self, idx: NDArray | int) -> None:
        """Registra un acceso: incrementa utilidad y anota el reloj del uso.

        `age` no se toca (el FIFO del espacio desaloja por edad de escritura, no
        por uso); el reloj del uso va a `last_use`, que solo leen las políticas
        externas.
        """
        self.utility[idx] += 1.0
        self.last_use[idx] = np.float32(self.t)
