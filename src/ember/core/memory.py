"""`PolicyMemory`: el motor de memoria configurable de EMBER.

Compone las cinco familias de política en una memoria que satisface el protocolo
`Memory`. Es a la vez el fenotipo de cualquier genotipo del NAS y una memoria
utilizable en producción — instanciarla con `FIFO_GENOTYPE` da exactamente el
`EpisodicBuffer` de e-MDB.

Un solo motor, no dos. En el código piloto el NAS y el benchmark de
arquitecturas tenían implementaciones separadas con semánticas distintas
(desalojo FIFO por edad vs. por posición, compuertas de fuerza diferentes,
umbrales de acierto diferentes), lo que hacía que sus tablas de resultados no
midieran lo mismo aunque el paper las presentara como comparables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.core.genotype import Genotype
from ember.core.policies import NearestNeighbour
from ember.core.store import TraceStore
from ember.core.types import EMPTY_READ, ReadResult, unit


class PolicyMemory:
    """Memoria asociativa cuyo comportamiento queda definido por un genotipo."""

    def __init__(self, dim: int, capacity: int, genotype: Genotype, seed: int = 0) -> None:
        self.dim = dim
        self.genotype = genotype
        self.store = TraceStore(dim=dim, capacity=capacity)
        self.rng = np.random.default_rng(seed)
        self.n_evictions = 0
        self.n_writes = 0
        self.n_merges = 0
        """Cuántas escrituras se consolidaron en una traza existente.

        Es la medición directa de si el mecanismo de fusión está actuando. Un
        umbral de fusión que el dominio no alcanza deja este contador en cero, y
        entonces no puede haber régimen de compresión por bajo que sea `r`.
        """

    @property
    def capacity(self) -> int:
        return self.store.capacity

    def __len__(self) -> int:
        return len(self.store)

    # ------------------------------------------------------------- escritura

    def write(self, key: NDArray, value: Any, pred_error: float = 0.5) -> None:
        g = self.genotype
        store = self.store
        k = unit(key)

        store.tick()
        g.decay.step(store.strength)
        self.n_writes += 1

        novelty = store.novelty(k)
        s = g.strength.initial(pred_error=pred_error, novelty=novelty)

        objetivo = g.write.route(store, k)
        if objetivo is not None:
            self.n_merges += 1
            # Consolidación Hebbiana: la experiencia refuerza un engrama existente
            # en vez de crear una copia redundante. No hay traza nueva, así que no
            # puede haber desalojo.
            store.strength[objetivo] += np.float32(s)
            store.utility[objetivo] += 1.0
            store.age[objetivo] = 0.0
            return

        store.append(k, value, strength=s)
        self._evict_until_fits()

    def _evict_until_fits(self) -> None:
        store = self.store
        while len(store) > store.capacity:
            store.remove(self.genotype.evict.victim(store, self.rng))
            self.n_evictions += 1

    # ---------------------------------------------------------------- lectura

    def read(self, query: NDArray) -> ReadResult:
        g = self.genotype
        store = self.store
        if len(store) == 0:
            return EMPTY_READ

        q = unit(query)
        sims = store.similarities(q)
        sel = g.read.select(sims)

        # LTP por reactivación: consultar una traza la fortalece, lo que la
        # protege de desalojos posteriores. Solo es observable si la tarea
        # intercala lecturas entre escrituras.
        store.reinforce(sel, g.reinforce)
        store.touch(sel)

        ganador = self._resolver(q, sims, sel)
        return ReadResult(
            value=store.values[ganador],
            similarity=float(np.clip(sims[ganador], 0.0, 1.0)),
            index=ganador,
        )

    def _resolver(self, q: NDArray, sims: NDArray, sel: NDArray) -> int:
        """Reduce la selección de lectura a una sola traza.

        Con una sola traza seleccionada es recuperación episódica directa. Con
        varias es la superposición de Kanerva: las trazas seleccionadas se suman
        ponderadas por su similitud, y gana la traza más parecida a esa
        reconstrucción — que no tiene por qué ser la más parecida a la consulta.

        Esa distinción es lo que hace que el eje de lectura sea observable. En el
        código piloto los tres modos devolvían el vecino más cercano, y por eso
        el eje explicaba exactamente 0.0 % de la varianza.
        """
        if sel.size == 1:
            return int(sel[0])

        pesos = np.clip(sims[sel], 0.0, None).astype(np.float32)
        if pesos.sum() <= 0.0:
            return int(sel[int(np.argmax(sims[sel]))])

        reconstruccion = unit((pesos[:, None] * self.store.keys[sel]).sum(axis=0))
        sims_recon = self.store.similarities(reconstruccion)
        return int(np.argmax(sims_recon))


def make_memory(dim: int, capacity: int, genotype: Genotype, seed: int = 0) -> PolicyMemory:
    """Fábrica con la firma que usan las tareas: `(capacity, seed) -> Memory`."""
    return PolicyMemory(dim=dim, capacity=capacity, genotype=genotype, seed=seed)


def nearest_neighbour_memory(dim: int, capacity: int, seed: int = 0) -> PolicyMemory:
    """Memoria de referencia sin ningún mecanismo bioinspirado activo."""
    from ember.core.genotype import FIFO_GENOTYPE

    assert isinstance(FIFO_GENOTYPE.read, NearestNeighbour)
    return PolicyMemory(dim=dim, capacity=capacity, genotype=FIFO_GENOTYPE, seed=seed)
