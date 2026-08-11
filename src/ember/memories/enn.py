"""Engram Neural Network: memoria Hebbiana de pesos rápidos.

Una fila por episodio guardado, actualizada en línea sin retropropagación. Cada
escritura crea una traza nueva o se fusiona con una existente si ya hay un
episodio suficientemente parecido — el análogo computacional del dictum de Hebb:
la exposición repetida a la misma experiencia fortalece un solo engrama en vez
de crear copias redundantes.

El modo de falla que documenta el paper
---------------------------------------
La ENN tiene la misma compuerta de saliencia y la misma política de desalojo que
la SDM, y aun así retiene mucho peor los eventos raros. La causa es la fusión:
un prototipo visitado k veces acumula fuerza en progresión geométrica
(Σ boost·decay^i), que supera holgadamente la fuerza de una escritura única por
sorpresiva que sea. La traza fusionada sobrevive al evento raro en la cola de
desalojo.

Es decir: la compuerta de saliencia funciona solo mientras la señal de fuerza no
quede ahogada por la dinámica de acumulación de otro mecanismo. `test_enn.py`
fija esa interacción.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.core.policies import EvictPolicy, MinStrength, PredErrorGated, StrengthPolicy
from ember.core.store import TraceStore
from ember.core.types import EMPTY_READ, ReadResult, unit


class ENNMemory:
    """Memoria Hebbiana con consolidación por fusión y decaimiento global."""

    def __init__(
        self,
        dim: int,
        capacity: int,
        seed: int = 0,
        merge_threshold: float = 0.85,
        decay: float = 0.97,
        eta: float = 0.6,
        reinforce: float = 0.3,
        strength: StrengthPolicy | None = None,
        evict: EvictPolicy | None = None,
    ) -> None:
        self.dim = dim
        self.merge_threshold = merge_threshold
        self.decay = decay
        self.eta = eta
        self.reinforce = reinforce
        self.strength_policy = strength or PredErrorGated()
        self.evict_policy = evict or MinStrength()

        self.rng = np.random.default_rng(seed)
        self.store = TraceStore(dim=dim, capacity=capacity)
        self.n_evictions = 0

    @property
    def capacity(self) -> int:
        return self.store.capacity

    def __len__(self) -> int:
        return len(self.store)

    # --------------------------------------------------------------- escritura

    def write(self, key: NDArray, value: Any, pred_error: float = 0.5) -> None:
        k = unit(key)
        store = self.store

        store.tick()
        store.strength *= np.float32(self.decay)

        sims = store.similarities(k)
        boost = self.strength_policy.initial(pred_error=pred_error, novelty=store.novelty(k))

        if sims.size and sims.max() >= self.merge_threshold:
            # Consolidación: la traza se mueve hacia la clave nueva y acumula
            # fuerza. No hay traza nueva, así que no puede haber desalojo.
            i = int(sims.argmax())
            store.keys[i] = unit(store.keys[i] + np.float32(self.eta * boost) * k)
            store.strength[i] += np.float32(boost)
            store.utility[i] += 1.0
            store.age[i] = 0.0
            return

        store.append(k, value, strength=boost, contribution=boost)

        while len(store) > store.capacity:
            store.remove(self.evict_policy.victim(store, self.rng))
            self.n_evictions += 1

    # ----------------------------------------------------------------- lectura

    def read(self, query: NDArray) -> ReadResult:
        if len(self.store) == 0:
            return EMPTY_READ

        q = unit(query)
        sims = self.store.similarities(q)
        ganador = int(np.argmax(sims))

        # LTP por reactivación.
        self.store.reinforce(ganador, self.reinforce)
        self.store.touch(ganador)

        return ReadResult(
            value=self.store.values[ganador],
            similarity=float(np.clip(sims[ganador], 0.0, 1.0)),
            index=ganador,
        )
