"""Spiking-SDM: la decisión de activación de la SDM la toma un circuito LIF.

Punto de convergencia entre la PoC-1 (SDM) y la PoC-3 (circuito spiking). En vez
de decidir qué hard locations se activan por comparación algebraica, cada
ubicación tiene una neurona LIF que integra un tren de spikes codificado por
tasa a partir del vector de entrada, y se activa solo si su potencial de
membrana cruza el umbral.

Eso vuelve la decisión de activación estocástica y temporalmente fundada, que es
lo que le da al enfoque su ángulo neuromórfico: el mismo cómputo puede correr en
hardware de spikes.

Hereda de la SDM la conservación de contadores: lo que se sumó al escribir es lo
que se resta al desalojar. La diferencia con la SDM algebraica es que el
conjunto activo depende del generador aleatorio, así que se guarda explícitamente
por traza en vez de recalcularse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.core.policies import EvictPolicy, MinStrength, PredErrorGated, StrengthPolicy
from ember.core.store import TraceStore
from ember.core.types import EMPTY_READ, ReadResult, unit, unit_rows


class SpikingSDMMemory:
    """SDM cuyo circuito de activación es una capa de neuronas LIF."""

    def __init__(
        self,
        dim: int,
        capacity: int,
        seed: int = 0,
        n_hard: int = 256,
        activation_frac: float = 0.05,
        T_window: int = 15,
        p_high: float = 0.85,
        p_low: float = 0.05,
        leak: float = 0.95,
        v_th: float = 0.5,
        strength: StrengthPolicy | None = None,
        evict: EvictPolicy | None = None,
    ) -> None:
        self.dim = dim
        self.n_hard = n_hard
        self.k_active = max(1, int(n_hard * activation_frac))
        self.T = T_window
        self.p_high = p_high
        self.p_low = p_low
        self.leak = leak
        self.v_th = v_th
        self.strength_policy = strength or PredErrorGated()
        self.evict_policy = evict or MinStrength()

        self.rng = np.random.default_rng(seed)
        self.store = TraceStore(dim=dim, capacity=capacity)
        self.n_evictions = 0

        self.H: NDArray[np.float32] = unit_rows(
            self.rng.standard_normal((n_hard, dim)).astype(np.float32)
        )
        # Pesos de integración separados por signo: una neurona no integra
        # magnitudes negativas, integra dos poblaciones de entrada.
        self.W_pos = np.clip(self.H, 0.0, None)
        self.W_neg = np.clip(-self.H, 0.0, None)

        self.V: NDArray[np.float32] = np.zeros((n_hard, dim), dtype=np.float32)
        # Conjunto activo por traza: la activación es estocástica, así que no se
        # puede recalcular de forma idéntica al desalojar.
        self._activas: list[NDArray[np.intp]] = []

    @property
    def capacity(self) -> int:
        return self.store.capacity

    def __len__(self) -> int:
        return len(self.store)

    # ----------------------------------------------------------------- interno

    def _codificar(self, vec: NDArray) -> tuple[NDArray, NDArray]:
        """Codificación por tasa: cada dimensión emite spikes según su magnitud."""
        v = unit(vec)
        p_pos = np.clip(np.where(v > 0, self.p_high * v + self.p_low, self.p_low), 0.0, 1.0)
        p_neg = np.clip(np.where(v < 0, self.p_high * (-v) + self.p_low, self.p_low), 0.0, 1.0)
        sp_pos = (self.rng.random((self.T, self.dim)) < p_pos).astype(np.float32)
        sp_neg = (self.rng.random((self.T, self.dim)) < p_neg).astype(np.float32)
        return sp_pos, sp_neg

    def _activadas(self, vec: NDArray) -> NDArray[np.intp]:
        """Hard locations cuya neurona LIF llegó a disparar en la ventana."""
        sp_pos, sp_neg = self._codificar(vec)
        Vmem = np.zeros(self.n_hard, dtype=np.float32)
        disparo = np.zeros(self.n_hard, dtype=bool)
        for t in range(self.T):
            entrada = (self.W_pos @ sp_pos[t] + self.W_neg @ sp_neg[t]) / self.dim
            Vmem = self.leak * Vmem + entrada
            nuevas = (Vmem >= self.v_th) & ~disparo
            disparo |= nuevas
            Vmem[nuevas] = 0.0
        if not disparo.any():
            # Ninguna cruzó el umbral: se toman las más integradas, para que la
            # memoria degrade en vez de fallar.
            disparo[np.argpartition(-Vmem, self.k_active - 1)[: self.k_active]] = True
        return np.where(disparo)[0].astype(np.intp)

    def _desalojar(self, idx: int) -> None:
        clave = self.store.keys[idx].copy()
        aporte = float(self.store.contribution[idx])
        self.V[self._activas[idx]] -= aporte * clave
        self.store.remove(idx)
        self._activas.pop(idx)
        self.n_evictions += 1

    # --------------------------------------------------------------- escritura

    def write(self, key: NDArray, value: Any, pred_error: float = 0.5) -> None:
        k = unit(key)
        self.store.tick()

        s = self.strength_policy.initial(pred_error=pred_error, novelty=self.store.novelty(k))
        activas = self._activadas(k)
        self.V[activas] += np.float32(s) * k

        self.store.append(k, value, strength=s, contribution=s)
        self._activas.append(activas)

        while len(self.store) > self.store.capacity:
            self._desalojar(self.evict_policy.victim(self.store, self.rng))

    # ----------------------------------------------------------------- lectura

    def read(self, query: NDArray) -> ReadResult:
        if len(self.store) == 0:
            return EMPTY_READ

        q = unit(query)
        crudo = self.V[self._activadas(q)].sum(axis=0)
        if float(np.linalg.norm(crudo)) < 1e-6:
            return EMPTY_READ

        sims_recon = self.store.similarities(unit(crudo))
        ganador = int(np.argmax(sims_recon))
        self.store.touch(ganador)

        sim = float(np.clip(self.store.similarities(q)[ganador], 0.0, 1.0))
        return ReadResult(value=self.store.values[ganador], similarity=sim, index=ganador)
