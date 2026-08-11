"""Circuito spiking con LIF y STDP: engramas al nivel del impulso.

Las neuronas biológicas se comunican con pulsos discretos, y el tiempo relativo
entre pulsos lleva información que una neurona de tasa descarta. STDP es la
regla Hebbiana a nivel de spike: una sinapsis de A a B se fortalece si A dispara
justo antes que B (orden causal) y se debilita si el orden es el inverso.

Esa regla agrupa neuronas en ensambles por sí sola: cuando un grupo coactiva
repetidamente, STDP refuerza las conexiones internas más que las externas,
dejando una traza estructural que es precisamente un engrama.

Reproducibilidad
----------------
El código piloto sembraba el generador del entrenamiento con
`id(neuron_ids) % 2**32` — la dirección de memoria del arreglo, que cambia entre
corridas y entre plataformas. Los resultados de esta arquitectura no eran
replicables. Aquí toda la aleatoriedad sale de `self.rng`, derivada de la semilla
que pasa el experimento.

Alcance
-------
Esta arquitectura opera nativamente sobre patrones de activación neuronal
discretos. Sobre vectores continuos, la selección de neuronas por producto
interno es sensible a perturbaciones pequeñas: una versión ruidosa de un vector
guardado activa un conjunto parcialmente distinto de neuronas, que puede no
pertenecer a ningún ensamble aprendido. Cerrar esa brecha requiere una capa de
codificación estable de R^n a ensambles consistentes, que es el objetivo de la
PoC-5. Hasta entonces vale como validador del concepto de engrama, no como
componente de memoria de producción.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.core.policies import EvictPolicy, MinStrength, PredErrorGated, StrengthPolicy
from ember.core.store import TraceStore
from ember.core.types import EMPTY_READ, ReadResult, unit, unit_rows


class SpikingMemory:
    """Red recurrente de neuronas LIF donde los engramas los forma STDP."""

    def __init__(
        self,
        dim: int,
        capacity: int,
        seed: int = 0,
        n_neurons: int = 128,
        sparsity: float = 0.15,
        connectivity: float = 0.25,
        # STDP
        a_plus: float = 0.02,
        a_minus: float = 0.022,
        w_max: float = 0.6,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        # LIF
        tau_m: float = 20.0,
        v_th: float = 1.0,
        t_ref: int = 2,
        # entrenamiento
        n_burst: int = 6,
        burst_gap: int = 4,
        n_presentations: int = 30,
        # lectura
        t_readout: int = 50,
        cue_frac: float = 0.5,
        strength: StrengthPolicy | None = None,
        evict: EvictPolicy | None = None,
    ) -> None:
        self.dim = dim
        self.n = n_neurons
        self.sparsity = sparsity
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_m = tau_m
        self.v_th = v_th
        self.t_ref = t_ref
        self.n_burst = n_burst
        self.burst_gap = burst_gap
        self.n_presentations = n_presentations
        self.t_readout = t_readout
        self.cue_frac = cue_frac
        self.strength_policy = strength or PredErrorGated()
        self.evict_policy = evict or MinStrength()

        self.rng = np.random.default_rng(seed)
        self.store = TraceStore(dim=dim, capacity=capacity)
        self.n_evictions = 0

        # Vectores de decodificación: mapean un vector de entrada a un subconjunto
        # de neuronas. Fijos durante la vida de la memoria.
        self.D: NDArray[np.float32] = unit_rows(
            self.rng.standard_normal((n_neurons, dim)).astype(np.float32)
        )

        # Topología y pesos sinápticos, compartidos por todos los ensambles.
        self.mask: NDArray[np.bool_] = self.rng.random((n_neurons, n_neurons)) < connectivity
        np.fill_diagonal(self.mask, False)
        self.W: NDArray[np.float32] = (
            self.rng.uniform(0.01, 0.03, (n_neurons, n_neurons)).astype(np.float32) * self.mask
        )

        # Ensambles registrados, en paralelo con las trazas del store.
        self.store_patterns: list[NDArray[np.intp]] = []
        self._reset_state()

    @property
    def capacity(self) -> int:
        return self.store.capacity

    def __len__(self) -> int:
        return len(self.store)

    # -------------------------------------------------------------- dinámica

    def _reset_state(self) -> None:
        self.V = np.zeros(self.n, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=int)
        self.x_trace = np.zeros(self.n, dtype=np.float32)
        self.y_trace = np.zeros(self.n, dtype=np.float32)
        self.last_spikes = np.zeros(self.n, dtype=np.float32)

    def _neuronas_del_item(self, key: NDArray) -> NDArray[np.intp]:
        """Las neuronas más alineadas con la clave forman su ensamble."""
        sims = self.D @ unit(key)
        k = max(2, int(self.n * self.sparsity))
        return np.argpartition(-sims, k - 1)[:k].astype(np.intp)

    def _stdp(self, spikes: NDArray[np.bool_]) -> None:
        self.x_trace *= np.exp(-1.0 / self.tau_plus)
        self.y_trace *= np.exp(-1.0 / self.tau_minus)
        if not spikes.any():
            return
        idx = np.where(spikes)[0]
        # Potenciación: presinápticas que dispararon poco antes de estas.
        self.W[idx, :] += self.a_plus * self.x_trace[None, :] * self.mask[idx, :]
        # Depresión: postsinápticas que dispararon poco antes de estas.
        self.W[:, idx] -= self.a_minus * self.y_trace[:, None] * self.mask[:, idx]
        np.clip(self.W, 0.0, self.w_max, out=self.W)
        self.x_trace[idx] += 1.0
        self.y_trace[idx] += 1.0

    def _paso_lif(self, ext: NDArray[np.float32], aprender: bool) -> NDArray[np.bool_]:
        entrada = self.W @ self.last_spikes + ext
        activas = self.refrac <= 0
        dV = -(self.V) / self.tau_m + entrada
        self.V = np.where(activas, self.V + dV, 0.0).astype(np.float32)
        spikes = activas & (self.v_th <= self.V)
        self.V[spikes] = 0.0
        self.refrac[spikes] = self.t_ref
        self.refrac[~spikes] = np.maximum(self.refrac[~spikes] - 1, 0)
        if aprender:
            self._stdp(spikes)
        self.last_spikes = spikes.astype(np.float32)
        return spikes

    def _entrenar_ensamble(self, neuronas: NDArray[np.intp]) -> None:
        """Ráfagas forzadas sobre el ensamble para que STDP fije el engrama."""
        largo = self.n_burst * self.burst_gap
        for _ in range(self.n_presentations):
            for t in range(largo + 10):
                forzadas = np.zeros(self.n, dtype=bool)
                for ki in range(self.n_burst):
                    if t == ki * self.burst_gap:
                        jitter = self.rng.integers(-1, 2, size=len(neuronas))
                        forzadas[neuronas[jitter == 0]] = True
                ext = np.zeros(self.n, dtype=np.float32)
                ext[forzadas] = 2.0
                self._paso_lif(ext, aprender=True)
            self.x_trace[:] = 0.0
            self.y_trace[:] = 0.0

    # -------------------------------------------------------------- escritura

    def write(self, key: NDArray, value: Any, pred_error: float = 0.5) -> None:
        k = unit(key)
        self.store.tick()

        neuronas = self._neuronas_del_item(k)
        self._entrenar_ensamble(neuronas)

        s = self.strength_policy.initial(pred_error=pred_error, novelty=self.store.novelty(k))
        self.store.append(k, value, strength=s, contribution=s)
        self.store_patterns.append(neuronas)

        while len(self.store) > self.store.capacity:
            victima = self.evict_policy.victim(self.store, self.rng)
            self.store.remove(victima)
            self.store_patterns.pop(victima)
            self.n_evictions += 1

    # ---------------------------------------------------------------- lectura

    def read(self, query: NDArray) -> ReadResult:
        if len(self.store) == 0:
            return EMPTY_READ

        q = unit(query)
        cue_completo = self._neuronas_del_item(q)
        cue = cue_completo[: max(1, int(len(cue_completo) * self.cue_frac))]

        self._reset_state()
        cuentas = np.zeros(self.n, dtype=np.int32)
        for t in range(self.t_readout):
            ext = np.zeros(self.n, dtype=np.float32)
            if t < 4:
                ext[cue] = 2.0
            cuentas += self._paso_lif(ext, aprender=False).astype(np.int32)

        # Vota el ensamble que más se reactivó.
        votos = np.array(
            [cuentas[p].sum() / max(len(p), 1) for p in self.store_patterns],
            dtype=np.float32,
        )
        ganador = int(np.argmax(votos))
        self.store.touch(ganador)

        sim = float(np.clip(self.store.similarities(q)[ganador], 0.0, 1.0))
        return ReadResult(value=self.store.values[ganador], similarity=sim, index=ganador)
