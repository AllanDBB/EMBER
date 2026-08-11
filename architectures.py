"""
architectures.py
----------------
Cuatro arquitecturas de memoria bio-inspirada con interfaz unificada:

    mem = <Arquitectura>(dim, capacity, **kwargs)
    mem.write(key: np.ndarray[float32], value: any, pred_error: float = 0.5)
    val, sim = mem.read(query: np.ndarray[float32])
        -> val: el objeto guardado con write, o None
        -> sim: similitud coseno efectiva [0, 1]

Todas operan sobre vectores continuos de dimensión `dim`.
El parámetro `capacity` limita cuántos ítems pueden coexistir en memoria.

Arquitecturas:
  SDMMemory        - Sparse Distributed Memory (Kanerva) sobre espacio continuo
  ENNMemory        - Engram Neural Network (pesos Hebbianos rápidos)
  SpikingMemory    - Circuito LIF + STDP (PoC-3 escalado a un patrón por vez)
  SpikingSDMMemory - SDM con circuito de activación LIF (PoC-4)
  FIFOMemory       - Baseline: buffer FIFO de e-MDB (sin importancia)
"""

import numpy as np


# ─── utilidades comunes ───────────────────────────────────────────────────────


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-8
    return v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _cosine_matrix(keys: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Similitud coseno de q contra cada fila de keys."""
    norms = np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8
    return (keys / norms) @ q / (np.linalg.norm(q) + 1e-8)


# ─── 1. SDM en espacio continuo ───────────────────────────────────────────────


class SDMMemory:
    """
    Sparse Distributed Memory de Kanerva adaptada a vectores continuos.

    Las 'hard locations' son vectores unitarios aleatorios fijos.  Una consulta
    activa las M*activation_frac ubicaciones más cercanas (en vez de radio fijo
    en Hamming), lo que hace el radio auto-adaptativo a la densidad del espacio.

    La memoria de valores es una matriz de acumuladores reales (no bipolares),
    porque los valores también son vectores continuos.
    """

    def __init__(
        self,
        dim: int,
        capacity: int,
        seed: int = 0,
        n_hard: int = 512,
        activation_frac: float = 0.05,
        pred_error_gate: bool = True,
    ):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.capacity = capacity
        self.n_hard = n_hard
        self.k_active = max(1, int(n_hard * activation_frac))
        self.pred_error_gate = pred_error_gate

        # hard locations: vectores unitarios fijos
        H = rng.standard_normal((n_hard, dim)).astype(np.float32)
        self.H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

        # contadores de valor (uno por hard location)
        self.V = np.zeros((n_hard, dim), dtype=np.float32)

        # para rastrear qué hard locations tienen contenido y cuándo
        self.write_counts = np.zeros(n_hard, dtype=np.int32)
        self.total_writes = 0

        # índice de valores guardados (para poder devolver el objeto original)
        self._keys: list[np.ndarray] = []
        self._vals: list = []
        self._strength: list[float] = []

    def _active_set(self, q: np.ndarray) -> np.ndarray:
        sims = self.H @ _unit(q)
        return np.argpartition(-sims, self.k_active)[: self.k_active]

    def write(self, key: np.ndarray, value, pred_error: float = 0.5):
        k = _unit(key.astype(np.float32))
        strength = 1.0 + (2.0 * pred_error if self.pred_error_gate else 0.0)

        idx = self._active_set(k)
        self.V[idx] += strength * k
        self.write_counts[idx] += 1
        self.total_writes += 1

        # tabla explícita de ítems para devolver el valor correcto
        self._keys.append(k)
        self._vals.append(value)
        self._strength.append(strength)

        # desalojo por capacidad: quitar el más antiguo si supera límite
        if len(self._vals) > self.capacity:
            # min-strength eviction (mejor que FIFO según el NAS)
            drop = int(np.argmin(self._strength))
            dk = self._keys.pop(drop)
            self._vals.pop(drop)
            self._strength.pop(drop)
            # restar su contribución de los contadores
            di = self._active_set(dk)
            self.V[di] -= dk  # aproximación: strength original era 1+ algo
            self.V = np.clip(self.V, -1e6, 1e6)

    def read(self, query: np.ndarray):
        if not self._keys:
            return None, 0.0
        q = _unit(query.astype(np.float32))
        idx = self._active_set(q)

        # reconstrucción SDM: suma de contadores activos
        raw = self.V[idx].sum(axis=0)
        if np.linalg.norm(raw) < 1e-8:
            return None, 0.0
        reconstructed = _unit(raw)

        # buscar el ítem guardado más cercano a la reconstrucción
        keys_arr = np.stack(self._keys)
        sims = _cosine_matrix(keys_arr, reconstructed)
        best = int(np.argmax(sims))
        return self._vals[best], float(np.clip(sims[best], 0, 1))


# ─── 2. ENN – Engram Neural Network ──────────────────────────────────────────


class ENNMemory:
    """
    Memoria Hebbiana explícita sin encoder entrenable (versión online, sin
    backprop). Cada ítem escrito crea o refuerza una traza en la matriz de
    engramas. La lectura retorna el ítem cuya traza tiene mayor similitud con
    la consulta.

    Esta versión usa pesos rápidos puros (sin red neuronal lenta), lo que la
    hace directamente comparable con SDM y Spiking en el benchmark.
    Equivale a la HebbianEngramMemory de PoC-2 sin el encoder de PyTorch.
    """

    MERGE_THRESHOLD = 0.85  # fusiona trazas muy similares (consolidación)

    def __init__(
        self,
        dim: int,
        capacity: int,
        seed: int = 0,
        decay: float = 0.97,
        eta: float = 0.6,
        pred_error_gate: bool = True,
    ):
        self.dim = dim
        self.capacity = capacity
        self.decay = decay
        self.eta = eta
        self.pred_error_gate = pred_error_gate

        self._engrams: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._vals: list = []
        self._strength: np.ndarray = np.zeros(0, dtype=np.float32)
        self._age: np.ndarray = np.zeros(0, dtype=np.float32)

    def _sims(self, q: np.ndarray) -> np.ndarray:
        if len(self._vals) == 0:
            return np.zeros(0)
        norms = np.linalg.norm(self._engrams, axis=1, keepdims=True) + 1e-8
        return (self._engrams / norms) @ q / (np.linalg.norm(q) + 1e-8)

    def write(self, key: np.ndarray, value, pred_error: float = 0.5):
        k = _unit(key.astype(np.float32))
        boost = 1.0 + (2.0 * pred_error if self.pred_error_gate else 0.0)

        # decaimiento global de trazas antes de cada escritura
        self._strength *= self.decay
        self._age += 1.0

        sims = self._sims(k)

        # fusión si hay una traza muy similar
        if len(sims) and sims.max() >= self.MERGE_THRESHOLD:
            i = int(sims.argmax())
            # refuerzo Hebbiano: mueve la traza hacia el nuevo vector
            self._engrams[i] = _unit(self._engrams[i] + self.eta * boost * k)
            self._strength[i] += boost
            self._age[i] = 0.0
            return

        # nueva traza
        self._engrams = np.vstack([self._engrams, k[None, :]]) if len(self._vals) else k[None, :]
        self._vals.append(value)
        self._strength = np.append(self._strength, boost)
        self._age = np.append(self._age, 0.0)

        # desalojo por mínima fuerza si supera capacidad
        while len(self._vals) > self.capacity:
            drop = int(np.argmin(self._strength))
            self._engrams = np.delete(self._engrams, drop, axis=0)
            self._vals.pop(drop)
            self._strength = np.delete(self._strength, drop)
            self._age = np.delete(self._age, drop)

    def read(self, query: np.ndarray):
        if not self._vals:
            return None, 0.0
        q = _unit(query.astype(np.float32))
        sims = self._sims(q)
        best = int(np.argmax(sims))
        # refuerzo por lectura
        self._strength[best] += 0.3
        return self._vals[best], float(np.clip(sims[best], 0, 1))


# ─── 3. Spiking LIF + STDP ───────────────────────────────────────────────────


class SpikingMemory:
    """
    Circuito LIF + STDP escalado para memoria episódica de ítems continuos.

    Cada ítem guardado se codifica como un patrón de disparo sobre un banco de
    `n_neurons` neuronas LIF. Durante el entrenamiento, STDP refuerza las
    conexiones dentro del ensamble que codifica ese ítem (=engrama). Durante la
    lectura, se presenta un cue parcial/ruidoso y se mide qué ensamble se
    reactiva — el ítem cuyo ensamble tiene más disparos gana.

    Diferencias respecto a PoC-3:
    - El espacio de patrones es continuo (no grupos fijos): cada ítem se mapea
      a un subconjunto de neuronas por umbral de similitud con vectores de
      decodificación fijos.
    - Soporta capacity por evicción explícita (borra el ensamble de menor
      refuerzo acumulado).

    Coste: es el más lento. Se recomienda n_neurons <= 200 para benchmarks.
    """

    def __init__(
        self,
        dim: int,
        capacity: int,
        seed: int = 0,
        n_neurons: int = 128,
        sparsity: float = 0.15,
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
        # training
        n_burst: int = 6,
        burst_gap: int = 4,
        n_presentations: int = 30,
        # readout
        t_readout: int = 50,
        cue_frac: float = 0.5,
    ):

        rng = np.random.default_rng(seed)
        self.dim = dim
        self.capacity = capacity
        self.n = n_neurons
        self.sparsity = sparsity
        self.cue_frac = cue_frac
        self.t_readout = t_readout
        self.n_burst = n_burst
        self.burst_gap = burst_gap
        self.n_pres = n_presentations

        # STDP / LIF params
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_m = tau_m
        self.v_th = v_th
        self.t_ref = t_ref

        # Vectores de decodificación: mapean item vector → conjunto de neuronas
        self.D = rng.standard_normal((n_neurons, dim)).astype(np.float32)
        self.D /= np.linalg.norm(self.D, axis=1, keepdims=True) + 1e-8

        # Pesos sinápticos compartidos (todos los ítems comparten la red)
        mask = rng.random((n_neurons, n_neurons)) < 0.25
        np.fill_diagonal(mask, False)
        self.mask = mask
        self.W = rng.uniform(0.01, 0.03, (n_neurons, n_neurons)).astype(np.float32) * mask

        # Registro de ensambles: qué neuronas pertenecen a cada ítem
        self._patterns: list[np.ndarray] = []  # índices de neuronas
        self._vals: list = []
        self._strength: list[float] = []
        self._keys: list[np.ndarray] = []  # vectores originales

        # Estado dinámico
        self._reset_state()

    def _reset_state(self):
        self.V = np.zeros(self.n, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=int)
        self.x_trace = np.zeros(self.n, dtype=np.float32)
        self.y_trace = np.zeros(self.n, dtype=np.float32)
        self.last_spikes = np.zeros(self.n, dtype=np.float32)

    def _item_neurons(self, key: np.ndarray) -> np.ndarray:
        """Selecciona las k neuronas más alineadas con el vector key."""
        sims = self.D @ _unit(key)
        k = max(2, int(self.n * self.sparsity))
        return np.argpartition(-sims, k)[:k]

    def _stdp_update(self, spikes: np.ndarray):
        dt = 1.0
        self.x_trace *= np.exp(-dt / self.tau_plus)
        self.y_trace *= np.exp(-dt / self.tau_minus)
        if spikes.any():
            self.W[np.ix_(spikes, np.arange(self.n))] += (
                self.a_plus * self.x_trace[None, :] * self.mask[np.ix_(spikes, np.arange(self.n))]
            )
            self.W[np.ix_(np.arange(self.n), spikes)] -= (
                self.a_minus * self.y_trace[:, None] * self.mask[np.ix_(np.arange(self.n), spikes)]
            )
            np.clip(self.W, 0, self.w_max, out=self.W)
            self.x_trace[spikes] += 1.0
            self.y_trace[spikes] += 1.0

    def _lif_step(self, ext: np.ndarray, learn: bool) -> np.ndarray:
        inp = self.W @ self.last_spikes + ext
        active = self.refrac <= 0
        dV = (-(self.V) / self.tau_m) + inp
        self.V = np.where(active, self.V + dV, 0.0)
        spikes = active & (self.V >= self.v_th)
        self.V[spikes] = 0.0
        self.refrac[spikes] = self.t_ref
        self.refrac[~spikes] = np.maximum(self.refrac[~spikes] - 1, 0)
        if learn:
            self._stdp_update(spikes)
        self.last_spikes = spikes.astype(np.float32)
        return spikes

    def _train_pattern(self, neuron_ids: np.ndarray):
        """Presenta ráfagas forzadas en las neuronas del ítem para que STDP aprenda."""
        rng = np.random.default_rng(id(neuron_ids) % 2**32)
        burst_len = self.n_burst * self.burst_gap
        for _ in range(self.n_pres):
            for t in range(burst_len + 10):
                forced = np.zeros(self.n, dtype=bool)
                for ki in range(self.n_burst):
                    if t == ki * self.burst_gap:
                        jitter = rng.integers(-1, 2, size=len(neuron_ids))
                        fire = (t - ki * self.burst_gap) == jitter
                        forced[neuron_ids[fire]] = True
                ext = np.zeros(self.n, dtype=np.float32)
                ext[forced] = 2.0
                self._lif_step(ext, learn=True)
            # reset trazas entre presentaciones
            self.x_trace[:] = 0.0
            self.y_trace[:] = 0.0

    def write(self, key: np.ndarray, value, pred_error: float = 0.5):
        neurons = self._item_neurons(key)
        self._train_pattern(neurons)

        self._patterns.append(neurons)
        self._vals.append(value)
        self._strength.append(1.0 + 2.0 * pred_error)
        self._keys.append(_unit(key.astype(np.float32)))

        # desalojo por mínima fuerza
        while len(self._vals) > self.capacity:
            drop = int(np.argmin(self._strength))
            self._patterns.pop(drop)
            self._vals.pop(drop)
            self._strength.pop(drop)
            self._keys.pop(drop)

    def read(self, query: np.ndarray):
        if not self._vals:
            return None, 0.0

        q = _unit(query.astype(np.float32))
        cue_neurons = self._item_neurons(q)
        # cue parcial: solo la fracción cue_frac de las neuronas del cue
        cue_size = max(1, int(len(cue_neurons) * self.cue_frac))
        cue = cue_neurons[:cue_size]

        self._reset_state()
        self.last_spikes = np.zeros(self.n, dtype=np.float32)
        spike_counts = np.zeros(self.n, dtype=np.int32)

        for t in range(self.t_readout):
            ext = np.zeros(self.n, dtype=np.float32)
            if t < 4:
                ext[cue] = 2.0
            spikes = self._lif_step(ext, learn=False)
            spike_counts += spikes.astype(np.int32)

        # votar: qué ensamble registrado tiene más disparos
        votes = np.array([spike_counts[p].sum() / (len(p) + 1e-8) for p in self._patterns])

        best = int(np.argmax(votes))
        # similitud como fracción del ensamble reclutado
        recruited = spike_counts[self._patterns[best]].sum()
        max_possible = self.t_readout * len(self._patterns[best])
        sim = float(np.clip(recruited / (max_possible + 1e-8), 0, 1))

        # también devolver similitud coseno con la clave original
        cos_sim = float(np.clip(_cosine(self._keys[best], q), 0, 1))
        return self._vals[best], max(sim, cos_sim)


# ─── 4. Spiking-SDM ──────────────────────────────────────────────────────────


class SpikingSDMMemory:
    """
    SDM cuyo circuito de activación es LIF en vez de comparación algebraica.
    Basada en PoC-4, adaptada a vectores continuos (vs. binarios en PoC-4).

    El vector de entrada se codifica como un tren de spikes por rate coding
    (p_fire[i] proporcional a la componente i normalizada). La similitud entre
    el input y cada hard location emerge de la integración LIF.
    """

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
        pred_error_gate: bool = True,
    ):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.capacity = capacity
        self.n_hard = n_hard
        self.k_active = max(1, int(n_hard * activation_frac))
        self.T = T_window
        self.p_high = p_high
        self.p_low = p_low
        self.leak = leak
        self.pred_error_gate = pred_error_gate
        self._rng = rng

        # hard locations: vectores unitarios aleatorios fijos
        H = rng.standard_normal((n_hard, dim)).astype(np.float32)
        self.H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

        # pesos de integración: H[j,i] = alineación entre hard-loc j y dim i
        self.W_pos = np.clip(self.H, 0, None)  # dims positivas
        self.W_neg = np.clip(-self.H, 0, None)  # dims negativas

        # contadores de valor
        self.V = np.zeros((n_hard, dim), dtype=np.float32)

        # tabla explícita de ítems
        self._keys: list[np.ndarray] = []
        self._vals: list = []
        self._strength: list[float] = []

    def _encode_spikes(self, vec: np.ndarray):
        """Rate coding: cada dimensión genera spikes proporcional a su magnitud."""
        v_norm = _unit(vec)
        p_pos = np.where(v_norm > 0, self.p_high * v_norm + self.p_low, self.p_low)
        p_neg = np.where(v_norm < 0, self.p_high * (-v_norm) + self.p_low, self.p_low)
        p_pos = np.clip(p_pos, 0, 1)
        p_neg = np.clip(p_neg, 0, 1)
        sp_pos = (self._rng.random((self.T, self.dim)) < p_pos).astype(np.float32)
        sp_neg = (self._rng.random((self.T, self.dim)) < p_neg).astype(np.float32)
        return sp_pos, sp_neg

    def _activated_lif(self, vec: np.ndarray) -> np.ndarray:
        """Devuelve índices de hard locations que dispararon al menos una vez."""
        sp_pos, sp_neg = self._encode_spikes(vec)
        Vmem = np.zeros(self.n_hard, dtype=np.float32)
        fired = np.zeros(self.n_hard, dtype=bool)
        for t in range(self.T):
            inp = (self.W_pos @ sp_pos[t] + self.W_neg @ sp_neg[t]) / self.dim
            Vmem = self.leak * Vmem + inp
            new_fire = (Vmem >= 0.5) & ~fired
            fired |= new_fire
            Vmem[new_fire] = 0.0
        # si ninguna disparó, tomar las k más integradas
        if not fired.any():
            fired[np.argpartition(-Vmem, self.k_active)[: self.k_active]] = True
        return np.where(fired)[0]

    def write(self, key: np.ndarray, value, pred_error: float = 0.5):
        k = _unit(key.astype(np.float32))
        strength = 1.0 + (2.0 * pred_error if self.pred_error_gate else 0.0)

        idx = self._activated_lif(k)
        self.V[idx] += strength * k

        self._keys.append(k)
        self._vals.append(value)
        self._strength.append(strength)

        while len(self._vals) > self.capacity:
            drop = int(np.argmin(self._strength))
            dk = self._keys.pop(drop)
            self._vals.pop(drop)
            self._strength.pop(drop)
            di = self._activated_lif(dk)
            self.V[di] -= dk
            self.V = np.clip(self.V, -1e6, 1e6)

    def read(self, query: np.ndarray):
        if not self._keys:
            return None, 0.0
        q = _unit(query.astype(np.float32))
        idx = self._activated_lif(q)

        raw = self.V[idx].sum(axis=0)
        if np.linalg.norm(raw) < 1e-8:
            return None, 0.0
        reconstructed = _unit(raw)

        keys_arr = np.stack(self._keys)
        sims = _cosine_matrix(keys_arr, reconstructed)
        best = int(np.argmax(sims))
        return self._vals[best], float(np.clip(sims[best], 0, 1))


# ─── 5. FIFO baseline ────────────────────────────────────────────────────────


class FIFOMemory:
    """EpisodicBuffer FIFO de e-MDB — sin criterio de importancia."""

    def __init__(self, dim: int, capacity: int, seed: int = 0, **kwargs):
        self.dim = dim
        self.capacity = capacity
        self._keys: list[np.ndarray] = []
        self._vals: list = []

    def write(self, key: np.ndarray, value, pred_error: float = 0.5):
        self._keys.append(_unit(key.astype(np.float32)))
        self._vals.append(value)
        if len(self._vals) > self.capacity:
            self._keys.pop(0)
            self._vals.pop(0)

    def read(self, query: np.ndarray):
        if not self._vals:
            return None, 0.0
        q = _unit(query.astype(np.float32))
        keys_arr = np.stack(self._keys)
        sims = _cosine_matrix(keys_arr, q)
        best = int(np.argmax(sims))
        return self._vals[best], float(np.clip(sims[best], 0, 1))


# ─── registro ─────────────────────────────────────────────────────────────────

ARCHITECTURES = {
    "SDM": SDMMemory,
    "ENN": ENNMemory,
    "Spiking": SpikingMemory,
    "SpikingSDM": SpikingSDMMemory,
    "FIFO": FIFOMemory,
}
