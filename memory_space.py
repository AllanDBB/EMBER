"""
Espacio de busqueda de arquitecturas de memoria bioinspirada.

Cada arquitectura es un genotipo discreto que combina mecanismos tomados de
tres familias:
  - Sparse Distributed Memory (Kanerva): lectura distribuida sobre un
    conjunto de ubicaciones dentro de un radio de similitud.
  - Engramas / plasticidad Hebbiana: consolidacion por fusion, refuerzo por
    uso, decaimiento de trazas no reforzadas.
  - Ciclo de vida del engrama: fuerza inicial modulada por novedad y error
    de prediccion; desalojo por fuerza, utilidad, edad o azar.

El punto clave: el EpisodicBuffer FIFO de e-MDB es UN PUNTO de este espacio
(read=nn, write=append, init=constant, decay=1.0, evict=fifo, reinforce=0),
no un baseline externo. Eso permite preguntar si la busqueda lo domina.
"""

import numpy as np
from itertools import product

# ---------------------------------------------------------------- genotipo

SEARCH_SPACE = {
    "read_mode": ["nn", "topk3", "radius"],  # nn=episodico, radius=SDM-like
    "write_mode": ["append", "merge"],  # merge = consolidacion
    "init_str": ["constant", "novelty", "perr", "both"],
    "decay": [1.0, 0.995, 0.98],
    "evict": ["fifo", "min_strength", "min_utility", "random"],
    "reinforce": [0.0, 0.5],
}

FIFO_GENOTYPE = {  # = e-MDB EpisodicBuffer
    "read_mode": "nn",
    "write_mode": "append",
    "init_str": "constant",
    "decay": 1.0,
    "evict": "fifo",
    "reinforce": 0.0,
}


def enumerate_space():
    keys = list(SEARCH_SPACE.keys())
    for combo in product(*[SEARCH_SPACE[k] for k in keys]):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------- memoria


class ConfigurableMemory:
    """Memoria asociativa cuyo comportamiento queda definido por el genotipo."""

    MERGE_THRESHOLD = 0.85
    RADIUS_THRESHOLD = 0.70

    def __init__(self, genotype, capacity, dim, seed=0):
        self.g = genotype
        self.capacity = capacity
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.keys = np.zeros((0, dim), dtype=np.float32)
        self.values = []
        self.strength = np.zeros(0, dtype=np.float32)
        self.age = np.zeros(0, dtype=np.float32)
        self.utility = np.zeros(0, dtype=np.float32)
        self.t = 0

    # --------------------------------------------------------- utilidades
    def _sims(self, key):
        if len(self.values) == 0:
            return np.zeros(0, dtype=np.float32)
        num = self.keys @ key
        den = np.linalg.norm(self.keys, axis=1) * np.linalg.norm(key) + 1e-8
        return num / den

    def _append(self, key, value, strength):
        self.keys = np.vstack([self.keys, key[None, :]])
        self.values.append(value)
        self.strength = np.append(self.strength, strength)
        self.age = np.append(self.age, 0.0)
        self.utility = np.append(self.utility, 0.0)

    def _remove(self, idx):
        self.keys = np.delete(self.keys, idx, axis=0)
        self.values.pop(idx)
        self.strength = np.delete(self.strength, idx)
        self.age = np.delete(self.age, idx)
        self.utility = np.delete(self.utility, idx)

    def _evict(self):
        while len(self.values) > self.capacity:
            mode = self.g["evict"]
            if mode == "fifo":
                idx = int(np.argmax(self.age))
            elif mode == "min_strength":
                idx = int(np.argmin(self.strength))
            elif mode == "min_utility":
                idx = int(np.argmin(self.utility))
            else:  # random
                idx = int(self.rng.integers(len(self.values)))
            self._remove(idx)

    # ------------------------------------------------------------ escritura
    def write(self, key, value, pred_error=0.0):
        self.t += 1
        self.age += 1.0
        if self.g["decay"] < 1.0:
            self.strength *= self.g["decay"]

        sims = self._sims(key)
        novelty = float(1.0 - sims.max()) if len(sims) else 1.0

        if self.g["write_mode"] == "merge" and len(sims) and sims.max() >= self.MERGE_THRESHOLD:
            idx = int(sims.argmax())
            self.strength[idx] += max(self.g["reinforce"], 0.5)  # consolidar siempre suma algo
            self.utility[idx] += 1.0
            self.age[idx] = 0.0
            return

        mode = self.g["init_str"]
        s = 1.0
        if mode == "novelty":
            s += 2.0 * novelty
        elif mode == "perr":
            s += 2.0 * pred_error
        elif mode == "both":
            s += 2.0 * novelty + 2.0 * pred_error
        self._append(key, value, s)
        self._evict()

    # -------------------------------------------------------------- lectura
    def read(self, key):
        """Devuelve (valor_recuperado, similitud_efectiva)."""
        sims = self._sims(key)
        if len(sims) == 0:
            return None, 0.0

        mode = self.g["read_mode"]
        if mode == "nn":
            sel = np.array([int(sims.argmax())])
        elif mode == "topk3":
            sel = np.argsort(-sims)[:3]
        else:  # radius: todas las ubicaciones dentro del radio (SDM-like)
            sel = np.where(sims >= self.RADIUS_THRESHOLD)[0]
            if len(sel) == 0:
                sel = np.array([int(sims.argmax())])

        if self.g["reinforce"] > 0:
            self.strength[sel] += self.g["reinforce"]
        self.utility[sel] += 1.0

        # lectura ponderada por similitud (superposicion tipo SDM)
        w = np.clip(sims[sel], 0, None)
        if w.sum() <= 0:
            best = sel[int(np.argmax(sims[sel]))]
            return self.values[best], float(sims[best])
        best_local = int(np.argmax(w))
        best = sel[best_local]
        return self.values[best], float(sims[best])
