"""Sparse Distributed Memory de Kanerva sobre un espacio continuo.

La idea central de Kanerva es que guardar información en un solo lugar es
frágil: los patrones deben distribuirse sobre muchas ubicaciones solapadas, de
modo que una consulta ruidosa o parcial active suficientes de las mismas
ubicaciones como para reconstruir el original. Esa es la propiedad que hace a
SDM útil para un robot: degradación suave en vez de fallo abrupto.

Adaptación al dominio continuo: en la SDM original las hard locations tienen
direcciones en {0,1}^n y se activan por radio de Hamming. Aquí son vectores
unitarios aleatorios fijos y se activan las `activation_frac` más alineadas por
producto interno, lo que hace el radio autoadaptativo a la densidad local del
espacio.

Conservación de contadores
--------------------------
Escribir suma `strength * key` a las ubicaciones activas. Desalojar tiene que
restar **exactamente eso**. El código piloto restaba `key` (sin el factor de
fuerza), de modo que cada desalojo dejaba un residuo proporcional a
`strength - 1` en los contadores; sobre un flujo de cientos de escrituras en
una memoria de capacidad 20, los contadores terminaban dominados por basura de
trazas ya borradas. `TraceStore.contribution` guarda el escalar exacto y
`_desalojar` lo usa.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.core.policies import EvictPolicy, MinStrength, PredErrorGated, StrengthPolicy
from ember.core.store import TraceStore
from ember.core.types import EMPTY_READ, ReadResult, unit, unit_rows


class SDMMemory:
    """SDM sobre vectores continuos, con ciclo de vida de traza del núcleo."""

    def __init__(
        self,
        dim: int,
        capacity: int,
        seed: int = 0,
        n_hard: int = 512,
        activation_frac: float = 0.05,
        strength: StrengthPolicy | None = None,
        evict: EvictPolicy | None = None,
    ) -> None:
        self.dim = dim
        self.n_hard = n_hard
        self.k_active = max(1, int(n_hard * activation_frac))
        self.strength_policy = strength or PredErrorGated()
        self.evict_policy = evict or MinStrength()

        self.rng = np.random.default_rng(seed)
        self.store = TraceStore(dim=dim, capacity=capacity)
        self.n_evictions = 0

        # Hard locations: direcciones fijas que no cambian durante la vida de la
        # memoria. Se derivan de la semilla, nunca del reloj ni de id().
        self.H: NDArray[np.float32] = unit_rows(
            self.rng.standard_normal((n_hard, dim)).astype(np.float32)
        )
        # Contadores de valor: la superposición distribuida propiamente dicha.
        self.V: NDArray[np.float32] = np.zeros((n_hard, dim), dtype=np.float32)

    @property
    def capacity(self) -> int:
        return self.store.capacity

    def __len__(self) -> int:
        return len(self.store)

    # ----------------------------------------------------------------- interno

    def _conjunto_activo(self, key: NDArray) -> NDArray[np.intp]:
        """Las `k_active` hard locations más alineadas con la clave.

        Determinista dada la clave: es lo que permite que el desalojo reste sobre
        exactamente el mismo conjunto sobre el que se escribió.
        """
        sims = self.H @ key
        return np.argpartition(-sims, self.k_active - 1)[: self.k_active].astype(np.intp)

    def _desalojar(self, idx: int) -> None:
        """Quita una traza y revierte su contribución a los contadores."""
        clave = self.store.keys[idx].copy()
        aporte = float(self.store.contribution[idx])
        activas = self._conjunto_activo(clave)
        self.V[activas] -= aporte * clave
        self.store.remove(idx)
        self.n_evictions += 1

    # --------------------------------------------------------------- escritura

    def write(self, key: NDArray, value: Any, pred_error: float = 0.5) -> None:
        k = unit(key)
        self.store.tick()

        s = self.strength_policy.initial(pred_error=pred_error, novelty=self.store.novelty(k))

        activas = self._conjunto_activo(k)
        self.V[activas] += np.float32(s) * k
        self.store.append(k, value, strength=s, contribution=s)

        while len(self.store) > self.store.capacity:
            self._desalojar(self.evict_policy.victim(self.store, self.rng))

    # ----------------------------------------------------------------- lectura

    def read(self, query: NDArray) -> ReadResult:
        if len(self.store) == 0:
            return EMPTY_READ

        q = unit(query)
        activas = self._conjunto_activo(q)

        # Lectura por superposición: se suman los contadores de las ubicaciones
        # activadas y el resultado se compara contra las trazas guardadas.
        crudo = self.V[activas].sum(axis=0)
        if float(np.linalg.norm(crudo)) < 1e-6:
            return EMPTY_READ
        reconstruccion = unit(crudo)

        sims_recon = self.store.similarities(reconstruccion)
        ganador = int(np.argmax(sims_recon))
        self.store.touch(ganador)

        # La similitud reportada es contra la consulta original, no contra la
        # reconstrucción: es lo comparable entre arquitecturas.
        sim = float(np.clip(self.store.similarities(q)[ganador], 0.0, 1.0))
        return ReadResult(value=self.store.values[ganador], similarity=sim, index=ganador)
