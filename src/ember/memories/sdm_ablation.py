"""Variantes de la SDM para separar qué aporta cada uno de sus componentes.

La `SDMMemory` del benchmark tiene dos estructuras de estado:

1. **Contadores distribuidos** `V` (más las direcciones `H` de las hard
   locations): la superposición de Kanerva propiamente dicha.
2. **La lista de trazas** (`TraceStore`): claves exactas, payloads y la
   metadata de ciclo de vida (fuerza, edad, utilidad, contribución).

La lectura usa las dos: reconstruye desde los contadores y después decide el
ganador comparando esa reconstrucción contra las claves exactas de la lista. La
revisión de BIP2026 pregunta, con razón, cuánto de lo que rinde la SDM es de
cada una. Este módulo construye las variantes que lo separan:

- `SDMCountersOnly(exact_erase=False)` — **(a) solo contadores**. No guarda
  claves: el payload se reconstruye desde los contadores, que almacenan la
  clave concatenada con un código aleatorio del payload (memoria
  heteroasociativa). Lo único por traza es una tabla de símbolos —el código, el
  payload y la fuerza— porque devolver un payload arbitrario exacto exige
  guardarlo en algún lado. Al desalojar un símbolo **no** se puede restar su
  aporte a los contadores (no hay clave para recalcular el conjunto activo), así
  que el residuo queda: es la SDM clásica, que olvida por interferencia.
- `SDMCountersOnly(exact_erase=True)` — (a+) igual, pero conserva la clave
  exacta de cada traza **solo** para poder restar su aporte al desalojar. Separa
  el papel de la lista como contabilidad de borrado de su papel como
  decodificador.
- `sdm_traces_only` — **(b) solo la lista de trazas**, direccionada por vecino
  más cercano y sin contadores. Es exactamente la SDM con la lectura por
  superposición quitada: mismo `PredErrorGated`, mismo `MinStrength`, sin
  decaimiento ni refuerzo. Se construye como un punto del espacio de genotipos,
  no como una clase nueva, para no duplicar el motor.
- La variante (c), completa, es `SDMMemory` sin cambios.

`AdmissionGated` envuelve cualquier memoria con una compuerta de admisión
(`ember.core.policies.AdmissionPolicy`). Ninguna de estas variantes está
registrada en `ARCHITECTURES`: no entran al benchmark de `exp04` ni al espacio
de búsqueda.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.core.genotype import Genotype
from ember.core.memory import PolicyMemory
from ember.core.policies import (
    AdmissionPolicy,
    Append,
    EvictPolicy,
    MinStrength,
    NearestNeighbour,
    NoDecay,
    PredErrorGated,
    StrengthPolicy,
)
from ember.core.store import TraceStore
from ember.core.types import EMPTY_READ, ReadResult, unit, unit_rows

SDM_TRACES_ONLY_GENOTYPE = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=PredErrorGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""El ciclo de vida de traza de la `SDMMemory`, sin sus contadores.

`SDMMemory` usa `PredErrorGated` y `MinStrength` por defecto, no decae ni
refuerza por lectura (solo incrementa utilidad). Este genotipo es eso mismo con
lectura por vecino más cercano sobre las claves exactas.
"""


def sdm_traces_only(dim: int, capacity: int, seed: int = 0) -> PolicyMemory:
    """Variante (b): la lista de trazas de la SDM, sin contadores."""
    return PolicyMemory(dim=dim, capacity=capacity, genotype=SDM_TRACES_ONLY_GENOTYPE, seed=seed)


class SDMCountersOnly:
    """Variante (a): SDM heteroasociativa que reconstruye el payload desde los contadores.

    Cada escritura suma `strength * [clave ; código]` a las hard locations
    activas, donde `código` es un vector unitario aleatorio de dimensión
    `code_dim` asignado al payload. La lectura suma los contadores activados por
    la consulta, separa la mitad-clave de la mitad-código, y decodifica el
    payload como el símbolo de la tabla cuyo código se parece más a la mitad
    código. La similitud reportada es la de la consulta contra la mitad-clave
    reconstruida: no hay clave exacta guardada contra la cual compararla.

    La tabla de símbolos es un `TraceStore` cuyas "claves" son los códigos: así
    el desalojo reutiliza la misma política del núcleo (`MinStrength` por
    defecto) sobre la misma fuerza de saliencia que la SDM completa.
    """

    def __init__(
        self,
        dim: int,
        capacity: int,
        seed: int = 0,
        n_hard: int = 512,
        activation_frac: float = 0.05,
        code_dim: int | None = None,
        exact_erase: bool = False,
        strength: StrengthPolicy | None = None,
        evict: EvictPolicy | None = None,
    ) -> None:
        self.dim = dim
        self.n_hard = n_hard
        self.k_active = max(1, int(n_hard * activation_frac))
        self.code_dim = code_dim or dim
        self.exact_erase = exact_erase
        self.strength_policy = strength or PredErrorGated()
        self.evict_policy = evict or MinStrength()

        self.rng = np.random.default_rng(seed)
        # Mismo orden de extracción que `SDMMemory`: con la misma semilla las
        # hard locations son idénticas, así que la ablación no cambia el azar.
        self.H: NDArray[np.float32] = unit_rows(
            self.rng.standard_normal((n_hard, dim)).astype(np.float32)
        )
        self.V: NDArray[np.float32] = np.zeros((n_hard, dim + self.code_dim), dtype=np.float32)

        # Tabla de símbolos: código del payload, payload y metadata de ciclo de vida.
        self.store = TraceStore(dim=self.code_dim, capacity=capacity)
        # Solo con `exact_erase`: la clave de cada traza, para restar su aporte.
        self.erase_keys: NDArray[np.float32] | None = (
            np.zeros((0, dim), dtype=np.float32) if exact_erase else None
        )
        self.n_evictions = 0

    @property
    def capacity(self) -> int:
        return self.store.capacity

    def __len__(self) -> int:
        return len(self.store)

    # ----------------------------------------------------------------- interno

    def _conjunto_activo(self, key: NDArray) -> NDArray[np.intp]:
        sims = self.H @ key
        return np.argpartition(-sims, self.k_active - 1)[: self.k_active].astype(np.intp)

    def _reconstruir(self, q: NDArray) -> NDArray[np.float32]:
        return self.V[self._conjunto_activo(q)].sum(axis=0)

    def _codigo_nuevo(self) -> NDArray[np.float32]:
        """Código bipolar aleatorio y unitario: casi ortogonal a los demás."""
        signos = self.rng.integers(0, 2, size=self.code_dim).astype(np.float32) * 2.0 - 1.0
        return unit(signos)

    def _desalojar(self, idx: int) -> None:
        if self.erase_keys is not None:
            clave = self.erase_keys[idx].copy()
            aporte = float(self.store.contribution[idx])
            patron = np.concatenate([clave, self.store.keys[idx]])
            self.V[self._conjunto_activo(clave)] -= np.float32(aporte) * patron
            self.erase_keys = np.delete(self.erase_keys, idx, axis=0)
        self.store.remove(idx)
        self.n_evictions += 1

    # --------------------------------------------------------------- escritura

    def write(self, key: NDArray, value: Any, pred_error: float = 0.5) -> None:
        k = unit(key)
        self.store.tick()

        # Sin claves guardadas, la novedad solo se puede medir contra lo que los
        # contadores reconstruyen para esta clave.
        crudo = self._reconstruir(k)[: self.dim]
        norma = float(np.linalg.norm(crudo))
        novedad = 1.0 if norma < 1e-6 else float(1.0 - np.clip(crudo @ k / norma, -1.0, 1.0))
        s = self.strength_policy.initial(pred_error=pred_error, novelty=novedad)

        codigo = self._codigo_nuevo()
        self.V[self._conjunto_activo(k)] += np.float32(s) * np.concatenate([k, codigo])
        self.store.append(codigo, value, strength=s, contribution=s)
        if self.erase_keys is not None:
            self.erase_keys = np.vstack([self.erase_keys, k[None, :]])

        while len(self.store) > self.store.capacity:
            self._desalojar(self.evict_policy.victim(self.store, self.rng))

    # ----------------------------------------------------------------- lectura

    def read(self, query: NDArray) -> ReadResult:
        if len(self.store) == 0:
            return EMPTY_READ

        q = unit(query)
        crudo = self._reconstruir(q)
        mitad_clave, mitad_codigo = crudo[: self.dim], crudo[self.dim :]
        if float(np.linalg.norm(mitad_codigo)) < 1e-6:
            return EMPTY_READ

        ganador = int(np.argmax(self.store.similarities(mitad_codigo)))
        self.store.touch(ganador)

        norma = float(np.linalg.norm(mitad_clave))
        sim = 0.0 if norma < 1e-6 else float(np.clip(mitad_clave @ q / norma, 0.0, 1.0))
        return ReadResult(value=self.store.values[ganador], similarity=sim, index=ganador)


class AdmissionGated:
    """Envuelve una memoria con una compuerta de admisión en la escritura.

    Lo rechazado no toca la memoria en absoluto: ni avanza su reloj ni compite
    por lugar. La lectura se delega sin cambios.
    """

    def __init__(self, inner: Any, admission: AdmissionPolicy) -> None:
        self.inner = inner
        self.admission = admission
        self.n_rejected = 0

    @property
    def capacity(self) -> int:
        return self.inner.capacity

    @property
    def n_evictions(self) -> int | None:
        return getattr(self.inner, "n_evictions", None)

    def __len__(self) -> int:
        return len(self.inner)

    def write(self, key: NDArray, value: Any, pred_error: float = 0.5) -> None:
        if self.admission.admit(pred_error):
            self.inner.write(key, value, pred_error)
        else:
            self.n_rejected += 1

    def read(self, query: NDArray) -> ReadResult:
        return self.inner.read(query)
