"""Políticas de ciclo de vida de traza.

Cada familia de política corresponde a un eje del espacio de diseño del NAS, y
cada eje corresponde a un mecanismo de la literatura de memoria bioinspirada:

| Familia          | Raíz biológica                                          |
|------------------|---------------------------------------------------------|
| `StrengthPolicy` | Compuerta de LTP por saliencia y sorpresa (dopamina)     |
| `WritePolicy`    | Creación de traza vs. consolidación Hebbiana por fusión  |
| `ReadPolicy`     | Recuperación episódica vs. círculo de activación (SDM)   |
| `EvictPolicy`    | Olvido por edad vs. por importancia                      |
| `DecayPolicy`    | Debilitamiento sináptico sin refuerzo                    |

Esta es la única implementación de cada mecanismo en todo el repositorio. El
genotipo del NAS es una tupla de estos objetos, y las arquitecturas concretas
(SDM, ENN, spiking) reutilizan las mismas piezas. Por eso el `EpisodicBuffer`
FIFO de e-MDB no es un baseline externo sino un punto del mismo espacio, que es
lo que el paper afirma.

Todas las políticas son `frozen` y sin estado, para poder compararlas por
igualdad, usarlas como clave de caché y pasarlas entre procesos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ember.core.store import TraceStore

# ══════════════════════════════════════════════════════ fuerza inicial de traza


@runtime_checkable
class StrengthPolicy(Protocol):
    """Cuánta fuerza recibe una traza en el momento de codificarse."""

    label: str

    def initial(self, pred_error: float, novelty: float) -> float: ...


@dataclass(frozen=True, slots=True)
class Constant:
    """Toda experiencia pesa lo mismo. Es lo que hace el `EpisodicBuffer`.

    Bajo desalojo por mínima fuerza esto degenera: si todas las trazas tienen
    fuerza idéntica, la política se vuelve pseudoaleatoria. Ese es el 0.017 de
    retención de eventos raros de la Tabla IV.
    """

    label: str = "constant"

    def initial(self, pred_error: float, novelty: float) -> float:
        return 1.0


@dataclass(frozen=True, slots=True)
class NoveltyGated:
    """La fuerza escala con cuán distinta es la experiencia de lo ya guardado."""

    gain: float = 2.0
    label: str = "novelty"

    def initial(self, pred_error: float, novelty: float) -> float:
        return 1.0 + self.gain * novelty


@dataclass(frozen=True, slots=True)
class PredErrorGated:
    """La fuerza escala con la sorpresa (compuerta dopaminérgica de codificación)."""

    gain: float = 2.0
    label: str = "pred_error"

    def initial(self, pred_error: float, novelty: float) -> float:
        return 1.0 + self.gain * pred_error


@dataclass(frozen=True, slots=True)
class BothGated:
    """Novedad y error de predicción, aditivos."""

    gain: float = 2.0
    label: str = "both"

    def initial(self, pred_error: float, novelty: float) -> float:
        return 1.0 + self.gain * novelty + self.gain * pred_error


# ══════════════════════════════════════════════════════════ modo de escritura


@runtime_checkable
class WritePolicy(Protocol):
    """Decide si una experiencia crea traza nueva o consolida en una existente."""

    label: str

    def route(self, store: TraceStore, key: NDArray) -> int | None: ...


@dataclass(frozen=True, slots=True)
class Append:
    """Cada escritura crea una traza nueva. Recuperación episódica pura."""

    label: str = "append"

    def route(self, store: TraceStore, key: NDArray) -> int | None:
        return None


@dataclass(frozen=True, slots=True)
class Merge:
    """Consolidación Hebbiana: si ya hay una traza suficientemente parecida, la refuerza.

    Es el análogo computacional de "las neuronas que se activan juntas se
    conectan": la exposición repetida a la misma experiencia fortalece un solo
    engrama en vez de crear copias redundantes.

    Paga solo cuando los prototipos recurrentes *caben* en la memoria. Por
    encima de `r = 1` llena la memoria de rutina y no queda espacio para lo raro:
    esa es la ley del umbral.
    """

    threshold: float = 0.85
    label: str = "merge"

    def route(self, store: TraceStore, key: NDArray) -> int | None:
        sims = store.similarities(key)
        if sims.size and sims.max() >= self.threshold:
            return int(sims.argmax())
        return None


# ════════════════════════════════════════════════════════ compuerta de admisión


@runtime_checkable
class AdmissionPolicy(Protocol):
    """Decide si una experiencia entra a la memoria o se descarta sin escribirse.

    No es un eje del espacio de búsqueda: ninguna de las 576 configuraciones la
    usa, y ninguna arquitectura registrada la aplica. Existe para el análisis de
    sensibilidad de `exp12_substrate_cost`, que barre un umbral de admisión
    sobre la SDM a través de `ember.memories.sdm_ablation.AdmissionGated`.
    """

    label: str

    def admit(self, pred_error: float) -> bool: ...


@dataclass(frozen=True, slots=True)
class AdmitAll:
    """Toda experiencia se escribe. Es lo que hacen todas las memorias hoy."""

    label: str = "all"

    def admit(self, pred_error: float) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class PredErrorAdmission:
    """Solo se escribe lo que llega con error de predicción `>= threshold`.

    Con `threshold=0` equivale a `AdmitAll`.
    """

    threshold: float = 0.0

    @property
    def label(self) -> str:
        return f"pe>={self.threshold:g}"

    def admit(self, pred_error: float) -> bool:
        return pred_error >= self.threshold


# ═════════════════════════════════════════════════════════════ modo de lectura


@runtime_checkable
class ReadPolicy(Protocol):
    """Qué trazas participan en la reconstrucción de una consulta."""

    label: str

    def select(self, sims: NDArray) -> NDArray[np.intp]: ...


@dataclass(frozen=True, slots=True)
class NearestNeighbour:
    """Devuelve la traza más parecida. Recuperación episódica clásica."""

    label: str = "nn"

    def select(self, sims: NDArray) -> NDArray[np.intp]:
        return np.array([int(sims.argmax())], dtype=np.intp)


@dataclass(frozen=True, slots=True)
class TopK:
    """Agrega las `k` trazas más parecidas."""

    k: int = 3
    label: str = "topk3"

    def select(self, sims: NDArray) -> NDArray[np.intp]:
        return np.argsort(-sims, kind="stable")[: self.k].astype(np.intp)


@dataclass(frozen=True, slots=True)
class Radius:
    """El círculo de activación de Kanerva: todo lo suficientemente cercano contribuye.

    La reconstrucción es una superposición ponderada de varias trazas, no una
    consulta a un solo lugar. `PolicyMemory.read` aplica la superposición sobre
    la selección que devuelve esta política.

    El radio es **relativo al mejor match**, no absoluto. Un umbral absoluto no
    es transportable entre dominios: en R^32 dos vectores gaussianos aleatorios
    tienen coseno ~0.18, así que cualquier umbral por encima de eso hace que el
    círculo contenga siempre exactamente una traza y el modo degenere en vecino
    más cercano — que es precisamente el fallo que hacía inobservable a este eje.

    Kanerva elige el radio de Hamming para que active una fracción conocida de
    las hard locations, aprovechando que en el espacio binario la distancia
    esperada se conoce de antemano. Sobre vectores continuos esa distancia
    depende de los datos, así que el análogo correcto es un radio que se adapte:
    entran las trazas cuya similitud llega al `fraction` de la mejor.
    """

    fraction: float = 0.70
    label: str = "radius"

    def select(self, sims: NDArray) -> NDArray[np.intp]:
        mejor = float(sims.max())
        if mejor <= 0.0:
            return np.array([int(sims.argmax())], dtype=np.intp)
        sel = np.where(sims >= self.fraction * mejor)[0]
        if sel.size == 0:
            return np.array([int(sims.argmax())], dtype=np.intp)
        return sel.astype(np.intp)


# ═══════════════════════════════════════════════════════════ política de olvido


@runtime_checkable
class EvictPolicy(Protocol):
    """A quién se descarta cuando la memoria está llena."""

    label: str

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int: ...


@dataclass(frozen=True, slots=True)
class FIFO:
    """Descarta lo más viejo. Es lo que hace el `EpisodicBuffer` de e-MDB.

    Insensible a la fuerza: por eso ninguna compuerta de saliencia tiene efecto
    sobre una memoria que desaloja así.
    """

    label: str = "fifo"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        return int(np.argmax(store.age))


@dataclass(frozen=True, slots=True)
class MinStrength:
    """Descarta la traza más débil. La única política que lee la señal de fuerza.

    De las cuatro políticas de desalojo, es la única sensible a la fuerza, y por
    eso es la única bajo la cual la compuerta de saliencia hace algo. Ese
    condicionamiento es lo que explica que el efecto principal de la fuerza
    inicial sea 0.2 % mientras su efecto condicional es un factor de 35.
    """

    label: str = "min_strength"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        return int(np.argmin(store.strength))


@dataclass(frozen=True, slots=True)
class MinUtility:
    """Descarta la traza menos consultada."""

    label: str = "min_utility"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        return int(np.argmin(store.utility))


@dataclass(frozen=True, slots=True)
class Random:
    """Descarta una traza al azar. Es el piso contra el que se mide todo lo demás."""

    label: str = "random"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        return int(rng.integers(len(store)))


# ════════════════════════════════════════════════════════ decaimiento de traza


@runtime_checkable
class DecayPolicy(Protocol):
    """Erosión de la fuerza de las trazas que no se refuerzan."""

    label: str

    def step(self, strength: NDArray) -> None: ...


@dataclass(frozen=True, slots=True)
class NoDecay:
    """Las trazas no se debilitan con el tiempo."""

    label: str = "1.0"

    def step(self, strength: NDArray) -> None:
        return


@dataclass(frozen=True, slots=True)
class ExponentialDecay:
    """Debilitamiento sináptico: se pierde una fracción de fuerza por paso."""

    rate: float = 0.98

    @property
    def label(self) -> str:
        return str(self.rate)

    def step(self, strength: NDArray) -> None:
        strength *= np.float32(self.rate)
