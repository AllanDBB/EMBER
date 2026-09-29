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


# ═══════════════════════════════════════ políticas externas (fuera del espacio)
#
# Todo lo que sigue son baselines de gestión de memoria NO bioinspirados, tomados
# de la literatura de cachés, de muestreo en flujo y de aprendizaje continuo. Se
# implementan acá —y no en otro módulo— por el invariante de un solo motor: una
# política externa se compone con `PolicyMemory` igual que cualquier otra, así
# que la comparación contra la frontera no tiene diferencias de código
# accidentales.
#
# **Ninguna de estas entra a `ember.nas.space.SEARCH_SPACE`.** Las 576
# configuraciones y los resultados de exp01–exp06 no cambian. Las lecturas y
# desalojos de esta sección consumen `TraceStore.last_use` y
# `TraceStore.priority`, que ninguna política del espacio lee.
#
# Qué cuenta como "uso" para las políticas de caché (LRU, LFU, utilidad): la
# escritura que crea la traza, una fusión que la consolida, y toda lectura que
# la selecciona (`TraceStore.touch`). Es la definición estándar de acceso en una
# caché: lectura o escritura de la entrada.
#
# Admisión: en una caché el ítem entrante siempre entra y se desaloja entre los
# residentes. `PolicyMemory` agrega antes de desalojar, así que las políticas de
# caché excluyen la traza recién agregada (`_residentes`). Las de muestreo y
# selección (reservorio, prioridad, cobertura) sí la dejan competir, como en sus
# fuentes: el reservorio puede rechazar el ítem nuevo, y la repetición
# priorizada selectiva se queda con el conjunto de mayor sorpresa.


def _residentes(store: TraceStore) -> int:
    """Cuántas trazas compiten en un desalojo de caché (todas menos la entrante)."""
    return len(store) - 1 if len(store) > store.capacity else len(store)


@dataclass(frozen=True, slots=True)
class SequentialScan:
    """Lectura del `EpisodicBuffer` real de e-MDB: barrido secuencial, sin similitud.

    Fuente: GII (Universidade da Coruña), repositorio
    https://github.com/GII/emdb_cognitive_nodes_gii, archivo
    `cognitive_nodes/cognitive_nodes/episodic_buffer.py` (commit `d15f96a`,
    feb. 2026). Ahí el buffer principal es `deque(maxlen=main_size)`;
    `add_episode` hace `append` (desalojo FIFO implícito del `deque`), y la
    única forma de recuperar es por posición (`get_sample(index)`) o volcando el
    buffer entero en orden de inserción (`get_dataset`, `get_train_samples`,
    con barajado opcional). No hay ninguna consulta por contenido.

    Para poder evaluarlo con las mismas tareas hay que traducir "consulta" a
    ese modelo de acceso. La traducción más fiel —y es un **supuesto**, porque
    e-MDB nunca responde consultas— es recorrer el buffer en orden de inserción
    (el orden de `get_dataset(shuffle=False)`) y devolver el **primer** episodio
    que coincide con la consulta, donde "coincidir" es el mismo criterio de
    acierto que usa toda la evaluación (`HIT_SIMILARITY = 0.85`). Si nada
    coincide, la lectura falla: devuelve una selección vacía y `PolicyMemory`
    responde `ReadResult(None, 0.0, None)`.

    Diferencia con el proxy `FIFO_GENOTYPE` (`read=nn`): el proxy le regala al
    buffer una búsqueda por vecino más cercano que no tiene. Con el mismo
    desalojo, las dos retienen exactamente las mismas trazas; lo que cambia es
    si una clave degradada todavía encuentra la suya.

    `TraceStore` agrega al final y compacta al desalojar, así que el índice
    creciente es el orden de inserción: el mismo orden que el `deque`.
    """

    threshold: float = 0.85
    label: str = "seq"

    def select(self, sims: NDArray) -> NDArray[np.intp]:
        return np.flatnonzero(sims >= self.threshold)[:1].astype(np.intp)


@dataclass(frozen=True, slots=True)
class Reservoir:
    """Muestreo de reservorio, algoritmo R de Vitter (1985).

    Vitter, J. S. "Random sampling with a reservoir". ACM TOMS 11(1):37–57.
    Mantiene una muestra uniforme de todo el flujo visto. Al llegar el ítem
    `n > C`, se sortea `j ~ U{0, …, n-1}`: si `j < C` el ítem nuevo reemplaza la
    ranura `j`; si no, el ítem nuevo se descarta. Es el baseline de memoria de
    repetición estándar en aprendizaje continuo (Chaudhry et al. 2019, "On tiny
    episodic memories in continual learning"; Isele y Cosgun 2018 lo llaman
    "distribution matching").

    `n` es `store.t`, el número de escrituras vistas. Con escritura `append`
    cada escritura es un ítem del flujo. `PolicyMemory` agrega antes de
    desalojar, así que el ítem nuevo está en el último índice.
    """

    label: str = "reservoir"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        nuevo = len(store) - 1
        n = max(int(store.t), len(store))
        j = int(rng.integers(n))
        return j if j < nuevo else nuevo


@dataclass(frozen=True, slots=True)
class LRU:
    """Desaloja la traza usada hace más tiempo (Least Recently Used).

    La política de reemplazo de caché clásica (Belady 1966; Mattson et al.
    1970, "Evaluation techniques for storage hierarchies"). "Uso" es escritura,
    fusión o lectura que la selecciona (ver encabezado de la sección). Empates:
    la de inserción más vieja.
    """

    label: str = "lru"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        return int(np.argmin(store.last_use))


@dataclass(frozen=True, slots=True)
class LFU:
    """Desaloja la traza con menos usos (Least Frequently Used), desempate LRU.

    LFU clásico con el desempate por recencia de la implementación O(1) de
    Shah, Mitra y Matani (2010), "An O(1) algorithm for implementing the LFU
    cache eviction scheme". La frecuencia es `TraceStore.utility` (lecturas que
    la seleccionan más fusiones).

    Se parece a `MinUtility` del espacio, que usa el mismo contador pero
    desempata por posición de inserción y deja competir a la traza entrante.
    Bajo empates masivos (casi toda traza tiene cero lecturas) el desempate y
    la admisión son lo que decide, y por eso son políticas distintas.
    """

    label: str = "lfu"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        n = _residentes(store)
        return int(np.lexsort((store.last_use[:n], store.utility[:n]))[0])


@dataclass(frozen=True, slots=True)
class PrioritySurprise:
    """Desaloja la traza de menor prioridad, con prioridad = sorpresa.

    Adaptación al desalojo de la repetición priorizada de Schaul et al. (2016),
    "Prioritized experience replay", ICLR, donde la prioridad es el error de
    TD. Es la variante "surprise" de Isele y Cosgun (2018), "Selective
    experience replay for lifelong learning", AAAI, que conserva en el buffer
    las experiencias de mayor error y descarta la de menor. Usa la **misma
    señal** que la compuerta de fuerza de EMBER, pero cruda
    (`TraceStore.priority`, el `pred_error` de escritura): sin novedad, sin
    decaimiento y sin refuerzo por lectura. Empates: la más vieja.
    """

    label: str = "per_min"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        return int(np.argmin(store.priority))


@dataclass(frozen=True, slots=True)
class StochasticPriority:
    """Desalojo estocástico con probabilidad inversa a la prioridad.

    La versión estocástica de Schaul et al. (2016): allí se *muestrea* para
    repetir con `P(i) ∝ p_i^α`, `p_i = |δ_i| + ε`. Acá se *desaloja* con
    `P(i) ∝ (p_i + ε)^(-α)`, que protege lo sorpresivo sin volver imposible
    desalojarlo — la razón por la que Schaul et al. prefieren la versión
    estocástica a la greedy. `α = 1` y `ε = 0.01` fijos, sin ajustar.
    """

    alpha: float = 1.0
    eps: float = 0.01
    label: str = "per_stoch"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        w = (store.priority.astype(np.float64) + self.eps) ** (-self.alpha)
        return int(rng.choice(len(store), p=w / w.sum()))


@dataclass(frozen=True, slots=True)
class UtilityCache:
    """Caché por utilidad: frecuencia de uso con decaimiento por recencia.

    Puntaje `u_i = (1 + usos_i) · 2^(-(t - último_uso_i) / h)`; se desaloja el
    menor. Combina frecuencia y recencia como la familia LRFU de Lee et al.
    (2001), "LRFU: a spectrum of policies that subsumes the LRU and LFU
    policies", IEEE Trans. Computers 50(12), que decae exponencialmente el
    aporte de cada uso. Esta es una aproximación: decae la cuenta total desde
    el último uso, en vez de cada uso por separado. La vida media `h` es la
    capacidad de la memoria (en escrituras), para que la escala sea relativa al
    dominio y no un número absoluto. Empates: la más vieja.
    """

    label: str = "utility_cache"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        n = _residentes(store)
        h = float(store.capacity)
        edad_uso = float(store.t) - store.last_use[:n].astype(np.float64)
        u = (1.0 + store.utility[:n].astype(np.float64)) * np.exp2(-edad_uso / h)
        return int(np.argmin(u))


@dataclass(frozen=True, slots=True)
class CoverageMax:
    """Maximización de cobertura del espacio de claves: desaloja lo más redundante.

    La estrategia "coverage maximization" de Isele y Cosgun (2018), "Selective
    experience replay for lifelong learning", AAAI: descartar experiencias de
    las regiones más densas del espacio de estados para que el buffer lo cubra
    lo más parejo posible. Isele y Cosgun cuentan vecinos dentro de un radio
    fijo; acá se usa la forma sin umbral —un radio absoluto no es transportable
    entre dominios, ver `Radius`—: se desaloja la traza cuyo vecino más cercano
    está más cerca (máxima similitud coseno con otra traza guardada). Es el paso
    greedy de la cobertura max–min (k-centro, Gonzalez 1985). Del par más
    parecido sale el de inserción más vieja.

    No lee ni la sorpresa ni el uso: solo la geometría de las claves.
    """

    label: str = "coverage"

    def victim(self, store: TraceStore, rng: np.random.Generator) -> int:
        sims = store.keys @ store.keys.T
        np.fill_diagonal(sims, -np.inf)
        return int(np.argmax(sims.max(axis=1)))
