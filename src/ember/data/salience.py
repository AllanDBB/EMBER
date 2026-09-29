"""Señales de saliencia imperfectas: degradar el error de predicción de un flujo.

En `clustered_stream` la sorpresa de una experiencia común sale de
N(0.10, 0.05) recortada a [0, 0.35], y la de un evento raro de N(0.90, 0.05)
recortada a [0.65, 1]. Los rangos no se solapan: la sorpresa es un clasificador
perfecto de importancia (AUC = 1). Con esa señal, que una política que desaloja
la traza más débil conserve lo raro es casi una consecuencia de la
construcción, no un hallazgo.

Este módulo reemplaza esa señal por una imperfecta, con cinco perillas
independientes que corresponden a las cinco formas en que una señal de
sorpresa real falla:

- **`auc`** — solapamiento de las distribuciones. La sorpresa latente es
  N(0, 1) para lo rutinario y N(d, 1) para lo importante, con
  `d = √2 · Φ⁻¹(AUC)`, y se mapea a [0, 1] con `Φ(z − d/2)`. El mapeo es
  monótono, así que el AUC de la sorpresa observada es exactamente el pedido
  en esperanza. `None` conserva la señal original del generador.
- **`noise`** — ruido gaussiano aditivo de desvío `noise`, recortado a [0, 1].
- **`delay`, `delay_prob`** — error de asignación de crédito: con probabilidad
  `delay_prob`, la experiencia `t` recibe la sorpresa que produjo la
  experiencia `t − delay`. Con `delay_prob = 1` la sorpresa de un evento raro
  cae entera sobre otra experiencia, `delay` pasos después.
- **`misleading`** — en esa fracción de los eventos, elegidos al azar, la
  señal se invierte (`1 − pe`): lo importante parece rutinario y viceversa.
- **`distractors`** — no degrada la sorpresa sino la *novedad*: esa fracción
  de las experiencias rutinarias se reemplaza por vectores nuevos, sin
  prototipo y sin importancia. Sin esta perilla la novedad (el otro canal de
  saliencia del espacio de diseño) es también un clasificador perfecto por
  construcción, porque lo raro es lo único que no se repite.

El orden de aplicación es fijo: distractores → base (`auc`) → inversión
(`misleading`) → ruido → recorte → retraso. El retraso va al final porque
modela cuándo llega la señal observada, no cómo se genera.

Toda la aleatoriedad sale de la semilla que se pasa. La señal degradada de un
flujo es función determinista de `(flujo, condición, semilla)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from ember.core.types import unit_rows
from ember.data.streams import Stream, StreamItem

SALT = 0x5A11
"""Sal de la semilla: separa esta corriente aleatoria de la del generador del flujo."""


@dataclass(frozen=True, slots=True)
class SignalCondition:
    """Una condición de calidad de señal. Inmutable y hashable, para cachear flujos."""

    auc: float | None = None
    noise: float = 0.0
    delay: int = 0
    delay_prob: float = 1.0
    misleading: float = 0.0
    distractors: float = 0.0

    def __post_init__(self) -> None:
        if self.auc is not None and not 0.5 <= self.auc < 1.0:
            raise ValueError(f"auc debe estar en [0.5, 1) o ser None, se recibió {self.auc}")
        if self.noise < 0.0:
            raise ValueError(f"noise debe ser >= 0, se recibió {self.noise}")
        if self.delay < 0:
            raise ValueError(f"delay debe ser >= 0, se recibió {self.delay}")
        for nombre in ("delay_prob", "misleading", "distractors"):
            v = getattr(self, nombre)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{nombre} debe estar en [0, 1], se recibió {v}")

    @property
    def is_clean(self) -> bool:
        """Si la condición deja la señal del generador intacta."""
        return self == SignalCondition()

    def label(self) -> str:
        """Identificador corto y estable, para las claves del JSON.

        No lleva puntos, porque `paper_sync` navega el JSON con rutas punteadas:
        el AUC va en milésimas (`auc700` = 0.70) y el resto en centésimas
        (`noise030` = 0.30, `p025` = 0.25).
        """
        partes = []
        if self.auc is not None:
            partes.append(f"auc{round(self.auc * 1000):03d}")
        if self.noise:
            partes.append(f"noise{round(self.noise * 100):03d}")
        if self.delay:
            partes.append(f"delay{self.delay}")
            if self.delay_prob < 1.0:
                partes.append(f"p{round(self.delay_prob * 100):03d}")
        if self.misleading:
            partes.append(f"mislead{round(self.misleading * 100):03d}")
        if self.distractors:
            partes.append(f"distr{round(self.distractors * 100):03d}")
        return "_".join(partes) or "clean"


_PHI = np.frompyfunc(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))), 1, 1)


def separation_for_auc(auc: float) -> float:
    """Distancia entre medias de dos gaussianas unitarias que da ese AUC.

    Para N(0, 1) contra N(d, 1), `AUC = Φ(d / √2)`.
    """
    if not 0.5 <= auc < 1.0:
        raise ValueError(f"auc debe estar en [0.5, 1), se recibió {auc}")
    return math.sqrt(2.0) * NormalDist().inv_cdf(auc)


def overlapping_signal(
    is_important: NDArray[np.bool_], auc: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sorpresa en [0, 1] cuyo AUC como clasificador de importancia es `auc`."""
    d = separation_for_auc(auc)
    z = rng.standard_normal(is_important.shape[0]) + d * is_important.astype(np.float64)
    return _PHI(z - d / 2.0).astype(np.float64)


def degrade_signal(
    pred_error: NDArray,
    is_important: NDArray[np.bool_],
    condition: SignalCondition,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Aplica base, inversión, ruido, recorte y retraso a una señal de sorpresa."""
    pe = np.asarray(pred_error, dtype=np.float64).copy()
    imp = np.asarray(is_important, dtype=bool)
    n = pe.shape[0]

    if condition.auc is not None:
        pe = overlapping_signal(imp, condition.auc, rng)

    if condition.misleading > 0.0:
        invertir = rng.random(n) < condition.misleading
        pe[invertir] = 1.0 - pe[invertir]

    if condition.noise > 0.0:
        pe = pe + condition.noise * rng.standard_normal(n)

    pe = np.clip(pe, 0.0, 1.0)

    if condition.delay > 0:
        k = condition.delay
        # La señal que llega en t es la que produjo t − k. Para los primeros k
        # pasos no hay experiencia anterior: reciben una sorpresa rutinaria
        # tomada de la propia señal, para no inventar una distribución nueva.
        rutinarias = pe[~imp] if (~imp).any() else pe
        corrida = np.empty_like(pe)
        corrida[k:] = pe[:-k] if k < n else pe[:0]
        corrida[: min(k, n)] = rng.choice(rutinarias, size=min(k, n))
        retrasar = rng.random(n) < condition.delay_prob
        pe = np.where(retrasar, corrida, pe)

    return pe


def with_novel_distractors(stream: Stream, fraction: float, rng: np.random.Generator) -> Stream:
    """Reemplaza una fracción de lo rutinario por experiencias nuevas sin importancia.

    Cada distractor es un vector unitario al azar que aparece una sola vez: es
    tan novedoso como un evento raro, pero no importa. Conserva el error de
    predicción rutinario que tenía la experiencia que reemplaza.
    """
    if fraction <= 0.0:
        return stream
    indices = [i for i, it in enumerate(stream.items) if not it.is_rare]
    n_distr = int(round(fraction * len(indices)))
    elegidos = sorted(rng.choice(indices, size=n_distr, replace=False).tolist())
    claves = unit_rows(rng.standard_normal((n_distr, stream.spec.dim)).astype(np.float32))

    items = list(stream.items)
    for j, i in enumerate(elegidos):
        items[i] = StreamItem(
            key=claves[j], value=f"distractor{j}", pred_error=items[i].pred_error, is_rare=False
        )
    return Stream(items=items, spec=stream.spec, rare_items=list(stream.rare_items))


def degrade_stream(stream: Stream, condition: SignalCondition, seed: int) -> Stream:
    """Copia del flujo con la señal de saliencia degradada según `condition`.

    Las claves, los valores y las etiquetas de importancia no cambian (salvo
    los distractores, que reemplazan experiencias rutinarias): solo cambia lo
    que la memoria ve como sorpresa.
    """
    rng = np.random.default_rng((seed, SALT))
    base = with_novel_distractors(stream, condition.distractors, rng)

    imp = np.array([it.is_rare for it in base.items], dtype=bool)
    pe = degrade_signal(
        np.array([it.pred_error for it in base.items], dtype=np.float64), imp, condition, rng
    )

    items = [
        StreamItem(key=it.key, value=it.value, pred_error=float(p), is_rare=it.is_rare)
        for it, p in zip(base.items, pe, strict=True)
    ]
    raros = [it for it in items if it.is_rare]
    return Stream(items=items, spec=base.spec, rare_items=raros)


def _rangos_promedio(x: NDArray) -> NDArray[np.float64]:
    """Rangos base 1 con empates promediados (lo que necesita Mann-Whitney)."""
    orden = np.argsort(x, kind="stable")
    ordenados = x[orden]
    rangos = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and ordenados[j + 1] == ordenados[i]:
            j += 1
        rangos[orden[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return rangos


def empirical_auc(scores: NDArray, labels: NDArray[np.bool_]) -> float:
    """AUC de `scores` como clasificador de `labels`, por Mann-Whitney con empates.

    Es la probabilidad de que una experiencia importante al azar tenga más
    sorpresa que una rutinaria al azar (los empates cuentan medio).
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rangos = _rangos_promedio(s)
    u = rangos[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def stream_auc(stream: Stream) -> float:
    """AUC de la sorpresa de un flujo como clasificador de sus eventos raros."""
    return empirical_auc(
        np.array([it.pred_error for it in stream.items]),
        np.array([it.is_rare for it in stream.items], dtype=bool),
    )


__all__ = [
    "SignalCondition",
    "degrade_signal",
    "degrade_stream",
    "empirical_auc",
    "overlapping_signal",
    "separation_for_auc",
    "stream_auc",
    "with_novel_distractors",
]
