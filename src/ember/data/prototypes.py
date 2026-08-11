"""Estimación del número de prototipos recurrentes de un flujo.

La ley del umbral se enuncia sobre `r = K_proto / C`. Sobre flujos sintéticos
`K_proto` se conoce por construcción, pero un robot no sabe cuántas situaciones
distintas contiene su ambiente: tiene un flujo de percepciones y nada más.

Para que la ley sea aplicable fuera del laboratorio hay que estimarlo. Este
módulo lo hace en numpy puro —el núcleo no puede depender de scikit-learn— con
k-means sembrado, barrido de k, y selección por coeficiente de silueta. El
intervalo de confianza sale de bootstrap sobre submuestras del flujo, y es lo
que se reporta: la ley sobre datos reales no se enuncia contra un número exacto
sino contra un rango.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ember.core.types import unit_rows

SILUETA_MINIMA = 0.15
"""Silueta por debajo de la cual no se puede afirmar que haya estructura de clúster.

Sobre ruido isotrópico puro el barrido de k produce siluetas de 0.04–0.07 para
todo k: no hay un óptimo, solo fluctuación. Tomar el argmax de esa curva y
reportarlo como `K_proto` sería inventar estructura. El umbral separa "el
estimador encontró algo" de "el estimador no encontró nada", y los experimentos
sobre datos reales tienen que consultarlo antes de calcular `r`.
"""


@dataclass(frozen=True, slots=True)
class PrototypeEstimate:
    """Cuántos prototipos recurrentes parece contener un flujo."""

    k_hat: int
    ci_low: int
    ci_high: int
    scores: dict[int, float]

    @property
    def ci(self) -> tuple[int, int]:
        return (self.ci_low, self.ci_high)

    @property
    def best_score(self) -> float:
        """La silueta del `k` elegido. Cuán creíble es la estimación."""
        return self.scores.get(self.k_hat, -1.0)

    @property
    def has_structure(self) -> bool:
        """Si el flujo tiene estructura de prototipos recurrente y detectable.

        Cuando es `False`, `k_hat` no significa nada y `r` no está definido para
        este flujo: no hay experiencias recurrentes que comprimir ni prototipos
        que quepan o no quepan en memoria.
        """
        return self.best_score >= SILUETA_MINIMA


def _kmeans(
    X: NDArray[np.float32], k: int, rng: np.random.Generator, n_iter: int = 50
) -> NDArray[np.intp]:
    """k-means esférico con inicialización k-means++ sembrada.

    Sobre la esfera unitaria la distancia euclídea es monótona en el coseno, así
    que basta maximizar el producto interno.
    """
    n = len(X)
    # k-means++ : el primer centro al azar, cada siguiente con probabilidad
    # proporcional a su distancia al centro más cercano ya elegido.
    centros = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d = 1.0 - (X @ np.stack(centros).T).max(axis=1)
        d = np.clip(d, 0.0, None) ** 2
        total = d.sum()
        if total <= 0:
            centros.append(X[rng.integers(n)])
            continue
        centros.append(X[int(rng.choice(n, p=d / total))])
    C = unit_rows(np.stack(centros))

    etiquetas = np.zeros(n, dtype=np.intp)
    for _ in range(n_iter):
        nuevas = np.argmax(X @ C.T, axis=1).astype(np.intp)
        if np.array_equal(nuevas, etiquetas):
            break
        etiquetas = nuevas
        for j in range(k):
            miembros = X[etiquetas == j]
            if len(miembros):
                C[j] = miembros.mean(axis=0)
        C = unit_rows(C)
    return etiquetas


def _distancias(X: NDArray[np.float32]) -> NDArray[np.float64]:
    """Matriz de distancia coseno. Se calcula una vez y se reutiliza en todo el barrido."""
    D = 1.0 - (X @ X.T).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    return D


def _silueta(D: NDArray[np.float64], etiquetas: NDArray[np.intp], k: int) -> float:
    """Coeficiente de silueta medio, sobre distancia coseno.

    Mide qué tan compacto es cada grupo frente a su vecino más cercano. Se
    maximiza cuando el número de grupos coincide con la estructura real.
    """
    if k < 2 or len(np.unique(etiquetas)) < 2:
        return -1.0

    puntajes = np.zeros(len(D), dtype=np.float64)
    for j in range(k):
        dentro = etiquetas == j
        n_dentro = int(dentro.sum())
        if n_dentro <= 1:
            continue
        a = D[np.ix_(dentro, dentro)].sum(axis=1) / (n_dentro - 1)
        b = np.full(n_dentro, np.inf)
        for m in range(k):
            if m == j or not (etiquetas == m).any():
                continue
            b = np.minimum(b, D[np.ix_(dentro, etiquetas == m)].mean(axis=1))
        puntajes[dentro] = (b - a) / np.maximum(a, b)
    return float(puntajes.mean())


def _mejor_k(
    X: NDArray[np.float32], ks: list[int], rng: np.random.Generator
) -> tuple[int, dict[int, float]]:
    D = _distancias(X)
    puntajes: dict[int, float] = {}
    for k in ks:
        if k >= len(X):
            continue
        puntajes[k] = _silueta(D, _kmeans(X, k, rng), k)
    if not puntajes:
        return 1, {}
    return max(puntajes, key=lambda k: puntajes[k]), puntajes


def estimate_n_prototypes(
    keys: NDArray,
    *,
    k_max: int = 64,
    n_boot: int = 12,
    subsample: int = 400,
    seed: int = 0,
) -> PrototypeEstimate:
    """Estima cuántos prototipos recurrentes contiene un conjunto de claves.

    `k_max` acota el barrido. `n_boot` submuestras dan el intervalo de confianza:
    sobre datos reales la respuesta honesta es un rango, no un entero.
    """
    X = unit_rows(np.asarray(keys, dtype=np.float32))
    if len(X) < 4:
        return PrototypeEstimate(k_hat=len(X), ci_low=len(X), ci_high=len(X), scores={})

    rng = np.random.default_rng(seed)
    tope = min(k_max, len(X) - 1)
    ks = sorted({max(2, int(round(v))) for v in np.geomspace(2, tope, num=min(16, tope))})

    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]

    k_hat, puntajes = _mejor_k(X, ks, rng)

    estimaciones = []
    for b in range(n_boot):
        rng_b = np.random.default_rng(seed + 1000 + b)
        idx = rng_b.choice(len(X), size=max(8, int(len(X) * 0.7)), replace=True)
        k_b, _ = _mejor_k(np.ascontiguousarray(X[idx]), ks, rng_b)
        estimaciones.append(k_b)

    lo = int(np.percentile(estimaciones, 2.5))
    hi = int(np.ceil(np.percentile(estimaciones, 97.5)))
    return PrototypeEstimate(
        k_hat=k_hat,
        ci_low=min(lo, k_hat),
        ci_high=max(hi, k_hat),
        scores=puntajes,
    )
