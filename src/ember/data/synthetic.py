"""Generadores sintéticos de flujo de experiencia, con estructura controlable.

El código piloto usaba vectores gaussianos i.i.d. sobre la esfera de R^32. En esa
dimensión dos vectores aleatorios son casi ortogonales con alta probabilidad, lo
que reduce la interferencia cruzada entre patrones guardados y probablemente
sobreestima la precisión de recuperación respecto de embeddings perceptuales
reales. Es la limitación principal que declara el paper.

Este módulo conserva ese caso —es `correlation=0, density_skew=0, drift=0`— pero
lo vuelve un punto de un espacio de estructuras, no la única opción:

- **`correlation`**: fracción de la varianza que vive en un subespacio compartido
  de rango bajo. Es lo que tienen los embeddings reales: las clases de un dataset
  no ocupan direcciones independientes, comparten factores.
- **`density_skew`**: las visitas por prototipo siguen una ley de potencias en
  vez de repartirse uniformemente. Un robot no visita todos los lugares de su
  ambiente con la misma frecuencia.
- **`drift`**: los centros se desplazan lentamente a lo largo del flujo. Es el
  cambio distribucional de un agente que opera durante mucho tiempo.

El control exacto sobre `n_prototypes` es lo que hace a estos generadores
irreemplazables para medir la ley del umbral: `r` es la variable independiente,
y sobre datos reales solo se la puede estimar.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ember.core.types import unit_rows
from ember.data.streams import Stream, StreamItem, StreamSpec

PE_COMUN = (0.10, 0.05, 0.0, 0.35)
"""Media, desviación y recorte del error de predicción de una experiencia común."""

PE_RARO = (0.90, 0.05, 0.65, 1.0)
"""Ídem para un evento raro. Los recortes garantizan la separación que la tarea define."""


def _pe(rng: np.random.Generator, params: tuple[float, float, float, float]) -> float:
    mu, sigma, lo, hi = params
    return float(np.clip(rng.normal(mu, sigma), lo, hi))


def _vectores_estructurados(
    rng: np.random.Generator, n: int, dim: int, correlation: float
) -> NDArray[np.float32]:
    """Vectores unitarios con una fracción de varianza en un subespacio compartido.

    Con `correlation=0` son gaussianos i.i.d. y por tanto casi ortogonales. A
    medida que crece, comparten cada vez más estructura y el coseno medio entre
    pares sube — que es lo que se observa en embeddings perceptuales reales.
    """
    z = rng.standard_normal((n, dim)).astype(np.float32)
    if correlation > 0.0:
        rango = max(1, int(dim * 0.15))
        base = unit_rows(rng.standard_normal((rango, dim)).astype(np.float32))
        coef = rng.standard_normal((n, rango)).astype(np.float32)
        peso = correlation / max(1.0 - correlation, 1e-3)
        z = z + np.float32(peso) * (coef @ base)
    return unit_rows(z)


def _pesos_de_visita(n: int, skew: float) -> NDArray[np.float64]:
    """Reparto de visitas entre prototipos. `skew=0` es uniforme; mayor es ley de potencias."""
    if skew <= 0.0:
        return np.full(n, 1.0 / n)
    w = (np.arange(1, n + 1, dtype=np.float64)) ** (-skew)
    return w / w.sum()


def clustered_stream(
    n_prototypes: int,
    capacity: int,
    *,
    n_common: int = 300,
    n_rare: int = 20,
    dim: int = 32,
    noise: float = 0.05,
    correlation: float = 0.0,
    density_skew: float = 0.0,
    drift: float = 0.0,
    seed: int = 0,
) -> Stream:
    """Flujo dominado por experiencia recurrente con eventos raros intercalados.

    Es la tarea T1 del paper: la memoria tiene que conservar lo raro-pero-importante
    contra el desplazamiento continuo de lo común. Los eventos raros llevan error
    de predicción alto; los comunes, bajo.
    """
    if n_prototypes < 1:
        raise ValueError(f"n_prototypes debe ser >= 1, se recibió {n_prototypes}")

    rng = np.random.default_rng(seed)
    centros = _vectores_estructurados(rng, n_prototypes, dim, correlation)
    pesos = _pesos_de_visita(n_prototypes, density_skew)

    # Dirección de deriva: un desplazamiento lento y común a todos los centros.
    direccion = _vectores_estructurados(rng, 1, dim, 0.0)[0]

    items: list[StreamItem] = []
    for t in range(n_common):
        p = int(rng.choice(n_prototypes, p=pesos))
        centro = centros[p]
        if drift > 0.0:
            centro = centro + np.float32(drift * t / max(n_common - 1, 1)) * direccion
        k = centro + rng.standard_normal(dim).astype(np.float32) * noise
        items.append(
            StreamItem(
                key=(k / np.linalg.norm(k)).astype(np.float32),
                value=p,
                pred_error=_pe(rng, PE_COMUN),
                is_rare=False,
            )
        )

    raros: list[StreamItem] = []
    if n_rare > 0:
        claves_raras = _vectores_estructurados(rng, n_rare, dim, correlation)
        posiciones = sorted(rng.choice(n_common + n_rare, size=n_rare, replace=False))
        for i, pos in enumerate(posiciones):
            it = StreamItem(
                key=claves_raras[i],
                value=f"rare{i}",
                pred_error=_pe(rng, PE_RARO),
                is_rare=True,
            )
            items.insert(int(pos), it)
            raros.append(it)

    return Stream(
        items=items,
        spec=StreamSpec(
            n_prototypes=n_prototypes,
            capacity=capacity,
            dim=dim,
            source="synthetic",
        ),
        rare_items=raros,
    )


def block_stream(
    *,
    n_blocks: int = 4,
    per_block: int = 8,
    repeats: int = 3,
    dim: int = 32,
    capacity: int = 20,
    pe_first: float = 0.9,
    pe_rest: float = 0.1,
    correlation: float = 0.0,
    seed: int = 0,
) -> Stream:
    """Bloques secuenciales de aprendizaje, para medir interferencia catastrófica.

    Cada ítem se repite `repeats` veces dentro de su bloque, simulando el
    aprendizaje de un contexto antes de pasar al siguiente. Se consulta el bloque
    1 al final.

    El error de predicción diferenciado (`pe_first` vs `pe_rest`) es deliberado.
    Con un valor uniforme para todos los bloques, el desalojo por mínima fuerza
    degenera —todas las trazas tienen la misma fuerza— y toda arquitectura
    puntúa cero. Eso es informativo, pero no es la comparación que la tarea
    pretende hacer: sin una señal que distinga qué experiencias importan, ninguna
    arquitectura puede resistir la interferencia.
    """
    rng = np.random.default_rng(seed)
    items: list[StreamItem] = []
    for b in range(n_blocks):
        claves = _vectores_estructurados(rng, per_block, dim, correlation)
        pe = pe_first if b == 0 else pe_rest
        for j, k in enumerate(claves):
            for _ in range(repeats):
                items.append(StreamItem(key=k, value=f"b{b}_{j}", pred_error=pe))

    primero = [it for it in items if it.value.startswith("b0_")]
    return Stream(
        items=items,
        spec=StreamSpec(
            n_prototypes=n_blocks * per_block,
            capacity=capacity,
            dim=dim,
            source="synthetic_blocks",
        ),
        rare_items=primero,
    )
