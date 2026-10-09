"""Familias de flujos para poner a prueba la robustez de la ley del umbral.

La ley se mostró sobre un único generador estacionario (`clustered_stream` con
sus valores por defecto): prototipos equiprobables, centros fijos, dispersión
intra-prototipo baja y una cantidad fija de eventos raros. Un revisor objeta,
con razón, que eso no alcanza para llamarla "ley". Este módulo define las
familias que la estresan una perilla a la vez:

- **Frecuencia desigual** (`zipf`): las visitas siguen una ley de Zipf con el
  exponente dado. Con exponente alto, buena parte de los `K` prototipos casi no
  aparece, y la "masa" de prototipos que la memoria tiene que alojar es menor
  que `K`.
- **Deriva por prototipo** (`drift_deg`): cada centro rota, a lo largo del
  flujo, un ángulo total dado hacia una dirección propia e independiente. Es
  distinta de la deriva de `clustered_stream`, que empuja a todos los centros
  en una **misma** dirección y por eso los vuelve cada vez más parecidos entre
  sí (con deriva grande, prototipos distintos terminarían fusionándose). Acá
  la deriva fragmenta cada prototipo sin acercarlo a los demás.
- **Dispersión intra-prototipo** (`noise`): el coseno esperado entre dos
  visitas al mismo prototipo es ≈ 1 / (1 + dim·σ²). Con `dim=32` cruza el
  umbral de fusión 0.85 cerca de σ ≈ 0.074, así que barrer σ alrededor de ese
  valor produce consolidación parcial.
- **Similitud entre prototipos** (`correlation`): la perilla homónima de
  `clustered_stream`.
- **Prevalencia de raros** (`rare_prevalence`): fracción del flujo que son
  eventos raros. En el diseño de `exp02` la cantidad de raros es fija
  (`max(5, C // 2)`), de modo que su prevalencia cambia con `K`; acá se fija la
  prevalencia.
- **Recurrencia y largo** (`visits`): visitas promedio por prototipo. Con el
  diseño de `exp02` el largo del flujo es `visits · K`, así que recurrencia y
  largo son la misma perilla al barrer `r`.

Con todas las perillas en su valor por defecto, `family_stream` delega en
`clustered_stream` con exactamente los mismos argumentos que `exp02`, y
reproduce su flujo bit a bit. Solo depende de numpy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from ember.core.types import unit_rows
from ember.data.streams import Stream, StreamItem, StreamSpec
from ember.data.synthetic import (
    PE_COMUN,
    PE_RARO,
    _pe,
    _pesos_de_visita,
    _vectores_estructurados,
    clustered_stream,
)

DIM = 32
VISITAS_POR_DEFECTO = 20
"""Las mismas visitas por prototipo que `exp02`."""


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """Una familia de flujos: el generador con una o más perillas movidas.

    `rare_prevalence=None` conserva el diseño de `exp02` (una cantidad fija de
    raros, `max(5, C // 2)`). `n_rare` fija la cantidad a mano y tiene prioridad
    sobre ambos.
    """

    name: str = "standard"
    zipf: float = 0.0
    drift_deg: float = 0.0
    noise: float = 0.05
    correlation: float = 0.0
    rare_prevalence: float | None = None
    visits: int = VISITAS_POR_DEFECTO
    n_rare: int | None = None

    def n_common(self, n_prototypes: int) -> int:
        """Largo de la parte recurrente: crece con `K` para fijar la recurrencia."""
        return self.visits * n_prototypes

    def rare_count(self, n_common: int, capacity: int) -> int:
        """Cuántos eventos raros lleva un flujo con `n_common` experiencias comunes."""
        if self.n_rare is not None:
            return self.n_rare
        if self.rare_prevalence is None:
            return max(5, capacity // 2)
        p = self.rare_prevalence
        return max(1, int(round(p * n_common / (1.0 - p))))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _rotar(
    centros: NDArray[np.float32], direcciones: NDArray[np.float32], angulo: float
) -> NDArray[np.float32]:
    """Rota cada centro `angulo` radianes hacia su dirección ortogonal propia."""
    return (np.cos(angulo) * centros + np.sin(angulo) * direcciones).astype(np.float32)


def _direcciones_ortogonales(
    rng: np.random.Generator, centros: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Una dirección unitaria por centro, ortogonal a él e independiente de las demás."""
    z = rng.standard_normal(centros.shape).astype(np.float32)
    z = z - (z * centros).sum(axis=1, keepdims=True) * centros
    return unit_rows(z)


def drifting_stream(
    n_prototypes: int,
    capacity: int,
    *,
    n_common: int,
    n_rare: int,
    drift_deg: float,
    dim: int = DIM,
    noise: float = 0.05,
    correlation: float = 0.0,
    zipf: float = 0.0,
    seed: int = 0,
) -> Stream:
    """Flujo de prototipos que derivan, cada uno hacia su propia dirección.

    En el instante `t` el centro del prototipo `p` es
    `cos(φ_t)·c_p + sin(φ_t)·u_p` con `φ_t = drift · t / (n_common - 1)` y `u_p`
    ortogonal a `c_p`. Como el umbral de fusión compara contra la clave guardada
    al crear la traza —que no se actualiza al fusionar—, una deriva mayor que el
    margen del umbral parte cada prototipo en varias trazas.
    """
    if n_prototypes < 1:
        raise ValueError(f"n_prototypes debe ser >= 1, se recibió {n_prototypes}")

    rng = np.random.default_rng(seed)
    centros = _vectores_estructurados(rng, n_prototypes, dim, correlation)
    direcciones = _direcciones_ortogonales(rng, centros)
    pesos = _pesos_de_visita(n_prototypes, zipf)
    angulo_total = np.deg2rad(drift_deg)

    items: list[StreamItem] = []
    for t in range(n_common):
        p = int(rng.choice(n_prototypes, p=pesos))
        phi = angulo_total * t / max(n_common - 1, 1)
        centro = _rotar(centros[p : p + 1], direcciones[p : p + 1], phi)[0]
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
            source="synthetic_drift",
        ),
        rare_items=raros,
    )


def family_stream(
    family: FamilySpec,
    n_prototypes: int,
    capacity: int,
    *,
    seed: int = 0,
    dim: int = DIM,
    n_common: int | None = None,
) -> Stream:
    """Un flujo de la familia dada, con `K` prototipos y capacidad declarada `C`.

    Sin deriva delega en `clustered_stream` (la frecuencia de Zipf es su
    `density_skew`), de modo que la familia estándar es exactamente el flujo de
    `exp02`. `n_common` permite fijar el largo a mano, como hace `exp01`.
    """
    n_c = family.n_common(n_prototypes) if n_common is None else n_common
    n_r = family.rare_count(n_c, capacity)
    if family.drift_deg > 0.0:
        return drifting_stream(
            n_prototypes,
            capacity,
            n_common=n_c,
            n_rare=n_r,
            drift_deg=family.drift_deg,
            dim=dim,
            noise=family.noise,
            correlation=family.correlation,
            zipf=family.zipf,
            seed=seed,
        )
    return clustered_stream(
        n_prototypes=n_prototypes,
        capacity=capacity,
        n_common=n_c,
        n_rare=n_r,
        dim=dim,
        noise=family.noise,
        correlation=family.correlation,
        density_skew=family.zipf,
        seed=seed,
    )


# ─────────────────────────────────────────── medidas de presión de capacidad


def visit_counts(stream: Stream) -> NDArray[np.int64]:
    """Visitas por prototipo en la parte recurrente del flujo (solo los vistos)."""
    valores = [it.value for it in stream if not it.is_rare]
    if not valores:
        return np.zeros(0, dtype=np.int64)
    _, cuentas = np.unique(np.asarray(valores), return_counts=True)
    return cuentas.astype(np.int64)


def hill_number(stream: Stream, order: float = 1.0) -> float:
    """Número efectivo de prototipos: número de Hill de la distribución de visitas.

    Con `order=1` es `exp(H)`, la exponencial de la entropía de Shannon de las
    frecuencias empíricas. Con visitas uniformes coincide con `K`; con Zipf
    fuerte cae muy por debajo, porque los prototipos casi nunca visitados pesan
    poco. `order=0` es la cantidad de prototipos distintos que aparecieron.
    """
    cuentas = visit_counts(stream)
    if cuentas.size == 0:
        return 0.0
    p = cuentas / cuentas.sum()
    if order == 0:
        return float(cuentas.size)
    if order == 1:
        return float(np.exp(-(p * np.log(p)).sum()))
    return float((p**order).sum() ** (1.0 / (1.0 - order)))
