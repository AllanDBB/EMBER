"""Piezas compartidas por los experimentos.

El evaluador vive acá porque tiene que ser importable a nivel de módulo: el
motor de búsqueda paraleliza con procesos, y una closure o un lambda no se
pueden serializar para mandarlos a un worker.
"""

from __future__ import annotations

from dataclasses import dataclass

from ember.core.genotype import Genotype
from ember.core.memory import PolicyMemory
from ember.data.synthetic import clustered_stream
from ember.tasks.battery import (
    t1_rare_retention,
    t2_noise_under_pressure,
    t3_sequential_interference,
)

DIM = 32
CAPACITY = 20
SEEDS = (0, 1, 2)
READ_EVERY = 5
"""Una lectura cada cinco escrituras: es lo que hace observable al eje de refuerzo."""


def make_factory(genotype: Genotype, dim: int = DIM):
    """`(capacity, seed) -> PolicyMemory` para un genotipo dado."""

    def factory(capacity: int, seed: int) -> PolicyMemory:
        return PolicyMemory(dim=dim, capacity=capacity, genotype=genotype, seed=seed)

    return factory


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Parámetros de evaluación de un genotipo. Serializable, para los workers."""

    capacity: int = CAPACITY
    n_prototypes: int = 5
    dim: int = DIM
    seeds: tuple[int, ...] = SEEDS
    read_every: int = READ_EVERY
    n_common: int = 300
    n_rare: int = 20


class GenotypeEvaluator:
    """Evalúa un genotipo sobre la batería completa.

    Es una clase y no una función parcial porque tiene que poder viajar a un
    proceso worker por pickle.
    """

    def __init__(self, config: EvalConfig | None = None) -> None:
        self.config = config or EvalConfig()

    def __call__(self, genotype: Genotype) -> dict[str, float]:
        c = self.config
        factory = make_factory(genotype, dim=c.dim)

        t1, t2, t3 = [], [], []
        for s in c.seeds:
            stream = clustered_stream(
                n_prototypes=c.n_prototypes,
                capacity=c.capacity,
                n_common=c.n_common,
                n_rare=c.n_rare,
                dim=c.dim,
                seed=s,
            )
            t1.append(t1_rare_retention(factory, stream, seed=s, read_every=c.read_every).score)
            t2.append(
                t2_noise_under_pressure(factory, seed=s, dim=c.dim, capacity=c.capacity).score
            )
            t3.append(
                t3_sequential_interference(factory, seed=s, dim=c.dim, capacity=c.capacity).score
            )

        n = len(c.seeds)
        return {
            "rare_retention": sum(t1) / n,
            "noise_under_pressure": sum(t2) / n,
            "sequential_interference": sum(t3) / n,
        }


class RareRetentionEvaluator:
    """Evalúa solo T1. Es la tarea sobre la que se define la ley del umbral."""

    def __init__(self, config: EvalConfig | None = None) -> None:
        self.config = config or EvalConfig()

    def __call__(self, genotype: Genotype) -> dict[str, float]:
        c = self.config
        factory = make_factory(genotype, dim=c.dim)
        puntajes = []
        for s in c.seeds:
            stream = clustered_stream(
                n_prototypes=c.n_prototypes,
                capacity=c.capacity,
                n_common=c.n_common,
                n_rare=c.n_rare,
                dim=c.dim,
                seed=s,
            )
            puntajes.append(
                t1_rare_retention(factory, stream, seed=s, read_every=c.read_every).score
            )
        return {"rare_retention": sum(puntajes) / len(puntajes)}


def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f} %"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson (95% por defecto) para una proporción binomial.

    Con `n` chico (muestras de decenas de eventos raros, no de miles) el
    intervalo normal puede salirse de [0, 1]; Wilson no. Ver Wilson (1927),
    "Probable inference, the law of succession, and statistical inference".
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centro = p + z**2 / (2 * n)
    margen = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return ((centro - margen) / denom, (centro + margen) / denom)
