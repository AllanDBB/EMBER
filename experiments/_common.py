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
