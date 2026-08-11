"""Motor de búsqueda exhaustiva sobre el espacio de arquitecturas.

El punto delicado de este módulo es cómo se reporta la posición de un genotipo
dentro del ranking.

El piloto reportaba que el FIFO de e-MDB quedaba "#415 de 576". Ese número salía
del orden de desempate del `sort`: 162 arquitecturas empataban exactamente con
él y ninguna puntuaba estrictamente peor, así que #415 era el mejor puesto
asignable y el rango defendible era #415–#576. Reportar el mejor extremo de un
empate masivo como si fuera la posición es engañoso, y engaña en la dirección
que perjudica el argumento: lo que el FIFO realmente ocupa es el mínimo exacto
del espacio, que es un resultado más fuerte.

Por eso `rank_of` devuelve una tupla `(optimista, pesimista)` y nunca un entero.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from ember.core.genotype import Genotype
from ember.nas.space import AXES, enumerate_space

Evaluator = Callable[[Genotype], dict[str, float]]
"""Evalúa un genotipo y devuelve un puntaje por tarea."""


@dataclass(frozen=True, slots=True)
class SearchRecord:
    """Un genotipo evaluado."""

    genotype: Genotype
    scores: dict[str, float]
    mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "genotype": self.genotype.as_dict(),
            "label": self.genotype.label(),
            "scores": dict(self.scores),
            "mean": self.mean,
        }


def _media(scores: dict[str, float]) -> float:
    return float(np.mean(list(scores.values()))) if scores else 0.0


@dataclass(frozen=True, slots=True)
class SearchResults:
    """Los resultados de una búsqueda, ordenados de mejor a peor."""

    records: list[SearchRecord]

    def __len__(self) -> int:
        return len(self.records)

    def best(self, n: int = 8) -> list[SearchRecord]:
        return self.records[:n]

    def score_of(self, genotype: Genotype) -> float:
        for r in self.records:
            if r.genotype == genotype:
                return r.mean
        raise KeyError(f"genotipo no evaluado: {genotype.label()}")

    def rank_of(self, genotype: Genotype, tol: float = 1e-9) -> tuple[int, int]:
        """Rango del genotipo como intervalo `(optimista, pesimista)`, base 1.

        El optimista cuenta cuántos puntúan estrictamente mejor; el pesimista
        suma además todos los que empatan. Cuando el intervalo es ancho, el
        ranking puntual no significa nada y hay que reportar el intervalo.
        """
        objetivo = self.score_of(genotype)
        mejores = sum(1 for r in self.records if r.mean > objetivo + tol)
        empates = sum(1 for r in self.records if abs(r.mean - objetivo) <= tol)
        return (mejores + 1, mejores + empates)

    def is_at_floor(self, genotype: Genotype, tol: float = 1e-9) -> bool:
        """Si nada en el espacio puntúa estrictamente peor que este genotipo."""
        objetivo = self.score_of(genotype)
        return not any(r.mean < objetivo - tol for r in self.records)

    def per_axis_means(self, axis: str, metric: str = "mean") -> dict[str, float]:
        """Puntaje medio por opción de un eje. La base del análisis de efectos."""
        acumulado: dict[str, list[float]] = {}
        for r in self.records:
            etiqueta = r.genotype.axis(axis)
            valor = r.mean if metric == "mean" else r.scores[metric]
            acumulado.setdefault(etiqueta, []).append(valor)
        return {k: float(np.mean(v)) for k, v in acumulado.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_genotypes": len(self.records),
            "axes": list(AXES),
            "records": [r.to_dict() for r in self.records],
        }


def _evaluar_uno(args: tuple[Evaluator, Genotype]) -> SearchRecord:
    evaluator, g = args
    scores = evaluator(g)
    return SearchRecord(genotype=g, scores=scores, mean=_media(scores))


def run_search(
    evaluator: Evaluator,
    *,
    genotypes: Iterable[Genotype] | None = None,
    n_jobs: int = -1,
    progress: bool = True,
) -> SearchResults:
    """Evalúa un conjunto de genotipos, en paralelo si se pide.

    El orden del resultado es determinista e independiente del paralelismo: se
    desempata por la etiqueta del genotipo, no por el orden de finalización.
    """
    espacio: Sequence[Genotype] = (
        list(genotypes) if genotypes is not None else list(enumerate_space())
    )

    if n_jobs == 1 or len(espacio) < 8:
        registros = []
        for i, g in enumerate(espacio, start=1):
            registros.append(_evaluar_uno((evaluator, g)))
            if progress and i % 100 == 0:
                print(f"  evaluadas {i}/{len(espacio)} arquitecturas")
    else:
        trabajadores = os.cpu_count() or 1 if n_jobs < 0 else n_jobs
        with ProcessPoolExecutor(max_workers=trabajadores) as pool:
            registros = list(pool.map(_evaluar_uno, [(evaluator, g) for g in espacio], chunksize=4))

    registros.sort(key=lambda r: (-r.mean, r.genotype.label()))
    return SearchResults(records=registros)
