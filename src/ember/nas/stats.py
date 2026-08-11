"""Análisis estadístico del espacio de búsqueda.

La tesis metodológica de este módulo: **en un espacio donde los mecanismos están
compuertados entre sí, los efectos principales engañan.**

El caso de estudio es la compuerta de saliencia. Su efecto principal sobre las
576 arquitecturas es de 0.2 %, lo que invita a concluir que el mecanismo más
citado por la literatura de memoria bioinspirada casi no importa. Pero de las
cuatro políticas de desalojo, tres son completamente insensibles a la fuerza de
las trazas: en tres cuartos del espacio la señal de saliencia se calcula y se
descarta sin que nada la lea. Condicionado a la única política que sí la lee, el
mismo mecanismo multiplica por 35 la retención de eventos raros.

Publicar el 0.2 % como resultado negativo es un autogol: un revisor ve la
interacción en la tabla y concluye que el ANOVA de efectos principales era la
herramienta equivocada — que es exactamente el caso. Por eso `conditional_effect`
está aquí junto a `eta_squared`, y los experimentos reportan ambos.

Todo en numpy puro: el análisis tiene que poder correr donde corre el núcleo.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from ember.nas.engine import SearchRecord
from ember.nas.space import AXES


def _valores(records: Sequence[SearchRecord], metric: str) -> np.ndarray:
    if metric == "mean":
        return np.array([r.mean for r in records], dtype=np.float64)
    return np.array([r.scores[metric] for r in records], dtype=np.float64)


def _etiquetas(records: Sequence[SearchRecord], axis: str) -> np.ndarray:
    return np.array([r.genotype.axis(axis) for r in records])


def eta_squared(records: Sequence[SearchRecord], axis: str, metric: str = "mean") -> float:
    """Fracción de la varianza total que explica un eje por sí solo.

    Es el efecto principal. Útil como descripción, engañoso como conclusión
    cuando el eje está compuertado por otro — ver `conditional_effect`.
    """
    y = _valores(records, metric)
    ss_total = float(((y - y.mean()) ** 2).sum())
    if ss_total <= 0.0:
        return 0.0

    etiquetas = _etiquetas(records, axis)
    ss_entre = 0.0
    for opcion in np.unique(etiquetas):
        grupo = y[etiquetas == opcion]
        ss_entre += len(grupo) * (grupo.mean() - y.mean()) ** 2
    return float(ss_entre / ss_total)


def sums_of_squares(
    records: Sequence[SearchRecord],
    factors: Sequence[str] = AXES,
    metric: str = "mean",
    max_order: int = 2,
) -> tuple[dict[str, float], float, float]:
    """Descomposición factorial de la suma de cuadrados.

    Devuelve `(por_termino, ss_error, ss_total)`. `max_order=2` incluye las
    interacciones de dos factores, que es donde vive el efecto de la saliencia;
    órdenes mayores rara vez son interpretables.

    La suma de cuadrados de cada término se calcula por medias marginales: el
    efecto de una interacción es lo que queda de la media de cada celda después
    de restar la gran media y los dos efectos principales.
    """
    y = _valores(records, metric)
    gran_media = y.mean()
    ss_total = float(((y - gran_media) ** 2).sum())
    if ss_total <= 0.0:
        return (dict.fromkeys(factors, 0.0), 0.0, 0.0)

    etiquetas = {f: _etiquetas(records, f) for f in factors}

    # Efectos principales.
    efectos: dict[str, np.ndarray] = {}
    ss: dict[str, float] = {}
    for f in factors:
        ajuste = np.zeros_like(y)
        for opcion in np.unique(etiquetas[f]):
            m = etiquetas[f] == opcion
            ajuste[m] = y[m].mean() - gran_media
        efectos[f] = ajuste
        ss[f] = float((ajuste**2).sum())

    # Interacciones: media de celda menos los principales menos la gran media.
    if max_order >= 2:
        for a, b in combinations(factors, 2):
            ajuste = np.zeros_like(y)
            for oa in np.unique(etiquetas[a]):
                for ob in np.unique(etiquetas[b]):
                    m = (etiquetas[a] == oa) & (etiquetas[b] == ob)
                    if not m.any():
                        continue
                    ajuste[m] = y[m].mean() - gran_media
            residuo = ajuste - efectos[a] - efectos[b]
            ss[f"{a}×{b}"] = float((residuo**2).sum())

    ss_error = max(ss_total - sum(ss.values()), 0.0)
    return (ss, ss_error, ss_total)


def partial_eta_squared(
    records: Sequence[SearchRecord],
    factors: Sequence[str] = AXES,
    metric: str = "mean",
    max_order: int = 2,
) -> dict[str, float]:
    """Eta cuadrado parcial por término: `SS_efecto / (SS_efecto + SS_error)`.

    Mide el efecto de un término descontando la varianza que explican los demás.
    Advertencia: cuando el modelo no deja varianza residual —lo que pasa en
    espacios de diseño puramente determinísticos como este, si los factores
    incluidos explican todo— el eta cuadrado parcial satura en 1.0 para todo
    término no nulo y deja de discriminar. Para comparar la magnitud de los
    términos entre sí, usar `variance_share`.
    """
    ss, ss_error, _ = sums_of_squares(records, factors, metric, max_order)
    return {k: v / (v + ss_error) if (v + ss_error) > 0 else 0.0 for k, v in ss.items()}


def variance_share(
    records: Sequence[SearchRecord],
    factors: Sequence[str] = AXES,
    metric: str = "mean",
    max_order: int = 2,
) -> dict[str, float]:
    """Fracción de la varianza total que explica cada término: `SS_efecto / SS_total`.

    Es la generalización del eta cuadrado a un modelo con interacciones, y la
    magnitud correcta para tabular: los términos son comparables entre sí y
    suman como mucho 1. Es lo que hace visible que la interacción entre la
    compuerta de saliencia y la política de desalojo pesa mucho más que el
    efecto principal de la compuerta.
    """
    ss, _, ss_total = sums_of_squares(records, factors, metric, max_order)
    if ss_total <= 0.0:
        return dict.fromkeys(ss, 0.0)
    return {k: v / ss_total for k, v in ss.items()}


def conditional_effect(
    records: Sequence[SearchRecord],
    axis: str,
    given: dict[str, str],
    metric: str = "mean",
) -> dict[str, float]:
    """Efecto de un eje restringido al subespacio donde el mecanismo puede actuar.

    `given` filtra por etiquetas de otros ejes. Es lo que produce la tabla del
    efecto condicional de la saliencia: restringido a `evict=min_strength` y
    `write=append`, que son las condiciones bajo las cuales la señal de fuerza
    llega a influir en algo.
    """
    seleccion = [r for r in records if all(r.genotype.axis(k) == v for k, v in given.items())]
    if not seleccion:
        raise ValueError(f"ningún genotipo cumple la condición {given}")

    y = _valores(seleccion, metric)
    etiquetas = _etiquetas(seleccion, axis)
    return {str(o): float(y[etiquetas == o].mean()) for o in np.unique(etiquetas)}


def axis_liveness(records: Sequence[SearchRecord], axis: str, metric: str = "mean") -> float:
    """Máxima diferencia de puntaje entre genotipos que solo difieren en `axis`.

    Cero exacto significa que el eje es inobservable: cambiarlo no cambia nada,
    en ningún punto del espacio. Es la auditoría que descubrió que el espacio del
    piloto tenía 96 arquitecturas funcionalmente distintas y no 576, convertida
    en una función que puede correr en CI.

    Un eje puede morir por dos causas distintas, y ambas cuentan como muerto:
    porque no está implementado, o porque el diseño de la tarea lo vuelve
    inobservable. La segunda es más difícil de ver por inspección del código.
    """
    y = _valores(records, metric)
    otros = [a for a in AXES if a != axis]

    grupos: dict[tuple[str, ...], list[float]] = {}
    for r, valor in zip(records, y, strict=True):
        clave = tuple(r.genotype.axis(a) for a in otros)
        grupos.setdefault(clave, []).append(float(valor))

    return max(
        (max(v) - min(v) for v in grupos.values() if len(v) > 1),
        default=0.0,
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Intervalo de confianza percentil por bootstrap sobre la media."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]))

    rng = np.random.default_rng(seed)
    medias = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo = float(np.percentile(medias, 100 * alpha / 2))
    hi = float(np.percentile(medias, 100 * (1 - alpha / 2)))
    return (lo, hi)


def main_effects_table(
    records: Sequence[SearchRecord], metric: str = "mean"
) -> list[dict[str, object]]:
    """Tabla de efectos principales con su observabilidad, ordenada por eta cuadrado.

    Reporta `liveness` junto a `eta2` a propósito: un eje con eta cuadrado 0.000
    y liveness 0.000 no es un resultado sobre el mecanismo, es un mecanismo que
    no está siendo medido.
    """
    filas = []
    for eje in AXES:
        filas.append(
            {
                "axis": eje,
                "eta2": eta_squared(records, eje, metric),
                "liveness": axis_liveness(records, eje, metric),
                "por_opcion": {
                    k: float(v)
                    for k, v in sorted(
                        conditional_effect(records, eje, given={}, metric=metric).items()
                    )
                },
            }
        )
    return sorted(filas, key=lambda f: -f["eta2"])
