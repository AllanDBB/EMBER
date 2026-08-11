"""Búsqueda de arquitecturas sobre el espacio de mecanismos de memoria bioinspirada."""

from ember.nas.engine import Evaluator, SearchRecord, SearchResults, run_search
from ember.nas.space import (
    AXES,
    BIOLOGICAL_ROOT,
    FIFO_GENOTYPE,
    SEARCH_SPACE,
    axis_options,
    enumerate_space,
    siblings,
    space_size,
)
from ember.nas.stats import (
    axis_liveness,
    bootstrap_ci,
    conditional_effect,
    eta_squared,
    main_effects_table,
    partial_eta_squared,
    sums_of_squares,
    variance_share,
)

__all__ = [
    "AXES",
    "BIOLOGICAL_ROOT",
    "FIFO_GENOTYPE",
    "SEARCH_SPACE",
    "Evaluator",
    "SearchRecord",
    "SearchResults",
    "axis_liveness",
    "axis_options",
    "bootstrap_ci",
    "conditional_effect",
    "enumerate_space",
    "eta_squared",
    "main_effects_table",
    "partial_eta_squared",
    "run_search",
    "sums_of_squares",
    "siblings",
    "space_size",
    "variance_share",
]
