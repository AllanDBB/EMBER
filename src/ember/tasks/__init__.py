"""Tareas de evaluación de memoria, en dos fases.

Fase 1 (`reconstruction`) es un gate: sin presión de capacidad, mide si la
arquitectura puede recuperar un episodio desde una clave degradada. Fase 2
(`battery`) mide qué conserva cuando no le alcanza el espacio.

La separación es deliberada. Sin ella, una arquitectura puede puntuar bien en
retención simplemente porque su lectura es mala y devuelve cualquier cosa, o mal
porque su lectura es mala aunque su política de olvido sea la correcta.
"""

from ember.tasks.battery import (
    BATTERY_TASKS,
    HIT_SIMILARITY,
    run_battery,
    t1_rare_retention,
    t2_noise_under_pressure,
    t3_sequential_interference,
)
from ember.tasks.protocol import GateResult, MemoryFactory, SuiteResult, TaskResult
from ember.tasks.reconstruction import (
    RECON_THRESHOLD,
    RECONSTRUCTION_TASKS,
    r1_pattern_completion,
    r2_noise_robustness,
    r3_ab_interference,
    r4_capacity_profile,
    reconstruction_gate,
)

__all__ = [
    "BATTERY_TASKS",
    "HIT_SIMILARITY",
    "RECONSTRUCTION_TASKS",
    "RECON_THRESHOLD",
    "GateResult",
    "MemoryFactory",
    "SuiteResult",
    "TaskResult",
    "r1_pattern_completion",
    "r2_noise_robustness",
    "r3_ab_interference",
    "r4_capacity_profile",
    "reconstruction_gate",
    "run_battery",
    "t1_rare_retention",
    "t2_noise_under_pressure",
    "t3_sequential_interference",
]
