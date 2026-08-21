"""Fase 1: el gate de reconstrucción.

Toda memoria asociativa útil tiene que hacer una cosa antes que cualquier otra:
recuperar un episodio completo desde una clave degradada. Un robot nunca
consulta con la clave exacta que guardó — la consulta viene con ruido de sensor,
oclusión parcial, o desde un contexto desplazado.

Estas cuatro subtareas miden esa capacidad **sin presión de capacidad**: la
memoria siempre tiene lugar para todo lo que se le escribe, así que nunca desaloja
y lo único que se mide es la calidad de la recuperación. Una arquitectura que no
llega al umbral no pasa a la fase 2, porque medirle retención no significaría nada.

Una consecuencia de que no haya desalojo: arquitecturas que difieren solo en su
política de olvido puntúan idéntico aquí. Eso es correcto y es informativo — su
diferencia aparece recién bajo presión, en la fase 2.
"""

from __future__ import annotations

import numpy as np

from ember.core.types import unit
from ember.tasks.protocol import GateResult, MemoryFactory, TaskResult

DIM = 32
RECON_THRESHOLD = 0.50


def _patrones(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def r1_pattern_completion(
    factory: MemoryFactory,
    *,
    seed: int = 0,
    dim: int = DIM,
    n_patterns: int = 10,
    cue_fracs: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> TaskResult:
    """Recuperar el patrón completo desde una clave con dimensiones enmascaradas.

    El puntaje pondera acierto por similitud: devolver el ítem correcto con
    similitud baja vale menos que devolverlo con similitud alta.
    """
    rng = np.random.default_rng(seed)
    mem = factory(n_patterns, seed)
    patrones = _patrones(rng, n_patterns, dim)

    for i, p in enumerate(patrones):
        mem.write(p, i, pred_error=0.8)

    puntajes = []
    for i, p in enumerate(patrones):
        for frac in cue_fracs:
            cue = p.copy()
            n_mask = int(dim * (1.0 - frac))
            cue[rng.choice(dim, n_mask, replace=False)] = 0.0
            if np.linalg.norm(cue) < 1e-6:
                continue
            r = mem.read(unit(cue))
            puntajes.append(r.similarity if r.value == i else 0.0)

    return TaskResult(
        name="pattern_completion",
        score=float(np.mean(puntajes)) if puntajes else 0.0,
        detail={"n_patterns": n_patterns, "cue_fracs": list(cue_fracs)},
    )


def r2_noise_robustness(
    factory: MemoryFactory,
    *,
    seed: int = 0,
    dim: int = DIM,
    n_patterns: int = 15,
    noise_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
) -> TaskResult:
    """Recuperar la identidad correcta desde una clave con ruido gaussiano aditivo."""
    rng = np.random.default_rng(seed + 100)
    mem = factory(n_patterns, seed)
    patrones = _patrones(rng, n_patterns, dim)

    for i, p in enumerate(patrones):
        mem.write(p, i, pred_error=0.5)

    aciertos, total = 0, 0
    por_nivel: dict[str, float] = {}
    for sigma in noise_levels:
        a = 0
        for i, p in enumerate(patrones):
            ruidosa = unit(p + rng.standard_normal(dim).astype(np.float32) * sigma)
            a += mem.read(ruidosa).value == i
        por_nivel[str(sigma)] = a / n_patterns
        aciertos += a
        total += n_patterns

    return TaskResult(
        name="noise_robustness",
        score=aciertos / total if total else 0.0,
        detail={"por_nivel": por_nivel},
    )


def r3_ab_interference(
    factory: MemoryFactory, *, seed: int = 0, dim: int = DIM, n_pairs: int = 10
) -> TaskResult:
    """Después de escribir B inmediatamente tras A, ¿se puede leer A?

    Mide interferencia en su forma más simple. Una arquitectura que falla aquí
    sobrescribe conocimiento previo con cada escritura nueva.
    """
    rng = np.random.default_rng(seed + 200)
    puntajes = []
    for j in range(n_pairs):
        mem = factory(2, seed + j)
        a = _patrones(rng, 1, dim)[0]
        b = _patrones(rng, 1, dim)[0]
        mem.write(a, "A", pred_error=0.9)
        mem.write(b, "B", pred_error=0.9)
        r = mem.read(a)
        puntajes.append(r.similarity if r.value == "A" else 0.0)

    return TaskResult(
        name="ab_interference",
        score=float(np.mean(puntajes)),
        detail={"n_pairs": n_pairs},
    )


def r4_capacity_profile(
    factory: MemoryFactory,
    *,
    seed: int = 0,
    dim: int = DIM,
    noise: float = 0.15,
    loads: tuple[int, ...] = (2, 5, 10, 15, 20, 30),
) -> TaskResult:
    """Curva de precisión contra carga. Muestra si degrada suave o de golpe.

    La capacidad siempre iguala a la carga, así que la curva aísla el efecto de
    la interferencia entre patrones guardados, no el del desalojo.
    """
    rng = np.random.default_rng(seed + 300)
    curva: dict[int, float] = {}
    for n in loads:
        mem = factory(n, seed)
        patrones = _patrones(rng, n, dim)
        for i, p in enumerate(patrones):
            mem.write(p, i, pred_error=0.5)
        aciertos = 0
        for i, p in enumerate(patrones):
            ruidosa = unit(p + rng.standard_normal(dim).astype(np.float32) * noise)
            aciertos += mem.read(ruidosa).value == i
        curva[n] = aciertos / n

    return TaskResult(
        name="capacity_profile",
        score=float(np.mean(list(curva.values()))),
        detail={"curva": {str(k): v for k, v in curva.items()}},
    )


RECONSTRUCTION_TASKS = (
    r1_pattern_completion,
    r2_noise_robustness,
    r3_ab_interference,
    r4_capacity_profile,
)


def reconstruction_gate(
    factory: MemoryFactory,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    dim: int = DIM,
    threshold: float = RECON_THRESHOLD,
    name: str = "",
) -> GateResult:
    """Corre las cuatro subtareas y decide si la arquitectura pasa a la fase 2."""
    por_tarea: dict[str, float] = {}
    por_tarea_std: dict[str, float] = {}
    detalle: dict[str, object] = {}
    for tarea in RECONSTRUCTION_TASKS:
        resultados = [tarea(factory, seed=s, dim=dim) for s in seeds]
        puntajes = [r.score for r in resultados]
        por_tarea[resultados[0].name] = float(np.mean(puntajes))
        por_tarea_std[resultados[0].name] = float(np.std(puntajes))
        detalle[resultados[0].name] = resultados[0].detail

    return GateResult(
        name=name or "reconstruction",
        per_task=por_tarea,
        per_task_std=por_tarea_std,
        detail={"seeds": list(seeds), **detalle},
        threshold=threshold,
    )
