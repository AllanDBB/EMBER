"""Variantes de una tarea de MiniGrid para un flujo de aprendizaje de por vida.

`exp06` mide si la memoria *retiene* lo recompensado. Este módulo existe para
medir otra cosa: si lo retenido *sirve para actuar*. Hace falta un entorno
donde varias tareas emparentadas se alternen y reaparezcan, para que olvidar
una tarea vieja cueste desempeño cuando vuelve.

`GoalVariantEnv` es una habitación de MiniGrid con la meta en una de cuatro
esquinas. Cada esquina es una tarea distinta; la política óptima de una es
mala para las otras, así que conservar experiencia de la tarea equivocada no
ayuda. El agente arranca en una celda y orientación al azar en cada episodio,
de modo que resolver una tarea exige recordar valor sobre muchos estados, no
una sola trayectoria.

La clave de un estado es una codificación centrada en objetos de la
observación *completa* (lo que da `FullyObsWrapper`): un one-hot de la celda y
orientación del agente más un one-hot de la celda de la meta, proyectados con
una proyección aleatoria fija. Se omite el fondo (paredes y piso vacío) porque
es idéntico en todos los estados y dominaría la similitud coseno: con él, dos
estados cualesquiera de la misma habitación tendrían similitud > 0.95 y el
umbral de fusión se dispararía entre estados distintos. Es una decisión de
representación, igual para todas las memorias que se comparan.

Este módulo importa `minigrid` al construir el entorno, nunca al importarse:
`ember.envs` no es parte del núcleo que va al robot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.core.types import unit

GRID_SIZE = 9
"""Habitación de 7x7 celdas libres más el borde de paredes."""

GOAL_CORNERS: tuple[tuple[int, int], ...] = ((1, 1), (7, 1), (1, 7), (7, 7))
"""Una tarea por esquina. El índice de la esquina es el identificador de tarea."""

N_ACTIONS = 3
"""Girar a la izquierda, girar a la derecha, avanzar. Las demás no se usan."""


def make_goal_env(task: int, *, max_steps: int = 60, size: int = GRID_SIZE) -> Any:
    """Construye la variante `task` de la habitación. Importa minigrid acá adentro."""
    from minigrid.core.grid import Grid
    from minigrid.core.mission import MissionSpace
    from minigrid.core.world_object import Goal
    from minigrid.minigrid_env import MiniGridEnv

    goal = GOAL_CORNERS[task]

    class GoalVariantEnv(MiniGridEnv):
        """Habitación vacía con la meta en una esquina fija y arranque al azar."""

        def __init__(self) -> None:
            super().__init__(
                mission_space=MissionSpace(mission_func=lambda: "get to the green goal square"),
                grid_size=size,
                see_through_walls=True,
                max_steps=max_steps,
            )

        def _gen_grid(self, width: int, height: int) -> None:
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.put_obj(Goal(), *goal)
            self.place_agent()
            self.mission = "get to the green goal square"

    return GoalVariantEnv()


class StateEncoder:
    """Codificación centrada en objetos de la observación completa, proyectada a `dim`.

    El vector disperso tiene tres bloques: la celda de la meta (una por celda),
    la celda y orientación del agente (cuatro por celda) y la acción (una por
    acción, con peso `action_weight`). La acción va en la clave porque la
    memoria guarda pares estado-acción: sin ella, la consolidación por fusión
    mezclaría acciones distintas del mismo estado en una sola traza.
    """

    def __init__(
        self,
        *,
        size: int = GRID_SIZE,
        n_actions: int = N_ACTIONS,
        dim: int = 64,
        seed: int = 0,
        action_weight: float = 1.0,
    ) -> None:
        self.size = size
        self.n_actions = n_actions
        self.dim = dim
        self.action_weight = action_weight
        n_cells = size * size
        self._n_raw = n_cells + 4 * n_cells + n_actions
        rng = np.random.default_rng(seed + 13_000)
        self.P = (rng.standard_normal((dim, self._n_raw)) / np.sqrt(dim)).astype(np.float32)

    def state_part(self, agent_pos: tuple[int, int], agent_dir: int, goal: tuple[int, int]):
        """Proyección del bloque de estado (sin acción), para reutilizar entre acciones."""
        n_cells = self.size * self.size
        x = np.zeros(self._n_raw, dtype=np.float32)
        x[goal[1] * self.size + goal[0]] = 1.0
        celda = agent_pos[1] * self.size + agent_pos[0]
        x[n_cells + 4 * celda + int(agent_dir)] = 1.0
        return self.P @ x

    def keys(self, state_proj: NDArray) -> NDArray[np.float32]:
        """Claves unitarias de los `n_actions` pares (estado, acción), una por fila."""
        base = 5 * self.size * self.size
        filas = [
            unit(state_proj + self.action_weight * self.P[:, base + a])
            for a in range(self.n_actions)
        ]
        return np.stack(filas).astype(np.float32)
