"""Políticas de comportamiento para los rollouts de MiniGrid.

`exp06` usaba dos políticas sin objetivo (uniforme y sesgada a avanzar), y eso
deja abierta la objeción de que la retención dependa de un comportamiento
degenerado. Acá hay dos más, con comportamiento dirigido:

- `NoisyPlanner`: un planificador heurístico con acceso privilegiado al mapa
  completo (BFS sobre la rejilla) y ruido epsilon. No aprende nada; representa
  a un agente competente que llega a la recompensa a menudo.
- `TabularQAgent`: Q-learning tabular sobre la observación parcial codificada,
  aprendiendo en línea durante el mismo rollout. Su comportamiento no es
  estacionario: la distribución de lo que visita cambia mientras aprende, que es
  justamente la condición de un robot que aprende de por vida.

Interfaz: `act(obs, env) -> int` y, opcionalmente,
`observe(obs, action, reward, next_obs, terminated, truncated)`. El adaptador
también acepta un invocable `policy(obs)`, que es la interfaz de `exp06`.

Toda aleatoriedad sale de la semilla explícita que recibe cada política.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

# Acciones de MiniGrid (`minigrid.core.actions.Actions`).
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE = range(7)

# Vector de avance por dirección: 0 = +x, 1 = +y, 2 = -x, 3 = -y.
_DIR_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))


class ForwardBiasedPolicy:
    """Sesgada a avanzar, generalizada a todo el espacio de acciones.

    La `ForwardBiasedPolicy` de `exp06` solo elige entre girar y avanzar, así
    que en un entorno con llave nunca podría recoger la llave ni abrir la
    puerta: la recompensa sería imposible por construcción. Esta versión reserva
    una fracción `p_interactuar` repartida entre las acciones de interacción.
    """

    def __init__(
        self,
        seed: int,
        n_actions: int = 7,
        p_avanzar: float = 0.6,
        p_interactuar: float = 0.1,
    ) -> None:
        self.rng = np.random.default_rng(seed + 90_000)
        pesos = np.zeros(n_actions, dtype=np.float64)
        pesos[FORWARD] = p_avanzar
        n_interaccion = max(n_actions - 3, 0)
        p_giro = (1.0 - p_avanzar - (p_interactuar if n_interaccion else 0.0)) / 2
        pesos[LEFT] = pesos[RIGHT] = p_giro
        if n_interaccion:
            pesos[3:] = p_interactuar / n_interaccion
        self.pesos = pesos / pesos.sum()

    def act(self, obs: Any, env: Any) -> int:
        return int(self.rng.choice(len(self.pesos), p=self.pesos))


class NoisyPlanner:
    """Planificador BFS sobre el mapa completo, con una acción al azar con prob. `epsilon`.

    Decide el objetivo con reglas fijas, en este orden:

    1. `success_pos` si el entorno lo define (Memory): pisar esa celda.
    2. `env.obj` si existe y no se lo lleva (KeyCorridor): recogerlo.
    3. La celda `goal` (DoorKey, FourRooms): pisarla.

    Si el objetivo no es alcanzable porque lo tapa una puerta cerrada con llave,
    el objetivo pasa a ser la llave (o soltar lo que se lleva, si no es la
    llave). Una puerta cerrada sin llave, o cerrada con llave llevando la llave
    de su color, se atraviesa: al llegar enfrente, el plan la abre (`toggle`).
    Si nada es alcanzable, actúa al azar.
    """

    def __init__(self, seed: int, n_actions: int = 7, epsilon: float = 0.2) -> None:
        self.rng = np.random.default_rng(seed + 91_000)
        self.n_actions = n_actions
        self.epsilon = epsilon
        # La meta no se mueve dentro de un episodio: se busca una vez por rejilla.
        self._rejilla: Any = None
        self._meta: tuple[int, int] | None = None

    # ------------------------------------------------------------ utilidades
    @staticmethod
    def _transitable(u: Any, x: int, y: int) -> bool:
        celda = u.grid.get(x, y)
        if celda is None:
            return True
        if celda.type == "door":
            if not celda.is_locked:
                return True
            llevando = u.carrying
            return llevando is not None and llevando.type == "key" and llevando.color == celda.color
        return celda.type in ("goal", "floor")

    def _bfs(self, u: Any, destinos: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        """Camino más corto (sin la celda inicial) desde el agente hasta algún destino."""
        inicio = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        if inicio in destinos:
            return []
        previo: dict[tuple[int, int], tuple[int, int] | None] = {inicio: None}
        cola = deque([inicio])
        while cola:
            actual = cola.popleft()
            for dx, dy in _DIR_VEC:
                sig = (actual[0] + dx, actual[1] + dy)
                if sig in previo or not (0 <= sig[0] < u.width and 0 <= sig[1] < u.height):
                    continue
                if not self._transitable(u, *sig):
                    continue
                previo[sig] = actual
                if sig in destinos:
                    camino = [sig]
                    while previo[camino[-1]] != inicio:
                        camino.append(previo[camino[-1]])  # type: ignore[arg-type]
                    return camino[::-1]
                cola.append(sig)
        return None

    @staticmethod
    def _vecinas(pos: tuple[int, int]) -> set[tuple[int, int]]:
        return {(pos[0] + dx, pos[1] + dy) for dx, dy in _DIR_VEC}

    @staticmethod
    def _buscar(u: Any, tipo: str) -> tuple[int, int] | None:
        for x in range(u.width):
            for y in range(u.height):
                c = u.grid.get(x, y)
                if c is not None and c.type == tipo:
                    return (x, y)
        return None

    def _girar_hacia(self, u: Any, celda: tuple[int, int]) -> int:
        dx, dy = celda[0] - int(u.agent_pos[0]), celda[1] - int(u.agent_pos[1])
        deseada = _DIR_VEC.index((dx, dy))
        return RIGHT if (deseada - int(u.agent_dir)) % 4 == 1 else LEFT

    def _mirando(self, u: Any, celda: tuple[int, int]) -> bool:
        fx, fy = u.front_pos
        return (int(fx), int(fy)) == celda

    def _avanzar_por(self, u: Any, camino: list[tuple[int, int]]) -> int:
        sig = camino[0]
        if not self._mirando(u, sig):
            return self._girar_hacia(u, sig)
        celda = u.grid.get(*sig)
        if celda is not None and celda.type == "door" and not celda.is_open:
            return TOGGLE
        return FORWARD

    def _ir_a_recoger(self, u: Any, pos: tuple[int, int]) -> int | None:
        """Acción para recoger el objeto en `pos`, o None si no es alcanzable."""
        agente = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        if pos in self._vecinas(agente):
            # Primero soltar lo que lleva: si girara hacia el objetivo antes, la
            # búsqueda de una celda libre lo haría girar de vuelta y oscilaría.
            if u.carrying is not None:
                return self._soltar(u)
            if not self._mirando(u, pos):
                return self._girar_hacia(u, pos)
            return PICKUP
        camino = self._bfs(u, {v for v in self._vecinas(pos) if self._transitable(u, *v)})
        return None if camino is None else self._avanzar_por(u, camino)

    def _libre(self, u: Any, c: tuple[int, int]) -> bool:
        return 0 <= c[0] < u.width and 0 <= c[1] < u.height and u.grid.get(*c) is None

    def _soltar(self, u: Any) -> int:
        """Suelta lo que lleva en una celda vecina libre, yendo a buscar una si no hay."""
        agente = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        libres = [v for v in sorted(self._vecinas(agente)) if self._libre(u, v)]
        for v in libres:
            if self._mirando(u, v):
                return DROP
        if libres:
            return self._girar_hacia(u, libres[0])
        # Ninguna vecina libre (p. ej. parado en una puerta): moverse a una celda que tenga.
        destinos = {
            (x, y)
            for x in range(u.width)
            for y in range(u.height)
            if (x, y) != agente
            and self._transitable(u, x, y)
            and any(self._libre(u, v) for v in self._vecinas((x, y)))
        }
        camino = self._bfs(u, destinos)
        if not camino:
            return int(self.rng.integers(self.n_actions))
        return self._avanzar_por(u, camino)

    def _ir_a_pisar(self, u: Any, pos: tuple[int, int]) -> int | None:
        camino = self._bfs(u, {pos})
        if camino is None:
            return None
        if not camino:
            return FORWARD
        return self._avanzar_por(u, camino)

    # ------------------------------------------------------------------ plan
    def _plan(self, u: Any) -> int | None:
        if getattr(u, "success_pos", None) is not None:
            accion = self._ir_a_pisar(u, tuple(u.success_pos))
        elif getattr(u, "obj", None) is not None and u.carrying is not u.obj:
            pos = u.obj.cur_pos
            accion = None if pos is None else self._ir_a_recoger(u, (int(pos[0]), int(pos[1])))
        else:
            if u.grid is not self._rejilla:
                self._rejilla, self._meta = u.grid, self._buscar(u, "goal")
            meta = self._meta
            accion = None if meta is None else self._ir_a_pisar(u, meta)
        if accion is not None:
            return accion

        # El objetivo no es alcanzable: lo más probable es una puerta con llave.
        llevando = u.carrying
        if llevando is not None and llevando.type == "key":
            return None
        if llevando is not None:
            return self._soltar(u)
        llave = self._buscar(u, "key")
        return None if llave is None else self._ir_a_recoger(u, llave)

    def act(self, obs: Any, env: Any) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        accion = self._plan(env.unwrapped)
        if accion is None:
            return int(self.rng.integers(self.n_actions))
        return int(accion)


class TabularQAgent:
    """Q-learning tabular sobre la observación parcial, aprendiendo en línea.

    El estado es la rejilla egocéntrica codificada tal cual (sus bytes), que es
    lo que el agente ve; no usa información privilegiada. La inicialización
    optimista (`q_init`) hace que explore sistemáticamente lo no probado antes
    de explotar, que es lo que le permite encontrar alguna recompensa en un
    presupuesto de pocos miles de pasos.
    """

    def __init__(
        self,
        seed: int,
        n_actions: int = 7,
        epsilon: float = 0.1,
        alpha: float = 0.3,
        gamma: float = 0.95,
        q_init: float = 1.0,
    ) -> None:
        self.rng = np.random.default_rng(seed + 92_000)
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_init = q_init
        self.q: dict[bytes, np.ndarray] = {}

    def _estado(self, obs: Any) -> bytes:
        return np.asarray(obs["image"], dtype=np.uint8).tobytes()

    def _fila(self, s: bytes) -> np.ndarray:
        fila = self.q.get(s)
        if fila is None:
            fila = np.full(self.n_actions, self.q_init, dtype=np.float64)
            self.q[s] = fila
        return fila

    def act(self, obs: Any, env: Any) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        fila = self._fila(self._estado(obs))
        mejores = np.flatnonzero(fila == fila.max())
        return int(mejores[int(self.rng.integers(len(mejores)))])

    def observe(
        self,
        obs: Any,
        action: int,
        reward: float,
        next_obs: Any,
        terminated: bool,
        truncated: bool,
    ) -> None:
        fila = self._fila(self._estado(obs))
        siguiente = 0.0 if terminated else float(self._fila(self._estado(next_obs)).max())
        objetivo = float(reward) + self.gamma * siguiente
        fila[action] += self.alpha * (objetivo - fila[action])
