"""Adaptador de MiniGrid a `Stream`: experiencia secuencial bajo observabilidad parcial.

Hasta acá las memorias se evalúan sobre flujos construidos a mano. Un robot no
recibe flujos: recibe la consecuencia de sus propias acciones, con observación
parcial y estructura temporal que ningún generador reproduce. MiniGrid es el
paso intermedio barato entre lo sintético y el Robotino.

De dónde sale el error de predicción
------------------------------------
En los flujos sintéticos, `pred_error` viene puesto por el generador: es una
etiqueta que declara qué experiencias son sorpresivas. Acá no hay etiqueta, y
poner una a mano sería hacer trampa con la variable que decide todo el
resultado. Se calcula, y hay dos maneras honestas de hacerlo, cada una una
lectura distinta de "la cuenta dopaminérgica":

- `OneStepPredictor` (`surprise_source="prediction"`, el default): un modelo de
  transición de un paso predice la próxima observación, y el error de esa
  predicción es sorpresa *perceptual*.
- `RewardPredictionError` (`surprise_source="reward"`): predice la recompensa
  esperada en vez de la observación, y el error de esa predicción es sorpresa
  de *recompensa* — el relato de Schultz sobre lo que la dopamina codifica en
  realidad. En un entorno de recompensa rara, esta señal puede separar lo
  recompensado de lo meramente novedoso donde la sorpresa perceptual no puede,
  porque perceptualmente un paso exitoso no tiene por qué distinguirse de
  cualquier otro paso nuevo.

Ambos modelos son deliberadamente simples (un mapa lineal ajustado en línea).
No hace falta que prediga bien: hace falta que sea *sorprendido* por lo que es
genuinamente nuevo (o genuinamente recompensado), y un modelo simple lo es más
honestamente que uno entrenado hasta el sobreajuste.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ember.core.types import unit
from ember.data.streams import Stream, StreamItem, StreamSpec


class OneStepPredictor:
    """Modelo lineal de transición ajustado en línea. Produce la señal de sorpresa.

    Predice `o_{t+1}` desde `[o_t, one_hot(a_t)]` con un mapa lineal actualizado
    por descenso en línea. El error normalizado de esa predicción es el
    `pred_error` del episodio.
    """

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 0.05) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.W = np.zeros((obs_dim, obs_dim + n_actions), dtype=np.float32)
        self._escala = 1.0

    def _entrada(self, obs: NDArray, action: int) -> NDArray[np.float32]:
        a = np.zeros(self.n_actions, dtype=np.float32)
        a[action] = 1.0
        return np.concatenate([obs.astype(np.float32), a])

    def surprise(self, obs: NDArray, action: int, next_obs: NDArray) -> float:
        """Error de predicción normalizado a [0, 1], y actualiza el modelo."""
        x = self._entrada(obs, action)
        pred = self.W @ x
        err = next_obs.astype(np.float32) - pred

        # Actualización en línea antes de normalizar, para que la sorpresa sea
        # la del modelo *previo* a ver el resultado.
        magnitud = float(np.linalg.norm(err))
        self.W += self.lr * np.outer(err, x)

        # Escala adaptativa: la sorpresa es relativa al error típico reciente,
        # no a una constante arbitraria.
        self._escala = 0.99 * self._escala + 0.01 * magnitud
        return float(np.clip(magnitud / (2.0 * self._escala + 1e-8), 0.0, 1.0))


class RewardPredictionError:
    """Error de predicción de recompensa. La cuenta dopaminérgica correcta.

    `OneStepPredictor` mide sorpresa perceptual: qué tan mal predice el modelo
    la próxima observación. La dopamina, en el relato de Schultz, codifica
    específicamente error de predicción de *recompensa* — no sorpresa
    perceptual genérica —, que es la lectura neurobiológica que motiva la
    compuerta de fuerza en el resto del programa. Predice
    la recompensa esperada desde `[obs, one_hot(action)]` con un mapa lineal
    ajustado en línea, y el error absoluto normalizado es la señal de fuerza.

    Sobre un entorno de recompensa rara, la recompensa esperada se queda cerca
    de 0 salvo en el paso que efectivamente la entrega, así que esta señal
    puede separar lo recompensado de lo meramente novedoso donde
    `OneStepPredictor` no puede: perceptualmente, un paso exitoso no tiene por
    qué verse distinto de cualquier otro paso nuevo.
    """

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 0.1) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.w = np.zeros(obs_dim + n_actions, dtype=np.float32)
        self._escala = 1.0

    def _entrada(self, obs: NDArray, action: int) -> NDArray[np.float32]:
        a = np.zeros(self.n_actions, dtype=np.float32)
        a[action] = 1.0
        return np.concatenate([obs.astype(np.float32), a])

    def surprise(self, obs: NDArray, action: int, reward: float) -> float:
        """Error de predicción de recompensa normalizado a [0, 1], y actualiza el modelo."""
        x = self._entrada(obs, action)
        pred = float(self.w @ x)
        err = float(reward) - pred

        self.w += self.lr * err * x
        self._escala = 0.99 * self._escala + 0.01 * abs(err)
        return float(np.clip(abs(err) / (2.0 * self._escala + 1e-8), 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class RolloutTrace:
    """Registro completo de un rollout, antes de decidir qué es sorpresa y qué importa.

    `rollout` fija en el momento de correr qué señal de sorpresa se calcula y
    qué cuenta como raro (la recompensa inmediata). Eso obliga a correr el
    entorno una vez por combinación. La traza guarda todo lo necesario para
    calcular *después*, sobre exactamente la misma experiencia, cualquier señal
    de saliencia y cualquier definición de importancia (ver
    `ember.envs.salience`): las celdas de la matriz importancia × señal difieren
    solo en eso, nunca en lo que el agente vivió.

    El ítem `t` es la transición desde la observación `keys[t]` con la acción
    `actions[t]` hacia `next_keys[t]`, con recompensa `rewards[t]`; su clave en
    memoria es `keys[t]`, igual que en `rollout`.
    """

    env_id: str
    seed: int
    n_actions: int
    keys: NDArray[np.float32]
    next_keys: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float64]
    """float64 a propósito: la recompensa se guarda tal cual la entrega el entorno, y
    redondearla a float32 movería en el último bit la señal de `exp06`."""
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    episode: NDArray[np.int64]
    obs_id: NDArray[np.int64]
    """Identificador de la observación parcial (entero por orden de aparición en el rollout)."""
    next_obs_id: NDArray[np.int64]
    agent_pos: NDArray[np.int64]
    """Celda del agente en el ítem `t` (información privilegiada, solo para etiquetar)."""
    key_pickup: NDArray[np.bool_]
    """El paso `t` recogió una llave."""
    door_unlocked: NDArray[np.bool_]
    """El paso `t` abrió una puerta que estaba cerrada con llave."""

    def __len__(self) -> int:
        return len(self.actions)

    def to_stream(
        self, pred_error: NDArray, is_rare: NDArray, *, capacity: int, source: str = ""
    ) -> Stream:
        """Arma el `Stream` con una señal de sorpresa y una definición de importancia."""
        items = [
            StreamItem(
                key=self.keys[t],
                value=t,
                pred_error=float(pred_error[t]),
                is_rare=bool(is_rare[t]),
            )
            for t in range(len(self))
        ]
        return Stream(
            items=items,
            spec=StreamSpec(
                n_prototypes=0,
                capacity=capacity,
                dim=int(self.keys.shape[1]),
                source=source or f"minigrid:{self.env_id}",
                n_prototypes_ci=(0, 0),
            ),
            rare_items=[it for it in items if it.is_rare],
        )


class MiniGridStreamAdapter:
    """Convierte rollouts de MiniGrid en un `Stream` evaluable.

    La observación parcial de MiniGrid (una rejilla egocéntrica codificada) se
    proyecta a un vector unitario de dimensión `dim` con una proyección aleatoria
    fija — que es la construcción de Johnson-Lindenstrauss, y también lo que hace
    la SDM con sus hard locations.
    """

    def __init__(
        self,
        env_id: str = "MiniGrid-MemoryS13-v0",
        *,
        dim: int = 32,
        capacity: int = 20,
        seed: int = 0,
        surprise_source: str = "prediction",
    ) -> None:
        if surprise_source not in ("prediction", "reward"):
            raise ValueError(
                f"surprise_source debe ser 'prediction' o 'reward', no {surprise_source!r}"
            )
        self.env_id = env_id
        self.dim = dim
        self.capacity = capacity
        self.seed = seed
        self.surprise_source = surprise_source
        self.rng = np.random.default_rng(seed)
        self._proyeccion: NDArray[np.float32] | None = None

    def _make_env(self) -> Any:
        import gymnasium as gym
        import minigrid  # noqa: F401  (registra los entornos en gymnasium)

        return gym.make(self.env_id)

    def _codificar(self, obs: dict[str, Any]) -> NDArray[np.float32]:
        """Aplana la observación parcial y la proyecta a la esfera de dimensión `dim`."""
        plano = np.asarray(obs["image"], dtype=np.float32).ravel()
        if self._proyeccion is None:
            # Proyección fija para toda la vida del adaptador: dos observaciones
            # iguales tienen que dar la misma clave.
            self._proyeccion = self.rng.standard_normal((self.dim, plano.size)).astype(
                np.float32
            ) / np.sqrt(self.dim)
        return unit(self._proyeccion @ plano)

    def rollout(self, n_steps: int, policy: Any = None) -> Stream:
        """Ejecuta `n_steps` pasos y devuelve el flujo de experiencia resultante."""
        env = self._make_env()
        obs, _ = env.reset(seed=self.seed)

        n_acciones = int(env.action_space.n)
        predictor = (
            OneStepPredictor(self.dim, n_acciones)
            if self.surprise_source == "prediction"
            else RewardPredictionError(self.dim, n_acciones)
        )
        clave = self._codificar(obs)

        items: list[StreamItem] = []
        for t in range(n_steps):
            accion = (
                int(self.rng.integers(env.action_space.n)) if policy is None else int(policy(obs))
            )
            siguiente, recompensa, terminado, truncado, _ = env.step(accion)
            clave_siguiente = self._codificar(siguiente)

            sorpresa = (
                predictor.surprise(clave, accion, clave_siguiente)
                if self.surprise_source == "prediction"
                else predictor.surprise(clave, accion, recompensa)
            )
            items.append(
                StreamItem(
                    key=clave,
                    value=t,
                    pred_error=sorpresa,
                    # Un episodio con recompensa es raro por construcción del
                    # entorno, y es lo que la memoria tiene que conservar.
                    is_rare=bool(recompensa > 0),
                )
            )

            if terminado or truncado:
                obs, _ = env.reset()
                clave = self._codificar(obs)
            else:
                obs, clave = siguiente, clave_siguiente

        env.close()
        return Stream(
            items=items,
            spec=StreamSpec(
                # Sobre un entorno el número de prototipos no se conoce: hay que
                # estimarlo con `ember.data.prototypes`, igual que con embeddings.
                n_prototypes=0,
                capacity=self.capacity,
                dim=self.dim,
                source=f"minigrid:{self.env_id}",
                n_prototypes_ci=(0, 0),
            ),
            rare_items=[it for it in items if it.is_rare],
        )

    def rollout_trace(self, n_steps: int, policy: Any = None) -> RolloutTrace:
        """Como `rollout`, pero devuelve la traza completa en vez de un `Stream`.

        Consume el generador del adaptador exactamente igual que `rollout`
        (proyección primero, después una acción por paso si `policy` es None),
        así que con la misma semilla la experiencia es idéntica y la señal
        `prediction`/`reward` calculada sobre la traza reproduce `rollout`.

        `policy` puede ser None (uniforme con el generador del adaptador), un
        invocable `policy(obs)` como en `exp06`, o un objeto con
        `act(obs, env)` y, opcionalmente, `observe(obs, a, r, obs2, term, trunc)`.
        """
        env = self._make_env()
        obs, _ = env.reset(seed=self.seed)
        u = env.unwrapped
        n_acciones = int(env.action_space.n)

        ids: dict[bytes, int] = {}

        def id_de(o: dict[str, Any]) -> int:
            b = np.asarray(o["image"], dtype=np.uint8).tobytes()
            return ids.setdefault(b, len(ids))

        clave = self._codificar(obs)
        oid = id_de(obs)
        episodio = 0

        keys, next_keys, acciones, recompensas = [], [], [], []
        terminados, truncados, episodios, oids, next_oids = [], [], [], [], []
        posiciones, llaves, puertas = [], [], []

        actuar = getattr(policy, "act", None)
        observar = getattr(policy, "observe", None)
        for _ in range(n_steps):
            if policy is None:
                accion = int(self.rng.integers(n_acciones))
            elif actuar is not None:
                accion = int(actuar(obs, env))
            else:
                accion = int(policy(obs))

            pos = (int(u.agent_pos[0]), int(u.agent_pos[1]))
            fx, fy = (int(v) for v in u.front_pos)
            dentro = 0 <= fx < u.width and 0 <= fy < u.height
            enfrente = u.grid.get(fx, fy) if dentro else None
            llevaba = u.carrying
            con_llave = enfrente is not None and enfrente.type == "door" and enfrente.is_locked

            siguiente, recompensa, terminado, truncado, _ = env.step(accion)
            if observar is not None:
                observar(obs, accion, float(recompensa), siguiente, terminado, truncado)

            clave_siguiente = self._codificar(siguiente)
            keys.append(clave)
            next_keys.append(clave_siguiente)
            acciones.append(accion)
            recompensas.append(float(recompensa))
            terminados.append(bool(terminado))
            truncados.append(bool(truncado))
            episodios.append(episodio)
            oids.append(oid)
            next_oids.append(id_de(siguiente))
            posiciones.append(pos)
            llaves.append(llevaba is None and u.carrying is not None and u.carrying.type == "key")
            puertas.append(bool(con_llave and enfrente.is_open))

            if terminado or truncado:
                obs, _ = env.reset()
                clave = self._codificar(obs)
                oid = id_de(obs)
                episodio += 1
            else:
                obs, clave, oid = siguiente, clave_siguiente, next_oids[-1]

        env.close()
        return RolloutTrace(
            env_id=self.env_id,
            seed=self.seed,
            n_actions=n_acciones,
            keys=np.stack(keys).astype(np.float32),
            next_keys=np.stack(next_keys).astype(np.float32),
            actions=np.asarray(acciones, dtype=np.int64),
            rewards=np.asarray(recompensas, dtype=np.float64),
            terminated=np.asarray(terminados, dtype=bool),
            truncated=np.asarray(truncados, dtype=bool),
            episode=np.asarray(episodios, dtype=np.int64),
            obs_id=np.asarray(oids, dtype=np.int64),
            next_obs_id=np.asarray(next_oids, dtype=np.int64),
            agent_pos=np.asarray(posiciones, dtype=np.int64),
            key_pickup=np.asarray(llaves, dtype=bool),
            door_unlocked=np.asarray(puertas, dtype=bool),
        )
