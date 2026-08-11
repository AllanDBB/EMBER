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
resultado. Se calcula: un modelo de transición de un paso predice la próxima
observación a partir de la actual y la acción, y el error de esa predicción es
la señal de sorpresa — que es exactamente el rol que la cuenta dopaminérgica le
asigna en el cerebro.

El modelo es deliberadamente simple (un mapa lineal ajustado en línea por
mínimos cuadrados recursivos). No hace falta que prediga bien: hace falta que
sea *sorprendido* por lo que es genuinamente nuevo, y un modelo simple lo es más
honestamente que uno entrenado hasta el sobreajuste.

Este módulo está cableado pero no se corre para el envío de BIP2026: la
validación en navegación pertenece al trabajo siguiente, y el claim de la ley
del umbral no la necesita.
"""

from __future__ import annotations

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
    ) -> None:
        self.env_id = env_id
        self.dim = dim
        self.capacity = capacity
        self.seed = seed
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

        predictor = OneStepPredictor(self.dim, int(env.action_space.n))
        clave = self._codificar(obs)

        items: list[StreamItem] = []
        for t in range(n_steps):
            accion = (
                int(self.rng.integers(env.action_space.n)) if policy is None else int(policy(obs))
            )
            siguiente, recompensa, terminado, truncado, _ = env.step(accion)
            clave_siguiente = self._codificar(siguiente)

            sorpresa = predictor.surprise(clave, accion, clave_siguiente)
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
