"""Señales de saliencia y definiciones de importancia sobre una `RolloutTrace`.

`exp06` definía lo importante como "recompensa inmediata" y probaba como señal
el error de predicción de recompensa: las dos cosas miden casi lo mismo, y el
revisor lo señala con razón. Este módulo separa las dos preguntas para poder
cruzarlas:

- **Importancia** (qué debería conservar la memoria): una etiqueta booleana
  por ítem. Se calcula con información que la memoria no ve —recompensas
  futuras, eventos del entorno, estado privilegiado—, porque es la vara con la
  que se mide, no una entrada.
- **Señal de saliencia** (lo que la memoria recibe como `pred_error`): un
  número en [0, 1] por ítem. Todas son causales (usan solo el pasado), salvo
  `retorno`, que es retrospectiva y está marcada como tal.

Cada combinación importancia × señal produce un `Stream` distinto sobre la misma
experiencia, y la matriz resultante dice cuándo la saliencia transfiere y
cuándo no.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ember.envs.minigrid import OneStepPredictor, RewardPredictionError, RolloutTrace

# ══════════════════════════════════════════════════════════════ señales

SIGNALS = ("perceptual", "reward_pe", "count_novelty", "td_error", "retorno", "azar")
"""Señales de saliencia, en el orden en que se reportan.

- `perceptual`: error del modelo lineal de transición (`OneStepPredictor`), la
  línea de base de `exp06`.
- `reward_pe`: error de predicción de recompensa inmediata
  (`RewardPredictionError`), la señal que funcionó en `exp06`.
- `count_novelty`: novedad por conteo, `1/sqrt(N(o))` sobre la observación
  parcial. Motivación intrínseca clásica, sin recompensa.
- `td_error`: |error TD(0)| de una función de valor lineal aprendida en línea.
  Propaga la recompensa hacia atrás con la experiencia, así que puede marcar
  pasos que *preceden* a la recompensa.
- `retorno`: retorno descontado dentro del episodio, **no causal**: exige
  conocer el futuro (una memoria podría implementarlo solo demorando la
  escritura hasta el fin del episodio). Es la cota de una señal de valor
  perfecta, no un mecanismo candidato.
- `azar`: uniforme en [0, 1], independiente de todo. Control nulo.
"""

CAUSAL_SIGNALS = ("perceptual", "reward_pe", "count_novelty", "td_error", "azar")


class TDErrorPredictor:
    """Error TD(0) de una función de valor lineal ajustada en línea.

    `V(o) = w · o` sobre la clave codificada. Con `delta = r + gamma * V(o')
    - V(o)` (sin arranque en estados terminales), la señal es `|delta|`
    normalizado por una escala adaptativa, igual que en
    `RewardPredictionError`. La diferencia con esa señal es que TD aprende a
    anticipar: después de algunas recompensas, `V` sube en los estados previos
    y el error aparece antes del paso recompensado.
    """

    def __init__(self, obs_dim: int, lr: float = 0.1, gamma: float = 0.9) -> None:
        self.w = np.zeros(obs_dim, dtype=np.float32)
        self.lr = lr
        self.gamma = gamma
        self._escala = 1.0

    def surprise(self, obs: NDArray, reward: float, next_obs: NDArray, terminal: bool) -> float:
        o = obs.astype(np.float32)
        v = float(self.w @ o)
        v_sig = 0.0 if terminal else float(self.w @ next_obs.astype(np.float32))
        delta = float(reward) + self.gamma * v_sig - v
        self.w += np.float32(self.lr * delta) * o
        self._escala = 0.99 * self._escala + 0.01 * abs(delta)
        return float(np.clip(abs(delta) / (2.0 * self._escala + 1e-8), 0.0, 1.0))


def discounted_return(trace: RolloutTrace, gamma: float = 0.9) -> NDArray[np.float64]:
    """Retorno descontado desde cada ítem hasta el fin de su episodio (retrospectivo)."""
    g = np.zeros(len(trace), dtype=np.float64)
    acumulado = 0.0
    for t in range(len(trace) - 1, -1, -1):
        fin = trace.terminated[t] or trace.truncated[t]
        if fin or t == len(trace) - 1:
            acumulado = 0.0
        acumulado = float(trace.rewards[t]) + gamma * acumulado
        g[t] = acumulado
    return np.clip(g, 0.0, 1.0)


def compute_signal(trace: RolloutTrace, name: str, *, seed: int = 0) -> NDArray[np.float64]:
    """Una señal de saliencia por ítem, en [0, 1].

    Se devuelve en float64 porque los predictores devuelven `float` de Python y
    `exp06` pasaba ese valor tal cual a la memoria: redondear a float32 acá
    movería el último bit y dejaría de reproducir `exp06` exactamente.
    """
    n = len(trace)
    dim = int(trace.keys.shape[1])
    out = np.zeros(n, dtype=np.float64)
    if name == "perceptual":
        p = OneStepPredictor(dim, trace.n_actions)
        for t in range(n):
            out[t] = p.surprise(trace.keys[t], int(trace.actions[t]), trace.next_keys[t])
    elif name == "reward_pe":
        r = RewardPredictionError(dim, trace.n_actions)
        for t in range(n):
            out[t] = r.surprise(trace.keys[t], int(trace.actions[t]), float(trace.rewards[t]))
    elif name == "count_novelty":
        cuentas: dict[int, int] = {}
        for t in range(n):
            c = cuentas.get(int(trace.obs_id[t]), 0) + 1
            cuentas[int(trace.obs_id[t])] = c
            out[t] = 1.0 / np.sqrt(c)
    elif name == "td_error":
        td = TDErrorPredictor(dim)
        for t in range(n):
            out[t] = td.surprise(
                trace.keys[t],
                float(trace.rewards[t]),
                trace.next_keys[t],
                bool(trace.terminated[t]),
            )
    elif name == "retorno":
        out = discounted_return(trace)
    elif name == "azar":
        out = np.random.default_rng(seed + 93_000).uniform(0.0, 1.0, size=n)
    else:
        raise ValueError(f"señal desconocida: {name!r}; opciones: {SIGNALS}")
    return out


# ══════════════════════════════════════════════════════════ importancia

IMPORTANCE = ("recompensa", "previa_k", "evento_clave", "novedad_estado", "sorpresa_transicion")
"""Definiciones de importancia, en el orden en que se reportan.

- `recompensa`: el paso recibe recompensa positiva. La definición de `exp06`.
- `previa_k`: los `k` pasos que preceden a una recompensa dentro del mismo
  episodio, sin el paso recompensado. Importancia **retrasada**: lo que llevó a
  la recompensa, no la recompensa.
- `evento_clave`: recoger una llave o abrir una puerta cerrada con llave.
  Subobjetivos que solo más adelante permiten la recompensa. Solo existe en
  entornos con llave (DoorKey, KeyCorridor).
- `novedad_estado`: primera visita del agente a una celda en todo el rollout,
  con la posición privilegiada. Importancia **intrínseca**, sin recompensa.
- `sorpresa_transicion`: un modelo tabular exacto de la transición
  observación × acción → observación se equivoca: ese par ya se había visto y
  el resultado es uno nunca visto para él. Intrínseca, sin recompensa, y
  definida con un modelo distinto (exacto) del que produce `perceptual`
  (lineal y aproximado).
"""

K_PREVIOS = 5


def importance_mask(trace: RolloutTrace, name: str, *, k: int = K_PREVIOS) -> NDArray[np.bool_]:
    """Etiqueta booleana de importancia por ítem."""
    n = len(trace)
    if name == "recompensa":
        return trace.rewards > 0
    if name == "previa_k":
        m = np.zeros(n, dtype=bool)
        for t in np.flatnonzero(trace.rewards > 0):
            for j in range(max(t - k, 0), t):
                if trace.episode[j] == trace.episode[t]:
                    m[j] = True
        return m
    if name == "evento_clave":
        return trace.key_pickup | trace.door_unlocked
    if name == "novedad_estado":
        m = np.zeros(n, dtype=bool)
        vistas: set[tuple[int, int]] = set()
        for t in range(n):
            pos = (int(trace.agent_pos[t, 0]), int(trace.agent_pos[t, 1]))
            if pos not in vistas:
                vistas.add(pos)
                m[t] = True
        return m
    if name == "sorpresa_transicion":
        m = np.zeros(n, dtype=bool)
        resultados: dict[tuple[int, int], set[int]] = {}
        for t in range(n):
            par = (int(trace.obs_id[t]), int(trace.actions[t]))
            destino = int(trace.next_obs_id[t])
            vistos = resultados.get(par)
            if vistos is None:
                resultados[par] = {destino}
            elif destino not in vistos:
                m[t] = True
                vistos.add(destino)
        return m
    raise ValueError(f"importancia desconocida: {name!r}; opciones: {IMPORTANCE}")


def separation_auc(signal: NDArray, mask: NDArray) -> float | None:
    """AUC de Mann-Whitney: P(señal de un ítem importante > la de uno común).

    0.5 es una señal que no separa; 1.0, una que separa perfectamente. None si
    alguna de las dos poblaciones está vacía. Los empates cuentan 1/2.
    """
    pos = np.asarray(signal[mask], dtype=np.float64)
    neg = np.asarray(signal[~mask], dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return None
    todos = np.concatenate([pos, neg])
    orden = todos.argsort(kind="mergesort")
    rangos = np.empty(len(todos), dtype=np.float64)
    ordenados = todos[orden]
    # Rango medio para empates.
    i = 0
    while i < len(ordenados):
        j = i
        while j + 1 < len(ordenados) and ordenados[j + 1] == ordenados[i]:
            j += 1
        rangos[orden[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = rangos[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))
