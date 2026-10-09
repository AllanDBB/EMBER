"""exp13 · ¿La mejor retención se traduce en mejor aprendizaje? Control episódico en MiniGrid.

Qué pide la revisión
--------------------
El revisor 1 observa que el programa mide retención y reconstrucción, pero no
el efecto de la memoria sobre el aprendizaje ni sobre el comportamiento. Este
experimento cierra esa brecha con un escenario *tipo* e-MDB (no el e-MDB real):
un agente cuya **única** fuente de experiencia es la memoria episódica, que
enfrenta un flujo no estacionario de tareas que se alternan y reaparecen, con
presión de capacidad y recompensa escasa.

El agente
---------
Control episódico (Blundell et al. 2016, MFEC; en la línea de Pritzel et al.
2017): cada paso se guarda como la traza `(clave(estado, acción), retorno)`, y
el agente actúa estimando `Q(s, a)` desde la memoria — el máximo retorno de las
trazas que coinciden exactamente con `(s, a)` o, si no hay ninguna, el promedio
de los `k` vecinos más cercanos con la misma acción. No hay red, tabla ni
parámetro que sobreviva fuera de la memoria: si la memoria olvida una tarea, el
agente la olvida. La estimación de valor es la misma para todas las
condiciones; lo único que cambia es *qué conserva* la memoria.

La saliencia es la que el agente puede calcular en línea: el error de
predicción de retorno `|G_t - Q̂(s_t, a_t)|`, con `Q̂` la estimación que el
agente usó al decidir. Es la señal de recompensa que en `exp06` sí separa lo
recompensado de lo novedoso. La novedad la calcula el motor.

Analogía con e-MDB (y dónde se rompe)
--------------------------------------
En e-MDB el `EpisodicBuffer` alimenta el aprendizaje de modelos y utilidades
cuando un contexto reaparece. Acá la memoria alimenta directamente la
estimación de utilidad de acciones. Es un escenario simulado con cuatro tareas
de navegación, no el e-MDB real: no hay ROS2, ni modelos del mundo aprendidos,
ni objetivos abiertos.

El flujo
--------
Cuatro variantes de una habitación de MiniGrid (meta en cada esquina), en
fases de `EPISODIOS_POR_FASE` episodios, recorridas en orden A B C D durante
`CICLOS` ciclos. Cada tarea reaparece tras tres fases de otras tareas. Con
capacidad `C`, un FIFO solo conserva los últimos ~C pasos y llega a cada
reaparición sin nada de la tarea.

Condiciones (misma capacidad, mismo agente)
-------------------------------------------
- `frontera`: el genotipo de frontera de `exp01`.
- `FIFO`: `FIFO_GENOTYPE` (el `EpisodicBuffer` con búsqueda por similitud).
- `reservorio`: muestreo de reservorio (Vitter 1985), la política `Reservoir` de
  `ember.core.policies` (la misma que evalúa exp09).
- `sin_saliencia`: el mejor genotipo del NAS de `exp01` con `strength=constant`
  (fusión + decaimiento 0.98 + desalojo por mínima utilidad; es el primero en
  orden de etiqueta de un empate de 0.673).
- `sin_limite`: FIFO sin presión de capacidad. Es el techo.

El FIFO con recuperación secuencial (el `deque` real, sin búsqueda por
contenido) **no se incluye como condición aparte**: para este agente,
recorrer el `deque` secuencialmente y quedarse con las coincidencias es
exactamente la misma estimación que la búsqueda por similitud sobre el mismo
conjunto de trazas. El conjunto retenido es idéntico al del `FIFO`, y con él el
desempeño. La diferencia entre las dos es de costo, no de comportamiento.

Hipótesis y qué la refutaría
----------------------------
H1 (retención funcional): a igual capacidad, la frontera obtiene mayor retorno
que el FIFO en los primeros episodios de una tarea que reaparece. Se refuta si
el IC95 bootstrap sobre semillas de la diferencia pareada frontera − FIFO en
ese retorno incluye el 0 o es negativo.

H2 (desempeño integrado): la frontera tiene mayor área bajo la curva de
aprendizaje (retorno medio sobre todo el flujo) que el FIFO. Misma regla de
refutación.

Se reporta también contra reservorio y la configuración sin saliencia, sin
hipótesis direccional: el reservorio es un baseline fuerte conocido de
aprendizaje continuo, y puede ganarle a la frontera.

Extensión: ¿la pérdida de plasticidad es por la falta de decaimiento?
---------------------------------------------------------------------
La primera corrida sostuvo H1 y refutó H2: la frontera recuerda mejor la
tarea que vuelve, pero integra menos retorno que el FIFO. La explicación
propuesta es que sin decaimiento (`decay=1.0`) el desalojo por mínima fuerza
congela la memoria en trazas viejas de fuerza alta. Se prueba directamente con
genotipos **del mismo espacio de 576**: la frontera con `decay=0.995` y
`decay=0.98` (los únicos valores < 1 que admite `ember.nas.space`; un 0.9 no
existe en el espacio y no se agrega), todo lo demás igual. El decaimiento es,
dentro del espacio, la única forma de combinar saliencia con recencia en el
desalojo: `min_strength` sobre una fuerza que decae desaloja lo débil *y*
viejo. `evict=fifo` con saliencia no combina nada: ignora la fuerza y es el
FIFO.

H3 (plasticidad): con decaimiento, la frontera sube su AUC respecto de la
frontera sin decaimiento. Se refuta si el IC95 de la diferencia pareada
incluye el 0 o es negativo.

Predicción registrada antes de correr: con 0.98 la ventaja de saliencia (un
factor ≤ ~3 de fuerza inicial) se borra en ~55 escrituras y con 0.995 en ~220,
ambas muy por debajo de las ~1500 escrituras entre reapariciones de una tarea.
Se espera que el decaimiento recupere AUC pero pierda la ventaja en
reaparición: un intercambio, no una configuración que gane en las dos.

Pregunta abierta: ¿algún genotipo del espacio gana a la vez en reaparición y
en AUC contra el FIFO? Se responde con los IC pareados, tal cual salgan.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.memory import PolicyMemory
from ember.core.policies import (
    Append,
    BothGated,
    Constant,
    ExponentialDecay,
    Merge,
    MinStrength,
    MinUtility,
    NearestNeighbour,
    NoDecay,
    Reservoir,
)
from ember.envs.lifelong import GOAL_CORNERS, StateEncoder, make_goal_env
from ember.experiment import ExperimentRun

# ════════════════════════════════════════════════════════════════ parámetros

SEMILLAS = tuple(range(10))
CAPACIDADES = (300, 1000)
"""300 es la condición principal (bastante menos de lo que usa una sola tarea);
1000 es presión moderada (cabe ~una tarea aprendida): para ver si el efecto depende
de cuánto apriete. Se probó agregar 3000 y no entró en el presupuesto de cómputo
con la máquina compartida."""
CAPACIDAD_PRINCIPAL = 300
CICLOS = 3
EPISODIOS_POR_FASE = 30
MAX_STEPS = 60
DIM = 128
GAMMA = 0.95
EPSILON = 0.1
K_VECINOS = 5
UMBRAL_EXACTO = 0.999
"""Similitud a partir de la cual dos claves son el mismo par estado-acción."""
PRIMEROS = 5
"""Episodios al inicio de una fase sobre los que se mide la retención funcional."""
N_BOOT = 10_000
SIN_LIMITE = 10**7


FRONTIER_GENOTYPE = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=BothGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
"""El genotipo de frontera de `exp01`."""

NO_SALIENCE_GENOTYPE = Genotype(
    read=NearestNeighbour(),
    write=Merge(),
    strength=Constant(),
    decay=ExponentialDecay(0.98),
    evict=MinUtility(),
    reinforce=0.0,
)
"""El mejor genotipo de `exp01` con `strength=constant` (0.673, primero del empate)."""

RESERVOIR_GENOTYPE = FIFO_GENOTYPE.with_axis("evict", Reservoir())

CONDICIONES: dict[str, tuple[Genotype, bool]] = {
    "frontera": (FRONTIER_GENOTYPE, False),
    "FIFO": (FIFO_GENOTYPE, False),
    "reservorio": (RESERVOIR_GENOTYPE, False),
    "sin_saliencia": (NO_SALIENCE_GENOTYPE, False),
    "sin_limite": (FIFO_GENOTYPE, True),
    "frontera_decay0.995": (FRONTIER_GENOTYPE.with_axis("decay", ExponentialDecay(0.995)), False),
    "frontera_decay0.98": (FRONTIER_GENOTYPE.with_axis("decay", ExponentialDecay(0.98)), False),
}
"""nombre -> (genotipo, sin límite de capacidad)."""


def programa_de_tareas(ciclos: int = CICLOS, n_tareas: int = len(GOAL_CORNERS)) -> list[int]:
    """Orden de las fases: A B C D, repetido `ciclos` veces."""
    return [t for _ in range(ciclos) for t in range(n_tareas)]


# ═══════════════════════════════════════════════════════════════════ el agente


class EpisodicControlAgent:
    """Control episódico sobre una `PolicyMemory`. La memoria es todo lo que sabe.

    Los valores de las trazas son identificadores enteros; la acción y el
    retorno de cada identificador viven en arreglos del agente, para no
    reconstruir tuplas en cada paso. Eso no le da al agente memoria propia: un
    identificador que la memoria desalojó ya no aparece en `store.values`, y
    por lo tanto no participa de ninguna decisión.
    """

    def __init__(self, memory: PolicyMemory, *, seed: int, epsilon: float = EPSILON) -> None:
        self.memory = memory
        self.rng = np.random.default_rng(seed + 7_000)
        self.epsilon = epsilon
        self._acciones: list[int] = []
        self._retornos: list[float] = []
        self._tareas: list[int] = []
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _vista(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cache is None:
            ids = np.asarray(self.memory.store.values, dtype=np.int64)
            acc = np.asarray(self._acciones, dtype=np.int64)
            ret = np.asarray(self._retornos, dtype=np.float32)
            self._cache = (ids, acc[ids] if ids.size else ids, ret[ids] if ids.size else ret[:0])
        return self._cache

    def q_values(self, keys: np.ndarray) -> np.ndarray:
        """`Q(s, a)` para cada fila de `keys` (una por acción), leyendo solo la memoria.

        Las trazas consultadas reciben los mismos efectos de lectura que
        aplica `PolicyMemory.read` (utilidad y refuerzo), para que los ejes que
        dependen del uso vean las lecturas del agente.
        """
        store = self.memory.store
        q = np.zeros(len(keys), dtype=np.float32)
        if len(store) == 0:
            return q
        ids, acc, ret = self._vista()
        sims = keys @ store.keys.T  # claves unitarias: producto interno = coseno
        leidas = []
        for a in range(len(keys)):
            idx = np.flatnonzero(acc == a)
            if idx.size == 0:
                continue
            s = sims[a, idx]
            exactas = idx[s >= UMBRAL_EXACTO]
            if exactas.size:
                q[a] = float(ret[exactas].max())
                leidas.append(exactas)
            else:
                if s.size > K_VECINOS:
                    vecinos = idx[np.argpartition(-s, K_VECINOS)[:K_VECINOS]]
                else:
                    vecinos = idx
                q[a] = float(ret[vecinos].mean())
                leidas.append(vecinos)
        sel = np.concatenate(leidas).astype(np.intp)
        store.reinforce(sel, self.memory.genotype.reinforce)
        store.touch(sel)
        return q

    def act(self, q: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(q)))
        mejores = np.flatnonzero(q >= q.max() - 1e-9)
        return int(self.rng.choice(mejores))

    def store_episode(
        self,
        keys: list[np.ndarray],
        actions: list[int],
        q_pred: list[float],
        reward: float,
        task: int,
    ) -> None:
        """Escribe el episodio al terminar, con retorno descontado y su error de predicción."""
        T = len(actions)
        for t in range(T):
            g = float(reward) * GAMMA ** (T - 1 - t)
            pred_error = float(np.clip(abs(g - q_pred[t]), 0.0, 1.0))
            ident = len(self._acciones)
            self._acciones.append(actions[t])
            self._retornos.append(g)
            self._tareas.append(task)
            self.memory.write(keys[t], ident, pred_error=pred_error)
        self._cache = None

    def composicion(self, n_tareas: int) -> dict[str, list[int]]:
        """Cuántas trazas de cada tarea hay en memoria, y cuántas con retorno positivo."""
        ids = np.asarray(self.memory.store.values, dtype=np.int64)
        tareas = np.asarray(self._tareas, dtype=np.int64)[ids] if ids.size else ids
        ret = np.asarray(self._retornos, dtype=np.float32)[ids] if ids.size else np.zeros(0)
        return {
            "trazas": [int((tareas == k).sum()) for k in range(n_tareas)],
            "trazas_recompensadas": [
                int(((tareas == k) & (ret > 0)).sum()) for k in range(n_tareas)
            ],
        }


# ═════════════════════════════════════════════════════════════ un flujo entero


def correr_flujo(
    condicion: str,
    capacity: int,
    seed: int,
    *,
    ciclos: int = CICLOS,
    episodios_por_fase: int = EPISODIOS_POR_FASE,
    max_steps: int = MAX_STEPS,
) -> dict:
    """Corre un flujo de por vida completo y devuelve el registro por episodio y por fase."""
    genotipo, sin_limite = CONDICIONES[condicion]
    cap = SIN_LIMITE if sin_limite else capacity
    memoria = PolicyMemory(dim=DIM, capacity=cap, genotype=genotipo, seed=seed)
    agente = EpisodicControlAgent(memoria, seed=seed)
    encoder = StateEncoder(dim=DIM, seed=seed)
    n_tareas = len(GOAL_CORNERS)
    envs = [make_goal_env(k, max_steps=max_steps) for k in range(n_tareas)]
    env_rng = np.random.default_rng(seed + 31_000)

    retornos: list[float] = []
    exitos: list[int] = []
    fases: list[dict] = []
    for fase, tarea in enumerate(programa_de_tareas(ciclos, n_tareas)):
        env = envs[tarea]
        goal = GOAL_CORNERS[tarea]
        antes = agente.composicion(n_tareas)
        for _ in range(episodios_por_fase):
            env.reset(seed=int(env_rng.integers(2**31)))
            claves, acciones, q_pred = [], [], []
            recompensa, terminado, truncado = 0.0, False, False
            while not (terminado or truncado):
                keys = encoder.keys(encoder.state_part(tuple(env.agent_pos), env.agent_dir, goal))
                q = agente.q_values(keys)
                a = agente.act(q)
                claves.append(keys[a])
                acciones.append(a)
                q_pred.append(float(q[a]))
                _, recompensa, terminado, truncado, _ = env.step(a)
            agente.store_episode(claves, acciones, q_pred, float(recompensa), tarea)
            retornos.append(float(recompensa))
            exitos.append(int(terminado and recompensa > 0))
        fases.append({"fase": fase, "tarea": tarea, "composicion_al_inicio": antes})
    for env in envs:
        env.close()

    return {
        "condicion": condicion,
        "capacity": capacity,
        "seed": seed,
        "retornos": retornos,
        "exitos": exitos,
        "fases": fases,
        "n_escrituras": memoria.n_writes,
        "n_fusiones": memoria.n_merges,
    }


def _correr_trabajo(args: tuple[str, int, int, dict]) -> dict:
    condicion, capacity, seed, kw = args
    return correr_flujo(condicion, capacity, seed, **kw)


# ═════════════════════════════════════════════════════════════════ estadística


def metricas_por_semilla(
    flujo: dict, *, episodios_por_fase: int, n_tareas: int, primeros: int = PRIMEROS
) -> dict:
    """Reduce un flujo a los escalares que se comparan entre condiciones.

    - `auc`: retorno medio sobre todo el flujo (área bajo la curva normalizada).
    - `exito`: tasa de éxito sobre todo el flujo.
    - `reaparicion`: retorno medio en los primeros `primeros` episodios de las
      fases en que la tarea ya se había visto (ciclos 2 en adelante).
    - `primera_vez`: lo mismo en las fases del primer ciclo (referencia: sin
      nada que retener).
    - `ultimo_ciclo`: retorno medio del último ciclo completo.
    """
    r = np.asarray(flujo["retornos"], dtype=np.float64).reshape(-1, episodios_por_fase)
    e = np.asarray(flujo["exitos"], dtype=np.float64).reshape(-1, episodios_por_fase)
    return {
        "auc": float(r.mean()),
        "exito": float(e.mean()),
        "reaparicion": float(r[n_tareas:, :primeros].mean()),
        "exito_reaparicion": float(e[n_tareas:, :primeros].mean()),
        "primera_vez": float(r[:n_tareas, :primeros].mean()),
        "ultimo_ciclo": float(r[-n_tareas:].mean()),
        "retorno_por_fase": r.mean(axis=1).tolist(),
    }


def bootstrap_ci(x: np.ndarray, *, seed: int = 0, n_boot: int = N_BOOT) -> tuple[float, float]:
    """IC95 percentil de la media, remuestreando la unidad independiente (la semilla)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    medias = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return (float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5)))


def resumir(flujos: list[dict], *, episodios_por_fase: int, n_tareas: int) -> dict:
    """Agrega por (capacidad, condición) con IC sobre semillas, y diferencias pareadas."""
    por = {}
    for f in flujos:
        m = metricas_por_semilla(f, episodios_por_fase=episodios_por_fase, n_tareas=n_tareas)
        por.setdefault(f["capacity"], {}).setdefault(f["condicion"], {})[f["seed"]] = (m, f)

    salida: dict = {}
    escalares = ("auc", "exito", "reaparicion", "exito_reaparicion", "primera_vez", "ultimo_ciclo")
    for cap, conds in por.items():
        bloque: dict = {}
        for cond, semillas in conds.items():
            orden = sorted(semillas)
            ms = [semillas[s][0] for s in orden]
            d: dict = {"n_semillas": len(orden)}
            for k in escalares:
                v = np.array([m[k] for m in ms])
                lo, hi = bootstrap_ci(v)
                d[k] = {
                    "media": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            curva = np.array([m["retorno_por_fase"] for m in ms])
            d["curva_por_fase"] = {
                "media": curva.mean(axis=0).tolist(),
                "ci_low": [bootstrap_ci(curva[:, j])[0] for j in range(curva.shape[1])],
                "ci_high": [bootstrap_ci(curva[:, j])[1] for j in range(curva.shape[1])],
            }
            # Composición de la memoria al empezar cada reaparición: cuánto de la
            # tarea que vuelve sigue en memoria. Conecta retención con desempeño.
            reap = []
            reap_rec = []
            for s in orden:
                fl = semillas[s][1]
                for fase in fl["fases"][n_tareas:]:
                    comp = fase["composicion_al_inicio"]
                    reap.append(comp["trazas"][fase["tarea"]])
                    reap_rec.append(comp["trazas_recompensadas"][fase["tarea"]])
            d["trazas_de_la_tarea_al_reaparecer"] = float(np.mean(reap))
            d["trazas_recompensadas_de_la_tarea_al_reaparecer"] = float(np.mean(reap_rec))
            d["fraccion_fusiones"] = float(
                np.mean(
                    [
                        semillas[s][1]["n_fusiones"] / max(semillas[s][1]["n_escrituras"], 1)
                        for s in orden
                    ]
                )
            )
            bloque[cond] = d

        # Diferencias pareadas por semilla contra el FIFO (y frontera contra el resto).
        difs: dict = {}
        for a, b in (
            ("frontera", "FIFO"),
            ("frontera", "reservorio"),
            ("frontera", "sin_saliencia"),
            ("reservorio", "FIFO"),
            ("sin_limite", "frontera"),
            ("frontera_decay0.995", "FIFO"),
            ("frontera_decay0.98", "FIFO"),
            ("frontera_decay0.995", "frontera"),
            ("frontera_decay0.98", "frontera"),
        ):
            if a not in conds or b not in conds:
                continue
            comunes = sorted(set(conds[a]) & set(conds[b]))
            for k in ("auc", "reaparicion", "exito_reaparicion"):
                v = np.array([conds[a][s][0][k] - conds[b][s][0][k] for s in comunes])
                lo, hi = bootstrap_ci(v)
                difs[f"{a}_menos_{b}"] = difs.get(f"{a}_menos_{b}", {})
                difs[f"{a}_menos_{b}"][k] = {
                    "media": float(v.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "semillas_a_favor": int((v > 0).sum()),
                    "n": int(v.size),
                }
        bloque["diferencias_pareadas"] = difs
        salida[f"C{cap}"] = bloque
    return salida


def correr_todo(
    seeds: tuple[int, ...] = SEMILLAS,
    capacidades: tuple[int, ...] = CAPACIDADES,
    condiciones: tuple[str, ...] = tuple(CONDICIONES),
    *,
    n_jobs: int | None = None,
    **kw,
) -> list[dict]:
    """Corre todas las combinaciones en paralelo. `sin_limite` no depende de C: se corre una vez."""
    trabajos = []
    for cap in capacidades:
        for cond in condiciones:
            if CONDICIONES[cond][1] and cap != capacidades[0]:
                continue
            trabajos += [(cond, cap, s, kw) for s in seeds]
    n_jobs = n_jobs or os.cpu_count() or 1
    if n_jobs == 1:
        flujos = [_correr_trabajo(t) for t in trabajos]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            flujos = list(ex.map(_correr_trabajo, trabajos))
    # El techo sin límite es el mismo para toda capacidad: se replica en cada bloque.
    extra = []
    for f in flujos:
        if CONDICIONES[f["condicion"]][1]:
            for cap in capacidades[1:]:
                extra.append({**f, "capacity": cap})
    return flujos + extra


# ═══════════════════════════════════════════════════════════════════════ main


def main() -> int:
    t0 = time.time()
    with ExperimentRun("exp13_downstream_learning") as run:
        run.set_seeds(SEMILLAS)
        run.note(
            "Control episódico (MFEC) cuya única fuente de experiencia es la memoria, sobre "
            f"4 variantes de una habitación MiniGrid {GOAL_CORNERS} (meta en cada esquina), "
            f"fases de {EPISODIOS_POR_FASE} episodios en orden A B C D x {CICLOS} ciclos, "
            f"max_steps={MAX_STEPS}, gamma={GAMMA}, epsilon={EPSILON}. Saliencia: error de "
            "predicción de retorno |G - Q̂| calculado en línea por el agente."
        )
        run.note(
            "Escenario tipo e-MDB, no el e-MDB real. El reservorio es `Reservoir` de "
            "ember.core.policies, la misma política que evalúa exp09."
        )
        flujos = correr_todo()
        n_tareas = len(GOAL_CORNERS)
        resumen = resumir(flujos, episodios_por_fase=EPISODIOS_POR_FASE, n_tareas=n_tareas)
        run.record(
            "config",
            {
                "semillas": list(SEMILLAS),
                "capacidades": list(CAPACIDADES),
                "capacidad_principal": CAPACIDAD_PRINCIPAL,
                "ciclos": CICLOS,
                "episodios_por_fase": EPISODIOS_POR_FASE,
                "max_steps": MAX_STEPS,
                "dim": DIM,
                "gamma": GAMMA,
                "epsilon": EPSILON,
                "k_vecinos": K_VECINOS,
                "primeros": PRIMEROS,
                "programa": programa_de_tareas(),
                "genotipos": {c: str(g) for c, (g, _) in CONDICIONES.items()},
            },
        )
        run.record("resumen", resumen)
        run.record(
            "retorno_por_episodio",
            {f"C{f['capacity']}/{f['condicion']}/{f['seed']}": f["retornos"] for f in flujos},
        )

        for cap_key, bloque in resumen.items():
            print(f"\n[{cap_key}]")
            for cond, d in bloque.items():
                if cond == "diferencias_pareadas":
                    continue
                print(
                    f"  {cond:<14} auc={d['auc']['media']:.3f} "
                    f"[{d['auc']['ci_low']:.3f},{d['auc']['ci_high']:.3f}]  "
                    f"reaparición={d['reaparicion']['media']:.3f} "
                    f"[{d['reaparicion']['ci_low']:.3f},{d['reaparicion']['ci_high']:.3f}]  "
                    f"1ra vez={d['primera_vez']['media']:.3f}  "
                    f"trazas tarea al reaparecer={d['trazas_de_la_tarea_al_reaparecer']:.0f} "
                    f"(recomp. {d['trazas_recompensadas_de_la_tarea_al_reaparecer']:.0f})"
                )
            for par, v in bloque["diferencias_pareadas"].items():
                print(
                    f"  {par:<28} Δauc={v['auc']['media']:+.3f} "
                    f"[{v['auc']['ci_low']:+.3f},{v['auc']['ci_high']:+.3f}]  "
                    f"Δreap={v['reaparicion']['media']:+.3f} "
                    f"[{v['reaparicion']['ci_low']:+.3f},{v['reaparicion']['ci_high']:+.3f}]"
                )

        principal = resumen[f"C{CAPACIDAD_PRINCIPAL}"]["diferencias_pareadas"][
            "frontera_menos_FIFO"
        ]
        for k, h in (("reaparicion", "H1"), ("auc", "H2")):
            v = principal[k]
            veredicto = "sostenida" if v["ci_low"] > 0 else "refutada (IC incluye 0 o negativo)"
            run.note(
                f"{h} ({k}) con C={CAPACIDAD_PRINCIPAL}: frontera − FIFO = {v['media']:+.3f} "
                f"IC95 [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}], "
                f"{v['semillas_a_favor']}/{v['n']} semillas a favor → {veredicto}."
            )
        try:
            from ember.figures import figura_aprendizaje_downstream

            destino = figura_aprendizaje_downstream(
                resumen,
                "paper/figures/fig_exp13_downstream.pdf",
                capacidad=CAPACIDAD_PRINCIPAL,
                n_tareas=n_tareas,
            )
            run.record("figura", str(destino))
            print(f"\nFigura: {destino}")
        except ImportError:
            run.note("matplotlib no disponible; figura no generada")

        run.note(f"duración total {time.time() - t0:.0f} s")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
