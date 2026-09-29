"""Pruebas de exp13: el agente de control episódico y sus condiciones de memoria."""

import numpy as np
import pytest

from ember.core.genotype import FIFO_GENOTYPE
from ember.core.memory import PolicyMemory
from ember.core.policies import Merge

pytest.importorskip("minigrid")

from ember.envs.lifelong import GOAL_CORNERS, StateEncoder, make_goal_env  # noqa: E402
from experiments.exp13_downstream_learning import (  # noqa: E402
    RESERVOIR_GENOTYPE,
    EpisodicControlAgent,
    correr_todo,
    metricas_por_semilla,
    programa_de_tareas,
    resumir,
)


class TestReservorio:
    def test_respeta_la_capacidad(self):
        mem = PolicyMemory(dim=8, capacity=20, genotype=RESERVOIR_GENOTYPE, seed=0)
        rng = np.random.default_rng(0)
        for i in range(500):
            mem.write(rng.standard_normal(8), i)
        assert len(mem) == 20

    def test_la_muestra_es_uniforme_en_el_tiempo(self):
        """Lo que distingue al reservorio del FIFO: lo viejo sobrevive tanto como lo nuevo."""
        viejas = 0
        for s in range(40):
            mem = PolicyMemory(dim=8, capacity=50, genotype=RESERVOIR_GENOTYPE, seed=s)
            rng = np.random.default_rng(s)
            for i in range(1000):
                mem.write(rng.standard_normal(8), i)
            viejas += sum(v < 500 for v in mem.store.values)
        assert viejas / (40 * 50) == pytest.approx(0.5, abs=0.05)


class TestCodificador:
    def test_mismo_par_misma_clave(self):
        enc = StateEncoder(dim=64, seed=0)
        a = enc.keys(enc.state_part((2, 3), 1, GOAL_CORNERS[0]))
        b = enc.keys(enc.state_part((2, 3), 1, GOAL_CORNERS[0]))
        np.testing.assert_array_equal(a, b)

    def test_acciones_distintas_no_se_fusionan(self):
        """Si la fusión mezclara acciones del mismo estado, borraría la decisión misma."""
        enc = StateEncoder(dim=128, seed=0)
        k = enc.keys(enc.state_part((2, 3), 1, GOAL_CORNERS[0]))
        sims = k @ k.T
        assert sims[np.triu_indices(3, 1)].max() < Merge().threshold


class TestAgente:
    def test_sin_memoria_no_sabe_nada(self):
        mem = PolicyMemory(dim=16, capacity=10, genotype=FIFO_GENOTYPE, seed=0)
        ag = EpisodicControlAgent(mem, seed=0)
        keys = np.eye(3, 16, dtype=np.float32)
        np.testing.assert_array_equal(ag.q_values(keys), 0.0)

    def test_recuerda_el_retorno_de_una_coincidencia_exacta(self):
        mem = PolicyMemory(dim=16, capacity=10, genotype=FIFO_GENOTYPE, seed=0)
        ag = EpisodicControlAgent(mem, seed=0)
        keys = np.eye(3, 16, dtype=np.float32)
        ag.store_episode([keys[2]], [2], [0.0], reward=0.8, task=0)
        assert ag.q_values(keys)[2] == pytest.approx(0.8)

    def test_lo_desalojado_deja_de_influir(self):
        """La memoria es la única fuente de experiencia: sin traza, no hay valor."""
        mem = PolicyMemory(dim=16, capacity=1, genotype=FIFO_GENOTYPE, seed=0)
        ag = EpisodicControlAgent(mem, seed=0)
        keys = np.eye(3, 16, dtype=np.float32)
        ag.store_episode([keys[2]], [2], [0.0], reward=0.8, task=0)
        ag.store_episode([keys[0]], [0], [0.0], reward=0.0, task=0)
        assert ag.q_values(keys)[2] == pytest.approx(0.0)


def test_programa_alterna_y_repite():
    assert programa_de_tareas(2, 4) == [0, 1, 2, 3, 0, 1, 2, 3]


def test_la_meta_esta_en_su_esquina():
    env = make_goal_env(0, max_steps=10)
    env.reset(seed=0)
    assert env.grid.get(*GOAL_CORNERS[0]).type == "goal"
    env.close()


def test_experimento_en_tamano_reducido():
    kw = {"ciclos": 2, "episodios_por_fase": 3, "max_steps": 12}
    flujos = correr_todo((0, 1), (20,), n_jobs=1, **kw)
    assert len(flujos) == 2 * 5
    for f in flujos:
        assert len(f["retornos"]) == 2 * 4 * 3
        m = metricas_por_semilla(f, episodios_por_fase=3, n_tareas=4)
        assert 0.0 <= m["auc"] <= 1.0
    res = resumir(flujos, episodios_por_fase=3, n_tareas=4)
    d = res["C20"]["diferencias_pareadas"]["frontera_menos_FIFO"]["reaparicion"]
    assert d["ci_low"] <= d["media"] <= d["ci_high"]
