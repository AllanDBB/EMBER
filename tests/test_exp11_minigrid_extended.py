"""Tests de exp11: estadística por rollout, trazas, señales, importancia y políticas."""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from experiments.exp11_minigrid_extended import (
    BEST_NO_SALIENCE_GENOTYPE,
    block_ci,
    cluster_bootstrap_diff,
    cluster_bootstrap_ratio,
    rate_summary,
    spearman,
)


def _hay_minigrid() -> bool:
    return find_spec("minigrid") is not None


# ══════════════════════════════════════════════════════════ estadística


class TestEstadisticaPorRollout:
    def test_el_cluster_se_ensancha_cuando_los_eventos_se_agrupan(self):
        """Todos los aciertos en pocos rollouts: el IC por clúster tiene que ser más ancho que Wilson.

        Es el defecto que señala el revisor: Wilson trata 100 eventos de 10
        rollouts como 100 observaciones independientes.
        """
        n = np.full(10, 10)
        hits = np.array([10, 10, 10, 10, 10, 0, 0, 0, 0, 0])
        r = rate_summary(hits, n, np.arange(10) // 2, seed=0)
        assert r["tasa"] == pytest.approx(0.5)
        assert r["ensanchamiento"] > 2.0
        assert r["deff"] > 5.0

    def test_sin_correlacion_el_efecto_de_diseno_ronda_uno(self):
        rng = np.random.default_rng(0)
        n = np.ones(400, dtype=int)
        hits = (rng.random(400) < 0.3).astype(int)
        r = rate_summary(hits, n, np.arange(400) // 80, seed=0)
        assert 0.7 < r["deff"] < 1.4

    def test_el_ic_contiene_la_tasa(self):
        n = np.array([3, 0, 5, 2, 4])
        hits = np.array([1, 0, 5, 0, 2])
        ci = cluster_bootstrap_ratio(hits, n, seed=1)
        assert ci["ci_low"] <= hits.sum() / n.sum() <= ci["ci_high"]

    def test_sin_eventos_no_inventa_intervalo(self):
        ci = cluster_bootstrap_ratio(np.zeros(4), np.zeros(4))
        assert ci["ci_low"] is None

    def test_la_diferencia_pareada(self):
        n = np.array([2, 2, 2, 2])
        d = cluster_bootstrap_diff(np.array([2, 2, 2, 2]), np.array([0, 0, 0, 0]), n)
        assert d["dif"] == pytest.approx(1.0)
        assert d["ci_low"] == pytest.approx(1.0)

    def test_bloques(self):
        r = block_ci(np.array([1, 1, 0, 0, 1, 0]), np.ones(6), np.array([0, 0, 1, 1, 2, 2]))
        assert r["por_bloque"] == [1.0, 0.0, 0.5]
        assert r["ci_low"] < r["media"] < r["ci_high"]

    def test_spearman(self):
        assert spearman(np.arange(5), np.arange(5) ** 2) == pytest.approx(1.0)
        assert spearman(np.arange(5), -np.arange(5)) == pytest.approx(-1.0)


def test_la_mejor_sin_saliencia_es_la_de_exp01():
    """Si exp01 se re-corre y cambia su mejor configuración sin saliencia, esto lo detecta."""
    ruta = Path("results/exp01_nas_full/data.json")
    if not ruta.exists():
        pytest.skip("sin resultados de exp01")
    registros = json.loads(ruta.read_text())["all_records"]
    sin = [r for r in registros if r["genotype"]["strength"] in ("constant", "novelty")]
    mejor = max(r["mean"] for r in sin)
    etiquetas = {r["label"] for r in sin if r["mean"] == mejor}
    assert BEST_NO_SALIENCE_GENOTYPE.label() in etiquetas


# ══════════════════════════════════════════════════ señales e importancia


class _TrazaFalsa:
    """Una traza mínima para probar etiquetas sin correr el entorno."""

    def __init__(self, rewards, episode, obs_id, next_obs_id, actions, pos):
        self.rewards = np.asarray(rewards, dtype=np.float64)
        self.episode = np.asarray(episode)
        self.obs_id = np.asarray(obs_id)
        self.next_obs_id = np.asarray(next_obs_id)
        self.actions = np.asarray(actions)
        self.agent_pos = np.asarray(pos)
        self.key_pickup = np.zeros(len(rewards), dtype=bool)
        self.door_unlocked = np.zeros(len(rewards), dtype=bool)
        fin = np.r_[self.episode[1:] != self.episode[:-1], True]
        self.terminated = fin & (self.rewards > 0)
        self.truncated = fin & ~(self.rewards > 0)

    def __len__(self):
        return len(self.rewards)


def _traza():
    return _TrazaFalsa(
        rewards=[0, 0, 0, 0.5, 0, 0, 0.7],
        episode=[0, 0, 0, 0, 1, 1, 1],
        obs_id=[0, 1, 0, 2, 0, 1, 0],
        next_obs_id=[1, 0, 2, 0, 1, 3, 0],
        actions=[2, 2, 2, 2, 2, 2, 2],
        pos=[(1, 1), (1, 2), (1, 1), (1, 3), (1, 1), (1, 2), (2, 2)],
    )


class TestImportancia:
    def test_previa_k_no_cruza_episodios_ni_incluye_la_recompensa(self):
        from ember.envs.salience import importance_mask

        m = importance_mask(_traza(), "previa_k", k=5)
        assert m.tolist() == [True, True, True, False, True, True, False]

    def test_novedad_estado_es_primera_visita(self):
        from ember.envs.salience import importance_mask

        m = importance_mask(_traza(), "novedad_estado")
        assert m.tolist() == [True, True, False, True, False, False, True]

    def test_sorpresa_transicion_exige_par_ya_visto(self):
        from ember.envs.salience import importance_mask

        # (0, 2) se vio en t=0 con destino 1: en t=2 va a 2 (sorpresa), en t=4
        # repite 1 (no), en t=6 va a 0 (sorpresa). (1, 2) cambia de 0 a 3 en t=5.
        m = importance_mask(_traza(), "sorpresa_transicion")
        assert m.tolist() == [False, False, True, False, False, True, True]

    def test_retorno_descontado_por_episodio(self):
        from ember.envs.salience import discounted_return

        g = discounted_return(_traza(), gamma=0.5)
        assert g[3] == pytest.approx(0.5)
        assert g[2] == pytest.approx(0.25)
        assert g[6] == pytest.approx(0.7)
        assert g[4] == pytest.approx(0.7 * 0.25)

    def test_auc(self):
        from ember.envs.salience import separation_auc

        s = np.array([0.1, 0.2, 0.9, 0.8])
        assert separation_auc(s, np.array([False, False, True, True])) == 1.0
        assert separation_auc(s, np.array([True, True, False, False])) == 0.0
        assert separation_auc(np.ones(4), np.array([True, False, False, False])) == 0.5
        assert separation_auc(s, np.zeros(4, dtype=bool)) is None

    def test_nombre_desconocido_falla(self):
        from ember.envs.salience import importance_mask

        with pytest.raises(ValueError):
            importance_mask(_traza(), "lo_que_sea")

    def test_td_anticipa_la_recompensa(self):
        """Tras repetir el mismo episodio, TD da error en el paso previo a la recompensa."""
        from ember.envs.salience import TDErrorPredictor

        a, b = np.eye(4, dtype=np.float32)[0], np.eye(4, dtype=np.float32)[1]
        td = TDErrorPredictor(4)
        for _ in range(200):
            td.surprise(a, 0.0, b, False)
            td.surprise(b, 1.0, a, True)
        assert td.w[1] > 0.5 and td.w[0] > 0.3


# ══════════════════════════════════════════════════════ con el entorno


@pytest.mark.skipif(not _hay_minigrid(), reason="requiere el extra envs")
class TestConEntorno:
    def test_la_traza_reproduce_exp06_bit_a_bit(self):
        """Misma semilla: mismas claves, misma señal y mismos aciertos que exp06."""
        from ember.envs.minigrid import MiniGridStreamAdapter
        from ember.envs.salience import compute_signal, importance_mask
        from ember.tasks.battery import t1_rare_retention
        from experiments._common import make_factory
        from experiments.exp06_minigrid import rollout
        from experiments.exp11_minigrid_extended import MEMORIAS, retention_per_step

        for seed in (0, 4):
            for fuente, senal in (("reward", "reward_pe"), ("prediction", "perceptual")):
                s = rollout(seed, n_steps=400, surprise_source=fuente)
                tr = MiniGridStreamAdapter(seed=seed).rollout_trace(400)
                v = compute_signal(tr, senal)
                assert [it.pred_error for it in s] == v.tolist()
                m = importance_mask(tr, "recompensa")
                assert int(m.sum()) == len(s.rare_items)
                for g in MEMORIAS.values():
                    r = t1_rare_retention(make_factory(g, 32), s, seed=seed, capacity=20, dim=32)
                    esperado = round(r.score * max(len(s.rare_items), 1))
                    # Consultar más pasos que los importantes no cambia los aciertos.
                    ac = retention_per_step(tr, v, g, np.ones(len(tr), dtype=bool), seed=seed)
                    assert int(ac[m].sum()) == esperado

    def test_el_planificador_resuelve_doorkey_y_registra_los_eventos(self):
        from ember.envs.agents import NoisyPlanner
        from ember.envs.minigrid import MiniGridStreamAdapter

        tr = MiniGridStreamAdapter("MiniGrid-DoorKey-6x6-v0", seed=0).rollout_trace(
            150, policy=NoisyPlanner(0, epsilon=0.0)
        )
        assert (tr.rewards > 0).sum() >= 3
        assert tr.key_pickup.sum() >= 3
        assert tr.door_unlocked.sum() >= 3

    def test_el_planificador_resuelve_keycorridor(self):
        from ember.envs.agents import NoisyPlanner
        from ember.envs.minigrid import MiniGridStreamAdapter

        tr = MiniGridStreamAdapter("MiniGrid-KeyCorridorS3R2-v0", seed=1).rollout_trace(
            200, policy=NoisyPlanner(1, epsilon=0.0)
        )
        assert (tr.rewards > 0).sum() >= 2

    def test_qlearning_aprende_y_es_reproducible(self):
        from ember.envs.agents import TabularQAgent
        from ember.envs.minigrid import MiniGridStreamAdapter

        def correr():
            ag = TabularQAgent(3)
            tr = MiniGridStreamAdapter("MiniGrid-DoorKey-6x6-v0", seed=3).rollout_trace(
                200, policy=ag
            )
            return tr.actions.tolist(), len(ag.q)

        a, b = correr(), correr()
        assert a == b
        assert a[1] > 10

    def test_la_sesgada_puede_interactuar(self):
        from ember.envs.agents import FORWARD, ForwardBiasedPolicy

        p = ForwardBiasedPolicy(0, 7)
        acciones = [p.act(None, None) for _ in range(3000)]
        assert acciones.count(FORWARD) / 3000 == pytest.approx(0.6, abs=0.03)
        assert any(a >= 3 for a in acciones)

    def test_experimento_reducido_de_punta_a_punta(self):
        """Fija que la orquestación no rompa: tamaño mínimo, sin procesos."""
        from experiments.exp11_minigrid_extended import (
            correr_matriz,
            reanalizar_exp06,
            resumen_matriz,
        )

        r = reanalizar_exp06((0, 1, 2), n_steps=300, n_jobs=1)
        assert set(r) == {"aleatoria_prediccion", "sesgada_prediccion", "aleatoria_recompensa"}
        m = correr_matriz(("DoorKey6x6",), ("planificador",), (0, 1, 40), n_steps=200, n_jobs=1)
        celda = m["DoorKey6x6"]["planificador"]
        assert celda["recompensa"]["n_eventos"] > 0
        f = celda["recompensa"]["senales"]["reward_pe"]["frontera"]
        assert 0.0 <= f["tasa"] <= 1.0
        res = resumen_matriz(m)
        assert res["recompensa"]["reward_pe"]["n_celdas"] == 1
        assert "frontera_menos_azar" in celda["recompensa"]["senales"]["reward_pe"]
        assert sum(t["n_celdas"] for t in res["por_auc"].values()) == res["n_celdas_total"]
