"""Pruebas de la señal de saliencia degradada (`ember.data.salience`).

Lo que se fija: que cada perilla degrade la señal en la dirección que dice, que
el AUC pedido sea el que sale, y que todo sea determinista dada la semilla.
"""

import numpy as np
import pytest

from ember.data.salience import (
    SignalCondition,
    degrade_signal,
    degrade_stream,
    empirical_auc,
    separation_for_auc,
    stream_auc,
    with_novel_distractors,
)
from ember.data.synthetic import clustered_stream


def _flujo(seed=0):
    return clustered_stream(n_prototypes=5, capacity=20, seed=seed)


class TestAUCEmpirico:
    def test_separacion_perfecta_da_uno(self):
        assert empirical_auc(np.array([0.1, 0.2, 0.9]), np.array([0, 0, 1], bool)) == 1.0

    def test_invertida_da_cero(self):
        assert empirical_auc(np.array([0.9, 0.8, 0.1]), np.array([0, 0, 1], bool)) == 0.0

    def test_los_empates_cuentan_medio(self):
        assert empirical_auc(np.array([0.5, 0.5]), np.array([0, 1], bool)) == 0.5

    def test_coincide_con_la_definicion_por_pares(self):
        rng = np.random.default_rng(0)
        s = rng.integers(0, 5, 60).astype(float)
        y = rng.random(60) < 0.3
        pares = [(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]]
        assert empirical_auc(s, y) == pytest.approx(np.mean(pares))


class TestCondicion:
    def test_la_condicion_vacia_deja_el_flujo_intacto(self):
        s = _flujo()
        d = degrade_stream(s, SignalCondition(), seed=0)
        assert [it.pred_error for it in d] == pytest.approx([it.pred_error for it in s])
        assert SignalCondition().is_clean

    def test_las_etiquetas_no_llevan_puntos(self):
        """`paper_sync` navega el JSON con rutas punteadas."""
        c = SignalCondition(auc=0.95, noise=0.3, delay=1, delay_prob=0.25, misleading=0.05)
        assert "." not in c.label()
        assert c.label() == "auc950_noise030_delay1_p025_mislead005"

    @pytest.mark.parametrize("kw", [{"auc": 1.0}, {"auc": 0.4}, {"noise": -1}, {"misleading": 2}])
    def test_rechaza_parametros_fuera_de_rango(self, kw):
        with pytest.raises(ValueError):
            SignalCondition(**kw)


class TestDegradacion:
    @pytest.mark.parametrize("auc", [0.95, 0.8, 0.6])
    def test_el_auc_pedido_es_el_que_sale(self, auc):
        imp = np.zeros(20000, bool)
        imp[:2000] = True
        pe = degrade_signal(
            np.zeros(20000), imp, SignalCondition(auc=auc), np.random.default_rng(0)
        )
        assert empirical_auc(pe, imp) == pytest.approx(auc, abs=0.01)
        assert pe.min() >= 0.0 and pe.max() <= 1.0

    def test_separacion_crece_con_el_auc(self):
        assert separation_for_auc(0.5) == pytest.approx(0.0)
        assert separation_for_auc(0.9) > separation_for_auc(0.7) > 0

    def test_mas_ruido_baja_el_auc(self):
        aucs = [
            np.mean(
                [
                    stream_auc(degrade_stream(_flujo(s), SignalCondition(noise=n), s))
                    for s in range(5)
                ]
            )
            for n in (0.0, 0.3, 1.0)
        ]
        assert aucs[0] > aucs[1] > aucs[2]

    def test_el_retraso_mueve_la_sorpresa_a_otra_experiencia(self):
        s = _flujo()
        d = degrade_stream(s, SignalCondition(delay=3), seed=0)
        pe, pe_d = [it.pred_error for it in s], [it.pred_error for it in d]
        assert pe_d[3:] == pytest.approx(pe[:-3])

    def test_invertir_todo_invierte_el_auc(self):
        d = degrade_stream(_flujo(), SignalCondition(misleading=1.0), seed=0)
        assert stream_auc(d) == 0.0

    def test_es_determinista_dada_la_semilla(self):
        c = SignalCondition(auc=0.7, noise=0.2, delay=1, delay_prob=0.5, misleading=0.1)
        a = [it.pred_error for it in degrade_stream(_flujo(), c, seed=3)]
        b = [it.pred_error for it in degrade_stream(_flujo(), c, seed=3)]
        otra = [it.pred_error for it in degrade_stream(_flujo(), c, seed=4)]
        assert a == b and a != otra

    def test_no_cambia_claves_ni_etiquetas(self):
        s = _flujo()
        d = degrade_stream(s, SignalCondition(auc=0.6, misleading=0.2), seed=0)
        assert [it.value for it in d] == [it.value for it in s]
        assert [it.is_rare for it in d] == [it.is_rare for it in s]
        assert len(d.rare_items) == len(s.rare_items)


class TestDistractores:
    def test_reemplaza_la_fraccion_pedida_de_lo_rutinario(self):
        s = _flujo()
        d = with_novel_distractors(s, 0.25, np.random.default_rng(0))
        n_rut = sum(not it.is_rare for it in s)
        distr = [it for it in d if str(it.value).startswith("distractor")]
        assert len(distr) == round(0.25 * n_rut)
        assert all(not it.is_rare for it in distr)
        assert len(d) == len(s) and len(d.rare_items) == len(s.rare_items)

    def test_los_distractores_son_nuevos(self):
        """Un distractor no se parece a ningún prototipo: es tan novedoso como lo raro."""
        s = _flujo()
        d = with_novel_distractors(s, 0.25, np.random.default_rng(0))
        prototipos = np.stack([it.key for it in s if not it.is_rare])
        for it in d:
            if str(it.value).startswith("distractor"):
                assert float((prototipos @ it.key).max()) < 0.8
