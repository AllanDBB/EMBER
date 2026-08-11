import numpy as np
import pytest

from ember.core.policies import (
    FIFO,
    Append,
    BothGated,
    Constant,
    ExponentialDecay,
    Merge,
    MinStrength,
    MinUtility,
    NearestNeighbour,
    NoDecay,
    NoveltyGated,
    PredErrorGated,
    Radius,
    Random,
    TopK,
)
from ember.core.store import TraceStore


def _clave(rng, dim=8):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestFuerzaInicial:
    def test_constant_ignora_ambas_senales(self):
        p = Constant()
        assert p.initial(pred_error=0.9, novelty=0.9) == pytest.approx(1.0)
        assert p.initial(pred_error=0.1, novelty=0.1) == pytest.approx(1.0)

    def test_pred_error_escala_con_la_sorpresa(self):
        p = PredErrorGated()
        assert p.initial(pred_error=0.9, novelty=0.0) == pytest.approx(2.8)
        assert p.initial(pred_error=0.1, novelty=0.0) == pytest.approx(1.2)

    def test_novelty_escala_con_la_novedad(self):
        assert NoveltyGated().initial(pred_error=0.0, novelty=0.5) == pytest.approx(2.0)

    def test_both_suma_ambas(self):
        assert BothGated().initial(pred_error=0.9, novelty=0.5) == pytest.approx(3.8)


class TestEscritura:
    def test_append_nunca_consolida(self):
        store = TraceStore(dim=8, capacity=4)
        rng = np.random.default_rng(0)
        k = _clave(rng)
        store.append(k, "a", strength=1.0)
        assert Append().route(store, k) is None

    def test_merge_consolida_si_supera_el_umbral(self):
        store = TraceStore(dim=8, capacity=4)
        rng = np.random.default_rng(0)
        k = _clave(rng)
        store.append(k, "a", strength=1.0)
        assert Merge(threshold=0.85).route(store, k) == 0

    def test_merge_crea_traza_nueva_si_no_lo_supera(self):
        store = TraceStore(dim=8, capacity=4)
        rng = np.random.default_rng(0)
        store.append(_clave(rng), "a", strength=1.0)
        assert Merge(threshold=0.85).route(store, _clave(rng)) is None


class TestLectura:
    def test_nn_devuelve_exactamente_un_indice_el_mayor(self):
        sims = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        assert NearestNeighbour().select(sims).tolist() == [1]

    def test_topk_devuelve_k_indices_ordenados_por_similitud(self):
        sims = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
        assert TopK(k=3).select(sims).tolist() == [1, 3, 2]

    def test_topk_no_falla_si_hay_menos_trazas_que_k(self):
        assert TopK(k=3).select(np.array([0.4], dtype=np.float32)).tolist() == [0]

    def test_radius_toma_todo_lo_que_llega_a_la_fraccion_del_mejor(self):
        # mejor = 0.9; el corte queda en 0.63, así que entran 0.9, 0.75 y 0.8.
        sims = np.array([0.1, 0.9, 0.75, 0.8], dtype=np.float32)
        assert sorted(Radius(fraction=0.70).select(sims).tolist()) == [1, 2, 3]

    def test_el_radio_es_relativo_y_no_degenera_en_dominios_poco_similares(self):
        """Con similitudes bajas un umbral absoluto colapsaría a vecino más cercano."""
        sims = np.array([0.10, 0.30, 0.25], dtype=np.float32)
        assert sorted(Radius(fraction=0.70).select(sims).tolist()) == [1, 2]

    def test_radius_cae_al_mas_cercano_si_el_mejor_no_es_positivo(self):
        sims = np.array([-0.4, -0.1, -0.9], dtype=np.float32)
        assert Radius(fraction=0.70).select(sims).tolist() == [1]

    def test_un_solo_candidato_devuelve_ese(self):
        assert Radius(fraction=0.70).select(np.array([0.5], dtype=np.float32)).tolist() == [0]


class TestDesalojo:
    def _store_poblado(self):
        store = TraceStore(dim=8, capacity=10)
        rng = np.random.default_rng(0)
        for nombre, s in [("a", 3.0), ("b", 1.0), ("c", 2.0)]:
            store.append(_clave(rng), nombre, strength=s)
        store.age[:] = [10.0, 5.0, 1.0]
        store.utility[:] = [0.0, 7.0, 3.0]
        return store

    def test_fifo_descarta_el_mas_viejo(self):
        assert FIFO().victim(self._store_poblado(), np.random.default_rng(0)) == 0

    def test_min_strength_descarta_el_mas_debil(self):
        assert MinStrength().victim(self._store_poblado(), np.random.default_rng(0)) == 1

    def test_min_utility_descarta_el_menos_usado(self):
        assert MinUtility().victim(self._store_poblado(), np.random.default_rng(0)) == 0

    def test_random_es_determinista_bajo_la_misma_semilla(self):
        store = self._store_poblado()
        a = Random().victim(store, np.random.default_rng(42))
        b = Random().victim(store, np.random.default_rng(42))
        assert a == b
        assert 0 <= a < 3


class TestDecaimiento:
    def test_no_decay_no_toca_la_fuerza(self):
        s = np.array([1.0, 2.0], dtype=np.float32)
        NoDecay().step(s)
        assert s.tolist() == pytest.approx([1.0, 2.0])

    def test_exponential_multiplica_in_place(self):
        s = np.array([1.0, 2.0], dtype=np.float32)
        ExponentialDecay(rate=0.98).step(s)
        assert s.tolist() == pytest.approx([0.98, 1.96], abs=1e-6)


class TestEtiquetas:
    def test_toda_politica_expone_una_etiqueta_para_el_analisis(self):
        politicas = [
            Constant(),
            NoveltyGated(),
            PredErrorGated(),
            BothGated(),
            Append(),
            Merge(),
            NearestNeighbour(),
            TopK(),
            Radius(),
            FIFO(),
            MinStrength(),
            MinUtility(),
            Random(),
            NoDecay(),
            ExponentialDecay(rate=0.98),
        ]
        for p in politicas:
            assert isinstance(p.label, str) and p.label

    def test_las_politicas_son_comparables_por_igualdad(self):
        assert Merge(threshold=0.85) == Merge(threshold=0.85)
        assert Merge(threshold=0.85) != Merge(threshold=0.90)

    def test_el_decaimiento_se_etiqueta_con_su_tasa(self):
        assert ExponentialDecay(rate=0.98).label == "0.98"
        assert NoDecay().label == "1.0"
