import numpy as np
import pytest

from ember.data.synthetic import block_stream, clustered_stream


def _coseno_medio(K):
    G = np.abs(K @ K.T)
    return float(G[~np.eye(len(K), dtype=bool)].mean())


def test_el_spec_reporta_el_ratio_r():
    s = clustered_stream(n_prototypes=10, capacity=20, seed=0)
    assert s.spec.r == pytest.approx(0.5)


def test_el_spec_clasifica_el_regimen():
    assert clustered_stream(2, 20, seed=0).spec.regime == "compression"
    assert clustered_stream(40, 20, seed=0).spec.regime == "selection"
    assert clustered_stream(15, 20, seed=0).spec.regime == "transition"


def test_hay_exactamente_n_rare_eventos_raros():
    s = clustered_stream(n_prototypes=5, capacity=20, n_common=100, n_rare=7, seed=0)
    assert sum(it.is_rare for it in s) == 7
    assert len(s) == 107
    assert len(s.rare_items) == 7


def test_los_raros_tienen_error_de_prediccion_alto_y_los_comunes_bajo():
    s = clustered_stream(n_prototypes=5, capacity=20, n_common=100, n_rare=10, seed=0)
    raros = [it.pred_error for it in s if it.is_rare]
    comunes = [it.pred_error for it in s if not it.is_rare]
    assert min(raros) > max(comunes)


def test_las_claves_son_unitarias_float32():
    s = clustered_stream(n_prototypes=5, capacity=20, seed=0)
    for it in list(s)[:20]:
        assert it.key.dtype == np.float32
        assert np.linalg.norm(it.key) == pytest.approx(1.0, abs=1e-5)


def test_la_misma_semilla_reproduce_el_stream():
    a = clustered_stream(n_prototypes=5, capacity=20, seed=3)
    b = clustered_stream(n_prototypes=5, capacity=20, seed=3)
    assert np.array_equal(a.keys(), b.keys())


def test_semillas_distintas_producen_streams_distintos():
    a = clustered_stream(n_prototypes=5, capacity=20, seed=1)
    b = clustered_stream(n_prototypes=5, capacity=20, seed=2)
    assert not np.array_equal(a.keys(), b.keys())


def test_correlation_reduce_la_ortogonalidad_media():
    """El gaussiano i.i.d. es casi ortogonal; con correlación deja de serlo."""

    def coseno(correlation):
        s = clustered_stream(
            n_prototypes=40,
            capacity=20,
            n_common=200,
            n_rare=0,
            correlation=correlation,
            seed=0,
        )
        return _coseno_medio(s.keys())

    assert coseno(0.9) > coseno(0.0) * 1.5


def test_density_skew_desbalancea_las_visitas_por_prototipo():
    def maxima_proporcion(skew):
        s = clustered_stream(
            n_prototypes=10, capacity=20, n_common=500, n_rare=0, density_skew=skew, seed=0
        )
        _, cuentas = np.unique([it.value for it in s], return_counts=True)
        return cuentas.max() / cuentas.sum()

    assert maxima_proporcion(2.0) > maxima_proporcion(0.0) * 1.5


def test_drift_aleja_las_claves_tardias_de_las_tempranas():
    def separacion(drift):
        s = clustered_stream(
            n_prototypes=3, capacity=20, n_common=300, n_rare=0, drift=drift, seed=0
        )
        K = s.keys()
        return float((K[:30] @ K[-30:].T).mean())

    assert separacion(1.5) < separacion(0.0)


def test_un_solo_prototipo_es_valido():
    s = clustered_stream(n_prototypes=1, capacity=20, n_common=50, n_rare=0, seed=0)
    assert s.spec.r == pytest.approx(0.05)
    assert len({it.value for it in s}) == 1


def test_rechaza_cero_prototipos():
    with pytest.raises(ValueError, match="n_prototypes"):
        clustered_stream(n_prototypes=0, capacity=20, seed=0)


class TestBlockStream:
    def test_repite_cada_item_dentro_de_su_bloque(self):
        s = block_stream(n_blocks=4, per_block=8, repeats=3, seed=0)
        assert len(s) == 4 * 8 * 3
        cuentas = {}
        for it in s:
            cuentas[it.value] = cuentas.get(it.value, 0) + 1
        assert set(cuentas.values()) == {3}

    def test_el_primer_bloque_lleva_error_de_prediccion_alto(self):
        s = block_stream(pe_first=0.9, pe_rest=0.1, seed=0)
        primeros = [it.pred_error for it in s if it.value.startswith("b0_")]
        resto = [it.pred_error for it in s if not it.value.startswith("b0_")]
        assert min(primeros) > max(resto)

    def test_con_pe_uniforme_no_hay_senal_que_distinga_bloques(self):
        s = block_stream(pe_first=0.5, pe_rest=0.5, seed=0)
        assert len({it.pred_error for it in s}) == 1

    def test_rare_items_apunta_al_primer_bloque(self):
        s = block_stream(n_blocks=4, per_block=8, repeats=3, seed=0)
        assert all(it.value.startswith("b0_") for it in s.rare_items)
