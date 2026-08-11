import numpy as np
import pytest

from ember.core.store import TraceStore


def _clave(rng, dim=8):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_append_incrementa_longitud_y_devuelve_indice():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    i = store.append(_clave(rng), "a", strength=2.5)
    assert i == 0
    assert len(store) == 1
    assert store.strength[0] == pytest.approx(2.5)


def test_remove_compacta_todos_los_arreglos_en_paralelo():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    for nombre, s in [("a", 1.0), ("b", 2.0), ("c", 3.0)]:
        store.append(_clave(rng), nombre, strength=s)
    store.remove(1)
    assert store.values == ["a", "c"]
    assert store.strength.tolist() == pytest.approx([1.0, 3.0])
    assert store.keys.shape == (2, 8)
    assert len(store.age) == len(store.utility) == len(store.contribution) == 2


def test_similarities_devuelve_coseno_y_vale_1_consigo_misma():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    k = _clave(rng)
    store.append(k, "a", strength=1.0)
    assert store.similarities(k)[0] == pytest.approx(1.0, abs=1e-5)


def test_similarities_sobre_store_vacio_devuelve_arreglo_vacio():
    store = TraceStore(dim=8, capacity=4)
    assert store.similarities(np.zeros(8, dtype=np.float32)).shape == (0,)


def test_novelty_es_1_en_store_vacio_y_0_ante_clave_repetida():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    k = _clave(rng)
    assert store.novelty(k) == pytest.approx(1.0)
    store.append(k, "a", strength=1.0)
    assert store.novelty(k) == pytest.approx(0.0, abs=1e-5)


def test_tick_envejece_las_trazas_existentes():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    store.append(_clave(rng), "a", strength=1.0)
    store.tick()
    store.tick()
    assert store.age[0] == pytest.approx(2.0)
    assert store.t == 2


def test_is_full_respeta_la_capacidad():
    store = TraceStore(dim=8, capacity=2)
    rng = np.random.default_rng(0)
    assert not store.is_full
    store.append(_clave(rng), "a", strength=1.0)
    store.append(_clave(rng), "b", strength=1.0)
    assert store.is_full


def test_la_contribucion_se_guarda_para_poder_revertirla_al_desalojar():
    """Sin esto, un sustrato distribuido no puede restar lo que sumó."""
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    store.append(_clave(rng), "a", strength=2.8, contribution=2.8)
    assert store.contribution[0] == pytest.approx(2.8)


def test_rechaza_claves_de_dimension_equivocada():
    store = TraceStore(dim=8, capacity=4)
    with pytest.raises(ValueError, match="forma"):
        store.append(np.ones(5, dtype=np.float32), "a", strength=1.0)


def test_rechaza_capacidad_no_positiva():
    with pytest.raises(ValueError, match="capacity"):
        TraceStore(dim=8, capacity=0)
