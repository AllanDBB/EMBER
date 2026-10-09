"""Variantes de ablación de la SDM y compuerta de admisión."""

import numpy as np
import pytest

from ember.core.policies import AdmitAll, MinStrength, PredErrorAdmission, PredErrorGated
from ember.memories.sdm import SDMMemory
from ember.memories.sdm_ablation import (
    SDM_TRACES_ONLY_GENOTYPE,
    AdmissionGated,
    SDMCountersOnly,
    sdm_traces_only,
)


def _claves(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(params=[False, True], ids=["a", "a+"])
def contadores(request):
    return lambda **kw: SDMCountersOnly(dim=32, exact_erase=request.param, **kw)


def test_lectura_vacia_devuelve_none(contadores):
    r = contadores(capacity=5, seed=0).read(_claves(1)[0])
    assert r.value is None and r.similarity == 0.0


def test_recupera_el_payload_desde_los_contadores(contadores):
    mem = contadores(capacity=10, seed=0)
    claves = _claves(8)
    for i, k in enumerate(claves):
        mem.write(k, f"p{i}", pred_error=0.5)
    aciertos = sum(mem.read(k).value == f"p{i}" for i, k in enumerate(claves))
    assert aciertos >= 7


def test_nunca_supera_la_capacidad(contadores):
    mem = contadores(capacity=4, seed=0)
    for i, k in enumerate(_claves(30)):
        mem.write(k, i, pred_error=0.5)
        assert len(mem) <= 4


def test_misma_semilla_mismas_hard_locations_que_la_sdm_completa():
    """La ablación no puede cambiar el azar: solo quita un componente."""
    a = SDMCountersOnly(dim=32, capacity=5, seed=3)
    b = SDMMemory(dim=32, capacity=5, seed=3)
    assert np.array_equal(a.H, b.H)


def test_misma_semilla_resultados_identicos():
    def correr(seed):
        mem = SDMCountersOnly(dim=32, capacity=5, seed=seed)
        for i, k in enumerate(_claves(20)):
            mem.write(k, i, pred_error=0.3)
        return mem.V.copy()

    assert np.array_equal(correr(1), correr(1))
    assert not np.array_equal(correr(1), correr(2))


def test_con_borrado_exacto_los_contadores_vuelven_a_cero():
    """Mismo invariante que la SDM completa: se resta lo que se sumó."""
    mem = SDMCountersOnly(dim=32, capacity=3, seed=0, exact_erase=True)
    for i, k in enumerate(_claves(40)):
        mem.write(k, i, pred_error=0.9 if i % 2 else 0.1)
    while len(mem):
        mem._desalojar(0)
    assert float(np.abs(mem.V).max()) == pytest.approx(0.0, abs=1e-3)


def test_sin_borrado_exacto_el_residuo_queda_por_diseno():
    """Sin claves guardadas no hay cómo recalcular el conjunto activo a restar."""
    mem = SDMCountersOnly(dim=32, capacity=3, seed=0, exact_erase=False)
    for i, k in enumerate(_claves(10)):
        mem.write(k, i, pred_error=0.5)
    while len(mem):
        mem._desalojar(0)
    assert float(np.abs(mem.V).max()) > 0.1


def test_la_variante_solo_trazas_es_el_ciclo_de_vida_de_la_sdm():
    sdm = SDMMemory(dim=32, capacity=5, seed=0)
    g = SDM_TRACES_ONLY_GENOTYPE
    assert type(g.strength) is type(sdm.strength_policy) is PredErrorGated
    assert type(g.evict) is type(sdm.evict_policy) is MinStrength
    assert g.reinforce == 0.0 and g.decay.label == "1.0"
    mem = sdm_traces_only(dim=32, capacity=5, seed=0)
    k = _claves(1)[0]
    mem.write(k, "x", pred_error=0.5)
    assert mem.read(k).value == "x"


def test_la_compuerta_de_admision_descarta_lo_poco_sorpresivo():
    mem = AdmissionGated(SDMMemory(dim=32, capacity=5, seed=0), PredErrorAdmission(0.5))
    claves = _claves(2)
    mem.write(claves[0], "bajo", pred_error=0.2)
    mem.write(claves[1], "alto", pred_error=0.8)
    assert len(mem) == 1 and mem.n_rejected == 1
    assert mem.read(claves[1]).value == "alto"


def test_umbral_cero_admite_todo():
    assert PredErrorAdmission(0.0).admit(0.0)
    assert AdmitAll().admit(0.0)
    assert not PredErrorAdmission(0.3).admit(0.29)
