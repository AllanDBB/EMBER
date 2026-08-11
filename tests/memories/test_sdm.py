import numpy as np
import pytest

from ember.memories.sdm import SDMMemory


def _claves(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_recupera_una_clave_exacta():
    mem = SDMMemory(dim=32, capacity=10, seed=0)
    claves = _claves(5)
    for i, k in enumerate(claves):
        mem.write(k, i, pred_error=0.5)
    for i, k in enumerate(claves):
        assert mem.read(k).value == i


def test_los_contadores_vuelven_a_cero_al_desalojar_todo():
    """Regresión: el piloto restaba 1x lo que la escritura había sumado strength x.

    Sobre 40 escrituras en capacidad 3 el piloto acumulaba ~37 residuos en los
    contadores, y todas las lecturas posteriores se hacían contra esa basura.
    """
    mem = SDMMemory(dim=32, capacity=3, seed=0)
    for i, k in enumerate(_claves(40)):
        mem.write(k, i, pred_error=0.9 if i % 2 else 0.1)
    while len(mem):
        mem._desalojar(0)
    assert float(np.abs(mem.V).max()) == pytest.approx(0.0, abs=1e-3)


def test_degradacion_suave_ante_ruido():
    """La propiedad que hace a SDM útil para un robot: no falla de golpe."""
    mem = SDMMemory(dim=32, capacity=10, seed=0)
    claves = _claves(10)
    for i, k in enumerate(claves):
        mem.write(k, i, pred_error=0.5)

    rng = np.random.default_rng(99)
    aciertos = 0
    for i, k in enumerate(claves):
        ruidosa = k + rng.standard_normal(32).astype(np.float32) * 0.2
        ruidosa /= np.linalg.norm(ruidosa)
        aciertos += mem.read(ruidosa).value == i
    assert aciertos >= 7


def test_misma_semilla_produce_hard_locations_identicas():
    a = SDMMemory(dim=32, capacity=5, seed=3)
    b = SDMMemory(dim=32, capacity=5, seed=3)
    assert np.array_equal(a.H, b.H)


def test_semillas_distintas_producen_hard_locations_distintas():
    a = SDMMemory(dim=32, capacity=5, seed=1)
    b = SDMMemory(dim=32, capacity=5, seed=2)
    assert not np.array_equal(a.H, b.H)


def test_retiene_el_evento_sorpresivo_bajo_presion():
    """Fuerza inicial 2.8 vs 1.2: el desalojo por mínima fuerza siempre saca lo común."""
    mem = SDMMemory(dim=32, capacity=5, seed=0)
    claves = _claves(30)
    mem.write(claves[0], "raro", pred_error=0.9)
    for i, k in enumerate(claves[1:], start=1):
        mem.write(k, f"comun{i}", pred_error=0.1)
    assert mem.read(claves[0]).value == "raro"


def test_el_conjunto_activo_es_determinista_dada_la_clave():
    """Requisito para que el desalojo pueda restar sobre el mismo conjunto."""
    mem = SDMMemory(dim=32, capacity=5, seed=0)
    k = _claves(1)[0]
    assert np.array_equal(mem._conjunto_activo(k), mem._conjunto_activo(k))
