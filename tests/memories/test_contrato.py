"""Contrato común a toda arquitectura de memoria.

Parametrizado sobre `ARCHITECTURES`: registrar una arquitectura nueva ahí la
somete automáticamente a estas pruebas. Es la red de seguridad de todo el repo.

`Spiking` se excluye de los casos rápidos porque cada escritura simula 30
presentaciones de ráfagas sobre 128 neuronas; corre en la clase marcada `slow`.
"""

import numpy as np
import pytest

from ember.memories import ARCHITECTURES

RAPIDAS = {k: v for k, v in ARCHITECTURES.items() if k != "Spiking"}


def _claves(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(params=sorted(RAPIDAS))
def constructor(request):
    return RAPIDAS[request.param]


def test_lectura_vacia_devuelve_none(constructor):
    mem = constructor(dim=32, capacity=5, seed=0)
    r = mem.read(_claves(1)[0])
    assert r.value is None and r.similarity == 0.0


def test_recupera_lo_que_acaba_de_escribir(constructor):
    mem = constructor(dim=32, capacity=5, seed=0)
    k = _claves(1)[0]
    mem.write(k, "x", pred_error=0.5)
    assert mem.read(k).value == "x"


def test_nunca_excede_la_capacidad(constructor):
    mem = constructor(dim=32, capacity=4, seed=0)
    for i, k in enumerate(_claves(30)):
        mem.write(k, i, pred_error=0.5)
        assert len(mem) <= 4


def test_la_similitud_esta_en_cero_uno(constructor):
    mem = constructor(dim=32, capacity=5, seed=0)
    claves = _claves(5)
    for i, k in enumerate(claves):
        mem.write(k, i, pred_error=0.5)
    for k in claves:
        assert 0.0 <= mem.read(k).similarity <= 1.0


def test_es_determinista_bajo_la_misma_semilla(constructor):
    claves = _claves(25)

    def correr():
        mem = constructor(dim=32, capacity=6, seed=13)
        for i, k in enumerate(claves):
            mem.write(k, i, pred_error=0.5 + 0.4 * (i % 2))
        return [mem.read(k).value for k in claves]

    assert correr() == correr()


def test_expone_la_capacidad_declarada(constructor):
    assert constructor(dim=32, capacity=7, seed=0).capacity == 7


def test_acepta_claves_no_normalizadas(constructor):
    """El protocolo normaliza en la frontera: quien llama no tiene que hacerlo."""
    mem = constructor(dim=32, capacity=5, seed=0)
    k = _claves(1)[0]
    mem.write(k * 17.0, "x", pred_error=0.5)
    assert mem.read(k * 0.03).value == "x"


@pytest.mark.slow
class TestArquitecturasLentas:
    def test_spiking_cumple_el_contrato_basico(self):
        cls = ARCHITECTURES["Spiking"]
        mem = cls(dim=32, capacity=4, seed=0, n_presentations=5)
        assert mem.read(_claves(1)[0]).value is None
        for i, k in enumerate(_claves(8)):
            mem.write(k, i, pred_error=0.5)
            assert len(mem) <= 4
