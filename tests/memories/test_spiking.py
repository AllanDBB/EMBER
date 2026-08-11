import numpy as np
import pytest

from ember.memories.spiking import SpikingMemory
from ember.memories.spiking_sdm import SpikingSDMMemory


def _claves(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestSpikingLIF:
    def test_misma_semilla_produce_pesos_identicos(self):
        """Regresión: el piloto sembraba con id(), la dirección de memoria del arreglo."""
        claves = _claves(3)
        pesos = []
        for _ in range(2):
            mem = SpikingMemory(dim=32, capacity=5, seed=5, n_presentations=3)
            for i, k in enumerate(claves):
                mem.write(k, i, pred_error=0.5)
            pesos.append(mem.W.copy())
        assert np.array_equal(pesos[0], pesos[1])

    def test_semillas_distintas_producen_pesos_distintos(self):
        claves = _claves(3)
        pesos = []
        for semilla in (1, 2):
            mem = SpikingMemory(dim=32, capacity=5, seed=semilla, n_presentations=3)
            for i, k in enumerate(claves):
                mem.write(k, i, pred_error=0.5)
            pesos.append(mem.W.copy())
        assert not np.array_equal(pesos[0], pesos[1])

    @pytest.mark.slow
    def test_stdp_fortalece_dentro_del_ensamble_mas_que_fuera(self):
        """El hallazgo de la PoC-3: intra ~0.21 vs inter ~0.02."""
        mem = SpikingMemory(dim=32, capacity=5, seed=0, n_presentations=30)
        mem.write(_claves(1)[0], "a", pred_error=0.5)

        ens = mem.store_patterns[0]
        fuera = np.setdiff1d(np.arange(mem.n), ens)
        intra = mem.W[np.ix_(ens, ens)][mem.mask[np.ix_(ens, ens)]].mean()
        inter = mem.W[np.ix_(ens, fuera)][mem.mask[np.ix_(ens, fuera)]].mean()
        assert intra > 3 * inter

    def test_nunca_excede_la_capacidad(self):
        mem = SpikingMemory(dim=32, capacity=3, seed=0, n_presentations=2)
        for i, k in enumerate(_claves(6)):
            mem.write(k, i, pred_error=0.5)
            assert len(mem) <= 3
            assert len(mem.store_patterns) == len(mem)

    def test_lectura_sobre_memoria_vacia_devuelve_none(self):
        mem = SpikingMemory(dim=32, capacity=3, seed=0, n_presentations=2)
        assert mem.read(_claves(1)[0]).value is None


class TestSpikingSDM:
    def test_recupera_una_clave_exacta(self):
        mem = SpikingSDMMemory(dim=32, capacity=10, seed=0)
        claves = _claves(5)
        for i, k in enumerate(claves):
            mem.write(k, i, pred_error=0.5)
        aciertos = sum(mem.read(k).value == i for i, k in enumerate(claves))
        assert aciertos >= 4

    def test_los_contadores_vuelven_a_cero_al_desalojar_todo(self):
        """Misma regresión que la SDM algebraica, sobre activación estocástica."""
        mem = SpikingSDMMemory(dim=32, capacity=3, seed=0)
        for i, k in enumerate(_claves(25)):
            mem.write(k, i, pred_error=0.9 if i % 2 else 0.1)
        while len(mem):
            mem._desalojar(0)
        assert float(np.abs(mem.V).max()) == pytest.approx(0.0, abs=1e-3)

    def test_misma_semilla_produce_el_mismo_tren_de_spikes(self):
        claves = _claves(6)

        def correr():
            mem = SpikingSDMMemory(dim=32, capacity=4, seed=9)
            for i, k in enumerate(claves):
                mem.write(k, i, pred_error=0.5)
            return [mem.read(k).value for k in claves]

        assert correr() == correr()

    def test_nunca_excede_la_capacidad(self):
        mem = SpikingSDMMemory(dim=32, capacity=4, seed=0)
        for i, k in enumerate(_claves(20)):
            mem.write(k, i, pred_error=0.5)
            assert len(mem) <= 4
            assert len(mem._activas) == len(mem)
