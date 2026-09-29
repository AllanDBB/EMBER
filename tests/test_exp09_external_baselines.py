"""Pruebas de los baselines externos y de exp09.

Fijan que cada política externa haga lo que dice su docstring (la política
publicada a la que corresponde), que ninguna se haya colado al espacio de 576, y
que el experimento mida lo que dice medir en tamaño reducido.
"""

import numpy as np
import pytest

from ember.baselines import (
    EMDB_SEQUENTIAL_GENOTYPE,
    EXTERNAL_BASELINES,
    FRONTIER_GENOTYPE,
    METHODS,
)
from ember.core.genotype import FIFO_GENOTYPE
from ember.core.memory import PolicyMemory
from ember.core.policies import (
    LFU,
    LRU,
    CoverageMax,
    PrioritySurprise,
    Reservoir,
    SequentialScan,
    StochasticPriority,
    UtilityCache,
)
from ember.core.store import TraceStore
from ember.nas.space import enumerate_space, space_size
from experiments.exp09_external_baselines import comparar, rango_en_espacio


def _claves(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _memoria(evict, capacity=3, dim=8, seed=0):
    return PolicyMemory(
        dim=dim, capacity=capacity, genotype=FIFO_GENOTYPE.with_axis("evict", evict), seed=seed
    )


class TestFueraDelEspacio:
    def test_el_espacio_sigue_teniendo_576(self):
        assert space_size() == 576

    def test_ningun_baseline_externo_ni_el_buffer_real_esta_en_el_espacio(self):
        """Invariante 3: las políticas externas no entran a la búsqueda."""
        espacio = set(enumerate_space())
        for g in [*EXTERNAL_BASELINES.values(), EMDB_SEQUENTIAL_GENOTYPE]:
            assert g not in espacio

    def test_la_frontera_y_el_proxy_si_son_puntos_del_espacio(self):
        espacio = set(enumerate_space())
        assert FRONTIER_GENOTYPE in espacio
        assert METHODS["fifo_nn_proxy"] in espacio


class TestStore:
    def test_last_use_y_priority_se_compactan_con_el_resto(self):
        store = TraceStore(dim=8, capacity=4)
        for i, k in enumerate(_claves(3)):
            store.tick()
            store.append(k, i, strength=1.0, priority=0.1 * (i + 1))
        store.remove(1)
        assert store.priority.tolist() == pytest.approx([0.1, 0.3])
        assert store.last_use.tolist() == pytest.approx([1.0, 3.0])


class TestPoliticasExternas:
    def test_lru_desaloja_la_menos_usada_recientemente(self):
        mem = _memoria(LRU())
        k = _claves(4)
        for i in range(3):
            mem.write(k[i], i)
        mem.read(k[0])  # 0 pasa a ser la más reciente; 1 queda como LRU
        mem.write(k[3], 3)
        assert sorted(mem.store.values) == [0, 2, 3]

    def test_lfu_desaloja_la_menos_frecuente_y_desempata_por_recencia(self):
        mem = _memoria(LFU())
        k = _claves(4)
        for i in range(3):
            mem.write(k[i], i)
        mem.read(k[0])
        mem.read(k[0])
        mem.read(k[1])
        mem.read(k[2])
        mem.read(k[1])  # frecuencias 2, 2, 1 → sale la 2
        mem.write(k[3], 3)
        assert sorted(mem.store.values) == [0, 1, 3]

    def test_per_min_lee_la_sorpresa_cruda_y_no_la_fuerza(self):
        """Con fuerza constante, lo único que la distingue es `priority`."""
        mem = _memoria(PrioritySurprise())
        k = _claves(4)
        for i, pe in enumerate([0.9, 0.1, 0.8]):
            mem.write(k[i], i, pred_error=pe)
        mem.write(k[3], 3, pred_error=0.7)
        assert sorted(mem.store.values) == [0, 2, 3]

    def test_per_estocastico_protege_lo_sorpresivo_sin_volverlo_imposible(self):
        store = TraceStore(dim=8, capacity=2)
        for i, pe in enumerate([0.95, 0.05]):
            store.append(_claves(2)[i], i, strength=1.0, priority=pe)
        rng = np.random.default_rng(0)
        victimas = [StochasticPriority().victim(store, rng) for _ in range(2000)]
        frac_sorpresiva = victimas.count(0) / len(victimas)
        # P(0) = (0.96^-1) / (0.96^-1 + 0.06^-1) ≈ 0.059
        assert 0.03 < frac_sorpresiva < 0.09

    def test_reservorio_retiene_cada_item_con_probabilidad_c_sobre_n(self):
        """Algoritmo R: la muestra es uniforme sobre todo el flujo."""
        c, n, ensayos = 3, 12, 400
        k = _claves(n, dim=6, seed=1)
        cuenta = np.zeros(n)
        for s in range(ensayos):
            mem = _memoria(Reservoir(), capacity=c, dim=6, seed=s)
            for i in range(n):
                mem.write(k[i], i)
            for v in mem.store.values:
                cuenta[v] += 1
        frecuencia = cuenta / ensayos
        assert frecuencia.mean() == pytest.approx(c / n)
        # Ni el principio ni el final del flujo están sobrerrepresentados (FIFO
        # se quedaría solo con los últimos c).
        assert frecuencia[:4].mean() == pytest.approx(c / n, abs=0.06)
        assert frecuencia[-4:].mean() == pytest.approx(c / n, abs=0.06)

    def test_cache_de_utilidad_protege_lo_usado_seguido(self):
        mem = _memoria(UtilityCache())
        k = _claves(4)
        for i in range(3):
            mem.write(k[i], i)
        for _ in range(3):
            mem.read(k[0])
        mem.read(k[2])
        mem.write(k[3], 3)
        assert 1 not in mem.store.values

    def test_cobertura_desaloja_una_del_par_mas_redundante(self):
        mem = _memoria(CoverageMax())
        k = _claves(3)
        casi_igual = k[0] + 0.01 * _claves(1, seed=9)[0]
        mem.write(k[0], "a")
        mem.write(k[1], "b")
        mem.write(k[2], "c")
        mem.write(casi_igual, "a2")
        assert "b" in mem.store.values and "c" in mem.store.values
        assert "a" not in mem.store.values  # del par, sale la más vieja


class TestBufferReal:
    def test_el_barrido_devuelve_el_primer_match_en_orden_de_insercion(self):
        sims = np.array([0.2, 0.9, 0.99, 0.95], dtype=np.float32)
        assert SequentialScan(threshold=0.85).select(sims).tolist() == [1]

    def test_sin_coincidencia_la_lectura_falla(self):
        mem = PolicyMemory(dim=8, capacity=4, genotype=EMDB_SEQUENTIAL_GENOTYPE)
        k = _claves(2)
        mem.write(k[0], "a")
        r = mem.read(k[1])
        assert r.value is None and r.index is None

    def test_retiene_exactamente_lo_mismo_que_el_proxy(self):
        """Mismo desalojo FIFO: la diferencia con el proxy es solo la lectura."""
        k = _claves(30)
        real = PolicyMemory(dim=8, capacity=5, genotype=EMDB_SEQUENTIAL_GENOTYPE)
        proxy = PolicyMemory(dim=8, capacity=5, genotype=FIFO_GENOTYPE)
        for i, kk in enumerate(k):
            real.write(kk, i, pred_error=0.3)
            proxy.write(kk, i, pred_error=0.3)
        assert real.store.values == proxy.store.values == list(range(25, 30))


class TestRango:
    def test_externo_se_inserta_y_cuenta_empates(self):
        r = rango_en_espacio(0.5, [0.9, 0.5, 0.5, 0.1], en_el_espacio=False)
        assert (r["rank_optimistic"], r["rank_pessimistic"], r["of"]) == (2, 4, 5)

    def test_punto_del_espacio_no_se_cuenta_dos_veces(self):
        r = rango_en_espacio(0.5, [0.9, 0.5, 0.5, 0.1], en_el_espacio=True)
        assert (r["rank_optimistic"], r["rank_pessimistic"], r["of"]) == (2, 3, 4)


class TestExperimento:
    def test_version_reducida_reporta_todo_con_intervalos(self):
        metodos = {n: METHODS[n] for n in ("frontier", "fifo_nn_proxy", "emdb_sequential")}
        res = comparar(
            seeds=(0, 1),
            ratios=(0.25,),
            metodos=metodos,
            evaluar_espacio=False,
            umbrales=(0.85,),
            n_jobs=1,
            verbose=False,
        )
        celda = res["exp01_cell"]
        assert set(celda) == set(metodos)
        for m in celda.values():
            lo, hi = m["battery_mean"]["ci"]
            assert lo <= m["battery_mean"]["mean"] <= hi
        # El buffer real no pasa el gate que el proxy sí pasa (H3).
        assert celda["fifo_nn_proxy"]["gate_passes"]
        assert (
            celda["emdb_sequential"]["gate_mean"]["mean"]
            < celda["fifo_nn_proxy"]["gate_mean"]["mean"]
        )
        # Mismo desalojo → misma retención de raros.
        assert (
            celda["emdb_sequential"]["rare_retention"]["per_seed"]
            == celda["fifo_nn_proxy"]["rare_retention"]["per_seed"]
        )
        assert res["ratio_sweep"][0]["r"] == pytest.approx(0.25)
        assert set(res["advantage_of_frontier"]) == {"fifo_nn_proxy", "emdb_sequential"}
