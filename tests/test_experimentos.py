"""Pruebas de los experimentos.

Corren versiones diminutas de cada barrido: lo que se fija acá es que el
experimento *mida lo que dice medir*, no el valor concreto que produce con sus
parámetros de publicación. Los valores publicados viven en `results/` y los
verifica `ember.paper_sync`.
"""

import numpy as np
import pytest

from ember.data.streams import Stream, StreamItem, StreamSpec
from ember.nas.space import AXES
from experiments.exp02_threshold_grid import barrer_grilla, localizar_umbral
from experiments.exp03_axis_liveness import genotipos_a_auditar, medir_liveness
from experiments.exp04_arch_benchmark import correr_benchmark
from experiments.exp06_minigrid import comparar, separacion_de_saliencia


def _hay_minigrid() -> bool:
    from importlib.util import find_spec

    return find_spec("minigrid") is not None


class TestAuditoriaDeEjes:
    def test_el_muestreo_por_hermanos_cubre_todas_las_opciones_de_cada_eje(self):
        """Un prefijo de la enumeración no las cubriría: los ejes lentos no varían."""
        from ember.nas.space import SEARCH_SPACE

        genos = genotipos_a_auditar(n_bases=8)
        for eje in AXES:
            vistos = {g.axis(eje) for g in genos}
            assert len(vistos) == len(SEARCH_SPACE[eje]), eje

    def test_el_espacio_completo_devuelve_los_576(self):
        assert len(genotipos_a_auditar(n_bases=None)) == 576

    @pytest.mark.slow
    def test_ningun_eje_es_inobservable(self):
        """El addendum encontró read y reinforce muertos. No pueden volver."""
        liveness = medir_liveness(seeds=(0,), n_bases=24)
        muertos = [eje for eje, v in liveness.items() if v < 1e-9]
        assert not muertos, f"ejes inobservables: {muertos}"

    @pytest.mark.slow
    def test_reporta_los_seis_ejes(self):
        assert set(medir_liveness(seeds=(0,), n_bases=8)) == set(AXES)


class TestGrillaDelUmbral:
    @pytest.mark.slow
    def test_cada_celda_reporta_el_ratio_y_sus_intervalos(self):
        celdas = barrer_grilla(capacities=(20,), ratios=(0.25, 2.0), seeds=(0, 1), verbose=False)
        assert len(celdas) == 2
        for c in celdas:
            assert c["r"] == pytest.approx(c["n_prototypes"] / c["capacity"])
            assert c["eta2_write_ci"][0] <= c["eta2_write"] <= c["eta2_write_ci"][1]
            assert c["eta2_evict_ci"][0] <= c["eta2_evict"] <= c["eta2_evict_ci"][1]

    @pytest.mark.slow
    def test_las_visitas_por_prototipo_se_mantienen_constantes(self):
        """Si no, el ratio queda confundido con la recurrencia del flujo."""
        celdas = barrer_grilla(capacities=(20,), ratios=(0.25, 2.0), seeds=(0,), verbose=False)
        assert len({c["visitas_por_prototipo"] for c in celdas}) == 1
        for c in celdas:
            assert c["n_common"] == c["visitas_por_prototipo"] * c["n_prototypes"]

    @pytest.mark.slow
    def test_la_escritura_domina_bajo_el_umbral_y_el_desalojo_por_encima(self):
        """El claim central del paper, en su forma mínima verificable."""
        celdas = barrer_grilla(capacities=(20,), ratios=(0.25, 2.0), seeds=(0, 1), verbose=False)
        bajo = next(c for c in celdas if c["r"] < 1.0)
        alto = next(c for c in celdas if c["r"] > 1.0)
        assert bajo["dominante"] == "write"
        assert alto["dominante"] == "evict"
        assert bajo["eta2_write"] > alto["eta2_write"]
        assert alto["eta2_evict"] > bajo["eta2_evict"]

    def test_localizar_umbral_encuentra_el_cruce(self):
        celdas = [
            {"capacity": 20, "r": 0.5, "dominante": "write"},
            {"capacity": 20, "r": 2.0, "dominante": "evict"},
        ]
        u = localizar_umbral(celdas, 20)
        assert u is not None
        assert u["r_below"] == 0.5 and u["r_above"] == 2.0

    def test_localizar_umbral_devuelve_none_si_no_hay_cruce(self):
        celdas = [
            {"capacity": 20, "r": 0.5, "dominante": "write"},
            {"capacity": 20, "r": 2.0, "dominante": "write"},
        ]
        assert localizar_umbral(celdas, 20) is None


class TestBenchmarkDeArquitecturas:
    @pytest.mark.slow
    def test_reporta_gate_y_bateria_para_cada_arquitectura(self):
        r = correr_benchmark(seeds=(0,), incluir_lentas=False, verbose=False)
        assert set(r) >= {"SDM", "ENN", "SpikingSDM", "FIFO"}
        for datos in r.values():
            assert "gate" in datos
            if datos["gate"]["passes"]:
                assert "battery" in datos

    @pytest.mark.slow
    def test_el_fifo_es_el_piso_en_retencion_de_eventos_raros(self):
        """Esta afirmación del draft sí sobrevive entre arquitecturas."""
        r = correr_benchmark(seeds=(0, 1), incluir_lentas=False, verbose=False)
        admitidas = {k: v for k, v in r.items() if v["gate"]["passes"]}
        raras = {k: v["battery"]["per_task"]["rare_retention"] for k, v in admitidas.items()}
        assert raras["FIFO"] == min(raras.values())

    @pytest.mark.slow
    def test_sdm_supera_al_fifo_en_retencion_de_eventos_raros(self):
        r = correr_benchmark(seeds=(0, 1), incluir_lentas=False, verbose=False)
        assert (
            r["SDM"]["battery"]["per_task"]["rare_retention"]
            > r["FIFO"]["battery"]["per_task"]["rare_retention"]
        )

    @pytest.mark.slow
    def test_el_gate_admite_a_todas_menos_al_spiking_puro(self):
        """Sin presión de capacidad, hasta el FIFO reconstruye bien."""
        r = correr_benchmark(seeds=(0,), incluir_lentas=False, verbose=False)
        assert all(d["gate"]["passes"] for d in r.values())


class TestSeparacionDeSaliencia:
    """`separacion_de_saliencia` no depende de MiniGrid: opera sobre `Stream`."""

    def _stream(self, raros: list[float], comunes: list[float]) -> Stream:
        items = [
            StreamItem(key=np.zeros(4, dtype=np.float32), value=i, pred_error=pe, is_rare=True)
            for i, pe in enumerate(raros)
        ] + [
            StreamItem(
                key=np.zeros(4, dtype=np.float32),
                value=f"c{i}",
                pred_error=pe,
                is_rare=False,
            )
            for i, pe in enumerate(comunes)
        ]
        spec = StreamSpec(n_prototypes=0, capacity=20, dim=4, source="test")
        return Stream(items=items, spec=spec, rare_items=items[: len(raros)])

    def test_sin_solapamiento_la_fraccion_es_cero(self):
        s = self._stream(raros=[0.9, 0.9], comunes=[0.1, 0.1, 0.1])
        r = separacion_de_saliencia([s])
        assert r["fraccion_comunes_sobre_minimo_raro"] == 0.0

    def test_con_solapamiento_total_la_fraccion_es_uno(self):
        s = self._stream(raros=[0.5], comunes=[0.5, 0.9, 0.6])
        r = separacion_de_saliencia([s])
        assert r["fraccion_comunes_sobre_minimo_raro"] == 1.0

    def test_sin_eventos_raros_no_calcula_solapamiento(self):
        s = self._stream(raros=[], comunes=[0.5, 0.5])
        r = separacion_de_saliencia([s])
        assert r["n_raros"] == 0
        assert "fraccion_comunes_sobre_minimo_raro" not in r


@pytest.mark.slow
@pytest.mark.skipif(not _hay_minigrid(), reason="requiere el extra envs")
class TestMiniGrid:
    def test_comparar_corre_de_punta_a_punta(self):
        """Fija que main() no rompa al orquestar: el bug de exp05 era exactamente esto."""
        r = comparar(seeds=(0, 1, 2), n_steps=200, verbose=False)
        assert r["n_seeds"] == 3
        assert set(r["resultados"]) == {"frontera", "FIFO"}
        for datos in r["resultados"].values():
            assert 0.0 <= datos["tasa"] <= 1.0

    def test_es_reproducible(self):
        a = comparar(seeds=(0, 1), n_steps=150, verbose=False)
        b = comparar(seeds=(0, 1), n_steps=150, verbose=False)
        assert a == b
