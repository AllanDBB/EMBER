"""Contabilidad de bytes y piezas de `exp12_substrate_cost` en tamaño reducido."""

import numpy as np
import pytest

from ember.footprint import (
    ARQUITECTURAS_CONTABLES,
    bytes_analiticos,
    capacidad_para_presupuesto,
    costo_fijo_y_por_traza,
)
from experiments.exp12_substrate_cost import (
    derivar_config,
    diagnostico_spiking_sdm,
    diferencia_pareada,
    evaluar_configs,
    ic_bootstrap,
    medir_una,
    pendiente_loglog,
    verificar_contabilidad,
)

RAPIDAS = [n for n in ARQUITECTURAS_CONTABLES if n != "Spiking"]


class TestContabilidad:
    @pytest.mark.parametrize("nombre", RAPIDAS)
    @pytest.mark.parametrize(("C", "d"), [(3, 16), (20, 32), (7, 64)])
    def test_bytes_reales_coinciden_con_la_formula(self, nombre, C, d):
        v = verificar_contabilidad(nombre, C, d)
        assert v["coincide"], v["por_componente"]
        assert v["arreglos_no_inventariados"] == []

    @pytest.mark.slow
    def test_bytes_reales_coinciden_con_la_formula_spiking(self):
        v = verificar_contabilidad("Spiking", 4, 32)
        assert v["coincide"] and v["arreglos_no_inventariados"] == []

    def test_la_formula_es_afin_en_C(self):
        for n in ARQUITECTURAS_CONTABLES:
            fijo, por_traza = costo_fijo_y_por_traza(n, 32)
            assert bytes_analiticos(n, 17, 32)["total"] == pytest.approx(fijo + 17 * por_traza)

    def test_la_sdm_por_defecto_paga_131_kib_fijos(self):
        """El punto de la crítica: 512 direcciones y 512 contadores de d = 32 en float32."""
        assert costo_fijo_y_por_traza("SDM", 32) == (2 * 512 * 32 * 4, 152.0)
        assert costo_fijo_y_por_traza("FIFO", 32) == (0, 152.0)

    @pytest.mark.parametrize("B", [3000, 4096, 150_000])
    def test_la_capacidad_derivada_es_la_maxima_que_entra(self, B):
        for n in ("FIFO", "SDM", "SpikingSDM"):
            C = capacidad_para_presupuesto(n, B, 32)
            if C == 0:
                assert costo_fijo_y_por_traza(n, 32)[0] > B - 152
                continue
            assert bytes_analiticos(n, C, 32)["total"] <= B
            assert bytes_analiticos(n, C + 1, 32)["total"] > B


class TestDerivacion:
    def test_con_valores_por_defecto_la_sdm_no_entra_en_4_kib(self):
        assert not derivar_config("SDM", 4096, "defaults")["factible"]
        assert derivar_config("FIFO", 4096, "defaults")["capacity"] == 4096 // 152

    def test_el_modo_escalado_reserva_a_lo_sumo_la_mitad_para_el_sustrato(self):
        for B in (4096, 65536, 524288):
            d = derivar_config("SDM", B, "sustrato_escalado")
            fijo, _ = costo_fijo_y_por_traza("SDM", 32, n_hard=d["hp"]["n_hard"])
            assert fijo <= B // 2
            assert d["bytes"] <= B

    def test_spiking_no_entra_al_modo_escalado(self):
        assert not derivar_config("Spiking", 524288, "sustrato_escalado")["factible"]


class TestEstadistica:
    def test_ic_bootstrap_contiene_la_media(self):
        v = [0.1, 0.5, 0.3, 0.7, 0.4]
        lo, hi = ic_bootstrap(v)
        assert lo <= np.mean(v) <= hi

    def test_diferencia_pareada_resuelve_un_efecto_consistente(self):
        d = diferencia_pareada([0.9, 0.8, 0.85, 0.95], [0.1, 0.2, 0.15, 0.05])
        assert d["resuelta"] and d["diff"] > 0

    def test_pendiente_loglog_recupera_una_ley_de_potencias(self):
        x = [10, 20, 40, 80]
        assert pendiente_loglog(x, [3 * xi**1.5 for xi in x]) == pytest.approx(1.5)
        assert pendiente_loglog(x, [1, 1, 1, 8], cola=2) == pytest.approx(3.0)


class TestMedicion:
    def test_medir_una_devuelve_tiempos_positivos(self):
        tw, tr = medir_una("SDM", 10, 32, seed=0, n_medidas=5, n_calentamiento=2)
        assert tw.shape == tr.shape == (5,)
        assert (tw > 0).all() and (tr > 0).all()

    def test_diagnostico_spiking_sdm_reporta_una_fraccion(self):
        d = diagnostico_spiking_sdm(dims=(32,), n=5)
        assert 0.0 <= d["d32"]["fraccion_respaldo"] <= 1.0

    def test_evaluacion_reducida_con_presupuesto(self):
        """Capacidad fija: la memoria no crece aunque la tarea pida más."""
        res = evaluar_configs(
            {
                "fifo2": {"nombre": "FIFO", "hp": None, "capacity": 2, "bateria": True},
                "sdm": {"nombre": "SDM", "hp": {"n_hard": 64}, "capacity": None, "bateria": True},
            },
            seeds=(0, 1),
            n_jobs=1,
        )
        assert res["fifo2"]["n_semillas"] == 2
        # Con dos ranuras no puede reconstruir 15 patrones sin presión.
        assert res["fifo2"]["gate"]["noise_robustness"] < 0.3
        assert 0.0 <= res["sdm"]["battery_mean"] <= 1.0
        lo, hi = res["sdm"]["battery_ci"]
        assert lo <= res["sdm"]["battery_mean"] <= hi
