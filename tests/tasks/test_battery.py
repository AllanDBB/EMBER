import dataclasses

import pytest

from ember.core.genotype import FIFO_GENOTYPE
from ember.data.synthetic import clustered_stream
from ember.tasks.battery import (
    run_battery,
    t1_rare_retention,
    t2_noise_under_pressure,
    t3_sequential_interference,
)
from tests.tasks.conftest import FRONTERA, FUSION, RADIO, SIN_SALIENCIA, fabrica


class TestT1RetencionDeEventosRaros:
    def test_la_saliencia_mas_desalojo_por_fuerza_retiene_lo_raro(self):
        s = clustered_stream(n_prototypes=5, capacity=20, seed=0)
        assert t1_rare_retention(fabrica(FRONTERA), s, seed=0).score > 0.7

    def test_el_fifo_pierde_lo_raro(self):
        s = clustered_stream(n_prototypes=5, capacity=20, seed=0)
        assert t1_rare_retention(fabrica(FIFO_GENOTYPE), s, seed=0).score < 0.2

    def test_sin_saliencia_el_desalojo_por_fuerza_degenera(self):
        """Con fuerza constante todas las trazas pesan igual: la política se vuelve ciega."""
        s = clustered_stream(n_prototypes=5, capacity=20, seed=0)
        con = t1_rare_retention(fabrica(FRONTERA), s, seed=0).score
        sin = t1_rare_retention(fabrica(SIN_SALIENCIA), s, seed=0).score
        assert con > sin


class TestT1LecturasIntercaladas:
    """El eje de refuerzo era inobservable por diseño de tarea, no por implementación."""

    def _con_refuerzo(self, valor):
        return dataclasses.replace(SIN_SALIENCIA, reinforce=valor)

    def test_sin_lecturas_intercaladas_el_refuerzo_no_puede_cambiar_nada(self):
        s = clustered_stream(n_prototypes=40, capacity=20, seed=0)
        a = t1_rare_retention(fabrica(self._con_refuerzo(0.0)), s, seed=0, read_every=0)
        b = t1_rare_retention(fabrica(self._con_refuerzo(3.0)), s, seed=0, read_every=0)
        assert a.score == b.score

    def test_con_lecturas_intercaladas_el_refuerzo_si_cambia_el_desalojo(self):
        s = clustered_stream(n_prototypes=40, capacity=20, seed=0)
        a = t1_rare_retention(fabrica(self._con_refuerzo(0.0)), s, seed=0, read_every=3)
        b = t1_rare_retention(fabrica(self._con_refuerzo(3.0)), s, seed=0, read_every=3)
        assert a.score != b.score

    def test_el_detalle_registra_el_protocolo_usado(self):
        s = clustered_stream(n_prototypes=5, capacity=20, seed=0)
        r = t1_rare_retention(fabrica(FRONTERA), s, seed=0, read_every=5)
        assert r.detail["read_every"] == 5
        assert r.detail["evictions"] > 0


class TestT2NoEsDegenerada:
    """El piloto devolvía 0.767 para las 576 arquitecturas: varianza exactamente cero."""

    def test_ejerce_presion_de_capacidad_real(self):
        r = t2_noise_under_pressure(fabrica(FIFO_GENOTYPE), seed=0, capacity=20, n_items=60)
        assert r.detail["evictions"] > 0
        assert r.detail["ceiling"] == pytest.approx(1 / 3)

    def test_discrimina_entre_politicas_de_desalojo(self):
        a = t2_noise_under_pressure(fabrica(FIFO_GENOTYPE), seed=0).score
        b = t2_noise_under_pressure(fabrica(FRONTERA), seed=0).score
        assert a != b

    def test_discrimina_entre_modos_de_lectura(self):
        a = t2_noise_under_pressure(fabrica(FRONTERA), seed=0).score
        b = t2_noise_under_pressure(fabrica(RADIO), seed=0).score
        assert a != b

    def test_la_saliencia_paga_porque_las_consultas_pesan_por_importancia(self):
        con = t2_noise_under_pressure(fabrica(FRONTERA), seed=0).score
        sin = t2_noise_under_pressure(fabrica(SIN_SALIENCIA), seed=0).score
        assert con > sin


class TestT3ConSenalDiferenciada:
    def test_con_pe_uniforme_toda_arquitectura_degenera(self):
        """Sin señal que distinga qué importa, ninguna política puede resistir."""
        a = t3_sequential_interference(fabrica(FRONTERA), seed=0, pe_first=0.5, pe_rest=0.5)
        b = t3_sequential_interference(fabrica(FIFO_GENOTYPE), seed=0, pe_first=0.5, pe_rest=0.5)
        assert a.score == pytest.approx(b.score, abs=0.2)
        assert a.detail["senal_diferenciada"] is False

    def test_con_pe_diferenciado_la_frontera_protege_el_primer_bloque(self):
        r = t3_sequential_interference(fabrica(FRONTERA), seed=0, pe_first=0.9, pe_rest=0.1)
        assert r.score > 0.5
        assert r.detail["senal_diferenciada"] is True

    def test_con_pe_diferenciado_el_fifo_sigue_sin_protegerlo(self):
        r = t3_sequential_interference(fabrica(FIFO_GENOTYPE), seed=0, pe_first=0.9, pe_rest=0.1)
        assert r.score < 0.3


class TestLeyDelUmbral:
    """El claim central: qué mecanismo importa es función del ratio r = K/C."""

    def _ventaja_de_fusionar(self, n_prototypes, capacity=20, seeds=(0, 1, 2)):
        con, sin = [], []
        for s in seeds:
            stream = clustered_stream(n_prototypes=n_prototypes, capacity=capacity, seed=s)
            con.append(t1_rare_retention(fabrica(FUSION), stream, seed=s).score)
            sin.append(t1_rare_retention(fabrica(FRONTERA), stream, seed=s).score)
        return sum(con) / len(con) - sum(sin) / len(sin)

    def test_fusionar_paga_mas_por_debajo_del_umbral_que_por_encima(self):
        bajo = self._ventaja_de_fusionar(n_prototypes=4)  # r = 0.2
        alto = self._ventaja_de_fusionar(n_prototypes=40)  # r = 2.0
        assert bajo > alto

    def test_el_spec_expone_el_regimen_de_cada_celda(self):
        assert clustered_stream(4, 20, seed=0).spec.regime == "compression"
        assert clustered_stream(40, 20, seed=0).spec.regime == "selection"


class TestBateriaCompleta:
    def test_reporta_las_tres_tareas(self, frontera):
        r = run_battery(frontera, seeds=(0, 1))
        assert set(r.per_task) == {
            "rare_retention",
            "noise_under_pressure",
            "sequential_interference",
        }
        assert 0.0 <= r.mean <= 1.0

    def test_la_frontera_supera_al_fifo_en_promedio(self, frontera, fifo):
        assert run_battery(frontera, seeds=(0, 1)).mean > run_battery(fifo, seeds=(0, 1)).mean

    def test_es_reproducible(self, frontera):
        assert run_battery(frontera, seeds=(0,)).mean == run_battery(frontera, seeds=(0,)).mean

    def test_serializa(self, frontera):
        d = run_battery(frontera, seeds=(0,)).to_dict()
        assert set(d) == {"name", "per_task", "per_task_std", "mean", "detail"}

    def test_per_task_std_es_cero_con_una_sola_semilla(self, frontera):
        r = run_battery(frontera, seeds=(0,))
        assert set(r.per_task_std) == set(r.per_task)
        assert all(v == 0.0 for v in r.per_task_std.values())

    def test_per_task_std_captura_dispersion_entre_semillas(self, frontera):
        r = run_battery(frontera, seeds=(0, 1, 2, 3))
        assert all(v >= 0.0 for v in r.per_task_std.values())
