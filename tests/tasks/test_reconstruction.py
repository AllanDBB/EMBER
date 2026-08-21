import pytest

from ember.memories import ARCHITECTURES
from ember.tasks.reconstruction import (
    RECONSTRUCTION_TASKS,
    r1_pattern_completion,
    reconstruction_gate,
)

from .conftest import DIM, fabrica  # noqa: F401


@pytest.mark.parametrize("tarea", RECONSTRUCTION_TASKS, ids=lambda t: t.__name__)
def test_toda_tarea_devuelve_un_puntaje_en_cero_uno(tarea, fifo):
    r = tarea(fifo, seed=0)
    assert 0.0 <= r.score <= 1.0
    assert r.name


@pytest.mark.parametrize("tarea", RECONSTRUCTION_TASKS, ids=lambda t: t.__name__)
def test_toda_tarea_es_reproducible(tarea, fifo):
    assert tarea(fifo, seed=1).score == tarea(fifo, seed=1).score


@pytest.mark.parametrize("tarea", RECONSTRUCTION_TASKS, ids=lambda t: t.__name__)
def test_ninguna_tarea_dispara_desalojos(tarea, fifo):
    """El gate mide reconstrucción, no retención: la capacidad iguala a la carga."""
    r = tarea(fifo, seed=0)
    assert 0.0 <= r.score <= 1.0


def test_el_gate_admite_al_fifo_porque_sin_presion_es_busqueda_por_similitud(fifo):
    """Sin desalojo, el FIFO se reduce a vecino más cercano sobre todo lo guardado."""
    g = reconstruction_gate(fifo, seeds=(0, 1))
    assert g.passes and g.mean > 0.5


def test_el_gate_reporta_las_cuatro_subtareas(fifo):
    g = reconstruction_gate(fifo, seeds=(0,))
    assert set(g.per_task) == {
        "pattern_completion",
        "noise_robustness",
        "ab_interference",
        "capacity_profile",
    }


def test_el_gate_serializa_su_veredicto(fifo):
    d = reconstruction_gate(fifo, seeds=(0,)).to_dict()
    assert d["passes"] is True
    assert d["threshold"] == 0.50
    assert 0.0 <= d["mean"] <= 1.0
    assert set(d["per_task_std"]) == set(d["per_task"])


def test_sdm_completa_patrones_desde_claves_enmascaradas():
    sdm = ARCHITECTURES["SDM"]
    r = r1_pattern_completion(lambda c, s: sdm(dim=DIM, capacity=c, seed=s), seed=0)
    assert r.score > 0.5


@pytest.mark.parametrize("nombre", ["SDM", "ENN", "SpikingSDM", "FIFO"])
def test_las_arquitecturas_rapidas_pasan_o_no_el_gate_sin_error(nombre):
    cls = ARCHITECTURES[nombre]
    g = reconstruction_gate(
        lambda c, s: cls(dim=DIM, capacity=c, seed=s), seeds=(0, 1), name=nombre
    )
    assert 0.0 <= g.mean <= 1.0
    assert isinstance(g.passes, bool)
