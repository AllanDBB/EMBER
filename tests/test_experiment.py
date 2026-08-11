import json

import pytest

from ember.experiment import ExperimentRun, load_results


def test_escribe_datos_y_manifiesto(tmp_path):
    with ExperimentRun("prueba", results_dir=tmp_path) as run:
        run.set_seeds([0, 1, 2])
        run.record("tabla", [{"a": 1}])

    datos = json.loads((tmp_path / "prueba" / "data.json").read_text())
    man = json.loads((tmp_path / "prueba" / "manifest.json").read_text())
    assert datos["tabla"] == [{"a": 1}]
    assert man["seeds"] == [0, 1, 2]


def test_el_manifiesto_registra_la_procedencia(tmp_path):
    with ExperimentRun("prueba", results_dir=tmp_path) as run:
        run.record("x", 1)

    man = json.loads((tmp_path / "prueba" / "manifest.json").read_text())
    for campo in (
        "git_sha",
        "git_dirty",
        "timestamp_utc",
        "python",
        "platform",
        "packages",
        "duration_s",
        "ember_version",
    ):
        assert campo in man, campo
    assert man["packages"]["numpy"]


def test_una_excepcion_no_deja_resultados_a_medias(tmp_path):
    """Un resultado parcial es peor que ninguno: parece completo."""
    with pytest.raises(RuntimeError), ExperimentRun("prueba", results_dir=tmp_path) as run:
        run.record("x", 1)
        raise RuntimeError("boom")

    assert not (tmp_path / "prueba" / "data.json").exists()


def test_las_notas_llegan_al_manifiesto(tmp_path):
    with ExperimentRun("prueba", results_dir=tmp_path) as run:
        run.note("el eje de lectura quedó al borde del umbral")
        run.record("x", 1)

    man = json.loads((tmp_path / "prueba" / "manifest.json").read_text())
    assert man["notes"] == ["el eje de lectura quedó al borde del umbral"]


def test_load_results_recupera_lo_escrito(tmp_path):
    with ExperimentRun("prueba", results_dir=tmp_path) as run:
        run.record("x", {"y": 2})

    assert load_results("prueba", results_dir=tmp_path)["x"] == {"y": 2}


def test_load_results_falla_claro_si_no_se_corrio(tmp_path):
    with pytest.raises(FileNotFoundError, match="no fue corrido"):
        load_results("inexistente", results_dir=tmp_path)
