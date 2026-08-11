import pytest

from ember.core.policies import MinStrength
from ember.nas.engine import run_search
from ember.nas.space import enumerate_space


def evaluador_por_desalojo(genotype):
    """Puntaje sintético que depende de un solo eje: produce empates masivos."""
    return {"t1": 1.0 if isinstance(genotype.evict, MinStrength) else 0.0}


def test_devuelve_un_registro_por_genotipo():
    genos = list(enumerate_space())[:32]
    r = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=1, progress=False)
    assert len(r) == 32


def test_los_registros_vienen_ordenados_de_mayor_a_menor():
    genos = list(enumerate_space())[:64]
    r = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=1, progress=False)
    assert [x.mean for x in r.records] == sorted((x.mean for x in r.records), reverse=True)


def test_rank_of_devuelve_un_intervalo_no_un_entero():
    """El '#415 de 576' del piloto era un artefacto del desempate del sort."""
    genos = list(enumerate_space())[:64]
    r = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=1, progress=False)
    perdedor = next(g for g in genos if not isinstance(g.evict, MinStrength))
    optimista, pesimista = r.rank_of(perdedor)
    assert isinstance(optimista, int) and isinstance(pesimista, int)
    assert optimista < pesimista, "con empates masivos el rango tiene que ser ancho"


def test_el_intervalo_cubre_a_todos_los_empatados():
    genos = [g for g in enumerate_space() if not isinstance(g.evict, MinStrength)][:40]
    r = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=1, progress=False)
    assert r.rank_of(genos[0]) == (1, 40)


def test_is_at_floor_detecta_que_nada_puntua_estrictamente_peor():
    genos = list(enumerate_space())[:64]
    r = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=1, progress=False)
    perdedor = next(g for g in genos if not isinstance(g.evict, MinStrength))
    ganador = next(g for g in genos if isinstance(g.evict, MinStrength))
    assert r.is_at_floor(perdedor)
    assert not r.is_at_floor(ganador)


def test_el_paralelismo_no_cambia_el_resultado():
    genos = list(enumerate_space())[:48]
    a = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=1, progress=False)
    b = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=4, progress=False)
    assert [x.genotype.label() for x in a.records] == [x.genotype.label() for x in b.records]
    assert [x.mean for x in a.records] == [x.mean for x in b.records]


def test_score_of_falla_para_un_genotipo_no_evaluado():
    genos = list(enumerate_space())[:8]
    r = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=1, progress=False)
    otro = list(enumerate_space())[-1]
    with pytest.raises(KeyError):
        r.score_of(otro)


def test_per_axis_means_agrupa_por_opcion():
    genos = list(enumerate_space())[:64]
    r = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=1, progress=False)
    medias = r.per_axis_means("evict")
    assert medias["min_strength"] == pytest.approx(1.0)
    assert medias["fifo"] == pytest.approx(0.0)


def test_serializa_a_diccionario():
    genos = list(enumerate_space())[:8]
    d = run_search(evaluador_por_desalojo, genotypes=genos, n_jobs=1, progress=False).to_dict()
    assert d["n_genotypes"] == 8
    assert len(d["records"]) == 8
    assert set(d["records"][0]) == {"genotype", "label", "scores", "mean"}
