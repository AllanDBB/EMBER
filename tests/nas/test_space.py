import math

from ember.core.genotype import FIFO_GENOTYPE
from ember.nas.space import (
    AXES,
    BIOLOGICAL_ROOT,
    SEARCH_SPACE,
    axis_options,
    enumerate_space,
    siblings,
    space_size,
)


def test_el_espacio_tiene_576_genotipos():
    assert space_size() == 576
    assert len(list(enumerate_space())) == 576


def test_el_producto_de_las_cardinalidades_da_576():
    assert math.prod(len(v) for v in SEARCH_SPACE.values()) == 576


def test_no_hay_genotipos_duplicados():
    todos = list(enumerate_space())
    assert len(set(todos)) == len(todos)


def test_el_genotipo_fifo_esta_en_el_espacio():
    """El incumbente de e-MDB es un punto del espacio, no un baseline externo."""
    assert FIFO_GENOTYPE in set(enumerate_space())


def test_los_seis_ejes_estan_presentes():
    assert set(AXES) == {"read", "write", "strength", "decay", "evict", "reinforce"}


def test_cada_eje_declara_su_raiz_biologica():
    assert set(BIOLOGICAL_ROOT) == set(AXES)
    assert all(BIOLOGICAL_ROOT[a] for a in AXES)


def test_las_etiquetas_de_cada_eje_son_unicas():
    for eje in AXES:
        opciones = axis_options(eje)
        assert len(set(opciones)) == len(opciones), eje


def test_los_hermanos_difieren_solo_en_el_eje_pedido():
    g = FIFO_GENOTYPE
    for hermano in siblings(g, "evict"):
        assert hermano.evict != g.evict
        for otro in AXES:
            if otro != "evict":
                assert hermano.axis(otro) == g.axis(otro)


def test_hay_un_hermano_menos_que_opciones_del_eje():
    assert len(siblings(FIFO_GENOTYPE, "evict")) == len(SEARCH_SPACE["evict"]) - 1


def test_la_enumeracion_es_estable_entre_llamadas():
    a = [g.label() for g in enumerate_space()]
    b = [g.label() for g in enumerate_space()]
    assert a == b
