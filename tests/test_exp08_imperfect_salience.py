"""Pruebas de `exp08_imperfect_salience` en tamaño reducido.

Fijan que el experimento mida lo que dice: que la condición limpia reproduzca
`exp01`, que la precisión cuente lo que hay en la memoria, que el oráculo sea
techo y que el resumen tenga las claves que usa el paper.
"""

import json

import pytest

from ember.core.genotype import FIFO_GENOTYPE
from ember.data.salience import SignalCondition
from ember.nas.space import enumerate_space
from experiments._common import make_factory
from experiments.exp08_imperfect_salience import (
    FRONTIER,
    ORACLE,
    SETTINGS,
    condiciones_unicas,
    correr,
    flujo,
    medir_retencion,
)

MAIN = SETTINGS["main"]


def test_la_condicion_limpia_reproduce_exp01():
    """exp01 reporta 0.9833 de rare_retention para la frontera con semillas 0-2."""
    rec = [
        medir_retencion(make_factory(FRONTIER), flujo(MAIN, SignalCondition(), s), s)[0]
        for s in (0, 1, 2)
    ]
    assert sum(rec) / 3 == pytest.approx(0.9833333333333334)


def test_la_precision_cuenta_lo_que_hay_en_memoria():
    m = medir_retencion(make_factory(FIFO_GENOTYPE), flujo(MAIN, SignalCondition(), 0), 0)
    recall, recall_set, precision, f1, n = m
    assert n == 20
    assert 0.0 <= precision <= 1.0
    # Con 20 raros en 20 ranuras y la memoria llena, precisión = recall de conjunto.
    assert precision == pytest.approx(recall_set)


def test_el_oraculo_retiene_todo_en_ambos_regimenes():
    for st in SETTINGS.values():
        f = flujo(st, SignalCondition(auc=0.5), 0, oracle=True)
        assert medir_retencion(make_factory(ORACLE), f, 0)[0] == 1.0


def test_condiciones_unicas_no_repite_la_limpia():
    ejes = {"a": (SignalCondition(), SignalCondition(noise=0.5)), "b": (SignalCondition(),)}
    assert [c.label() for c in condiciones_unicas(ejes)] == ["clean", "noise050"]


@pytest.mark.slow
def test_corrida_reducida_produce_el_resumen():
    genos = [
        g
        for g in enumerate_space()
        if g.axis("read") == "nn" and g.axis("decay") == "1.0" and g.axis("reinforce") == "0.0"
    ]
    ejes = {"main": {"overlap": (SignalCondition(), SignalCondition(auc=0.5))}}
    datos, _ = correr(
        ejes, seeds=(0, 1), genotipos=genos, arquitecturas=("FIFO",), n_jobs=1, log=lambda *_: None
    )
    d = datos["main"]
    json.dumps(d)
    limpia = d["conditions"]["clean"]
    assert limpia["auc"]["mean"] == 1.0
    assert limpia["policies"]["oracle"]["recall"]["mean"] == 1.0
    assert (
        limpia["policies"]["frontier"]["recall"]["mean"]
        > limpia["policies"]["fifo"]["recall"]["mean"]
    )
    assert d["conditions"]["auc500"]["auc"]["mean"] < 0.7
    assert "frontier_vs_fifo" in d["breakdown"]["overlap"]
    assert limpia["search"] is None  # subconjunto: no hay búsqueda completa
