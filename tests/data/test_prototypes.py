import numpy as np
import pytest

from ember.data.prototypes import estimate_n_prototypes
from ember.data.synthetic import clustered_stream


@pytest.mark.parametrize("k_real", [3, 5, 12])
def test_recupera_el_numero_de_prototipos_conocido(k_real):
    """Validación del estimador contra terreno conocido, antes de usarlo a ciegas."""
    s = clustered_stream(
        n_prototypes=k_real, capacity=20, n_common=400, n_rare=0, noise=0.05, seed=0
    )
    est = estimate_n_prototypes(s.keys(), k_max=24, seed=0)
    assert est.ci_low <= k_real <= est.ci_high


def test_el_intervalo_de_confianza_contiene_la_estimacion_puntual():
    s = clustered_stream(n_prototypes=6, capacity=20, n_common=300, n_rare=0, seed=0)
    est = estimate_n_prototypes(s.keys(), k_max=20, seed=0)
    assert est.ci_low <= est.k_hat <= est.ci_high
    assert est.ci == (est.ci_low, est.ci_high)


def test_es_determinista_bajo_la_misma_semilla():
    keys = clustered_stream(n_prototypes=5, capacity=20, n_rare=0, seed=0).keys()
    a = estimate_n_prototypes(keys, k_max=16, seed=1)
    b = estimate_n_prototypes(keys, k_max=16, seed=1)
    assert a.k_hat == b.k_hat and a.ci == b.ci


def test_datos_sin_estructura_se_reportan_como_sin_estructura():
    """Ruido isotrópico no tiene prototipos: el estimador no debe inventarlos.

    El argmax de la curva de silueta siempre devuelve *algún* k. Lo que distingue
    una estimación de una alucinación es la altura de la silueta en ese k, no el
    k mismo.
    """
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((300, 32)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)

    est = estimate_n_prototypes(keys, k_max=32, seed=0)
    assert not est.has_structure
    assert est.best_score < 0.15


def test_datos_con_estructura_se_reportan_como_con_estructura():
    s = clustered_stream(n_prototypes=6, capacity=20, n_common=400, n_rare=0, seed=0)
    est = estimate_n_prototypes(s.keys(), k_max=24, seed=0)
    assert est.has_structure
    assert est.best_score >= 0.15


def test_el_intervalo_es_mas_ancho_cuando_no_hay_estructura():
    rng = np.random.default_rng(0)
    ruido = rng.standard_normal((300, 32)).astype(np.float32)
    ruido /= np.linalg.norm(ruido, axis=1, keepdims=True)
    sin = estimate_n_prototypes(ruido, k_max=32, seed=0)

    s = clustered_stream(n_prototypes=6, capacity=20, n_common=300, n_rare=0, seed=0)
    con = estimate_n_prototypes(s.keys(), k_max=32, seed=0)

    assert (sin.ci_high - sin.ci_low) > (con.ci_high - con.ci_low)


def test_mas_prototipos_produce_una_estimacion_mayor():
    """Monotonía: es la propiedad que la ley del umbral necesita del estimador."""

    def estimar(k):
        s = clustered_stream(n_prototypes=k, capacity=20, n_common=400, n_rare=0, seed=0)
        return estimate_n_prototypes(s.keys(), k_max=48, seed=0).k_hat

    assert estimar(3) < estimar(20)


def test_conjuntos_diminutos_no_rompen_el_estimador():
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((3, 32)).astype(np.float32)
    est = estimate_n_prototypes(keys, seed=0)
    assert est.k_hat == 3
