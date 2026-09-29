"""Pruebas de `ember.data.families` y de la lógica de `exp10`.

Lo que se fija acá es que cada familia mueva la perilla que dice mover (y solo
esa), que la familia estándar sea exactamente el flujo de `exp02`, y que el
cruce y las correlaciones de rango se calculen bien.
"""

import numpy as np
import pytest

from ember.data.families import FamilySpec, family_stream, hill_number, visit_counts
from ember.data.synthetic import clustered_stream
from experiments.exp10_law_robustness import (
    VARIABLES,
    FamiliaA,
    barrer_familia,
    comparar_rankings,
    kendall_tau_b,
    localizar_cruce,
    medir_presion,
    spearman,
)


class TestFamilias:
    def test_la_familia_estandar_es_el_flujo_de_exp02_bit_a_bit(self):
        """Sin esto, la familia estándar no replicaría la Figura 1."""
        for k, c in ((5, 20), (40, 20), (8, 10)):
            a = family_stream(FamilySpec(), k, c, seed=3)
            b = clustered_stream(
                n_prototypes=k, capacity=c, n_common=20 * k, n_rare=max(5, c // 2), seed=3
            )
            assert len(a) == len(b)
            assert np.array_equal(a.keys(), b.keys())
            assert [it.pred_error for it in a] == [it.pred_error for it in b]

    def test_las_visitas_por_prototipo_se_mantienen_al_mover_k(self):
        """Si no, r queda confundido con la recurrencia, como advierte exp02."""
        f = FamilySpec(visits=10)
        for k in (5, 20, 40):
            s = family_stream(f, k, 20, seed=0)
            assert sum(not it.is_rare for it in s) == 10 * k

    def test_zipf_baja_el_numero_efectivo_de_prototipos(self):
        uniforme = family_stream(FamilySpec(), 40, 20, seed=0)
        zipf = family_stream(FamilySpec(zipf=1.5), 40, 20, seed=0)
        assert hill_number(uniforme) == pytest.approx(40, rel=0.05)
        assert hill_number(zipf) < 0.4 * 40
        assert visit_counts(zipf).max() > 5 * visit_counts(uniforme).max() / 2

    def test_la_prevalencia_de_raros_se_fija(self):
        for p in (0.01, 0.05, 0.2):
            for k in (10, 40):
                s = family_stream(FamilySpec(rare_prevalence=p), k, 20, seed=0)
                frac = len(s.rare_items) / len(s)
                assert frac == pytest.approx(p, abs=0.01)

    def test_la_deriva_aleja_al_prototipo_de_su_primera_visita(self):
        def similitud_primera_ultima(drift):
            s = family_stream(FamilySpec(drift_deg=drift, noise=0.0), 4, 20, seed=1)
            comunes = [it for it in s if not it.is_rare and it.value == 0]
            return float(comunes[0].key @ comunes[-1].key)

        assert similitud_primera_ultima(0.0) == pytest.approx(1.0, abs=1e-5)
        assert similitud_primera_ultima(60.0) < similitud_primera_ultima(15.0) < 0.999
        # La primera y la última visita no caen exactamente en t=0 y t=T-1.
        assert np.cos(np.deg2rad(60)) - 0.05 < similitud_primera_ultima(60.0) < 0.75

    def test_la_deriva_no_acerca_prototipos_distintos(self):
        """La deriva común de clustered_stream sí lo hace; esta no debe."""
        s = family_stream(FamilySpec(drift_deg=60.0, noise=0.0), 10, 20, seed=2)
        ultimos = {}
        for it in s:
            if not it.is_rare:
                ultimos[it.value] = it.key
        m = np.stack(list(ultimos.values()))
        sims = m @ m.T
        fuera = sims[~np.eye(len(m), dtype=bool)]
        assert np.abs(fuera).max() < 0.8

    def test_la_deriva_y_el_ruido_fragmentan_la_consolidacion(self):
        base = medir_presion(family_stream(FamilySpec(), 10, 20, seed=0), 20, 0)
        deriva = medir_presion(family_stream(FamilySpec(drift_deg=60.0), 10, 20, seed=0), 20, 0)
        ruido = medir_presion(family_stream(FamilySpec(noise=0.085), 10, 20, seed=0), 20, 0)
        assert base["k_eff"] == 10
        assert deriva["k_eff"] > 10
        assert ruido["k_eff"] > 10
        assert set(VARIABLES) <= set(base)

    def test_mismas_semillas_mismo_flujo(self):
        f = FamilySpec(zipf=1.0, drift_deg=30.0, rare_prevalence=0.1)
        a = family_stream(f, 12, 20, seed=7)
        b = family_stream(f, 12, 20, seed=7)
        c = family_stream(f, 12, 20, seed=8)
        assert np.array_equal(a.keys(), b.keys())
        assert not np.array_equal(a.keys(), c.keys())


class TestCruce:
    def test_interpola_en_escala_logaritmica(self):
        c = localizar_cruce(np.array([0.5, 2.0]), np.array([0.3, -0.3]))
        assert c["estado"] == "cruza"
        assert c["x_cruce"] == pytest.approx(1.0)

    def test_ordena_por_la_variable(self):
        c = localizar_cruce(np.array([2.0, 0.5, 0.25]), np.array([-0.2, 0.2, 0.5]))
        assert c["x_below"] == 0.5 and c["x_above"] == 2.0

    def test_sin_dominancia_de_escritura(self):
        assert localizar_cruce(np.array([0.5, 1.0]), np.array([-0.1, -0.2]))["estado"] == (
            "siempre_desalojo"
        )
        assert localizar_cruce(np.array([0.5, 1.0]), np.array([0.1, 0.2]))["estado"] == (
            "siempre_escritura"
        )


class TestRangos:
    def test_spearman_y_kendall_coinciden_con_scipy(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        a = rng.integers(0, 10, size=80).astype(float)  # con empates
        b = a + rng.normal(0, 3, size=80).round()
        assert spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic)
        assert kendall_tau_b(a, b) == pytest.approx(stats.kendalltau(a, b).statistic)

    def test_la_frontera_y_el_fifo_se_reportan_como_intervalo(self):
        from ember.core.genotype import FIFO_GENOTYPE
        from ember.nas.space import enumerate_space

        gs = [g for g, _ in zip(enumerate_space(), range(20), strict=False)]
        if FIFO_GENOTYPE not in gs:
            gs.append(FIFO_GENOTYPE)
        estandar = np.arange(len(gs), dtype=float)
        heldout = np.zeros(len(gs))  # todo empatado
        r = comparar_rankings(gs, estandar, heldout, gs[0])
        assert r["frontera_rango_optimista"] == 1
        assert r["frontera_rango_pesimista"] == len(gs)
        assert r["fifo_percentil"] == 0.0


@pytest.mark.slow
def test_barrer_una_familia_reducida():
    """El experimento mide lo que dice: η² por celda, cruce y variables de presión."""
    from ember.nas.space import enumerate_space

    gs = [
        g
        for g in enumerate_space()
        if g.axis("read") == "nn" and g.axis("decay") == "1.0" and g.axis("reinforce") == "0.0"
    ]
    fam = FamiliaA("estandar", "estandar", FamilySpec(), ratios=(0.25, 2.0), seeds=(0, 1))
    resumen, boot = barrer_familia(fam, genotypes=gs, n_boot=20, verbose=False)
    assert [c["K"] for c in resumen["celdas"]] == [5, 40]
    bajo, alto = resumen["celdas"]
    assert bajo["dominante"] == "write" and alto["dominante"] == "evict"
    assert resumen["cruce"]["r_nom"]["estado"] == "cruza"
    assert 0.25 < resumen["cruce"]["r_nom"]["x_cruce"] < 2.0
    assert boot.shape == (20, len(VARIABLES))
