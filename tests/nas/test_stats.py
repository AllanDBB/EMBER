import pytest

from ember.core.policies import Constant, MinStrength
from ember.nas.engine import SearchRecord
from ember.nas.space import enumerate_space
from ember.nas.stats import (
    axis_liveness,
    bootstrap_ci,
    conditional_effect,
    eta_squared,
    main_effects_table,
    partial_eta_squared,
    sums_of_squares,
    variance_share,
)


def registros(fn):
    return [SearchRecord(genotype=g, scores={"m": fn(g)}, mean=fn(g)) for g in enumerate_space()]


def solo_desalojo(g):
    return 1.0 if isinstance(g.evict, MinStrength) else 0.0


def saliencia_compuertada(g):
    """La fuerza solo se lee bajo desalojo por mínima fuerza. Tres cuartos del espacio la ignoran."""
    if not isinstance(g.evict, MinStrength):
        return 0.5
    return 0.0 if isinstance(g.strength, Constant) else 1.0


class TestEfectosPrincipales:
    def test_eta2_es_1_cuando_un_solo_eje_explica_todo(self):
        assert eta_squared(registros(solo_desalojo), "evict") == pytest.approx(1.0)

    def test_eta2_es_0_cuando_el_eje_no_influye(self):
        assert eta_squared(registros(solo_desalojo), "read") == pytest.approx(0.0, abs=1e-9)

    def test_eta2_es_0_si_no_hay_varianza(self):
        assert eta_squared(registros(lambda g: 0.7), "evict") == 0.0


class TestObservabilidadDeEjes:
    def test_detecta_un_eje_muerto(self):
        """Reproduce el hallazgo del addendum: diferencia máxima exactamente 0.000."""
        recs = registros(solo_desalojo)
        assert axis_liveness(recs, "read") == pytest.approx(0.0)
        assert axis_liveness(recs, "reinforce") == pytest.approx(0.0)

    def test_detecta_un_eje_vivo(self):
        assert axis_liveness(registros(solo_desalojo), "evict") == pytest.approx(1.0)

    def test_un_eje_puede_estar_vivo_solo_en_parte_del_espacio(self):
        """La saliencia está muerta en tres cuartos del espacio y viva en el otro."""
        assert axis_liveness(registros(saliencia_compuertada), "strength") == pytest.approx(1.0)


class TestEfectoCondicional:
    def test_revela_lo_que_el_efecto_principal_esconde(self):
        """El caso de estudio: 0.2 % de efecto principal, factor 35 condicionado."""
        recs = registros(saliencia_compuertada)

        assert eta_squared(recs, "strength") < 0.30

        cond = conditional_effect(recs, "strength", given={"evict": "min_strength"})
        assert max(cond.values()) - min(cond.values()) == pytest.approx(1.0)

    def test_sin_condicion_equivale_a_las_medias_marginales(self):
        recs = registros(solo_desalojo)
        cond = conditional_effect(recs, "evict", given={})
        assert cond["min_strength"] == pytest.approx(1.0)
        assert cond["fifo"] == pytest.approx(0.0)

    def test_falla_si_la_condicion_no_selecciona_nada(self):
        with pytest.raises(ValueError, match="ningún genotipo"):
            conditional_effect(registros(solo_desalojo), "read", given={"evict": "inexistente"})


class TestAnovaFactorial:
    def test_incluye_interacciones_de_dos_factores(self):
        assert "strength×evict" in variance_share(registros(saliencia_compuertada))

    def test_la_interaccion_supera_al_efecto_principal_de_la_saliencia(self):
        """Es la tesis metodológica: con mecanismos compuertados, mirar principales engaña."""
        r = variance_share(registros(saliencia_compuertada))
        assert r["strength×evict"] > r["strength"]

    def test_las_fracciones_de_varianza_suman_como_mucho_uno(self):
        r = variance_share(registros(saliencia_compuertada))
        assert all(0.0 <= v <= 1.0 for v in r.values())
        assert sum(r.values()) <= 1.0 + 1e-9

    def test_la_descomposicion_cierra_contra_la_varianza_total(self):
        recs = registros(saliencia_compuertada)
        ss, ss_error, ss_total = sums_of_squares(recs)
        assert sum(ss.values()) + ss_error == pytest.approx(ss_total)

    def test_el_eta_parcial_satura_sin_varianza_residual(self):
        """Documenta por qué la tabla del paper usa variance_share y no eta parcial."""
        recs = registros(saliencia_compuertada)
        _, ss_error, _ = sums_of_squares(recs)
        assert ss_error == pytest.approx(0.0)
        assert partial_eta_squared(recs)["strength×evict"] == pytest.approx(1.0)

    def test_el_eta_parcial_discrimina_cuando_hay_residuo(self):
        import numpy as np

        rng = np.random.default_rng(0)
        recs = registros(lambda g: saliencia_compuertada(g))
        ruidosos = [
            SearchRecord(
                genotype=r.genotype,
                scores={"m": r.mean + float(rng.normal(0, 0.3))},
                mean=r.mean + float(rng.normal(0, 0.3)),
            )
            for r in recs
        ]
        r = partial_eta_squared(ruidosos)
        assert all(v < 1.0 for v in r.values())


class TestBootstrap:
    def test_contiene_la_media_y_es_reproducible(self):
        valores = [0.4, 0.5, 0.6, 0.55, 0.45, 0.5, 0.52, 0.48]
        lo, hi = bootstrap_ci(valores, seed=0)
        assert lo < sum(valores) / len(valores) < hi
        assert (lo, hi) == bootstrap_ci(valores, seed=0)

    def test_un_solo_valor_da_un_intervalo_degenerado(self):
        assert bootstrap_ci([0.3], seed=0) == (0.3, 0.3)

    def test_mas_datos_estrechan_el_intervalo(self):
        import numpy as np

        rng = np.random.default_rng(0)
        corto = bootstrap_ci(rng.normal(0.5, 0.1, 10).tolist(), seed=0)
        largo = bootstrap_ci(rng.normal(0.5, 0.1, 500).tolist(), seed=0)
        assert (largo[1] - largo[0]) < (corto[1] - corto[0])


class TestTablaDeEfectos:
    def test_reporta_eta2_y_observabilidad_juntos(self):
        tabla = main_effects_table(registros(solo_desalojo))
        assert [f["axis"] for f in tabla][0] == "evict"
        muertos = [f["axis"] for f in tabla if f["liveness"] == 0.0]
        assert "read" in muertos and "reinforce" in muertos

    def test_cubre_los_seis_ejes(self):
        assert len(main_effects_table(registros(solo_desalojo))) == 6
