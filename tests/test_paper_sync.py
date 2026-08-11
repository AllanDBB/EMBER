import json

import pytest

from ember.paper_sync import render_tables, resolver, verify_paper


@pytest.fixture
def resultados(tmp_path):
    """Un `results/` mínimo con la forma real de exp01."""
    d = tmp_path / "results" / "exp01_nas_full"
    d.mkdir(parents=True)
    (d / "data.json").write_text(
        json.dumps(
            {
                "main_effects": [
                    {"axis": "write", "eta2": 0.566, "liveness": 0.583, "por_opcion": {}},
                    {"axis": "evict", "eta2": 0.170, "liveness": 0.728, "por_opcion": {}},
                ],
                "incumbent": {"score": 0.106, "rank_optimistic": 416},
                "salience_conditional_ratio": 14.444,
            }
        )
    )
    return tmp_path / "results"


class TestResolver:
    def test_navega_una_ruta_punteada_con_indices(self, resultados):
        valor, _ = resolver("exp01_nas_full:main_effects.0.eta2", resultados)
        assert valor == pytest.approx(0.566)

    def test_el_formato_por_defecto_son_tres_decimales(self, resultados):
        assert resolver("exp01_nas_full:incumbent.score", resultados)[1] == "f3"

    def test_pct_multiplica_por_cien(self, resultados):
        _, formato = resolver("exp01_nas_full:main_effects.0.eta2|pct", resultados)
        assert formato == "pct"

    def test_falla_claro_si_el_experimento_no_se_corrio(self, resultados):
        with pytest.raises(FileNotFoundError, match="no fue corrido"):
            resolver("exp99_inexistente:algo", resultados)

    def test_falla_claro_ante_un_formato_desconocido(self, resultados):
        with pytest.raises(ValueError, match="formato desconocido"):
            resolver("exp01_nas_full:incumbent.score|raro", resultados)


class TestVerificacion:
    def test_no_reporta_nada_si_los_numeros_coinciden(self, resultados, tmp_path):
        tex = tmp_path / "main.tex"
        tex.write_text(
            r"El modo de escritura explica "
            r"\result{exp01_nas_full:main_effects.0.eta2|pct}{56.6}\% de la varianza."
        )
        assert verify_paper(tex, resultados) == []

    def test_detecta_un_numero_publicado_que_ya_no_se_mide(self, resultados, tmp_path):
        """El caso real: se arregla un bug, el resultado cambia, el paper no."""
        tex = tmp_path / "main.tex"
        tex.write_text(r"\result{exp01_nas_full:main_effects.0.eta2|pct}{63.5}\%")
        d = verify_paper(tex, resultados)
        assert len(d) == 1
        assert d[0].publicado == 63.5
        assert d[0].medido == 56.6

    def test_detecta_una_clave_que_no_existe(self, resultados, tmp_path):
        tex = tmp_path / "main.tex"
        tex.write_text(r"\result{exp01_nas_full:no.existe}{1.0}")
        d = verify_paper(tex, resultados)
        assert len(d) == 1 and "no se pudo resolver" in d[0].motivo

    def test_reporta_la_linea_del_problema(self, resultados, tmp_path):
        tex = tmp_path / "main.tex"
        tex.write_text("bla\nbla\n" + r"\result{exp01_nas_full:incumbent.score}{0.261}")
        assert verify_paper(tex, resultados)[0].linea == 3

    def test_verifica_varios_numeros_en_un_solo_documento(self, resultados, tmp_path):
        tex = tmp_path / "main.tex"
        tex.write_text(
            r"\result{exp01_nas_full:main_effects.0.eta2|pct}{56.6} y "
            r"\result{exp01_nas_full:main_effects.1.eta2|pct}{9.9} y "
            r"\result{exp01_nas_full:incumbent.score}{0.106}"
        )
        d = verify_paper(tex, resultados)
        assert len(d) == 1
        assert d[0].publicado == 9.9

    def test_un_documento_sin_macros_no_tiene_discrepancias(self, resultados, tmp_path):
        tex = tmp_path / "main.tex"
        tex.write_text("Texto sin ningún número verificado.")
        assert verify_paper(tex, resultados) == []


class TestGeneracionDeTablas:
    def test_genera_el_macro_y_la_tabla_de_efectos(self, resultados, tmp_path):
        salidas = render_tables(resultados, tmp_path / "tex")
        assert (tmp_path / "tex" / "result_macro.tex").exists()
        assert any(p.name == "tab_main_effects.tex" for p in salidas)

    def test_la_tabla_marca_los_ejes_inobservables(self, resultados, tmp_path):
        datos = json.loads((resultados / "exp01_nas_full" / "data.json").read_text())
        datos["main_effects"][0]["liveness"] = 0.0
        (resultados / "exp01_nas_full" / "data.json").write_text(json.dumps(datos))

        render_tables(resultados, tmp_path / "tex")
        assert "inobservable" in (tmp_path / "tex" / "tab_main_effects.tex").read_text()

    def test_saltea_en_silencio_los_experimentos_no_corridos(self, resultados, tmp_path):
        salidas = render_tables(resultados, tmp_path / "tex")
        assert not any("benchmark" in p.name for p in salidas)

    def test_el_macro_renderiza_el_valor_publicado(self, resultados, tmp_path):
        render_tables(resultados, tmp_path / "tex")
        macro = (tmp_path / "tex" / "result_macro.tex").read_text()
        assert "\\newcommand{\\result}[2]{#2}" in macro
