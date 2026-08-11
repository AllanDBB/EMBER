import numpy as np
import pytest

from ember.envs.minigrid import OneStepPredictor


class TestPredictorDeSorpresa:
    """El error de predicción es la señal que gobierna todo el resultado.

    Sobre flujos sintéticos lo pone el generador; sobre un entorno hay que
    calcularlo, y estas pruebas fijan que la señal calculada tenga las
    propiedades que la tarea asume.
    """

    def test_la_sorpresa_esta_en_cero_uno(self):
        rng = np.random.default_rng(0)
        p = OneStepPredictor(obs_dim=8, n_actions=3)
        for _ in range(50):
            o = rng.standard_normal(8).astype(np.float32)
            s = p.surprise(o, int(rng.integers(3)), rng.standard_normal(8).astype(np.float32))
            assert 0.0 <= s <= 1.0

    def test_una_transicion_repetida_deja_de_sorprender(self):
        """Si el modelo no aprende, la señal no distingue lo nuevo de lo rutinario."""
        rng = np.random.default_rng(0)
        p = OneStepPredictor(obs_dim=8, n_actions=2, lr=0.2)
        o = rng.standard_normal(8).astype(np.float32)
        siguiente = rng.standard_normal(8).astype(np.float32)

        primera = p.surprise(o, 0, siguiente)
        for _ in range(80):
            p.surprise(o, 0, siguiente)
        ultima = p.surprise(o, 0, siguiente)

        assert ultima < primera

    def test_una_transicion_nueva_sorprende_mas_que_la_rutinaria(self):
        rng = np.random.default_rng(0)
        p = OneStepPredictor(obs_dim=8, n_actions=2, lr=0.2)
        o = rng.standard_normal(8).astype(np.float32)
        rutina = rng.standard_normal(8).astype(np.float32)

        for _ in range(100):
            p.surprise(o, 0, rutina)

        rutinaria = p.surprise(o, 0, rutina)
        novedosa = p.surprise(o, 0, rng.standard_normal(8).astype(np.float32) * 5.0)
        assert novedosa > rutinaria

    def test_es_determinista(self):
        def correr():
            rng = np.random.default_rng(3)
            p = OneStepPredictor(obs_dim=8, n_actions=2)
            return [
                p.surprise(
                    rng.standard_normal(8).astype(np.float32),
                    int(rng.integers(2)),
                    rng.standard_normal(8).astype(np.float32),
                )
                for _ in range(20)
            ]

        assert correr() == correr()


def _hay_minigrid() -> bool:
    from importlib.util import find_spec

    return find_spec("minigrid") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _hay_minigrid(), reason="requiere el extra envs")
class TestAdaptador:
    def test_el_rollout_produce_un_stream_del_largo_pedido(self):
        from ember.envs.minigrid import MiniGridStreamAdapter

        s = MiniGridStreamAdapter(seed=0).rollout(n_steps=50)
        assert len(s) == 50

    def test_las_claves_son_unitarias_de_la_dimension_pedida(self):
        from ember.envs.minigrid import MiniGridStreamAdapter

        for it in list(MiniGridStreamAdapter(dim=32, seed=0).rollout(n_steps=20))[:10]:
            assert it.key.shape == (32,)
            assert np.linalg.norm(it.key) == pytest.approx(1.0, abs=1e-5)

    def test_el_error_de_prediccion_esta_en_cero_uno(self):
        from ember.envs.minigrid import MiniGridStreamAdapter

        assert all(
            0.0 <= it.pred_error <= 1.0 for it in MiniGridStreamAdapter(seed=0).rollout(n_steps=30)
        )

    def test_es_reproducible(self):
        from ember.envs.minigrid import MiniGridStreamAdapter

        x = MiniGridStreamAdapter(seed=4).rollout(20).keys()
        y = MiniGridStreamAdapter(seed=4).rollout(20).keys()
        assert np.array_equal(x, y)

    def test_el_spec_declara_que_los_prototipos_hay_que_estimarlos(self):
        from ember.envs.minigrid import MiniGridStreamAdapter

        spec = MiniGridStreamAdapter(seed=0).rollout(20).spec
        assert spec.is_estimated
        assert spec.source.startswith("minigrid:")
