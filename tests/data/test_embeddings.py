import numpy as np
import pytest

from ember.data.embeddings import EmbeddingBank, embedding_stream
from ember.data.synthetic import clustered_stream


def _coseno_medio(K):
    G = np.abs(K @ K.T)
    return float(G[~np.eye(len(K), dtype=bool)].mean())


@pytest.fixture
def banco():
    """Banco sintético con la forma de uno real: 20 clases, 50 muestras cada una.

    Permite ejercitar `embedding_stream` sin descargar CIFAR-100 ni requerir torch.
    """
    rng = np.random.default_rng(0)
    centros = rng.standard_normal((20, 32)).astype(np.float32)
    vectores, etiquetas = [], []
    for c in range(20):
        v = centros[c] + rng.standard_normal((50, 32)).astype(np.float32) * 0.3
        vectores.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        etiquetas.append(np.full(50, c, dtype=np.int64))
    return EmbeddingBank(
        vectors=np.concatenate(vectores).astype(np.float32),
        labels=np.concatenate(etiquetas),
        label_names=[f"clase{i}" for i in range(20)],
    )


def test_el_stream_usa_exactamente_n_prototypes_clases_comunes(banco):
    s = embedding_stream(banco, n_prototypes=6, capacity=20, n_common=120, n_rare=5, seed=0)
    assert len({it.value for it in s if not it.is_rare}) == 6


def test_las_clases_raras_no_aparecen_entre_las_comunes(banco):
    s = embedding_stream(banco, n_prototypes=6, capacity=20, n_common=120, n_rare=5, seed=0)
    comunes = {it.value for it in s if not it.is_rare}
    raros = {it.value for it in s if it.is_rare}
    assert not (comunes & raros)


def test_el_spec_registra_el_origen_y_el_ratio(banco):
    s = embedding_stream(banco, n_prototypes=10, capacity=20, seed=0)
    assert s.spec.source == "cifar100"
    assert s.spec.r == pytest.approx(0.5)
    assert s.spec.dim == 32


def test_la_correlacion_sintetica_se_acerca_al_dominio_real(banco):
    """El generador sintético tiene que poder alcanzar la estructura del dominio real.

    No se puede probar la afirmación "los embeddings reales son menos ortogonales
    que el gaussiano" contra un banco sintético — el banco es gaussiano también.
    Lo que sí se puede fijar aquí es que la perilla `correlation` mueve el
    coseno medio en el rango correcto, para que `exp05` pueda comparar dominios
    de forma controlada. La afirmación sobre datos reales la prueba
    `test_el_banco_real_es_menos_ortogonal_que_el_gaussiano`, más abajo.
    """
    sin = clustered_stream(
        n_prototypes=10, capacity=20, n_common=200, n_rare=0, correlation=0.0, seed=0
    ).keys()
    con = clustered_stream(
        n_prototypes=10, capacity=20, n_common=200, n_rare=0, correlation=0.8, seed=0
    ).keys()
    assert _coseno_medio(con) > _coseno_medio(sin)


@pytest.mark.slow
def test_el_banco_real_es_menos_ortogonal_que_el_gaussiano():
    """La objeción que el paper deja abierta, probada contra CIFAR-100 real.

    Requiere el extra `lab` y el banco ya extraído en `data/cache/`. Se salta si
    no está: no tiene sentido descargar 170 MB dentro de una prueba unitaria.
    """
    from pathlib import Path

    cache = Path("data/cache/cifar100_resnet18_d32.npz")
    if not cache.exists():
        pytest.skip("banco de CIFAR-100 no extraído; correr exp05 primero")

    real_bank = EmbeddingBank.load(cache)
    real = embedding_stream(real_bank, 10, 20, n_common=300, n_rare=0, seed=0).keys()
    sint = clustered_stream(n_prototypes=10, capacity=20, n_common=300, n_rare=0, seed=0).keys()
    assert _coseno_medio(real) > _coseno_medio(sint)


def test_cada_visita_a_un_prototipo_muestrea_una_imagen_distinta(banco):
    """Variación intra-clase real, no un centro más ruido gaussiano."""
    s = embedding_stream(banco, n_prototypes=3, capacity=20, n_common=90, n_rare=0, seed=0)
    de_una_clase = [it.key for it in s if it.value == s.items[0].value]
    assert len({k.tobytes() for k in de_una_clase}) > 1


def test_es_reproducible(banco):
    a = embedding_stream(banco, n_prototypes=5, capacity=20, seed=2)
    b = embedding_stream(banco, n_prototypes=5, capacity=20, seed=2)
    assert np.array_equal(a.keys(), b.keys())


def test_density_skew_desbalancea_las_visitas(banco):
    def maxima_proporcion(skew):
        s = embedding_stream(
            banco,
            n_prototypes=10,
            capacity=20,
            n_common=400,
            n_rare=0,
            density_skew=skew,
            seed=0,
        )
        _, cuentas = np.unique([it.value for it in s], return_counts=True)
        return cuentas.max() / cuentas.sum()

    assert maxima_proporcion(2.0) > maxima_proporcion(0.0) * 1.5


def test_rechaza_pedir_mas_prototipos_que_clases(banco):
    with pytest.raises(ValueError, match="prototipos"):
        embedding_stream(banco, n_prototypes=50, capacity=20, seed=0)


def test_el_banco_hace_ida_y_vuelta_a_disco(banco, tmp_path):
    destino = tmp_path / "banco.npz"
    banco.save(destino)
    recuperado = EmbeddingBank.load(destino)
    assert np.array_equal(recuperado.vectors, banco.vectors)
    assert recuperado.label_names == banco.label_names
    assert recuperado.source == banco.source


def test_rechaza_vectores_y_etiquetas_desalineados():
    with pytest.raises(ValueError, match="largo"):
        EmbeddingBank(
            vectors=np.zeros((5, 32), dtype=np.float32),
            labels=np.zeros(3, dtype=np.int64),
            label_names=["a"],
        )
