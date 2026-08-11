import numpy as np

from ember.memories.enn import ENNMemory


def _clave(rng, dim=32):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_recupera_lo_escrito():
    rng = np.random.default_rng(0)
    mem = ENNMemory(dim=32, capacity=10, seed=0)
    claves = [_clave(rng) for _ in range(5)]
    for i, k in enumerate(claves):
        mem.write(k, i, pred_error=0.5)
    for i, k in enumerate(claves):
        assert mem.read(k).value == i


def test_claves_casi_identicas_se_consolidan_en_una_sola_traza():
    rng = np.random.default_rng(0)
    mem = ENNMemory(dim=32, capacity=10, seed=0)
    base = _clave(rng)
    for i in range(10):
        ruido = rng.standard_normal(32).astype(np.float32) * 0.02
        mem.write((base + ruido) / np.linalg.norm(base + ruido), i, pred_error=0.5)
    assert len(mem) == 1


def test_claves_ortogonales_no_se_consolidan():
    rng = np.random.default_rng(0)
    mem = ENNMemory(dim=32, capacity=10, seed=0)
    for i in range(5):
        mem.write(_clave(rng), i, pred_error=0.5)
    assert len(mem) == 5


def test_la_fusion_acumula_fuerza_por_encima_de_un_evento_raro_unico():
    """El modo de falla de la ENN: la fusión ahoga la señal de saliencia.

    Un prototipo visitado 30 veces acumula fuerza en progresión geométrica y
    supera a un evento raro codificado con sorpresa 0.9 una sola vez. Por eso la
    ENN retiene peor los eventos raros que la SDM pese a tener la misma
    compuerta y la misma política de desalojo.
    """
    rng = np.random.default_rng(0)
    mem = ENNMemory(dim=32, capacity=10, seed=0)
    base = _clave(rng)
    for _ in range(30):
        ruido = rng.standard_normal(32).astype(np.float32) * 0.02
        mem.write((base + ruido) / np.linalg.norm(base + ruido), "comun", pred_error=0.1)
    mem.write(_clave(rng), "raro", pred_error=0.9)

    fuerzas = dict(zip(mem.store.values, mem.store.strength.tolist(), strict=True))
    assert fuerzas["comun"] > fuerzas["raro"]


def test_misma_semilla_produce_resultados_identicos():
    def correr():
        rng = np.random.default_rng(0)
        mem = ENNMemory(dim=32, capacity=4, seed=11)
        claves = [_clave(rng) for _ in range(20)]
        for i, k in enumerate(claves):
            mem.write(k, i, pred_error=0.5)
        return [mem.read(k).value for k in claves]

    assert correr() == correr()


def test_el_decaimiento_erosiona_las_trazas_no_reforzadas():
    rng = np.random.default_rng(0)
    mem = ENNMemory(dim=32, capacity=20, seed=0, decay=0.9)
    primera = _clave(rng)
    mem.write(primera, "primera", pred_error=0.5)
    inicial = float(mem.store.strength[0])
    for _ in range(5):
        mem.write(_clave(rng), "otra", pred_error=0.5)
    assert float(mem.store.strength[0]) < inicial
