import numpy as np
import pytest

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.memory import PolicyMemory
from ember.core.policies import (
    FIFO,
    Append,
    Constant,
    ExponentialDecay,
    MinStrength,
    NearestNeighbour,
    NoDecay,
    PredErrorGated,
    Radius,
    Random,
)


def _claves(n, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _memoria(genotype, capacity=4, dim=16, seed=0):
    return PolicyMemory(dim=dim, capacity=capacity, genotype=genotype, seed=seed)


FRONTERA = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=PredErrorGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)


def test_lectura_sobre_memoria_vacia_devuelve_none():
    mem = _memoria(FIFO_GENOTYPE)
    r = mem.read(_claves(1)[0])
    assert r.value is None and r.similarity == 0.0


def test_escribir_y_leer_la_misma_clave_la_recupera():
    mem = _memoria(FIFO_GENOTYPE)
    k = _claves(1)[0]
    mem.write(k, "objetivo", pred_error=0.5)
    r = mem.read(k)
    assert r.value == "objetivo"
    assert r.similarity == pytest.approx(1.0, abs=1e-5)


def test_nunca_se_excede_la_capacidad():
    mem = _memoria(FIFO_GENOTYPE, capacity=3)
    for i, k in enumerate(_claves(20)):
        mem.write(k, i, pred_error=0.5)
        assert len(mem) <= 3


def test_fifo_conserva_los_mas_recientes():
    mem = _memoria(FIFO_GENOTYPE, capacity=3)
    claves = _claves(5)
    for i, k in enumerate(claves):
        mem.write(k, i, pred_error=0.5)
    assert {mem.read(k).value for k in claves[2:]} == {2, 3, 4}


def test_min_strength_con_gating_conserva_lo_sorpresivo():
    """El resultado central del paper: la fuerza solo importa si el desalojo la lee."""
    mem = _memoria(FRONTERA, capacity=3)
    claves = _claves(10)
    mem.write(claves[0], "raro", pred_error=0.9)
    for i, k in enumerate(claves[1:], start=1):
        mem.write(k, f"comun{i}", pred_error=0.1)
    assert mem.read(claves[0]).value == "raro"


def test_el_mismo_stream_con_fifo_pierde_lo_sorpresivo():
    mem = _memoria(FIFO_GENOTYPE, capacity=3)
    claves = _claves(10)
    mem.write(claves[0], "raro", pred_error=0.9)
    for i, k in enumerate(claves[1:], start=1):
        mem.write(k, f"comun{i}", pred_error=0.1)
    assert mem.read(claves[0]).value != "raro"


def test_reinforce_en_lectura_protege_del_desalojo():
    """Cierra el eje que el addendum declaró inobservable: lecturas intercaladas."""
    g = Genotype(
        read=NearestNeighbour(),
        write=Append(),
        strength=Constant(),
        decay=NoDecay(),
        evict=MinStrength(),
        reinforce=5.0,
    )
    mem = _memoria(g, capacity=3)
    claves = _claves(8)
    mem.write(claves[0], "consultado", pred_error=0.5)
    for i, k in enumerate(claves[1:], start=1):
        mem.write(k, f"otro{i}", pred_error=0.5)
        mem.read(claves[0])  # se refuerza en cada paso
    assert mem.read(claves[0]).value == "consultado"


def test_sin_lecturas_intercaladas_reinforce_no_puede_influir_en_el_desalojo():
    """La causa raíz del eje muerto: la fuerza modificada nunca llega a un desalojo."""
    claves = _claves(12)

    def correr(reinforce):
        g = Genotype(
            read=NearestNeighbour(),
            write=Append(),
            strength=Constant(),
            decay=NoDecay(),
            evict=MinStrength(),
            reinforce=reinforce,
        )
        mem = _memoria(g, capacity=4, seed=0)
        for i, k in enumerate(claves):
            mem.write(k, i, pred_error=0.5)
        return [mem.read(k).value for k in claves]

    assert correr(0.0) == correr(2.0)


def test_decaimiento_erosiona_la_fuerza_no_reforzada():
    g = Genotype(
        read=NearestNeighbour(),
        write=Append(),
        strength=Constant(),
        decay=ExponentialDecay(rate=0.5),
        evict=FIFO(),
        reinforce=0.0,
    )
    mem = _memoria(g, capacity=10)
    claves = _claves(3)
    mem.write(claves[0], "a", pred_error=0.5)
    inicial = float(mem.store.strength[0])
    mem.write(claves[1], "b", pred_error=0.5)
    mem.write(claves[2], "c", pred_error=0.5)
    assert float(mem.store.strength[0]) == pytest.approx(inicial * 0.25, abs=1e-5)


def test_misma_semilla_produce_resultados_identicos():
    g = Genotype(
        read=NearestNeighbour(),
        write=Append(),
        strength=Constant(),
        decay=NoDecay(),
        evict=Random(),
        reinforce=0.0,
    )
    claves = _claves(20)
    salidas = []
    for _ in range(2):
        mem = _memoria(g, capacity=5, seed=7)
        for i, k in enumerate(claves):
            mem.write(k, i, pred_error=0.5)
        salidas.append([mem.read(k).value for k in claves])
    assert salidas[0] == salidas[1]


def test_el_genotipo_fifo_es_el_episodic_buffer_de_emdb():
    assert FIFO_GENOTYPE.as_dict() == {
        "read": "nn",
        "write": "append",
        "strength": "constant",
        "decay": "1.0",
        "evict": "fifo",
        "reinforce": "0.0",
    }


def test_la_lectura_por_radio_hace_superposicion_no_vecino_mas_cercano():
    """El eje de lectura tiene que ser observable: en el piloto explicaba 0.0 %."""
    dim = 16
    rng = np.random.default_rng(0)
    base = rng.standard_normal(dim).astype(np.float32)
    base /= np.linalg.norm(base)

    # Un racimo de trazas parecidas entre sí. Ahí la reconstrucción por
    # superposición se acerca al centroide del racimo, y puede ganar una traza
    # distinta de la más parecida a la consulta.
    claves = []
    for _ in range(6):
        v = base + rng.standard_normal(dim).astype(np.float32) * 0.35
        claves.append(v / np.linalg.norm(v))

    def leer(read_policy, consulta):
        g = Genotype(
            read=read_policy,
            write=Append(),
            strength=Constant(),
            decay=NoDecay(),
            evict=FIFO(),
            reinforce=0.0,
        )
        mem = PolicyMemory(dim=dim, capacity=20, genotype=g, seed=0)
        for i, k in enumerate(claves):
            mem.write(k, i, pred_error=0.5)
        return mem.read(consulta)

    desacuerdos = 0
    for _ in range(40):
        q = base + rng.standard_normal(dim).astype(np.float32) * 0.5
        q /= np.linalg.norm(q)
        if leer(NearestNeighbour(), q).value != leer(Radius(fraction=0.85), q).value:
            desacuerdos += 1

    assert desacuerdos > 0, "el eje de lectura volvió a ser inobservable"


def test_la_consolidacion_no_dispara_desalojos():
    from ember.core.policies import Merge

    g = Genotype(
        read=NearestNeighbour(),
        write=Merge(threshold=0.85),
        strength=Constant(),
        decay=NoDecay(),
        evict=FIFO(),
        reinforce=0.0,
    )
    mem = _memoria(g, capacity=3, dim=16)
    base = _claves(1)[0]
    rng = np.random.default_rng(1)
    for i in range(20):
        v = base + rng.standard_normal(16).astype(np.float32) * 0.02
        mem.write(v / np.linalg.norm(v), i, pred_error=0.5)
    assert len(mem) == 1
    assert mem.n_evictions == 0
