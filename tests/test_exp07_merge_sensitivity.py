"""Pruebas de `exp07`: sensibilidad al umbral de fusión.

Lo que se fija acá es que el barrido mida lo que dice medir —que variar el
umbral no toque el espacio por defecto, que la auditoría de fusiones no cambie
el comportamiento de la memoria, y que la pureza cuente lo que debe— no los
valores publicados, que viven en `results/exp07_merge_sensitivity/`.
"""

import numpy as np
import pytest

from ember.core.genotype import Genotype
from ember.core.memory import PolicyMemory
from ember.core.policies import (
    FIFO,
    Constant,
    Merge,
    NearestNeighbour,
    NoDecay,
)
from ember.data.embeddings import EmbeddingBank
from ember.nas.space import SEARCH_SPACE, enumerate_space
from experiments.exp07_merge_sensitivity import (
    FRONTIER_GENOTYPE,
    AuditedMemory,
    auditar_sin_presion,
    barrer_ganancia,
    barrer_umbral,
    clave_ratio,
    clave_umbral,
    con_ganancia,
    construir_stream,
    genotipos_append,
    genotipos_merge,
    localizar_cruce,
    puntaje_t1,
    resumen_fusiones,
)


def _banco_falso(n_clases: int = 30, por_clase: int = 20, dim: int = 32) -> EmbeddingBank:
    """Banco chico con estructura de clase, para no depender de CIFAR-100 en CI."""
    rng = np.random.default_rng(0)
    centros = rng.standard_normal((n_clases, dim)).astype(np.float32)
    x = np.repeat(centros, por_clase, axis=0) + rng.standard_normal(
        (n_clases * por_clase, dim)
    ).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return EmbeddingBank(
        vectors=x.astype(np.float32),
        labels=np.repeat(np.arange(n_clases), por_clase).astype(np.int64),
        label_names=[f"c{i}" for i in range(n_clases)],
    )


class TestElEspacioPorDefectoNoSeMueve:
    def test_el_umbral_por_defecto_sigue_en_085(self):
        assert Merge().threshold == 0.85
        assert Merge() in SEARCH_SPACE["write"]

    def test_con_085_los_genotipos_son_los_del_espacio(self):
        """Si esto falla, el barrido compara contra otro espacio que exp01–exp06."""
        del_espacio = [g for g in enumerate_space() if isinstance(g.write, Merge)]
        assert genotipos_merge(0.85) == del_espacio
        assert len(genotipos_append()) + len(genotipos_merge(0.5)) == 576

    def test_cambiar_el_umbral_no_cambia_la_etiqueta_del_eje(self):
        """η² agrupa por etiqueta: el umbral tiene que caer dentro del nivel `merge`."""
        assert {g.axis("write") for g in genotipos_merge(0.3)} == {"merge"}

    def test_la_ganancia_por_defecto_reproduce_el_espacio(self):
        assert [con_ganancia(g, 2.0) for g in enumerate_space()] == list(enumerate_space())

    def test_claves_sin_puntos(self):
        assert clave_umbral(0.85) == "t085"
        assert clave_umbral(0.3) == "t030"
        assert clave_ratio(0.25) == "r0p25"
        assert clave_ratio(1.0) == "r1"


class TestAuditoria:
    def test_auditar_no_cambia_el_resultado(self):
        """La memoria auditada tiene que dar exactamente lo mismo que la original."""
        st = construir_stream("synthetic", 0.5, 3, capacity=10, banco=None)
        g = FRONTIER_GENOTYPE.with_axis("write", Merge(threshold=0.5))
        a = PolicyMemory(dim=32, capacity=10, genotype=g, seed=3)
        b = AuditedMemory(dim=32, capacity=10, genotype=g, seed=3)
        for it in st:
            a.write(it.key, it.value, it.pred_error)
            b.write(it.key, it.value, it.pred_error)
        assert a.store.values == b.store.values
        assert np.array_equal(a.store.strength, b.store.strength)
        assert a.n_merges == len(b.fusiones)

    def test_pureza_y_absorcion_de_raros(self):
        g = Genotype(
            read=NearestNeighbour(),
            write=Merge(threshold=0.9),
            strength=Constant(),
            decay=NoDecay(),
            evict=FIFO(),
            reinforce=0.0,
        )
        m = AuditedMemory(dim=2, capacity=10, genotype=g, seed=0)
        m.write(np.array([1.0, 0.0]), 0)
        m.write(np.array([1.0, 0.01]), 0)  # pura
        m.write(np.array([1.0, 0.02]), 1)  # impura: otra clase
        m.write(np.array([1.0, 0.03]), "rare0")  # raro absorbido
        m.write(np.array([0.0, 1.0]), "rare1")  # raro guardado
        r = resumen_fusiones(m)
        assert r["n_fusiones"] == 3
        assert r["tasa_fusion"] == pytest.approx(3 / 5)
        assert r["pureza"] == pytest.approx(1 / 3)
        assert r["raros_absorbidos"] == pytest.approx(1 / 2)

    def test_sin_fusiones_la_pureza_no_esta_definida(self):
        st = construir_stream("synthetic", 0.25, 0, capacity=8, banco=None)
        a = auditar_sin_presion(st, 0.999, 0)
        assert a["n_fusiones"] == 0 and a["pureza"] is None
        assert a["k_efectivo"] == a["n_escrituras"]

    def test_con_umbral_alto_en_sintetico_la_rutina_ocupa_k_trazas(self):
        """Con fusión que se dispara, K efectivo es el K del generador (como exp05)."""
        st = construir_stream("synthetic", 0.5, 0, capacity=10, banco=None)
        a = auditar_sin_presion(st, 0.85, 0)
        assert a["k_efectivo"] == st.spec.n_prototypes
        assert a["pureza"] == 1.0


class TestCruce:
    def test_localiza_el_primer_cambio_de_dominancia(self):
        assert localizar_cruce([0.25, 0.5, 1.0], [0.8, 0.6, 0.1], [0.1, 0.2, 0.5]) == (
            pytest.approx(np.sqrt(0.5))
        )

    def test_sin_dominancia_de_escritura_no_hay_cruce(self):
        assert localizar_cruce([0.25, 0.5], [0.0, 0.0], [0.4, 0.3]) is None


class TestBarridoReducido:
    def test_puntaje_de_la_frontera_coincide_con_la_auditoria(self):
        from experiments.exp07_merge_sensitivity import auditar_bajo_presion

        st = construir_stream("synthetic", 0.5, 1, capacity=10, banco=None)
        g = FRONTIER_GENOTYPE.with_axis("write", Merge(threshold=0.7))
        assert auditar_bajo_presion(st, 0.7, 1)["puntaje"] == puntaje_t1(g, st, 1)

    def test_barrido_minimo(self):
        """Tamaño mínimo, fuera de `slow`: el pipeline completo corre y resume."""
        res = barrer_umbral(
            dominios=("synthetic",),
            umbrales=(0.85,),
            ratios=(0.25, 1.0),
            seeds=(0,),
            capacity=4,
            n_jobs=1,
            verbose=False,
        )
        u = res["synthetic"]["umbrales"]["t085"]
        assert set(u["celdas"]) == {"r0p25", "r1"}
        assert "cruce_r_nominal" in u and "cruce_r_efectivo" in u

    @pytest.mark.slow
    def test_celdas_con_intervalos_y_ambos_dominios(self):
        banco = _banco_falso()
        res = barrer_umbral(
            umbrales=(0.5, 0.85),
            ratios=(0.4, 2.0),
            seeds=(0, 1),
            capacity=5,
            banco=banco,
            n_jobs=2,
            verbose=False,
        )
        assert set(res) == {"synthetic", "cifar100"}
        for d in res.values():
            assert set(d["umbrales"]) == {"t050", "t085"}
            for u in d["umbrales"].values():
                for c in u["celdas"].values():
                    lo, hi = c["eta2_write_ci"]
                    assert lo - 1e-12 <= c["eta2_write"] <= hi + 1e-12
                    assert 0.0 <= c["tasa_fusion"]["media"] <= 1.0
                    assert len(c["eta2_write_por_semilla"]) == 2

    @pytest.mark.slow
    def test_barrido_de_ganancia(self):
        res = barrer_ganancia(
            dominios=("synthetic",),
            ganancias=(1.0, 4.0),
            ratios=(0.4,),
            seeds=(0,),
            capacity=5,
            n_jobs=2,
        )
        assert set(res["synthetic"]) == {"g1", "g4"}
