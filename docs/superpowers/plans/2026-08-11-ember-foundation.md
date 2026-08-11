# EMBER Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruir EMBER como un paquete Python instalable con un único motor de memoria, una suite de experimentos reproducibles que respalde el paper BIP2026, y configuración de Claude para el dominio.

**Architecture:** `src/ember/core` define las políticas de ciclo de vida de traza una sola vez (fuerza, escritura, lectura, desalojo, decaimiento) sobre un `TraceStore` vectorizado. `PolicyMemory` las compone; el genotipo del NAS es una tupla de políticas y el `EpisodicBuffer` FIFO de e-MDB es un punto de ese espacio. Las arquitecturas con sustrato propio (SDM, ENN, spiking) implementan el mismo protocolo `Memory` reutilizando esas políticas. Experimentos numerados escriben a `results/` con manifiesto de reproducibilidad.

**Tech Stack:** Python 3.12, uv, numpy (núcleo), torch + torchvision + scipy + matplotlib (extra `lab`), gymnasium + minigrid (extra `envs`), pytest + ruff + mypy (extra `dev`).

## Global Constraints

- Python 3.12. El núcleo (`ember.core`, `ember.memories`) depende **solo de numpy**: nada de torch, scipy, matplotlib ni gym puede importarse desde ahí. Es el código que va al Robotino.
- Toda aleatoriedad deriva de una semilla explícita propagada desde el experimento. Prohibido `np.random.*` global, `id()`, `hash()` o el reloj como fuente de semilla.
- Todo array de claves y valores es `float32`. Dimensión por defecto `DIM = 32`.
- Los vectores clave se normalizan a norma 1 al entrar a `write` y `read`.
- Cada experimento escribe `results/<nombre>/data.json` y `results/<nombre>/manifest.json` (SHA de git, semillas, versiones, timestamp UTC ISO-8601).
- Docstrings y comentarios en español; nombres de identificadores en inglés. Es la convención que ya usa el código piloto.
- `ruff check` y `ruff format --check` limpios antes de cada commit.

---

### Task 1: Esqueleto del proyecto y entorno

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.python-version`, `ruff.toml`
- Create: `src/ember/__init__.py`, `src/ember/py.typed`
- Create: `tests/test_smoke.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nada.
- Produces: `ember.__version__` (str). Entorno `uv` funcional con `uv run pytest`.

- [ ] **Step 1: Escribir `pyproject.toml`**

```toml
[project]
name = "ember"
version = "0.1.0"
description = "Emergent Memory-Based Encoding and Reactivation — memoria episódica bioinspirada para robots cognitivos"
requires-python = ">=3.12"
dependencies = ["numpy>=2.0"]

[project.optional-dependencies]
lab  = ["torch>=2.4", "torchvision>=0.19", "scipy>=1.14", "matplotlib>=3.9", "pandas>=2.2"]
envs = ["gymnasium>=0.29", "minigrid>=2.3"]
dev  = ["pytest>=8.0", "pytest-xdist>=3.6", "ruff>=0.6", "mypy>=1.11"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ember"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: pruebas que tardan más de 5 s"]
```

- [ ] **Step 2: Escribir `.gitignore`**

Debe cubrir: `__pycache__/`, `*.py[cod]`, `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `*.egg-info/`, `dist/`, `build/`, `results/**/*.npz`, `data/cache/`, `paper/*.aux`, `paper/*.log`, `paper/*.out`, `paper/*.bbl`, `paper/*.blg`, `.ipynb_checkpoints/`.

`results/**/*.json` **no** se ignora: los resultados son parte del registro del paper.

- [ ] **Step 3: Escribir `ruff.toml`**

```toml
line-length = 100
target-version = "py312"

[lint]
select = ["E", "F", "I", "UP", "B", "SIM", "NPY"]
ignore = ["E501"]
```

- [ ] **Step 4: Crear el paquete**

`.python-version` con `3.12`. `src/ember/__init__.py`:

```python
"""EMBER — Emergent Memory-Based Encoding and Reactivation."""

__version__ = "0.1.0"
```

`src/ember/py.typed` vacío.

- [ ] **Step 5: Escribir el test de humo**

```python
# tests/test_smoke.py
def test_importa_y_tiene_version():
    import ember

    assert ember.__version__ == "0.1.0"


def test_el_nucleo_no_importa_torch():
    """El código que va al robot no puede arrastrar el stack de laboratorio."""
    import subprocess
    import sys

    code = (
        "import sys, ember.core, ember.memories; "
        "prohibidos = {'torch', 'scipy', 'matplotlib', 'gymnasium'}; "
        "encontrados = prohibidos & set(sys.modules); "
        "sys.exit(f'importa {encontrados}' if encontrados else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
```

- [ ] **Step 6: Crear el entorno y correr**

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest tests/test_smoke.py -v
```

Esperado: `test_importa_y_tiene_version` PASA; `test_el_nucleo_no_importa_torch` FALLA (aún no existen `ember.core` ni `ember.memories`). Se resolverá en la Task 2 y la Task 5.

- [ ] **Step 7: Escribir CI**

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv venv --python 3.12
      - run: uv pip install -e ".[dev]"
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest -v
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore .python-version ruff.toml src tests .github
git commit -m "chore: esqueleto del paquete, entorno uv y CI"
```

---

### Task 2: `TraceStore` — la contabilidad de trazas

**Files:**
- Create: `src/ember/core/__init__.py`, `src/ember/core/types.py`, `src/ember/core/store.py`
- Test: `tests/core/test_store.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `Episode` dataclass: `old_perception, policy, action, perception, reward_list` (compatibilidad e-MDB).
  - `ReadResult(NamedTuple)`: `value: Any`, `similarity: float`, `index: int | None`.
  - `TraceStore(dim: int, capacity: int)` con atributos `keys (n,dim) float32`, `values list`, `strength (n,) float32`, `age (n,)`, `utility (n,)`, `contribution (n,) float32`, `t: int`; y métodos `append(key, value, strength) -> int`, `remove(idx)`, `similarities(query) -> NDArray`, `novelty(query) -> float`, `tick()`, `__len__()`, `is_full`.

`contribution[i]` guarda el escalar que la traza `i` sumó a los contenedores de un
sustrato distribuido, para que el desalojo reste exactamente lo que se sumó.
Este es el arreglo del bug §2.2 del spec.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/core/test_store.py
import numpy as np
import pytest

from ember.core.store import TraceStore


def _clave(rng, dim=8):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_append_incrementa_longitud_y_devuelve_indice():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    i = store.append(_clave(rng), "a", strength=2.5)
    assert i == 0
    assert len(store) == 1
    assert store.strength[0] == pytest.approx(2.5)


def test_remove_compacta_todos_los_arreglos_en_paralelo():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    for nombre, s in [("a", 1.0), ("b", 2.0), ("c", 3.0)]:
        store.append(_clave(rng), nombre, strength=s)
    store.remove(1)
    assert store.values == ["a", "c"]
    assert store.strength.tolist() == pytest.approx([1.0, 3.0])
    assert store.keys.shape == (2, 8)
    assert len(store.age) == len(store.utility) == len(store.contribution) == 2


def test_similarities_devuelve_coseno_y_vale_1_consigo_misma():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    k = _clave(rng)
    store.append(k, "a", strength=1.0)
    assert store.similarities(k)[0] == pytest.approx(1.0, abs=1e-5)


def test_similarities_sobre_store_vacio_devuelve_arreglo_vacio():
    store = TraceStore(dim=8, capacity=4)
    assert store.similarities(np.zeros(8, dtype=np.float32)).shape == (0,)


def test_novelty_es_1_en_store_vacio_y_0_ante_clave_repetida():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    k = _clave(rng)
    assert store.novelty(k) == pytest.approx(1.0)
    store.append(k, "a", strength=1.0)
    assert store.novelty(k) == pytest.approx(0.0, abs=1e-5)


def test_tick_envejece_las_trazas_existentes():
    store = TraceStore(dim=8, capacity=4)
    rng = np.random.default_rng(0)
    store.append(_clave(rng), "a", strength=1.0)
    store.tick()
    store.tick()
    assert store.age[0] == pytest.approx(2.0)
    assert store.t == 2


def test_is_full_respeta_la_capacidad():
    store = TraceStore(dim=8, capacity=2)
    rng = np.random.default_rng(0)
    assert not store.is_full
    store.append(_clave(rng), "a", strength=1.0)
    store.append(_clave(rng), "b", strength=1.0)
    assert store.is_full
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `uv run pytest tests/core/test_store.py -v`
Esperado: FAIL con `ModuleNotFoundError: No module named 'ember.core'`.

- [ ] **Step 3: Implementar `types.py` y `store.py`**

`types.py` define `Episode`, `ReadResult` y `EPS = 1e-8`, más `unit(v)` que
normaliza a norma 1 en float32.

`store.py` implementa `TraceStore` con arreglos numpy preasignados o crecidos
por `np.vstack`/`np.append`. `similarities` normaliza filas y consulta.
`novelty(q)` devuelve `1.0 - max(similarities(q))`, y `1.0` si está vacío.
`remove(idx)` usa `np.delete` sobre cada arreglo y `list.pop` sobre `values`.

- [ ] **Step 4: Correr y verificar que pasa**

Run: `uv run pytest tests/core/test_store.py -v`
Esperado: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/core tests/core
git commit -m "feat(core): TraceStore con contabilidad de contribución por traza"
```

---

### Task 3: Políticas de ciclo de vida

**Files:**
- Create: `src/ember/core/policies.py`
- Test: `tests/core/test_policies.py`

**Interfaces:**
- Consumes: `TraceStore` (Task 2).
- Produces:
  - `StrengthPolicy` con `.initial(pred_error: float, novelty: float) -> float`. Implementaciones: `Constant()`, `NoveltyGated()`, `PredErrorGated()`, `BothGated()`. Fórmula: `1.0 + 2.0*novelty` y/o `1.0 + 2.0*pred_error`, aditivas en `BothGated` (`1.0 + 2*novelty + 2*pred_error`).
  - `WritePolicy` con `.route(store, key) -> int | None` (índice a consolidar, o `None` para traza nueva). Implementaciones: `Append()`, `Merge(threshold=0.85)`.
  - `ReadPolicy` con `.select(sims) -> NDArray[int]`. Implementaciones: `NearestNeighbour()`, `TopK(k=3)`, `Radius(threshold=0.70)`.
  - `EvictPolicy` con `.victim(store, rng) -> int`. Implementaciones: `FIFO()`, `MinStrength()`, `MinUtility()`, `Random()`.
  - `DecayPolicy` con `.step(strength: NDArray) -> None` (in-place). Implementaciones: `NoDecay()`, `ExponentialDecay(rate)`.

Todas las políticas son `@dataclass(frozen=True)` para poder usarlas como clave
de caché y compararlas por igualdad.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/core/test_policies.py
import numpy as np
import pytest

from ember.core.policies import (
    Append,
    BothGated,
    Constant,
    ExponentialDecay,
    FIFO,
    Merge,
    MinStrength,
    MinUtility,
    NearestNeighbour,
    NoDecay,
    NoveltyGated,
    PredErrorGated,
    Radius,
    Random,
    TopK,
)
from ember.core.store import TraceStore


def _clave(rng, dim=8):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestFuerzaInicial:
    def test_constant_ignora_ambas_senales(self):
        p = Constant()
        assert p.initial(pred_error=0.9, novelty=0.9) == pytest.approx(1.0)
        assert p.initial(pred_error=0.1, novelty=0.1) == pytest.approx(1.0)

    def test_pred_error_escala_con_la_sorpresa(self):
        p = PredErrorGated()
        assert p.initial(pred_error=0.9, novelty=0.0) == pytest.approx(2.8)
        assert p.initial(pred_error=0.1, novelty=0.0) == pytest.approx(1.2)

    def test_novelty_escala_con_la_novedad(self):
        assert NoveltyGated().initial(pred_error=0.0, novelty=0.5) == pytest.approx(2.0)

    def test_both_suma_ambas(self):
        assert BothGated().initial(pred_error=0.9, novelty=0.5) == pytest.approx(3.8)


class TestEscritura:
    def test_append_nunca_consolida(self):
        store = TraceStore(dim=8, capacity=4)
        rng = np.random.default_rng(0)
        k = _clave(rng)
        store.append(k, "a", strength=1.0)
        assert Append().route(store, k) is None

    def test_merge_consolida_si_supera_el_umbral(self):
        store = TraceStore(dim=8, capacity=4)
        rng = np.random.default_rng(0)
        k = _clave(rng)
        store.append(k, "a", strength=1.0)
        assert Merge(threshold=0.85).route(store, k) == 0

    def test_merge_crea_traza_nueva_si_no_lo_supera(self):
        store = TraceStore(dim=8, capacity=4)
        rng = np.random.default_rng(0)
        store.append(_clave(rng), "a", strength=1.0)
        assert Merge(threshold=0.85).route(store, _clave(rng)) is None


class TestLectura:
    def test_nn_devuelve_exactamente_un_indice_el_mayor(self):
        sims = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        assert NearestNeighbour().select(sims).tolist() == [1]

    def test_topk_devuelve_k_indices_ordenados_por_similitud(self):
        sims = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
        assert TopK(k=3).select(sims).tolist() == [1, 3, 2]

    def test_topk_no_falla_si_hay_menos_trazas_que_k(self):
        assert TopK(k=3).select(np.array([0.4], dtype=np.float32)).tolist() == [0]

    def test_radius_devuelve_todos_los_que_superan_el_umbral(self):
        sims = np.array([0.1, 0.9, 0.75, 0.8], dtype=np.float32)
        assert sorted(Radius(threshold=0.70).select(sims).tolist()) == [1, 2, 3]

    def test_radius_cae_al_mas_cercano_si_ninguno_supera_el_umbral(self):
        sims = np.array([0.1, 0.3, 0.2], dtype=np.float32)
        assert Radius(threshold=0.70).select(sims).tolist() == [1]


class TestDesalojo:
    def _store_poblado(self):
        store = TraceStore(dim=8, capacity=10)
        rng = np.random.default_rng(0)
        for nombre, s in [("a", 3.0), ("b", 1.0), ("c", 2.0)]:
            store.append(_clave(rng), nombre, strength=s)
        store.age[:] = [10.0, 5.0, 1.0]
        store.utility[:] = [0.0, 7.0, 3.0]
        return store

    def test_fifo_descarta_el_mas_viejo(self):
        assert FIFO().victim(self._store_poblado(), np.random.default_rng(0)) == 0

    def test_min_strength_descarta_el_mas_debil(self):
        assert MinStrength().victim(self._store_poblado(), np.random.default_rng(0)) == 1

    def test_min_utility_descarta_el_menos_usado(self):
        assert MinUtility().victim(self._store_poblado(), np.random.default_rng(0)) == 0

    def test_random_es_determinista_bajo_la_misma_semilla(self):
        store = self._store_poblado()
        a = Random().victim(store, np.random.default_rng(42))
        b = Random().victim(store, np.random.default_rng(42))
        assert a == b
        assert 0 <= a < 3


class TestDecaimiento:
    def test_no_decay_no_toca_la_fuerza(self):
        s = np.array([1.0, 2.0], dtype=np.float32)
        NoDecay().step(s)
        assert s.tolist() == pytest.approx([1.0, 2.0])

    def test_exponential_multiplica_in_place(self):
        s = np.array([1.0, 2.0], dtype=np.float32)
        ExponentialDecay(rate=0.98).step(s)
        assert s.tolist() == pytest.approx([0.98, 1.96], abs=1e-6)
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `uv run pytest tests/core/test_policies.py -v`
Esperado: FAIL con `ModuleNotFoundError: No module named 'ember.core.policies'`.

- [ ] **Step 3: Implementar `policies.py`**

Cada familia como `Protocol` con `@runtime_checkable`, y las implementaciones
como `@dataclass(frozen=True)`. `TopK.select` usa `np.argsort(-sims)[:k]`.
`Radius.select` usa `np.where(sims >= threshold)[0]` con caída a
`np.array([int(sims.argmax())])` si queda vacío.

- [ ] **Step 4: Correr y verificar que pasa**

Run: `uv run pytest tests/core/test_policies.py -v`
Esperado: 17 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/core/policies.py tests/core/test_policies.py
git commit -m "feat(core): políticas componibles de fuerza, escritura, lectura, desalojo y decaimiento"
```

---

### Task 4: `PolicyMemory` y el genotipo

**Files:**
- Create: `src/ember/core/protocol.py`, `src/ember/core/memory.py`, `src/ember/core/genotype.py`
- Test: `tests/core/test_memory.py`

**Interfaces:**
- Consumes: `TraceStore` (Task 2), políticas (Task 3).
- Produces:
  - `Memory` Protocol: `write(key, value, pred_error=0.5) -> None`, `read(query) -> ReadResult`, `__len__() -> int`, propiedad `capacity: int`.
  - `Genotype` frozen dataclass con campos `read, write, strength, decay, evict, reinforce: float`; método `.label() -> str` y `.as_dict() -> dict[str, str]` para el análisis del NAS.
  - `PolicyMemory(dim, capacity, genotype, seed=0)` que implementa `Memory`.
  - `FIFO_GENOTYPE: Genotype` — el `EpisodicBuffer` de e-MDB.

Semántica de `write`:
1. `store.tick()` (envejece), luego `decay.step(store.strength)`.
2. Calcula `novelty = store.novelty(key)`.
3. `idx = write_policy.route(store, key)`. Si no es `None`: consolida — suma
   `strength.initial(...)` a `store.strength[idx]`, incrementa `utility[idx]`,
   pone `age[idx] = 0`, y **retorna sin desalojar**.
4. Si es `None`: `store.append(key, value, strength.initial(pred_error, novelty))`.
5. Mientras `len(store) > capacity`: `store.remove(evict.victim(store, rng))`.

Semántica de `read`: selecciona con `read_policy`, suma `reinforce` a la fuerza
de las trazas seleccionadas, incrementa su utilidad, y devuelve el `ReadResult`
de la traza con mayor similitud dentro de la selección.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/core/test_memory.py
import numpy as np
import pytest

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.memory import PolicyMemory
from ember.core.policies import (
    Append,
    Constant,
    ExponentialDecay,
    FIFO,
    MinStrength,
    NearestNeighbour,
    NoDecay,
    PredErrorGated,
)


def _claves(n, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _memoria(genotype, capacity=4, dim=16, seed=0):
    return PolicyMemory(dim=dim, capacity=capacity, genotype=genotype, seed=seed)


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
    g = Genotype(
        read=NearestNeighbour(),
        write=Append(),
        strength=PredErrorGated(),
        decay=NoDecay(),
        evict=MinStrength(),
        reinforce=0.0,
    )
    mem = _memoria(g, capacity=3)
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
        evict=__import__("ember.core.policies", fromlist=["Random"]).Random(),
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
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `uv run pytest tests/core/test_memory.py -v`
Esperado: FAIL con `ModuleNotFoundError: No module named 'ember.core.genotype'`.

- [ ] **Step 3: Implementar**

`PolicyMemory` expone `.store` como atributo público (los tests y el análisis lo
inspeccionan) y `.rng = np.random.default_rng(seed)`.

- [ ] **Step 4: Correr y verificar que pasa**

Run: `uv run pytest tests/core/test_memory.py -v`
Esperado: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/core tests/core/test_memory.py
git commit -m "feat(core): PolicyMemory y genotipo; el FIFO de e-MDB es un punto del espacio"
```

---

### Task 5: `SDMMemory` con superposición real y desalojo correcto

**Files:**
- Create: `src/ember/memories/__init__.py`, `src/ember/memories/sdm.py`
- Test: `tests/memories/test_sdm.py`

**Interfaces:**
- Consumes: `Memory`, `ReadResult`, `TraceStore`, políticas.
- Produces: `SDMMemory(dim, capacity, seed=0, n_hard=512, activation_frac=0.05, strength=PredErrorGated(), evict=MinStrength())`.

Dos arreglos respecto del piloto:
- **Contadores conservados.** `write` suma `s * k` a las hard locations activas y
  guarda `s` en `store.contribution[i]`. `_desalojar(i)` resta
  `contribution[i] * keys[i]` **sobre el mismo conjunto activo** que se usó al
  escribir, que se recalcula de forma determinista desde la clave.
- **Lectura distribuida real.** `read` suma los contadores activos, normaliza, y
  devuelve la traza más similar a la reconstrucción. Esto es la superposición de
  Kanerva que el eje `radius` del piloto nunca implementó.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/memories/test_sdm.py
import numpy as np
import pytest

from ember.memories.sdm import SDMMemory


def _claves(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_recupera_una_clave_exacta():
    mem = SDMMemory(dim=32, capacity=10, seed=0)
    claves = _claves(5)
    for i, k in enumerate(claves):
        mem.write(k, i, pred_error=0.5)
    for i, k in enumerate(claves):
        assert mem.read(k).value == i


def test_los_contadores_vuelven_a_cero_al_desalojar_todo():
    """Regresión del bug §2.2: el desalojo restaba 1x lo que la escritura sumó strength x."""
    mem = SDMMemory(dim=32, capacity=3, seed=0)
    for i, k in enumerate(_claves(40)):
        mem.write(k, i, pred_error=0.9 if i % 2 else 0.1)
    while len(mem):
        mem._desalojar(0)
    assert float(np.abs(mem.V).max()) == pytest.approx(0.0, abs=1e-4)


def test_degradacion_suave_ante_ruido():
    mem = SDMMemory(dim=32, capacity=10, seed=0)
    claves = _claves(10)
    for i, k in enumerate(claves):
        mem.write(k, i, pred_error=0.5)
    rng = np.random.default_rng(99)
    aciertos = sum(
        mem.read(
            (k + rng.standard_normal(32).astype(np.float32) * 0.2)
            / np.linalg.norm(k + rng.standard_normal(32).astype(np.float32) * 0.2)
        ).value
        == i
        for i, k in enumerate(claves)
    )
    assert aciertos >= 7


def test_misma_semilla_produce_hard_locations_identicas():
    a = SDMMemory(dim=32, capacity=5, seed=3)
    b = SDMMemory(dim=32, capacity=5, seed=3)
    assert np.array_equal(a.H, b.H)


def test_retiene_el_evento_sorpresivo_bajo_presion():
    mem = SDMMemory(dim=32, capacity=5, seed=0)
    claves = _claves(30)
    mem.write(claves[0], "raro", pred_error=0.9)
    for i, k in enumerate(claves[1:], start=1):
        mem.write(k, f"comun{i}", pred_error=0.1)
    assert mem.read(claves[0]).value == "raro"
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `uv run pytest tests/memories/test_sdm.py -v`
Esperado: FAIL con `ModuleNotFoundError: No module named 'ember.memories'`.

- [ ] **Step 3: Implementar**

`_conjunto_activo(k)` devuelve `np.argpartition(-(self.H @ k), self.k_active)[:self.k_active]`.
Es determinista dada `k`, que es lo que hace válido restar en el desalojo.

- [ ] **Step 4: Correr y verificar que pasa**

Run: `uv run pytest tests/memories/test_sdm.py -v`
Esperado: 5 PASS. También `uv run pytest tests/test_smoke.py -v` debe pasar ahora completo.

- [ ] **Step 5: Commit**

```bash
git add src/ember/memories tests/memories
git commit -m "feat(memories): SDM con superposición real y conservación de contadores"
```

---

### Task 6: `ENNMemory`

**Files:**
- Create: `src/ember/memories/enn.py`
- Test: `tests/memories/test_enn.py`

**Interfaces:**
- Produces: `ENNMemory(dim, capacity, seed=0, merge_threshold=0.85, decay=0.97, eta=0.6, reinforce=0.3)`.

Memoria Hebbiana: consolida por fusión si alguna traza supera el umbral coseno,
moviendo la traza hacia la clave nueva; si no, crea traza. Desaloja por mínima
fuerza.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/memories/test_enn.py
import numpy as np
import pytest

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
    """Documenta el modo de falla de la Tabla VI: la fusión ahoga la señal de saliencia."""
    rng = np.random.default_rng(0)
    mem = ENNMemory(dim=32, capacity=10, seed=0)
    base = _clave(rng)
    for i in range(30):
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
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `uv run pytest tests/memories/test_enn.py -v`
Esperado: FAIL, módulo inexistente.

- [ ] **Step 3: Implementar `enn.py`**

- [ ] **Step 4: Correr y verificar que pasa**

Run: `uv run pytest tests/memories/test_enn.py -v` → 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/memories/enn.py tests/memories/test_enn.py
git commit -m "feat(memories): ENN Hebbiana con consolidación por fusión"
```

---

### Task 7: `SpikingMemory` reproducible

**Files:**
- Create: `src/ember/memories/spiking.py`
- Test: `tests/memories/test_spiking.py`

**Interfaces:**
- Produces: `SpikingMemory(dim, capacity, seed=0, n_neurons=128, sparsity=0.15, ...)` con los mismos hiperparámetros LIF/STDP del piloto.

Arreglo respecto del piloto: `_entrenar_patron` recibe un `np.random.Generator`
derivado de la semilla de la instancia (`self.rng`), nunca de `id()`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/memories/test_spiking.py
import numpy as np
import pytest

from ember.memories.spiking import SpikingMemory


def _claves(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_misma_semilla_produce_pesos_identicos():
    """Regresión del bug §2.3: la semilla venía de id(), la dirección de memoria."""
    claves = _claves(3)
    pesos = []
    for _ in range(2):
        mem = SpikingMemory(dim=32, capacity=5, seed=5, n_presentations=3)
        for i, k in enumerate(claves):
            mem.write(k, i, pred_error=0.5)
        pesos.append(mem.W.copy())
    assert np.array_equal(pesos[0], pesos[1])


def test_semillas_distintas_producen_pesos_distintos():
    claves = _claves(3)
    pesos = []
    for semilla in (1, 2):
        mem = SpikingMemory(dim=32, capacity=5, seed=semilla, n_presentations=3)
        for i, k in enumerate(claves):
            mem.write(k, i, pred_error=0.5)
        pesos.append(mem.W.copy())
    assert not np.array_equal(pesos[0], pesos[1])


def test_stdp_fortalece_dentro_del_ensamble_mas_que_fuera():
    """El hallazgo de PoC-3: intra ~0.21 vs inter ~0.02."""
    mem = SpikingMemory(dim=32, capacity=5, seed=0, n_presentations=30)
    k = _claves(1)[0]
    mem.write(k, "a", pred_error=0.5)
    ens = mem.store_patterns[0]
    fuera = np.setdiff1d(np.arange(mem.n), ens)
    intra = mem.W[np.ix_(ens, ens)][mem.mask[np.ix_(ens, ens)]].mean()
    inter = mem.W[np.ix_(ens, fuera)][mem.mask[np.ix_(ens, fuera)]].mean()
    assert intra > 3 * inter


def test_nunca_excede_la_capacidad():
    mem = SpikingMemory(dim=32, capacity=3, seed=0, n_presentations=2)
    for i, k in enumerate(_claves(6)):
        mem.write(k, i, pred_error=0.5)
        assert len(mem) <= 3


def test_lectura_sobre_memoria_vacia_devuelve_none():
    mem = SpikingMemory(dim=32, capacity=3, seed=0, n_presentations=2)
    assert mem.read(_claves(1)[0]).value is None
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `uv run pytest tests/memories/test_spiking.py -v` → módulo inexistente.

- [ ] **Step 3: Implementar `spiking.py`**

Marcar la clase de tests con `@pytest.mark.slow` si `n_presentations >= 30`.

- [ ] **Step 4: Correr y verificar que pasa**

Run: `uv run pytest tests/memories/test_spiking.py -v` → 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/memories/spiking.py tests/memories/test_spiking.py
git commit -m "fix(memories): Spiking LIF+STDP con semilla explícita, ahora reproducible"
```

---

### Task 8: `SpikingSDMMemory` y el contrato común

**Files:**
- Create: `src/ember/memories/spiking_sdm.py`
- Modify: `src/ember/memories/__init__.py` (registro `ARCHITECTURES`)
- Test: `tests/memories/test_spiking_sdm.py`, `tests/memories/test_contrato.py`

**Interfaces:**
- Produces:
  - `SpikingSDMMemory(dim, capacity, seed=0, n_hard=256, activation_frac=0.05, T_window=15, ...)`.
  - `ARCHITECTURES: dict[str, type]` con claves `"SDM"`, `"ENN"`, `"Spiking"`, `"SpikingSDM"`, `"FIFO"`. `"FIFO"` mapea a una fábrica que devuelve `PolicyMemory` con `FIFO_GENOTYPE` — no una clase aparte.

El test de contrato es parametrizado sobre `ARCHITECTURES` y es la red de
seguridad de todo el repo: cualquier arquitectura nueva lo hereda gratis.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/memories/test_contrato.py
import numpy as np
import pytest

from ember.memories import ARCHITECTURES

RAPIDAS = {k: v for k, v in ARCHITECTURES.items() if k != "Spiking"}


def _claves(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(params=sorted(RAPIDAS))
def arquitectura(request):
    return request.param, RAPIDAS[request.param]


def test_lectura_vacia_devuelve_none(arquitectura):
    _, cls = arquitectura
    mem = cls(dim=32, capacity=5, seed=0)
    r = mem.read(_claves(1)[0])
    assert r.value is None and r.similarity == 0.0


def test_recupera_lo_que_acaba_de_escribir(arquitectura):
    _, cls = arquitectura
    mem = cls(dim=32, capacity=5, seed=0)
    k = _claves(1)[0]
    mem.write(k, "x", pred_error=0.5)
    assert mem.read(k).value == "x"


def test_nunca_excede_la_capacidad(arquitectura):
    _, cls = arquitectura
    mem = cls(dim=32, capacity=4, seed=0)
    for i, k in enumerate(_claves(30)):
        mem.write(k, i, pred_error=0.5)
        assert len(mem) <= 4


def test_la_similitud_esta_en_cero_uno(arquitectura):
    _, cls = arquitectura
    mem = cls(dim=32, capacity=5, seed=0)
    claves = _claves(5)
    for i, k in enumerate(claves):
        mem.write(k, i, pred_error=0.5)
    for k in claves:
        assert 0.0 <= mem.read(k).similarity <= 1.0


def test_es_determinista_bajo_la_misma_semilla(arquitectura):
    _, cls = arquitectura
    claves = _claves(25)

    def correr():
        mem = cls(dim=32, capacity=6, seed=13)
        for i, k in enumerate(claves):
            mem.write(k, i, pred_error=0.5 + 0.4 * (i % 2))
        return [mem.read(k).value for k in claves]

    assert correr() == correr()
```

Además `tests/memories/test_spiking_sdm.py` con: recuperación exacta,
conservación de contadores al desalojar (mismo test de regresión que SDM), y
determinismo del tren de spikes bajo misma semilla.

- [ ] **Step 2: Correr y verificar que falla**

Run: `uv run pytest tests/memories/ -v` → FAIL en los nuevos.

- [ ] **Step 3: Implementar**

`SpikingSDMMemory` hereda la contabilidad de contribución de la Task 5.
El generador de spikes usa `self.rng`, sembrado desde `seed`.

- [ ] **Step 4: Correr y verificar que pasa**

Run: `uv run pytest tests/memories/ -v` → todo PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/memories tests/memories
git commit -m "feat(memories): Spiking-SDM y test de contrato parametrizado sobre todas las arquitecturas"
```

---

### Task 9: Streams y generadores sintéticos estructurados

**Files:**
- Create: `src/ember/data/__init__.py`, `src/ember/data/streams.py`, `src/ember/data/synthetic.py`
- Test: `tests/data/test_synthetic.py`

**Interfaces:**
- Produces:
  - `StreamItem(NamedTuple)`: `key: NDArray`, `value: Any`, `pred_error: float`, `is_rare: bool`.
  - `StreamSpec(frozen dataclass)`: `n_prototypes: int`, `capacity: int`, `dim: int`, `source: str`; propiedad `r -> float` = `n_prototypes / capacity`.
  - `Stream(frozen dataclass)`: `items: list[StreamItem]`, `spec: StreamSpec`, `rare_keys: NDArray`; `__len__`, `__iter__`.
  - `clustered_stream(n_prototypes, capacity, n_common=300, n_rare=20, dim=32, noise=0.05, *, correlation=0.0, density_skew=0.0, drift=0.0, seed=0) -> Stream`.

Los tres parámetros de estructura son lo que el spec §3.3 pide y el gaussiano
i.i.d. del piloto es el caso `correlation=0, density_skew=0, drift=0`:
- `correlation`: covarianza no isotrópica en el espacio latente.
- `density_skew`: proporción de visitas por prototipo vía ley de potencias.
- `drift`: desplazamiento lento de los centros a lo largo del stream.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/data/test_synthetic.py
import numpy as np
import pytest

from ember.data.synthetic import clustered_stream


def test_el_spec_reporta_el_ratio_r():
    s = clustered_stream(n_prototypes=10, capacity=20, seed=0)
    assert s.spec.r == pytest.approx(0.5)


def test_hay_exactamente_n_rare_eventos_raros():
    s = clustered_stream(n_prototypes=5, capacity=20, n_common=100, n_rare=7, seed=0)
    assert sum(it.is_rare for it in s) == 7
    assert len(s) == 107


def test_los_raros_tienen_error_de_prediccion_alto_y_los_comunes_bajo():
    s = clustered_stream(n_prototypes=5, capacity=20, n_common=100, n_rare=10, seed=0)
    raros = [it.pred_error for it in s if it.is_rare]
    comunes = [it.pred_error for it in s if not it.is_rare]
    assert min(raros) > max(comunes)


def test_las_claves_son_unitarias_float32():
    s = clustered_stream(n_prototypes=5, capacity=20, seed=0)
    for it in list(s)[:20]:
        assert it.key.dtype == np.float32
        assert np.linalg.norm(it.key) == pytest.approx(1.0, abs=1e-5)


def test_la_misma_semilla_reproduce_el_stream():
    a = clustered_stream(n_prototypes=5, capacity=20, seed=3)
    b = clustered_stream(n_prototypes=5, capacity=20, seed=3)
    assert np.array_equal(np.stack([it.key for it in a]), np.stack([it.key for it in b]))


def test_correlation_reduce_la_ortogonalidad_media():
    """El gaussiano i.i.d. es casi ortogonal; con correlación deja de serlo."""

    def coseno_medio(correlation):
        s = clustered_stream(
            n_prototypes=40,
            capacity=20,
            n_common=200,
            n_rare=0,
            correlation=correlation,
            seed=0,
        )
        K = np.stack([it.key for it in s])
        G = np.abs(K @ K.T)
        return float(G[~np.eye(len(K), dtype=bool)].mean())

    assert coseno_medio(0.9) > coseno_medio(0.0) * 1.5


def test_density_skew_desbalancea_las_visitas_por_prototipo():
    def maxima_proporcion(skew):
        s = clustered_stream(
            n_prototypes=10,
            capacity=20,
            n_common=500,
            n_rare=0,
            density_skew=skew,
            seed=0,
        )
        _, cuentas = np.unique([it.value for it in s], return_counts=True)
        return cuentas.max() / cuentas.sum()

    assert maxima_proporcion(2.0) > maxima_proporcion(0.0) * 1.5


def test_drift_aleja_las_claves_tardias_de_las_tempranas():
    s = clustered_stream(n_prototypes=3, capacity=20, n_common=300, n_rare=0, drift=1.0, seed=0)
    K = np.stack([it.key for it in s])
    assert float(K[:20] @ K[-20:].T.mean(axis=1)).mean() < 0.9
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `uv run pytest tests/data/test_synthetic.py -v` → módulo inexistente.

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Correr y verificar que pasa**

Run: `uv run pytest tests/data/test_synthetic.py -v` → 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/data tests/data
git commit -m "feat(data): streams sintéticos con correlación, sesgo de densidad y drift"
```

---

### Task 10: Estimación de `K_proto`

**Files:**
- Create: `src/ember/data/prototypes.py`
- Test: `tests/data/test_prototypes.py`

**Interfaces:**
- Produces: `estimate_n_prototypes(keys: NDArray, *, k_max: int = 64, seed: int = 0) -> PrototypeEstimate` donde `PrototypeEstimate` es un frozen dataclass con `k_hat: int`, `ci_low: int`, `ci_high: int`, `scores: dict[int, float]`.

Implementación en numpy puro (el núcleo no puede depender de scipy/sklearn):
k-means con inicialización k-means++ sembrada, barrido de k, selección por
máximo del coeficiente de silueta medio, e IC por bootstrap sobre submuestras.

Esto es lo que hace la ley del ratio aplicable a datos reales, donde `K_proto`
no se conoce (spec §3.3).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/data/test_prototypes.py
import numpy as np
import pytest

from ember.data.prototypes import estimate_n_prototypes
from ember.data.synthetic import clustered_stream


@pytest.mark.parametrize("k_real", [3, 5, 12])
def test_recupera_el_numero_de_prototipos_conocido(k_real):
    s = clustered_stream(
        n_prototypes=k_real, capacity=20, n_common=400, n_rare=0, noise=0.05, seed=0
    )
    est = estimate_n_prototypes(np.stack([it.key for it in s]), k_max=24, seed=0)
    assert est.ci_low <= k_real <= est.ci_high


def test_el_intervalo_de_confianza_contiene_la_estimacion_puntual():
    s = clustered_stream(n_prototypes=6, capacity=20, n_common=300, n_rare=0, seed=0)
    est = estimate_n_prototypes(np.stack([it.key for it in s]), k_max=20, seed=0)
    assert est.ci_low <= est.k_hat <= est.ci_high


def test_es_determinista_bajo_la_misma_semilla():
    keys = np.stack([it.key for it in clustered_stream(n_prototypes=5, capacity=20, seed=0)])
    a = estimate_n_prototypes(keys, k_max=16, seed=1)
    b = estimate_n_prototypes(keys, k_max=16, seed=1)
    assert a.k_hat == b.k_hat and a.ci_low == b.ci_low


def test_datos_sin_estructura_no_producen_pocos_clusters():
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((300, 32)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    assert estimate_n_prototypes(keys, k_max=32, seed=0).k_hat > 8
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/data/test_prototypes.py -v`

- [ ] **Step 3: Implementar `prototypes.py`**

- [ ] **Step 4: Correr y verificar que pasa** → 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/data/prototypes.py tests/data/test_prototypes.py
git commit -m "feat(data): estimador de K_proto con IC bootstrap para datos sin etiquetas"
```

---

### Task 11: Embeddings perceptuales reales

**Files:**
- Create: `src/ember/data/embeddings.py`
- Test: `tests/data/test_embeddings.py`

**Interfaces:**
- Produces:
  - `extract_cifar100_embeddings(cache_dir: Path, *, dim: int = 32, device: str = "auto", seed: int = 0) -> EmbeddingBank` — descarga CIFAR-100, corre un ResNet-18 preentrenado de torchvision, reduce a `dim` por PCA, normaliza a la esfera, cachea a `.npz`.
  - `EmbeddingBank(frozen dataclass)`: `vectors: NDArray (n,dim) float32`, `labels: NDArray (n,) int`, `label_names: list[str]`.
  - `embedding_stream(bank, n_prototypes, capacity, n_common=300, n_rare=20, *, density_skew=0.0, seed=0) -> Stream` — construye un `Stream` usando `n_prototypes` clases como experiencias comunes y clases retenidas como eventos raros.

`embeddings.py` importa torch **dentro** de la función, no a nivel de módulo,
para no romper la restricción global del núcleo. `embedding_stream` solo usa
numpy y funciona sobre un banco ya cacheado.

Los tests que necesitan torch se marcan
`@pytest.mark.skipif(not torch_disponible, reason="requiere extra lab")` y los
que operan sobre un banco sintético corren siempre.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/data/test_embeddings.py
import numpy as np
import pytest

from ember.data.embeddings import EmbeddingBank, embedding_stream


@pytest.fixture
def banco():
    rng = np.random.default_rng(0)
    centros = rng.standard_normal((20, 32)).astype(np.float32)
    vectores, etiquetas = [], []
    for c in range(20):
        v = centros[c] + rng.standard_normal((50, 32)).astype(np.float32) * 0.3
        vectores.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        etiquetas.append(np.full(50, c))
    return EmbeddingBank(
        vectors=np.concatenate(vectores).astype(np.float32),
        labels=np.concatenate(etiquetas),
        label_names=[f"clase{i}" for i in range(20)],
    )


def test_el_stream_usa_exactamente_n_prototypes_clases_comunes(banco):
    s = embedding_stream(banco, n_prototypes=6, capacity=20, n_common=120, n_rare=5, seed=0)
    comunes = {it.value for it in s if not it.is_rare}
    assert len(comunes) == 6


def test_las_clases_raras_no_aparecen_entre_las_comunes(banco):
    s = embedding_stream(banco, n_prototypes=6, capacity=20, n_common=120, n_rare=5, seed=0)
    assert not ({it.value for it in s if it.is_rare} & {it.value for it in s if not it.is_rare})


def test_el_spec_registra_el_origen_y_el_ratio(banco):
    s = embedding_stream(banco, n_prototypes=10, capacity=20, seed=0)
    assert s.spec.source == "cifar100" and s.spec.r == pytest.approx(0.5)


def test_los_embeddings_reales_son_menos_ortogonales_que_el_gaussiano(banco):
    """La objeción de la §VI del paper: los vectores aleatorios inflan la precisión."""
    from ember.data.synthetic import clustered_stream

    real = np.stack([it.key for it in embedding_stream(banco, 10, 20, n_rare=0, seed=0)])
    sint = np.stack(
        [it.key for it in clustered_stream(n_prototypes=10, capacity=20, n_rare=0, seed=0)]
    )

    def coseno_medio(K):
        G = np.abs(K @ K.T)
        return float(G[~np.eye(len(K), dtype=bool)].mean())

    assert coseno_medio(real) > coseno_medio(sint)


def test_es_reproducible(banco):
    a = embedding_stream(banco, n_prototypes=5, capacity=20, seed=2)
    b = embedding_stream(banco, n_prototypes=5, capacity=20, seed=2)
    assert np.array_equal(np.stack([it.key for it in a]), np.stack([it.key for it in b]))
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/data/test_embeddings.py -v`

- [ ] **Step 3: Implementar `embeddings.py`**

- [ ] **Step 4: Correr y verificar que pasa** → 5 PASS.

- [ ] **Step 5: Instalar el extra y extraer el banco real**

```bash
uv pip install -e ".[lab]"
uv run python -c "
from pathlib import Path
from ember.data.embeddings import extract_cifar100_embeddings
b = extract_cifar100_embeddings(Path('data/cache'))
print(b.vectors.shape, len(b.label_names))
"
```
Esperado: `(50000, 32) 100`, y `data/cache/cifar100_resnet18_d32.npz` creado.

- [ ] **Step 6: Commit**

```bash
git add src/ember/data/embeddings.py tests/data/test_embeddings.py
git commit -m "feat(data): embeddings perceptuales de CIFAR-100 vía ResNet-18 + PCA"
```

---

### Task 12: Tareas de reconstrucción (R1–R4)

**Files:**
- Create: `src/ember/tasks/__init__.py`, `src/ember/tasks/protocol.py`, `src/ember/tasks/reconstruction.py`
- Test: `tests/tasks/test_reconstruction.py`

**Interfaces:**
- Produces:
  - `TaskResult(frozen dataclass)`: `name: str`, `score: float`, `detail: dict[str, float]`.
  - `r1_pattern_completion(factory, *, seed=0, dim=32, n_patterns=10, cue_fracs=(0.3,0.5,0.7)) -> TaskResult`
  - `r2_noise_robustness(factory, *, seed=0, dim=32, n_patterns=15, noise_levels=(0.1,0.2,0.3,0.4)) -> TaskResult`
  - `r3_ab_interference(factory, *, seed=0, dim=32, n_pairs=10) -> TaskResult`
  - `r4_capacity_profile(factory, *, seed=0, dim=32, noise=0.15, loads=(2,5,10,15,20,30)) -> TaskResult`
  - `reconstruction_gate(factory, *, seeds=(0,1,2,3), threshold=0.50) -> GateResult` con `.mean`, `.passes`, `.per_task: dict[str, float]`.

`factory` es `Callable[[int, int], Memory]` — recibe `(capacity, seed)` y
devuelve una memoria de dimensión `dim`. Uniformiza SDM, ENN, spiking y
genotipos bajo una sola firma.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/tasks/test_reconstruction.py
import pytest

from ember.core.genotype import FIFO_GENOTYPE
from ember.core.memory import PolicyMemory
from ember.memories import ARCHITECTURES
from ember.tasks.reconstruction import (
    r1_pattern_completion,
    r2_noise_robustness,
    r3_ab_interference,
    r4_capacity_profile,
    reconstruction_gate,
)

DIM = 32


def fifo_factory(capacity, seed):
    return PolicyMemory(dim=DIM, capacity=capacity, genotype=FIFO_GENOTYPE, seed=seed)


@pytest.mark.parametrize(
    "tarea", [r1_pattern_completion, r2_noise_robustness, r3_ab_interference, r4_capacity_profile]
)
def test_toda_tarea_devuelve_un_puntaje_en_cero_uno(tarea):
    r = tarea(fifo_factory, seed=0)
    assert 0.0 <= r.score <= 1.0
    assert r.name


@pytest.mark.parametrize(
    "tarea", [r1_pattern_completion, r2_noise_robustness, r3_ab_interference, r4_capacity_profile]
)
def test_toda_tarea_es_reproducible(tarea):
    assert tarea(fifo_factory, seed=1).score == tarea(fifo_factory, seed=1).score


def test_el_gate_admite_al_fifo_porque_sin_presion_es_busqueda_por_similitud():
    """Sin desalojo, el FIFO se reduce a vecino más cercano sobre todo lo guardado."""
    g = reconstruction_gate(fifo_factory, seeds=(0, 1))
    assert g.passes and g.mean > 0.5


def test_el_gate_reporta_las_cuatro_subtareas():
    g = reconstruction_gate(fifo_factory, seeds=(0,))
    assert set(g.per_task) == {
        "pattern_completion",
        "noise_robustness",
        "ab_interference",
        "capacity_profile",
    }


def test_sdm_supera_al_fifo_en_completado_de_patrones():
    sdm = ARCHITECTURES["SDM"]
    r_sdm = r1_pattern_completion(lambda c, s: sdm(dim=DIM, capacity=c, seed=s), seed=0)
    assert r_sdm.score > 0.5
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/tasks/test_reconstruction.py -v`

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Correr y verificar que pasa** → 11 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/tasks tests/tasks
git commit -m "feat(tasks): gate de reconstrucción R1-R4 sobre una firma de fábrica común"
```

---

### Task 13: Batería T1–T3 con los tres arreglos del addendum

**Files:**
- Create: `src/ember/tasks/battery.py`
- Test: `tests/tasks/test_battery.py`

**Interfaces:**
- Produces:
  - `t1_rare_retention(factory, stream, *, seed=0, read_every=0) -> TaskResult`
  - `t2_noise_under_pressure(factory, *, seed=0, dim=32, capacity=20, n_items=60, noise=0.30, n_queries=100) -> TaskResult`
  - `t3_sequential_interference(factory, *, seed=0, dim=32, capacity=20, n_blocks=4, per_block=8, pe_first=0.9, pe_rest=0.1) -> TaskResult`
  - `run_battery(factory, *, seeds=(0,1,2,3), stream_fn=None, read_every=0) -> BatteryResult` con `.per_task` y `.mean`.

Los tres arreglos del spec §2.4 / §3.4, cada uno con un test que lo fija:

- **T1** acepta `read_every=n`: emite una lectura cada `n` escrituras, de modo
  que `reinforce` pueda influir en desalojos. Con `read_every=0` conserva el
  comportamiento del piloto, para poder medir el efecto del arreglo.
- **T2** escribe `n_items=60` en capacidad 20. El piloto escribía 20 en 20: nunca
  desalojaba y devolvía 0.767 constante para las 576 arquitecturas.
- **T3** usa `pe_first=0.9` / `pe_rest=0.1`. El piloto usaba 0.5 uniforme, con lo
  que el desalojo por mínima fuerza degeneraba y todas las arquitecturas
  puntuaban 0.000.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/tasks/test_battery.py
import pytest

from ember.core.genotype import FIFO_GENOTYPE, Genotype
from ember.core.memory import PolicyMemory
from ember.core.policies import (
    Append,
    Constant,
    MinStrength,
    NearestNeighbour,
    NoDecay,
    PredErrorGated,
)
from ember.data.synthetic import clustered_stream
from ember.tasks.battery import (
    t1_rare_retention,
    t2_noise_under_pressure,
    t3_sequential_interference,
)

DIM = 32


def _fab(genotype, capacity_por_defecto=20):
    def factory(capacity, seed):
        return PolicyMemory(dim=DIM, capacity=capacity, genotype=genotype, seed=seed)

    return factory


FRONTERA = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=PredErrorGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)


class TestT2NoEsDegenerada:
    def test_t2_discrimina_entre_genotipos(self):
        """El piloto devolvía 0.767 para las 576 arquitecturas: varianza exactamente 0."""
        a = t2_noise_under_pressure(_fab(FIFO_GENOTYPE), seed=0).score
        b = t2_noise_under_pressure(_fab(FRONTERA), seed=0).score
        assert a != b

    def test_t2_ejerce_presion_de_capacidad(self):
        r = t2_noise_under_pressure(_fab(FIFO_GENOTYPE), seed=0, capacity=20, n_items=60)
        assert r.detail["evictions"] > 0


class TestT3ConSenalDiferenciada:
    def test_min_strength_protege_el_primer_bloque(self):
        """Con pe diferenciado la comparación deja de degenerar en 0.000 para todos."""
        r = t3_sequential_interference(_fab(FRONTERA), seed=0, pe_first=0.9, pe_rest=0.1)
        assert r.score > 0.5

    def test_el_fifo_no_lo_protege(self):
        r = t3_sequential_interference(_fab(FIFO_GENOTYPE), seed=0, pe_first=0.9, pe_rest=0.1)
        assert r.score < 0.3

    def test_con_pe_uniforme_ambos_degeneran(self):
        a = t3_sequential_interference(_fab(FRONTERA), seed=0, pe_first=0.5, pe_rest=0.5).score
        b = t3_sequential_interference(_fab(FIFO_GENOTYPE), seed=0, pe_first=0.5, pe_rest=0.5).score
        assert a == pytest.approx(b, abs=0.15)


class TestT1ConLecturasIntercaladas:
    def test_reinforce_no_hace_nada_sin_lecturas_intercaladas(self):
        """La causa raíz que identificó el addendum: el eje era inobservable por diseño."""
        stream = clustered_stream(n_prototypes=5, capacity=20, seed=0)
        sin = Genotype(**{**FRONTERA.__dict__, "reinforce": 0.0})
        con = Genotype(**{**FRONTERA.__dict__, "reinforce": 0.5})
        a = t1_rare_retention(_fab(sin), stream, seed=0, read_every=0).score
        b = t1_rare_retention(_fab(con), stream, seed=0, read_every=0).score
        assert a == pytest.approx(b)

    def test_reinforce_si_hace_algo_con_lecturas_intercaladas(self):
        stream = clustered_stream(n_prototypes=40, capacity=20, seed=0)
        sin = Genotype(**{**FRONTERA.__dict__, "strength": Constant(), "reinforce": 0.0})
        con = Genotype(**{**FRONTERA.__dict__, "strength": Constant(), "reinforce": 2.0})
        a = t1_rare_retention(_fab(sin), stream, seed=0, read_every=5).score
        b = t1_rare_retention(_fab(con), stream, seed=0, read_every=5).score
        assert a != b


def test_la_ley_del_ratio_se_ve_en_t1():
    """r<1: la fusión paga. r>1: no hay nada que comprimir."""
    from ember.core.policies import Merge

    fusion = Genotype(**{**FRONTERA.__dict__, "write": Merge(threshold=0.85)})
    bajo = clustered_stream(n_prototypes=5, capacity=20, seed=0)
    alto = clustered_stream(n_prototypes=40, capacity=20, seed=0)
    ventaja_bajo = (
        t1_rare_retention(_fab(fusion), bajo, seed=0).score
        - t1_rare_retention(_fab(FRONTERA), bajo, seed=0).score
    )
    ventaja_alto = (
        t1_rare_retention(_fab(fusion), alto, seed=0).score
        - t1_rare_retention(_fab(FRONTERA), alto, seed=0).score
    )
    assert ventaja_bajo > ventaja_alto
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/tasks/test_battery.py -v`

- [ ] **Step 3: Implementar `battery.py`**

`t2_noise_under_pressure` cuenta desalojos consultando `len(mem)` contra el
número de escrituras y lo reporta en `detail["evictions"]`.

- [ ] **Step 4: Correr y verificar que pasa** → 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/tasks/battery.py tests/tasks/test_battery.py
git commit -m "feat(tasks): batería T1-T3 con lecturas intercaladas, presión real en T2 y pe diferenciado en T3"
```

---

### Task 14: Espacio de búsqueda y motor del NAS

**Files:**
- Create: `src/ember/nas/__init__.py`, `src/ember/nas/space.py`, `src/ember/nas/engine.py`
- Test: `tests/nas/test_space.py`, `tests/nas/test_engine.py`

**Interfaces:**
- Produces:
  - `SEARCH_SPACE: dict[str, tuple]` con los seis ejes y sus opciones como objetos de política.
  - `AXIS_LABELS: dict[str, dict]` — mapea cada objeto de política a su etiqueta corta para el análisis.
  - `enumerate_space() -> Iterator[Genotype]` — 576 genotipos.
  - `run_search(evaluator, *, genotypes=None, n_jobs=-1, progress=True) -> SearchResults` con `.records: list[SearchRecord]`, `.to_json()`, `.best(n)`, `.rank_of(genotype)`.
  - `SearchRecord(frozen dataclass)`: `genotype: Genotype`, `scores: dict[str, float]`, `mean: float`.
  - `rank_of` devuelve `(rango_optimista, rango_pesimista)` — **una tupla, no un entero**. Es el arreglo del error que el addendum señala: el `#415 de 576` del piloto salió del orden de desempate del `sort`; el rango defendible del FIFO era #415–#576.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/nas/test_space.py
from ember.core.genotype import FIFO_GENOTYPE
from ember.nas.space import SEARCH_SPACE, enumerate_space


def test_el_espacio_tiene_576_genotipos():
    assert len(list(enumerate_space())) == 576


def test_no_hay_genotipos_duplicados():
    todos = list(enumerate_space())
    assert len(set(todos)) == len(todos)


def test_el_genotipo_fifo_esta_en_el_espacio():
    assert FIFO_GENOTYPE in set(enumerate_space())


def test_los_seis_ejes_estan_presentes():
    assert set(SEARCH_SPACE) == {"read", "write", "strength", "decay", "evict", "reinforce"}


def test_el_producto_de_las_cardinalidades_da_576():
    import math

    assert math.prod(len(v) for v in SEARCH_SPACE.values()) == 576
```

```python
# tests/nas/test_engine.py
import pytest

from ember.core.genotype import FIFO_GENOTYPE
from ember.nas.engine import run_search
from ember.nas.space import enumerate_space


def _evaluador_falso(genotype):
    """Puntaje sintético: depende solo del desalojo, para tener empates masivos."""
    from ember.core.policies import MinStrength

    return {"t1": 1.0 if isinstance(genotype.evict, MinStrength) else 0.0}


def test_devuelve_un_registro_por_genotipo():
    genos = list(enumerate_space())[:32]
    r = run_search(_evaluador_falso, genotypes=genos, n_jobs=1, progress=False)
    assert len(r.records) == 32


def test_los_registros_vienen_ordenados_de_mayor_a_menor():
    genos = list(enumerate_space())[:64]
    r = run_search(_evaluador_falso, genotypes=genos, n_jobs=1, progress=False)
    assert r.records == sorted(r.records, key=lambda x: -x.mean)


def test_rank_of_devuelve_un_rango_no_un_entero_cuando_hay_empates():
    """El '#415 de 576' del piloto era un artefacto del desempate del sort."""
    genos = list(enumerate_space())[:64]
    r = run_search(_evaluador_falso, genotypes=genos, n_jobs=1, progress=False)
    optimista, pesimista = r.rank_of(genos[0])
    assert optimista <= pesimista
    assert isinstance(optimista, int) and isinstance(pesimista, int)


def test_el_rango_es_ancho_cuando_todos_empatan():
    from ember.core.policies import MinStrength

    genos = [g for g in enumerate_space() if not isinstance(g.evict, MinStrength)][:40]
    r = run_search(_evaluador_falso, genotypes=genos, n_jobs=1, progress=False)
    optimista, pesimista = r.rank_of(genos[0])
    assert optimista == 1 and pesimista == 40


def test_es_reproducible_y_el_paralelismo_no_cambia_el_resultado():
    genos = list(enumerate_space())[:32]
    a = run_search(_evaluador_falso, genotypes=genos, n_jobs=1, progress=False)
    b = run_search(_evaluador_falso, genotypes=genos, n_jobs=4, progress=False)
    assert [x.mean for x in a.records] == [x.mean for x in b.records]
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/nas/ -v`

- [ ] **Step 3: Implementar**

`run_search` usa `concurrent.futures.ProcessPoolExecutor` cuando `n_jobs != 1`,
y ordena de forma estable por `(-mean, genotype.label())` para que el orden sea
determinista independientemente del orden de finalización.

- [ ] **Step 4: Correr y verificar que pasa** → 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/nas tests/nas
git commit -m "feat(nas): espacio de 576 genotipos y motor paralelo con rangos de empate honestos"
```

---

### Task 15: Estadística del NAS

**Files:**
- Create: `src/ember/nas/stats.py`
- Test: `tests/nas/test_stats.py`

**Interfaces:**
- Produces:
  - `eta_squared(records, axis, metric="mean") -> float` — η² de un factor.
  - `partial_eta_squared(records, factors, metric="mean") -> dict[str, float]` — ANOVA factorial con interacciones.
  - `conditional_effect(records, axis, given: dict, metric) -> dict[str, float]` — el efecto de un eje condicionado a otros. Es lo que produce la Tabla IV.
  - `axis_liveness(records, axis, metric="mean") -> float` — máxima diferencia entre genotipos hermanos que solo difieren en `axis`. Devuelve `0.0` si el eje es inobservable.
  - `bootstrap_ci(values, *, n_boot=2000, alpha=0.05, seed=0) -> tuple[float, float]`.

`axis_liveness` es la auditoría del addendum convertida en función: el piloto
tenía diferencia máxima exactamente 0.000 en `read_mode` y `reinforce`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/nas/test_stats.py
import pytest

from ember.core.genotype import Genotype
from ember.nas.engine import SearchRecord
from ember.nas.space import enumerate_space
from ember.nas.stats import axis_liveness, bootstrap_ci, conditional_effect, eta_squared


def _registros(fn):
    return [SearchRecord(genotype=g, scores={"m": fn(g)}, mean=fn(g)) for g in enumerate_space()]


def test_eta2_es_1_cuando_un_solo_eje_explica_todo():
    from ember.core.policies import MinStrength

    recs = _registros(lambda g: 1.0 if isinstance(g.evict, MinStrength) else 0.0)
    assert eta_squared(recs, "evict") == pytest.approx(1.0)


def test_eta2_es_0_cuando_el_eje_no_influye():
    from ember.core.policies import MinStrength

    recs = _registros(lambda g: 1.0 if isinstance(g.evict, MinStrength) else 0.0)
    assert eta_squared(recs, "read") == pytest.approx(0.0, abs=1e-9)


def test_axis_liveness_detecta_un_eje_muerto():
    """Reproduce el hallazgo del addendum: diferencia máxima exactamente 0.000."""
    from ember.core.policies import MinStrength

    recs = _registros(lambda g: 1.0 if isinstance(g.evict, MinStrength) else 0.0)
    assert axis_liveness(recs, "read") == pytest.approx(0.0)
    assert axis_liveness(recs, "reinforce") == pytest.approx(0.0)


def test_axis_liveness_detecta_un_eje_vivo():
    from ember.core.policies import MinStrength

    recs = _registros(lambda g: 1.0 if isinstance(g.evict, MinStrength) else 0.0)
    assert axis_liveness(recs, "evict") == pytest.approx(1.0)


def test_el_efecto_condicional_revela_lo_que_el_efecto_principal_esconde():
    """El resultado de la Tabla IV: la saliencia solo importa si el desalojo la lee."""
    from ember.core.policies import Constant, MinStrength

    def puntaje(g):
        if not isinstance(g.evict, MinStrength):
            return 0.5  # la fuerza nunca se lee
        return 0.0 if isinstance(g.strength, Constant) else 1.0

    recs = _registros(puntaje)
    assert eta_squared(recs, "strength") < 0.30  # efecto principal chico
    cond = conditional_effect(recs, "strength", given={"evict": "min_strength"}, metric="mean")
    assert max(cond.values()) - min(cond.values()) == pytest.approx(1.0)


def test_bootstrap_ci_contiene_la_media_y_es_reproducible():
    valores = [0.4, 0.5, 0.6, 0.55, 0.45, 0.5, 0.52, 0.48]
    lo, hi = bootstrap_ci(valores, seed=0)
    assert lo < sum(valores) / len(valores) < hi
    assert (lo, hi) == bootstrap_ci(valores, seed=0)
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/nas/test_stats.py -v`

- [ ] **Step 3: Implementar `stats.py`** (numpy puro, sin scipy).

- [ ] **Step 4: Correr y verificar que pasa** → 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/nas/stats.py tests/nas/test_stats.py
git commit -m "feat(nas): eta2 parcial, efectos condicionales, liveness de ejes e IC bootstrap"
```

---

### Task 16: Infraestructura de experimentos

**Files:**
- Create: `src/ember/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Produces:
  - `ExperimentRun(name: str, *, results_dir: Path = Path("results"))` como context manager: `with ExperimentRun("exp01_nas_full") as run: run.record(clave, valor)`.
  - Al salir escribe `results/<name>/data.json` y `results/<name>/manifest.json`.
  - El manifiesto contiene: `git_sha`, `git_dirty` (bool), `timestamp_utc`, `python`, `packages` (nombre→versión de numpy/torch si están), `seeds`, `duration_s`, `ember_version`.
  - `run.set_seeds(seq)` registra las semillas usadas.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_experiment.py
import json

from ember.experiment import ExperimentRun


def test_escribe_datos_y_manifiesto(tmp_path):
    with ExperimentRun("prueba", results_dir=tmp_path) as run:
        run.set_seeds([0, 1, 2])
        run.record("tabla", [{"a": 1}])
    datos = json.loads((tmp_path / "prueba" / "data.json").read_text())
    man = json.loads((tmp_path / "prueba" / "manifest.json").read_text())
    assert datos["tabla"] == [{"a": 1}]
    assert man["seeds"] == [0, 1, 2]


def test_el_manifiesto_registra_la_procedencia(tmp_path):
    with ExperimentRun("prueba", results_dir=tmp_path) as run:
        run.record("x", 1)
    man = json.loads((tmp_path / "prueba" / "manifest.json").read_text())
    for campo in (
        "git_sha",
        "git_dirty",
        "timestamp_utc",
        "python",
        "packages",
        "duration_s",
        "ember_version",
    ):
        assert campo in man, campo
    assert man["packages"]["numpy"]


def test_una_excepcion_no_deja_resultados_a_medias(tmp_path):
    try:
        with ExperimentRun("prueba", results_dir=tmp_path) as run:
            run.record("x", 1)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not (tmp_path / "prueba" / "data.json").exists()
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/test_experiment.py -v`

- [ ] **Step 3: Implementar `experiment.py`**

- [ ] **Step 4: Correr y verificar que pasa** → 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/experiment.py tests/test_experiment.py
git commit -m "feat: ExperimentRun con manifiesto de procedencia"
```

---

### Task 17: `exp01` búsqueda exhaustiva y `exp03` auditoría de ejes

**Files:**
- Create: `experiments/__init__.py`, `experiments/exp01_nas_full.py`, `experiments/exp03_axis_liveness.py`
- Test: `tests/test_exp03_liveness.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: `results/exp01_nas_full/`, `results/exp03_axis_liveness/`.

`exp01` evalúa los 576 genotipos sobre T1/T2/T3 con 3 semillas y reporta η² por
eje (Tabla II), el puntaje del FIFO y su rango honesto (optimista–pesimista), y
los mejores genotipos.

`exp03` corre `axis_liveness` sobre cada uno de los seis ejes y **falla con
código de salida distinto de cero si alguno da 0.000**. Corre en CI.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_exp03_liveness.py
import pytest

from experiments.exp03_axis_liveness import medir_liveness

AXES = ("read", "write", "strength", "decay", "evict", "reinforce")


@pytest.mark.slow
def test_ningun_eje_del_espacio_es_inobservable():
    """El addendum encontró read_mode y reinforce muertos. No pueden volver."""
    liveness = medir_liveness(seeds=(0,), muestra=96)
    muertos = [eje for eje, v in liveness.items() if v < 1e-6]
    assert not muertos, f"ejes inobservables: {muertos}"


@pytest.mark.slow
def test_reporta_los_seis_ejes():
    assert set(medir_liveness(seeds=(0,), muestra=48)) == set(AXES)
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/test_exp03_liveness.py -v -m slow`

- [ ] **Step 3: Implementar ambos scripts**

`medir_liveness(seeds, muestra)` es importable y testeable; el `__main__` del
script la llama con los parámetros completos y escribe el `ExperimentRun`.

- [ ] **Step 4: Correr experimento y tests**

```bash
uv run pytest tests/test_exp03_liveness.py -v -m slow
uv run python -m experiments.exp01_nas_full
uv run python -m experiments.exp03_axis_liveness
```
Esperado: tests PASS; `exp03` sale con código 0 y ningún eje en 0.000.

- [ ] **Step 5: Commit**

```bash
git add experiments tests/test_exp03_liveness.py results/exp01_nas_full results/exp03_axis_liveness
git commit -m "feat(exp): búsqueda exhaustiva y auditoría de observabilidad de ejes"
```

---

### Task 18: `exp02` — la grilla del umbral (Figura 1)

**Files:**
- Create: `experiments/exp02_threshold_grid.py`, `src/ember/figures.py`
- Test: `tests/test_exp02_grid.py`

**Interfaces:**
- Produces:
  - `barrer_grilla(capacities, prototypes, *, seeds, n_jobs) -> list[dict]` — cada celda con `capacity`, `n_prototypes`, `r`, `eta2_write`, `eta2_evict`, IC bootstrap de cada uno, y el genotipo de la frontera.
  - `results/exp02_threshold_grid/`, `paper/figures/fig1_threshold.pdf`.

Grilla expandida según el addendum §B2: capacidades 10–160, prototipos 1–320,
10 semillas, IC bootstrap. El piloto tenía 14 celdas con 3 semillas y sin IC.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_exp02_grid.py
import pytest

from experiments.exp02_threshold_grid import barrer_grilla


@pytest.mark.slow
def test_cada_celda_reporta_ratio_e_intervalos():
    celdas = barrer_grilla(capacities=(20,), prototypes=(5, 40), seeds=(0, 1), n_jobs=1)
    assert len(celdas) == 2
    for c in celdas:
        assert c["r"] == pytest.approx(c["n_prototypes"] / c["capacity"])
        assert c["eta2_write_ci"][0] <= c["eta2_write"] <= c["eta2_write_ci"][1]


@pytest.mark.slow
def test_la_transicion_de_regimen_ocurre_en_r_igual_a_1():
    """El claim central: bajo r=1 domina la fusión, sobre r=1 domina el desalojo."""
    celdas = barrer_grilla(capacities=(20,), prototypes=(4, 10, 40, 80), seeds=(0, 1, 2), n_jobs=-1)
    bajo = [c for c in celdas if c["r"] <= 0.5]
    alto = [c for c in celdas if c["r"] >= 1.0]
    assert min(c["eta2_write"] for c in bajo) > max(c["eta2_write"] for c in alto)
    assert min(c["eta2_evict"] for c in alto) > max(c["eta2_evict"] for c in bajo)


@pytest.mark.slow
def test_el_umbral_no_se_mueve_con_la_capacidad():
    """La variable de control es el ratio, no la redundancia del stream."""
    celdas = barrer_grilla(capacities=(10, 40), prototypes=(5, 20, 80), seeds=(0, 1), n_jobs=-1)
    for cap in (10, 40):
        de_esta = sorted((c for c in celdas if c["capacity"] == cap), key=lambda c: c["r"])
        dominios = ["write" if c["eta2_write"] > c["eta2_evict"] else "evict" for c in de_esta]
        cruce = next(i for i, d in enumerate(dominios) if d == "evict")
        assert de_esta[cruce - 1]["r"] < 1.0 <= de_esta[cruce]["r"]
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/test_exp02_grid.py -v -m slow`

- [ ] **Step 3: Implementar**

`src/ember/figures.py` genera la Figura 1: η² de escritura y de desalojo contra
`r` en escala log, con bandas de IC y una línea vertical en `r=1`, una serie por
capacidad. Importa matplotlib de forma perezosa (extra `lab`).

- [ ] **Step 4: Correr y verificar**

```bash
uv run pytest tests/test_exp02_grid.py -v -m slow
uv run python -m experiments.exp02_threshold_grid
```

- [ ] **Step 5: Commit**

```bash
git add experiments/exp02_threshold_grid.py src/ember/figures.py tests/test_exp02_grid.py results/exp02_threshold_grid paper/figures
git commit -m "feat(exp): grilla del umbral expandida con IC bootstrap — Figura 1"
```

---

### Task 19: `exp04` — benchmark de arquitecturas en dos fases

**Files:**
- Create: `experiments/exp04_arch_benchmark.py`
- Test: `tests/test_exp04_benchmark.py`

**Interfaces:**
- Produces: `results/exp04_arch_benchmark/` con las Tablas V y VI, ahora medidas sobre el motor unificado.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_exp04_benchmark.py
import pytest

from experiments.exp04_arch_benchmark import correr_benchmark


@pytest.mark.slow
def test_reporta_gate_y_bateria_para_cada_arquitectura():
    r = correr_benchmark(seeds=(0,), incluir_lentas=False)
    assert set(r) >= {"SDM", "ENN", "SpikingSDM", "FIFO"}
    for datos in r.values():
        assert "gate" in datos
        if datos["gate"]["passes"]:
            assert "battery" in datos


@pytest.mark.slow
def test_el_fifo_queda_en_el_piso_de_retencion_de_eventos_raros():
    """El invariante que el paper defiende, ahora sobre contadores no contaminados."""
    r = correr_benchmark(seeds=(0, 1), incluir_lentas=False)
    admitidas = {k: v for k, v in r.items() if v["gate"]["passes"]}
    raras = {k: v["battery"]["per_task"]["rare_retention"] for k, v in admitidas.items()}
    assert raras["FIFO"] == min(raras.values())


@pytest.mark.slow
def test_sdm_supera_al_fifo_en_retencion_de_eventos_raros():
    r = correr_benchmark(seeds=(0, 1), incluir_lentas=False)
    assert (
        r["SDM"]["battery"]["per_task"]["rare_retention"]
        > r["FIFO"]["battery"]["per_task"]["rare_retention"]
    )
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/test_exp04_benchmark.py -v -m slow`

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Correr y comparar contra el piloto**

```bash
uv run pytest tests/test_exp04_benchmark.py -v -m slow
uv run python -m experiments.exp04_arch_benchmark
```

Anotar en `docs/notas/2026-08-numeros-movidos.md` qué números cambiaron respecto
del draft y por qué (arreglo de contadores, motor unificado, umbrales de acierto
unificados). Es el riesgo declarado en §5 del spec.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp04_arch_benchmark.py tests/test_exp04_benchmark.py results/exp04_arch_benchmark docs/notas
git commit -m "feat(exp): benchmark de dos fases sobre el motor unificado"
```

---

### Task 20: `exp05` — la ley del ratio sobre embeddings reales

**Files:**
- Create: `experiments/exp05_real_embeddings.py`
- Test: `tests/test_exp05_embeddings.py`

**Interfaces:**
- Produces: `results/exp05_real_embeddings/`, `paper/figures/fig2_real_vs_synthetic.pdf`.

Corre la grilla de `exp02` sobre streams de CIFAR-100 en vez de sintéticos, con
`K_proto` estimado por `estimate_n_prototypes` y reportado con su IC. Compara la
posición del umbral entre dominio sintético y dominio real — que es exactamente
la pregunta que la §VI del paper deja abierta.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_exp05_embeddings.py
import pytest

from experiments.exp05_real_embeddings import comparar_dominios


@pytest.mark.slow
def test_reporta_el_umbral_en_ambos_dominios_con_intervalo():
    r = comparar_dominios(capacities=(20,), prototypes=(5, 40), seeds=(0, 1))
    for dominio in ("synthetic", "cifar100"):
        assert dominio in r
        assert "r_umbral" in r[dominio]
        lo, hi = r[dominio]["r_umbral_ci"]
        assert lo <= r[dominio]["r_umbral"] <= hi


@pytest.mark.slow
def test_el_ratio_real_usa_k_estimado_no_el_numero_de_clases():
    """Con datos reales K_proto no se conoce: la ley se enuncia sobre K estimado."""
    r = comparar_dominios(capacities=(20,), prototypes=(10,), seeds=(0,))
    celda = r["cifar100"]["celdas"][0]
    assert "k_hat" in celda and "k_hat_ci" in celda
    assert celda["r"] == pytest.approx(celda["k_hat"] / celda["capacity"])
```

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/test_exp05_embeddings.py -v -m slow`

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Correr**

```bash
uv run pytest tests/test_exp05_embeddings.py -v -m slow
uv run python -m experiments.exp05_real_embeddings
```

- [ ] **Step 5: Commit**

```bash
git add experiments/exp05_real_embeddings.py tests/test_exp05_embeddings.py results/exp05_real_embeddings paper/figures
git commit -m "feat(exp): ley del ratio sobre embeddings de CIFAR-100 con K estimado"
```

---

### Task 21: `ember.envs` — adaptadores secuenciales

**Files:**
- Create: `src/ember/envs/__init__.py`, `src/ember/envs/minigrid.py`
- Test: `tests/envs/test_minigrid.py`

**Interfaces:**
- Produces:
  - `MiniGridStreamAdapter(env_id="MiniGrid-MemoryS13-v0", *, dim=32, seed=0)` con `.rollout(n_steps, policy=None) -> Stream`.
  - Codifica la observación parcial a un vector unitario de dimensión `dim` y usa el error de predicción de un modelo de transición de un paso como `pred_error`.

Cableado ahora, corrido después del envío (spec §4). Los tests se saltan si el
extra `envs` no está instalado.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/envs/test_minigrid.py
import numpy as np
import pytest

minigrid = pytest.importorskip("minigrid", reason="requiere el extra envs")

from ember.envs.minigrid import MiniGridStreamAdapter


def test_el_rollout_produce_un_stream_del_largo_pedido():
    a = MiniGridStreamAdapter(seed=0)
    s = a.rollout(n_steps=50)
    assert len(s) == 50


def test_las_claves_son_unitarias_de_la_dimension_pedida():
    a = MiniGridStreamAdapter(dim=32, seed=0)
    for it in list(a.rollout(n_steps=20))[:10]:
        assert it.key.shape == (32,)
        assert np.linalg.norm(it.key) == pytest.approx(1.0, abs=1e-5)


def test_el_error_de_prediccion_esta_en_cero_uno():
    a = MiniGridStreamAdapter(seed=0)
    assert all(0.0 <= it.pred_error <= 1.0 for it in a.rollout(n_steps=30))


def test_es_reproducible():
    x = np.stack([it.key for it in MiniGridStreamAdapter(seed=4).rollout(20)])
    y = np.stack([it.key for it in MiniGridStreamAdapter(seed=4).rollout(20)])
    assert np.array_equal(x, y)
```

- [ ] **Step 2: Correr y verificar que falla**

```bash
uv pip install -e ".[envs]"
uv run pytest tests/envs/test_minigrid.py -v
```

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Correr y verificar que pasa** → 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ember/envs tests/envs
git commit -m "feat(envs): adaptador MiniGrid a Stream con pred_error de modelo de transición"
```

---

### Task 22: Configuración de Claude y documentación

**Files:**
- Create: `CLAUDE.md`, `README.md` (reescribir)
- Create: `.claude/settings.json`
- Create: `.claude/skills/ember-domain/SKILL.md`
- Create: `.claude/skills/ember-experiment/SKILL.md`
- Create: `.claude/skills/ember-memory/SKILL.md`
- Create: `.claude/skills/paper-sync/SKILL.md`

**Interfaces:**
- Produces: contexto de proyecto para sesiones futuras.

Contenido de cada skill, según spec §3.7:

- **`ember-domain`** — glosario operativo: SDM y hard locations, engrama y LTP,
  STDP, e-MDB y el problema LOLA, `EpisodicBuffer`, la ley del ratio `r=K/C` y
  sus dos regímenes, qué es un genotipo. Incluye los cuatro defectos del piloto
  (spec §2) para que nadie los reintroduzca.
- **`ember-experiment`** — cómo agregar un experimento: `ExperimentRun`,
  propagación de semillas, dónde escribe, cómo se cita desde el paper, y la
  regla de que ningún experimento usa aleatoriedad no sembrada.
- **`ember-memory`** — cómo agregar una arquitectura: implementar el protocolo
  `Memory`, registrarla en `ARCHITECTURES`, y que los tests de contrato de
  `tests/memories/test_contrato.py` la recogen automáticamente. Incluye la regla
  de conservación de contadores para sustratos distribuidos.
- **`paper-sync`** — regenerar tablas y figuras del LaTeX desde `results/*.json`
  y verificar que ningún número publicado difiera del medido.

- [ ] **Step 1: Escribir `CLAUDE.md`**

Debe cubrir: qué es EMBER, comandos (`uv run pytest`, `uv run ruff check .`,
`uv run python -m experiments.expNN_...`), el layout de `src/ember/`, y los
invariantes de la sección "Global Constraints" de este plan.

- [ ] **Step 2: Escribir las cuatro skills**

Cada `SKILL.md` con frontmatter `name` y `description` que empiece con "Use
when...".

- [ ] **Step 3: Escribir `.claude/settings.json`**

```json
{
  "permissions": {
    "allow": [
      "Bash(uv run pytest:*)",
      "Bash(uv run ruff:*)",
      "Bash(uv run python -m experiments.:*)",
      "Bash(uv pip install:*)",
      "Bash(git status:*)",
      "Bash(git diff:*)",
      "Bash(git log:*)"
    ]
  }
}
```

- [ ] **Step 4: Reescribir `README.md`**

Qué es EMBER, instalación (`uv pip install -e ".[lab]"`), cómo reproducir cada
tabla y figura del paper, y el layout.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md .claude
git commit -m "docs: CLAUDE.md, README y skills de dominio"
```

---

### Task 23: Migración del código piloto y del paper

**Files:**
- Move: `PoC*.ipynb` → `notebooks/`
- Move: `*.docx`, `BIP2026_EMBER_draft.pdf`, `Addendum_piloto_BIP2026.md` → `docs/borradores/`
- Delete: `memory_space.py`, `search.py`, `analyze.py`, `architectures.py`, `benchmark.py`, `*.png` de la raíz
- Create: `paper/main.tex`, `paper/refs.bib`, `paper/tables/.gitkeep`, `paper/figures/.gitkeep`
- Create: `src/ember/paper_sync.py`
- Test: `tests/test_paper_sync.py`

**Interfaces:**
- Produces: `render_tables(results_dir, out_dir) -> list[Path]` — genera los `.tex` de las Tablas II–VI desde los JSON. `verify_paper(tex_path, results_dir) -> list[Discrepancia]` — devuelve vacío si todo número publicado coincide con el medido.

El código piloto se borra, no se conserva: su historia está en git y mantenerlo
al lado del motor nuevo invita a que alguien lo importe por error. El PDF del
draft y el addendum se conservan como documentos.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_paper_sync.py
import json

from ember.paper_sync import render_tables, verify_paper


def test_genera_un_tex_por_tabla(tmp_path):
    resultados = tmp_path / "results" / "exp01_nas_full"
    resultados.mkdir(parents=True)
    (resultados / "data.json").write_text(json.dumps({"eta2": {"write": 0.635, "evict": 0.099}}))
    salidas = render_tables(tmp_path / "results", tmp_path / "tex")
    assert salidas and all(p.suffix == ".tex" and p.exists() for p in salidas)


def test_verify_no_reporta_nada_si_los_numeros_coinciden(tmp_path):
    resultados = tmp_path / "results" / "exp01_nas_full"
    resultados.mkdir(parents=True)
    (resultados / "data.json").write_text(json.dumps({"eta2": {"write": 0.635}}))
    tex = tmp_path / "main.tex"
    tex.write_text(r"write mode explains \result{exp01_nas_full:eta2.write}{63.5}\% ")
    assert verify_paper(tex, tmp_path / "results") == []


def test_verify_detecta_un_numero_publicado_que_ya_no_se_mide(tmp_path):
    resultados = tmp_path / "results" / "exp01_nas_full"
    resultados.mkdir(parents=True)
    (resultados / "data.json").write_text(json.dumps({"eta2": {"write": 0.412}}))
    tex = tmp_path / "main.tex"
    tex.write_text(r"write mode explains \result{exp01_nas_full:eta2.write}{63.5}\% ")
    d = verify_paper(tex, tmp_path / "results")
    assert len(d) == 1 and d[0].publicado == 63.5 and d[0].medido == 41.2
```

El macro `\result{clave}{valor}` en el LaTeX se renderiza como el valor, y
`verify_paper` compara ese valor contra el JSON. Así el paper no puede
desincronizarse en silencio.

- [ ] **Step 2: Correr y verificar que falla.** Run: `uv run pytest tests/test_paper_sync.py -v`

- [ ] **Step 3: Implementar `paper_sync.py` y mover los archivos**

```bash
mkdir -p notebooks docs/borradores paper/tables paper/figures
git mv PoC1_SDM_EMBER.ipynb PoC2_ENN_EMBER.ipynb PoC3_Spiking_STDP_EMBER.ipynb PoC4_SpikingSDM_EMBER.ipynb notebooks/
git mv BIP2026_EMBER_draft.pdf Addendum_piloto_BIP2026.md docs/borradores/
git mv BIP2026_NAS_Memory_Draft.docx EMBER_documento_de_trabajo.docx Propuesta_BIP2026_NAS_Memoria_Bioinspirada.docx docs/borradores/
git rm memory_space.py search.py analyze.py architectures.py benchmark.py
git rm ember_eval_comparison.png nas_pilot_analysis.png
```

- [ ] **Step 4: Correr la suite completa**

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -v
uv run python -m experiments.exp03_axis_liveness
```
Esperado: todo verde, `exp03` con código de salida 0.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: migrar notebooks y borradores, retirar el código piloto"
```

---

## Self-Review

**Cobertura del spec:**

| Sección del spec | Task |
|---|---|
| §2.1 dos motores | 3, 4, 8 (`ARCHITECTURES["FIFO"]` es un genotipo) |
| §2.2 bug de contadores | 5 (`test_los_contadores_vuelven_a_cero_al_desalojar_todo`), 8 |
| §2.3 semilla por `id()` | 7 (`test_misma_semilla_produce_pesos_identicos`) |
| §2.4 `radius` sin superposición | 5 (lectura distribuida real) |
| §2.4 `reinforce` inobservable | 13 (`read_every`), 15 (`axis_liveness`), 17 (`exp03`) |
| §2.4 T2 degenerada | 13 (`TestT2NoEsDegenerada`) |
| §3.1 layout y extras | 1, 22 |
| §3.2 políticas componibles | 2, 3, 4 |
| §3.3 datos sintéticos estructurados | 9 |
| §3.3 K̂ para datos reales | 10, 20 |
| §3.3 embeddings perceptuales | 11 |
| §3.4 tareas | 12, 13 |
| §3.5 experimentos | 16, 17, 18, 19, 20 |
| §3.6 testing (4 categorías) | contrato 8, regresión 5/7, liveness 15/17, golden 19 |
| §3.7 configuración de Claude | 22 |
| §3.8 entorno y CI | 1 |
| §4 `exp06` post-envío | 21 (cableado, no corrido) |
| §5 riesgo de números movidos | 19 (`docs/notas/`) |

**Sin placeholders:** cada paso de código lleva el código real. Los pasos de
implementación que no muestran cuerpo completo (Tasks 5–11, 13–21) están
determinados por los tests, que sí están completos y son la especificación
ejecutable.

**Consistencia de tipos:** `ReadResult` (Task 2) se usa en 4, 5, 8, 12. `Stream`
y `StreamSpec` (Task 9) en 11, 13, 18, 20, 21. `Genotype` (Task 4) en 14, 15,
17, 18. `TaskResult` (Task 12) en 13. `factory: Callable[[int, int], Memory]`
es la misma firma en 12, 13, 19. `SearchRecord` (Task 14) en 15.

**Un ajuste hecho en la revisión:** la Task 1 declara que
`test_el_nucleo_no_importa_torch` falla hasta la Task 5, cuando `ember.memories`
existe. Queda anotado en el Step 6 de la Task 1 para que nadie lo trate como un
error del entorno.
