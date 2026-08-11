# EMBER — Diseño del repositorio

**Fecha:** 2026-08-11
**Estado:** aprobado
**Contexto:** BIP2026, fecha límite estimada a fines de agosto de 2026.

## 1. Qué es EMBER y qué tiene que ser este repo

EMBER (*Emergent Memory-Based Encoding and Reactivation*) es un programa de
investigación sobre memoria episódica de largo plazo para robots cognitivos.
El repo tiene que servir simultáneamente a tres metas, en este orden de urgencia:

1. **Paper BIP2026** — resultados reproducibles que respalden cada tabla y figura.
2. **Librería instalable** — `ember` como paquete que termine siendo un
   `Cognitive Node` de la arquitectura e-MDB corriendo en un Robotino.
3. **Repo de investigación** — las PoCs, los experimentos y los borradores.

La restricción que ordena todo: la librería que va al robot no puede arrastrar
torch, datasets ni gym. La separación se hace con extras de instalación, no con
paquetes separados (ver §3).

## 2. Qué está mal hoy, y por qué el rediseño no es cosmético

El código actual (`memory_space.py`, `search.py`, `analyze.py`,
`architectures.py`, `benchmark.py`) produjo los números del draft. Tiene cuatro
problemas que invalidan comparaciones que el paper presenta como válidas.

### 2.1 Dos motores de memoria que no son el mismo motor

`ConfigurableMemory` (usada por el NAS) y las clases de `architectures.py`
(usadas por el benchmark) reimplementan los mismos mecanismos con semánticas
distintas:

| Mecanismo | NAS (`memory_space.py`) | Benchmark (`architectures.py`) |
|---|---|---|
| Desalojo FIFO | `argmax(age)` sobre contador incremental | `pop(0)` sobre lista |
| Fuerza inicial | `1 + 2·novelty + 2·perr` | `1 + 2·perr` (sin novedad) |
| Umbral de acierto | `sim >= 0.90` | `sim >= 0.75` |
| Decaimiento | eje del genotipo (1.0 / 0.995 / 0.98) | fijo en 0.97, solo ENN |
| Refuerzo por lectura | eje del genotipo (0 / 0.5) | fijo en 0.3, solo ENN |

La Tabla II (varianza explicada sobre el espacio NAS) y la Tabla VI (batería de
arquitecturas) se presentan en el paper como mediciones del mismo fenómeno. No
lo son. La afirmación central de la sección I —"el FIFO de e-MDB no es un
baseline externo, es un punto del mismo espacio de diseño"— solo es cierta si
hay un solo motor.

**Decisión:** un único motor. `ember.core` define las políticas una vez;
`PolicyMemory` las compone; el genotipo NAS es literalmente una tupla de
políticas; y las arquitecturas concretas del benchmark se construyen sobre las
mismas piezas. El FIFO es un genotipo, no una clase aparte.

### 2.2 Bug numérico en las dos variantes de SDM

`architectures.py:108` y `architectures.py:491`, en el desalojo:

```python
self.V[idx] += strength * k  # al escribir
...
self.V[di] -= dk  # al desalojar  ← resta 1×, no strength×
```

Con `strength = 1 + 2·pred_error`, un evento raro (pe≈0.9) suma 2.8·k y resta
1.0·k. Cada desalojo deja un residuo de 1.8·k en los contadores. En T1, donde
se escriben ~215 ítems en capacidad 20, se acumulan ~195 residuos. El 1.000 de
retención de eventos raros de SDM en la Tabla VI está medido sobre contadores
contaminados.

**Decisión:** el store guarda la contribución exacta escrita por traza y la
resta al desalojar. Test de regresión: escribir N ítems, desalojarlos todos,
verificar `‖V‖ ≈ 0`.

### 2.3 `SpikingMemory` no es reproducible

`architectures.py:331`:

```python
rng = np.random.default_rng(id(neuron_ids) % 2**32)
```

`id()` es la dirección de memoria del array. Cambia entre corridas y entre
plataformas. Los resultados de `Spiking` en la Tabla V (el que falla el gate con
0.483) no son replicables.

**Decisión:** toda fuente de aleatoriedad deriva de una semilla explícita
propagada desde el experimento. Test: dos instancias con la misma semilla
producen resultados idénticos bit a bit.

### 2.4 Los tres defectos que el propio addendum ya declaró

El paper los declara abiertamente como limitaciones (§III-B, §VI). Están en el
alcance de este rediseño porque sin ellos no se puede escribir "576
arquitecturas" ni "seis mecanismos", que es la ventaja metodológica que vende la
§III:

- **`read_mode=radius` no hace superposición.** Devuelve el vecino más cercano
  en los tres modos. Hay que implementar la lectura distribuida real de Kanerva:
  sumar los valores ponderados de todas las trazas dentro del radio.
- **`reinforce` es inobservable por diseño de tarea.** Las tres tareas emiten
  todas las lecturas después de todas las escrituras, así que la fuerza
  modificada nunca influye en un desalojo. Hay que **intercalar lecturas y
  escrituras** — que además es lo que hace un robot: consulta mientras opera.
- **T2 (ruido) es degenerada.** Devuelve 0.767 para las 576 arquitecturas
  (varianza exactamente 0) porque 20 ítems entran exactos en capacidad 20 y
  nunca se dispara un desalojo. Aporta una constante de 0.256 al promedio de
  todo el mundo; el 98 % del 0.261 del FIFO es esa constante.

**Decisión:** los tres se arreglan, y un experimento dedicado
(`exp03_axis_liveness`) verifica en CI que ningún eje del espacio sea
inobservable. La auditoría del addendum se convierte en un test.

## 3. Arquitectura

### 3.1 Layout

```
src/ember/
  core/         protocolo Memory, Episode, Trace, TraceStore, políticas
  memories/     sdm, enn, spiking, spiking_sdm, policy (incluye FIFO)
  data/         streams sintéticos controlados, embeddings perceptuales, K̂proto
  tasks/        R1-R4 (reconstrucción), T1-T3 (batería)
  nas/          espacio, motor de búsqueda, estadística
  envs/         adaptadores MiniGrid / POPGym
experiments/    scripts numerados → results/ con manifiesto
paper/          LaTeX + generadores de tabla y figura desde results/
notebooks/      PoC-1 a PoC-4
tests/
```

Un solo `pyproject.toml`. Extras de instalación:

| Extra | Trae | Para |
|---|---|---|
| (base) | numpy | El robot. `core/` + `memories/` y nada más. |
| `[lab]` | torch, torchvision, matplotlib, scipy, pandas | Experimentos y figuras. |
| `[envs]` | gymnasium, minigrid, popgym | Validación secuencial. |
| `[dev]` | pytest, ruff, mypy | Desarrollo. |

Esto se convierte en un workspace multi-paquete (`ember-core` / `ember-lab` /
`ember-ros`) sin dolor el día que haga falta, porque `core/` ya es la frontera.

### 3.2 El núcleo: políticas componibles

`ember.core` define cinco familias de política, cada una con una interfaz
mínima. Son la única implementación de cada mecanismo en todo el repo.

```python
class StrengthPolicy(Protocol):  # fuerza inicial de una traza nueva
    def initial(self, pred_error: float, novelty: float) -> float: ...


class WritePolicy(Protocol):  # crear traza nueva o consolidar en una existente
    def route(self, store: TraceStore, key: NDArray) -> WriteTarget: ...


class ReadPolicy(Protocol):  # qué trazas participan en la reconstrucción
    def select(self, sims: NDArray) -> NDArray: ...
    def reconstruct(self, store, sel, sims) -> tuple[Any, float]: ...


class EvictPolicy(Protocol):  # a quién se descarta cuando no cabe
    def victim(self, store: TraceStore, rng: Generator) -> int: ...


class DecayPolicy(Protocol):  # erosión de fuerza sin refuerzo
    def step(self, strength: NDArray) -> None: ...
```

`TraceStore` es el contenedor vectorizado (matriz de claves, vectores de fuerza,
edad, utilidad, contribución escrita). Es el único lugar donde vive la
contabilidad, y por eso el bug de §2.2 no se puede repetir en dos sitios.

`PolicyMemory` compone las cinco políticas y satisface el protocolo `Memory`.
El genotipo del NAS es una tupla de políticas. El `EpisodicBuffer` de e-MDB es:

```python
FIFO_GENOTYPE = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=Constant(),
    decay=NoDecay(),
    evict=FIFO(),
    reinforce=0.0,
)
```

Las arquitecturas concretas (`SDMMemory`, `ENNMemory`, `SpikingMemory`,
`SpikingSDMMemory`) implementan el mismo protocolo `Memory` pero con sustratos
propios (hard locations, circuito LIF), reutilizando las políticas de ciclo de
vida de traza.

### 3.3 Datos

Un `Stream` es una secuencia de `StreamItem(key, value, pred_error, is_rare)`
más un `StreamSpec` con los metadatos que la ley del ratio necesita
(`n_prototypes`, `capacity`, `r`).

Dos fuentes:

- **`synthetic`** — generadores controlados. Mantienen el control exacto sobre
  `K_proto`, que es la variable independiente de la ley del ratio, e incorporan
  las estructuras que la §VI del paper dice que faltan: correlación entre
  dimensiones, densidad de clúster desigual, drift temporal, dimensión
  intrínseca baja. El gaussiano i.i.d. actual queda como un caso degenerado del
  generador, no como el único modo.
- **`embeddings`** — embeddings perceptuales reales de CIFAR-100 vía un encoder
  preentrenado (torchvision), cacheados a `.npz`. Da estructura correlacionada y
  densidad no uniforme sin depender del robot.

**El problema nuevo que esto abre**, y que hay que resolver explícitamente: con
datos reales `K_proto` deja de ser conocido a priori. La ley del ratio se
enuncia sobre `r = K_proto/C`, así que sobre datos reales hay que *estimar*
`K̂_proto`. `ember.data.prototypes` implementa el estimador (clustering + criterio
de selección de k) y los experimentos reportan la ley contra `K̂` con su
intervalo de confianza, no contra un número exacto. Esto es una contribución,
no un parche: hace la ley aplicable a un robot que no sabe cuántos prototipos
tiene su ambiente.

### 3.4 Tareas

Reconstrucción (gate): R1 pattern completion, R2 robustez al ruido, R3
interferencia A→B, R4 perfil de capacidad. Se mantienen como están
conceptualmente.

Batería, con los arreglos de §2.4:

- **T1 retención de eventos raros** — con lecturas intercaladas entre
  escrituras, de modo que `reinforce` sea observable.
- **T2 robustez al ruido bajo presión** — rediseñada: se escriben más ítems que
  la capacidad, de modo que la tarea discrimine en vez de devolver una constante.
- **T3 interferencia secuencial** — con error de predicción diferenciado
  (bloque 1 pe=0.9, resto pe=0.1). El paper ya reconoce que la versión con pe
  uniforme "es informativa pero no es la comparación pretendida".

Cada tarea declara si consume `pred_error` diferenciado y si intercala
lecturas, para que el motor de NAS pueda verificar la observabilidad de ejes.

### 3.5 Experimentos

Scripts numerados en `experiments/`, cada uno escribe a `results/<nombre>/` un
JSON con los datos y un `manifest.json` con SHA de git, semillas, versiones de
dependencias y timestamp.

| Script | Qué produce |
|---|---|
| `exp01_nas_full` | Búsqueda exhaustiva. Tabla II. |
| `exp02_threshold_grid` | Grilla capacidad 10–160 × prototipos 1–320, 10 semillas, IC bootstrap. **Figura 1.** Tabla III. |
| `exp03_axis_liveness` | Auditoría: cada eje mueve el resultado. Corre en CI. |
| `exp04_arch_benchmark` | Dos fases. Tablas V y VI. |
| `exp05_real_embeddings` | La ley del ratio sobre embeddings de CIFAR-100 con K̂. |
| `exp06_envs` | MiniGrid con observabilidad parcial. Post-envío. |

### 3.6 Testing

`pytest`, con cuatro categorías:

1. **Contrato** — parametrizados sobre todas las memorias: escribir k y leerlo
   devuelve k cuando hay capacidad; nunca se excede la capacidad; `read` sobre
   memoria vacía devuelve `None`.
2. **Regresión de los bugs de §2** — conservación de contadores SDM tras
   desalojo; determinismo bit a bit bajo misma semilla.
3. **Observabilidad de ejes** — falla si algún eje del espacio de búsqueda
   produce diferencia máxima 0.000 entre genotipos hermanos. Convierte la
   auditoría del addendum en CI.
4. **Golden values** — los números publicados quedan fijados con tolerancia, de
   modo que un cambio de código que mueva una tabla del paper falle ruidosamente.

### 3.7 Configuración de Claude

- `CLAUDE.md` en la raíz: comandos, layout, invariantes del proyecto.
- `.claude/skills/ember-domain` — glosario y contexto (SDM, engrama, STDP,
  e-MDB, LOLA, ley del ratio). Para que una sesión nueva arranque con el
  vocabulario correcto.
- `.claude/skills/ember-experiment` — cómo agregar y correr un experimento
  reproducible: semillas, manifiesto, escritura a `results/`.
- `.claude/skills/ember-memory` — cómo agregar una arquitectura de memoria
  respetando el protocolo, con los tests de contrato obligatorios.
- `.claude/skills/paper-sync` — regenerar tablas y figuras del LaTeX desde
  `results/` y verificar que lo publicado coincide con lo medido.
- `.claude/settings.json` — permisos para `uv`, `pytest`, `ruff`.

### 3.8 Entorno y CI

`uv` con lockfile, Python 3.12 (3.14 aún no tiene ruedas estables para el stack
científico). GPU disponible: RTX 4050 de 6 GB — suficiente para extraer
embeddings, no para entrenar agentes grandes, lo que refuerza dejar `exp06` para
después del envío.

CI en GitHub Actions: `ruff check`, `ruff format --check`, `pytest`, y
`exp03_axis_liveness` como prueba de humo.

## 4. Alcance de la primera iteración

Con ~2-3 semanas al envío, el orden es:

1. Núcleo, memorias, tareas, tests. (§3.2, §3.4, §3.6)
2. `exp01`, `exp03`, `exp04` — reproducir el paper con el motor unificado y los
   bugs corregidos. **Los números van a moverse**; hay que ver cuánto.
3. `exp02` — la grilla del umbral expandida. Es la Figura 1 y el resultado
   principal.
4. `exp05` — embeddings reales. Es lo que responde la objeción del revisor.
5. `.claude/`, CI, docs.

`exp06` (MiniGrid/POPGym) queda cableado en `ember.envs` pero se corre después
del envío. El paper ya dice que la validación en navegación pertenece al
trabajo siguiente.

## 5. Riesgo principal

Corregir §2.1 y §2.2 va a mover los números del draft. El 1.000 de retención de
SDM y el 0.908 de fidelidad de reconstrucción están medidos sobre contadores
contaminados, y las tablas II y VI pasarán a ser comparables por primera vez.
Es posible que alguna afirmación del paper no sobreviva.

Eso es el resultado correcto, no un problema: es exactamente lo que le pasó al
piloto anterior cuando se auditó, y el addendum muestra que lo que apareció a
cambio fue un paper mejor. La mitigación es orden: reproducir primero los
números viejos con el motor nuevo (`exp01`, `exp04`) antes de agregar datos
nuevos, para saber qué se movió por el arreglo y qué por el cambio de dominio.
