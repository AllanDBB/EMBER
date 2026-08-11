# EMBER

**Emergent Memory-Based Encoding and Reactivation** — memoria episódica de largo
plazo bioinspirada para robots cognitivos.

Un robot que aprende durante toda su vida tiene que decidir qué vale la pena
guardar, mantenerlo accesible en proporción a su relevancia, y recuperarlo desde
una clave degradada. El `EpisodicBuffer` de la arquitectura e-MDB —un `deque` de
tamaño fijo— no hace ninguna de las tres: conserva lo último que percibió, trate
de lo que trate.

EMBER define un espacio de diseño discreto donde cada mecanismo de memoria
bioinspirada es un eje, lo busca de forma exhaustiva, y trata al buffer FIFO
desplegado como **un punto de ese mismo espacio** en vez de como un baseline
externo. Eso convierte una comparación potencialmente circular en una pregunta
falsable: dado el espacio completo, ¿dónde cae el incumbente y qué combinación
define la frontera?

## Instalación

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"     # desarrollo
uv pip install -e ".[lab]"     # + experimentos y figuras
uv pip install -e ".[envs]"    # + MiniGrid
```

El paquete base depende **solo de numpy**: es el código que va al robot. Los
extras existen para que un Robotino no tenga que instalar torch.

## Uso

```python
from ember.memories import SDMMemory

mem = SDMMemory(dim=32, capacity=20, seed=0)
mem.write(clave, episodio, pred_error=0.9)  # sorpresa alta: se codifica fuerte
resultado = mem.read(consulta_ruidosa)  # ReadResult(value, similarity, index)
```

Cualquier punto del espacio de diseño se puede instanciar directamente:

```python
from ember.core import Genotype, PolicyMemory
from ember.core.policies import Append, MinStrength, NearestNeighbour, NoDecay, PredErrorGated

frontera = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=PredErrorGated(),
    decay=NoDecay(),
    evict=MinStrength(),
    reinforce=0.0,
)
mem = PolicyMemory(dim=32, capacity=20, genotype=frontera, seed=0)
```

## Reproducir los resultados

```bash
uv run python -m experiments.exp01_nas_full        # búsqueda exhaustiva   ~16 s
uv run python -m experiments.exp02_threshold_grid  # la ley del umbral    ~12 min
uv run python -m experiments.exp03_axis_liveness   # auditoría de ejes     ~15 s
uv run python -m experiments.exp04_arch_benchmark  # benchmark 2 fases
uv run python -m experiments.exp05_real_embeddings # CIFAR-100 (baja el dataset)
```

Cada uno escribe a `results/<nombre>/` los datos y un manifiesto con el SHA de
git, las semillas y las versiones. Las tablas y figuras del paper se generan
desde ahí con `ember.paper_sync`.

## Cómo está organizado

```
src/ember/
  core/       protocolo Memory, TraceStore, políticas, PolicyMemory, Genotype
  memories/   SDM, ENN, Spiking (LIF+STDP), Spiking-SDM
  data/       streams sintéticos, estimador de K_proto, embeddings de CIFAR-100
  tasks/      R1-R4 gate de reconstrucción, T1-T3 batería bajo presión
  nas/        espacio de 576 genotipos, motor paralelo, estadística
  envs/       adaptador MiniGrid
experiments/  los cinco experimentos
docs/         ESTADO.md, notas/, superpowers/{specs,plans}, borradores/
notebooks/    PoC-1 a PoC-4
```

El `EpisodicBuffer` FIFO **no es una clase**: es `PolicyMemory` con
`FIFO_GENOTYPE`. Esa decisión es lo que hace literal la afirmación de que el
incumbente es un punto del espacio que se busca.

## La ley del umbral

El resultado central. Qué mecanismo de memoria bioinspirada importa no es una
propiedad del mecanismo: es función del cociente

$$r = K_{proto} / C$$

entre prototipos de experiencia recurrentes y capacidad de memoria. Por debajo
de 1 domina la compresión (consolidación por fusión); por encima domina la
selección (olvido por mínima fuerza, modulado por saliencia).

Para un robot tiene lectura directa: en un ambiente temprano y estructurado
`r < 1` y consolidar es lo eficiente; en uno heterogéneo `r` crece y lo que
manda es el olvido selectivo. La recomendación de diseño no es elegir un
régimen — es que **el régimen cambia a lo largo de la vida del robot**, y la
arquitectura de la frontera funciona en ambos porque nunca depende de la fusión.

## Documentación

- `docs/ESTADO.md` — dónde está el trabajo ahora mismo.
- `docs/notas/` — qué números se movieron y por qué.
- `docs/superpowers/specs/` — el diseño y sus razones.
- `CLAUDE.md` — los siete invariantes del proyecto.

## Estado

Investigación en curso. La API puede cambiar hasta que se estabilice el
`Cognitive Node` para e-MDB.
