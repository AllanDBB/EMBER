# EMBER

Memoria episódica de largo plazo bioinspirada para robots cognitivos. El repo
sirve a tres cosas a la vez: el paper de BIP2026, una librería instalable que
termine siendo un `Cognitive Node` de e-MDB en un Robotino, y el registro de la
investigación.

Antes de tocar nada, leer `docs/ESTADO.md`.

## Comandos

```bash
uv run pytest -q                      # suite completa
uv run pytest -q -m "not slow"        # sin las lentas
uv run ruff check . && uv run ruff format --check .

uv run python -m experiments.exp01_nas_full        # búsqueda exhaustiva (~16 s)
uv run python -m experiments.exp02_threshold_grid  # la ley del umbral (~12 min)
uv run python -m experiments.exp03_axis_liveness   # auditoría de ejes (~15 s)
uv run python -m experiments.exp04_arch_benchmark  # benchmark de arquitecturas
uv run python -m experiments.exp05_real_embeddings # CIFAR-100 (baja el dataset)
```

Extras: `uv pip install -e ".[dev]"` para trabajar, `".[lab]"` para experimentos
y figuras, `".[envs]"` para MiniGrid.

## Layout

```
src/ember/
  core/       protocolo Memory, TraceStore, las cinco familias de política,
              PolicyMemory, Genotype. Solo numpy.
  memories/   SDM, ENN, Spiking, Spiking-SDM. FIFO no es una clase: es
              PolicyMemory con FIFO_GENOTYPE.
  data/       streams sintéticos, estimador de K_proto, embeddings de CIFAR-100
  tasks/      R1-R4 (gate de reconstrucción), T1-T3 (batería bajo presión)
  nas/        espacio de 576 genotipos, motor paralelo, estadística
  envs/       adaptador MiniGrid (cableado, no corrido para el envío)
  experiment.py  ExperimentRun: manifiesto de procedencia
  figures.py     figuras del paper
experiments/  expNN_*.py → results/expNN_*/
docs/         ESTADO.md, notas/, superpowers/{specs,plans}, borradores/
```

## Invariantes

Estos no son preferencias de estilo. Cada uno corresponde a un defecto concreto
que ya rompió resultados publicados una vez.

1. **El núcleo solo depende de numpy.** `ember.core` y `ember.memories` no
   pueden importar torch, scipy, matplotlib ni gymnasium — es el código que va
   al robot. `tests/test_smoke.py` lo verifica en un subproceso.

2. **Toda aleatoriedad sale de una semilla explícita** propagada desde el
   experimento. Prohibido `np.random.*` global, `id()`, `hash()` o el reloj.
   El piloto sembraba con `id(neuron_ids)` —la dirección de memoria del
   arreglo— y sus resultados no eran replicables.

3. **Un solo motor de memoria.** Cada mecanismo se implementa una vez en
   `ember.core.policies`. Si te encontrás escribiendo una segunda versión de un
   desalojo o de una compuerta de fuerza, parás: eso es lo que hacía que las
   tablas del piloto no fueran comparables entre sí.

4. **Un sustrato distribuido resta al desalojar exactamente lo que sumó al
   escribir.** `TraceStore.contribution` guarda el escalar. El piloto restaba
   `key` habiendo sumado `strength * key`, y dejaba residuos.

5. **Ningún eje del espacio de búsqueda puede ser inobservable.**
   `exp03_axis_liveness` falla con código ≠ 0 si alguno da 0.000, y corre en CI.
   Un eje puede morir por no estar implementado o porque el diseño de la tarea
   lo vuelve inobservable; la segunda causa es la difícil de ver.

6. **Los rangos dentro del espacio se reportan como intervalo, nunca como
   puesto puntual.** Con empates masivos, el extremo optimista de un empate no
   es una posición.

7. **Todo número que va al paper sale de un `ExperimentRun`** con su manifiesto.

## Convenciones

- Docstrings y comentarios en español; identificadores en inglés.
- Todo `float32`. Las claves se normalizan a norma 1 en la frontera de `write`
  y `read` — quien llama no tiene que hacerlo.
- Los tests explican *por qué* existe la prueba cuando fija una regresión real.
- `results/**/*.json` se versiona: es el registro del paper.

## Al agregar cosas

- **Una arquitectura de memoria**: implementá el protocolo `Memory`, registrala
  en `ARCHITECTURES`, y los tests de contrato la recogen solos. Ver la skill
  `ember-memory`.
- **Un experimento**: usá `ExperimentRun`. Ver la skill `ember-experiment`.
- **Un número al paper**: usá el macro `\result{}`. Ver la skill `paper-sync`.
- Vocabulario del dominio (SDM, engrama, STDP, e-MDB, LOLA, la ley del umbral):
  skill `ember-domain`.
