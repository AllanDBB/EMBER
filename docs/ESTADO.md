# Estado del trabajo — EMBER

**Última actualización:** 2026-08-11
**Rama:** `feat/ember-foundation` (aún no mergeada a `main`)

Retomar leyendo, en este orden:
1. `docs/superpowers/specs/2026-08-11-ember-repo-design.md` — el diseño y por qué.
2. `docs/superpowers/plans/2026-08-11-ember-foundation.md` — las 23 tareas.
3. `docs/notas/2026-08-11-numeros-movidos.md` — **qué resultados del draft no sobrevivieron.**

## Hecho

| Tarea del plan | Estado |
|---|---|
| 1–4 · Esqueleto, entorno uv, CI, núcleo (`ember.core`) | listo |
| 5–8 · Arquitecturas (`ember.memories`) | listo |
| 9–11 · Datos (`ember.data`) | listo |
| 12–13 · Tareas R1–R4 y T1–T3 (`ember.tasks`) | listo |
| 14–16 · NAS y `ExperimentRun` | listo |
| 17 · `exp01` y `exp03` | **corridos** |
| 18 · `exp02` grilla del umbral | **corriendo la última celda** |
| 19 · `exp04` benchmark de arquitecturas | **corrido** |
| 20 · `exp05` embeddings reales | escrito; bajando CIFAR-100 |
| 21 · `ember.envs` (MiniGrid) | listo y verificado contra el entorno real |
| 22 · `CLAUDE.md`, skills, README | listo |
| 23 · `paper/` y `paper_sync` | listo (el texto del paper falta) |

239 tests pasan. `ruff` limpio. CI tiene tres jobs: tests, auditoría de ejes y
verificación de sincronía del paper.

## Los cuatro defectos del piloto: arreglados y fijados con tests

1. **Dos motores de memoria con semánticas distintas.** Ahora hay uno solo:
   `ember.core.policies` define cada mecanismo una vez, `PolicyMemory` los
   compone, y `ARCHITECTURES["FIFO"]` es `FIFO_GENOTYPE`, no una clase aparte.
2. **Contadores de SDM corrompidos al desalojar.** `TraceStore.contribution`
   guarda el escalar exacto escrito. Test: desalojar todo deja `‖V‖ ≈ 0`.
3. **`SpikingMemory` sembrada con `id()`.** Toda aleatoriedad sale de la semilla.
   Test: misma semilla, pesos idénticos bit a bit.
4. **Tres ejes/tareas inobservables.** Los seis ejes viven ahora. `exp03` falla
   con código ≠ 0 si alguno vuelve a morir, y corre en CI.

## Resultados

### `exp03` — los seis ejes son observables

El espacio tiene **576 arquitecturas funcionalmente distintas, no 96**.
`reinforce` pasó de 0.000000 a 0.206 y `read` de 0.000000 a 0.119. La ventaja
metodológica que vende la sección de método del paper ya es verificable.

### `exp01` — la búsqueda exhaustiva (16 s)

- Frontera: `append + both + decay 1.0 + min_strength`, puntaje 0.818.
  **Es el mismo genotipo que reportaba el draft.**
- Saliencia: efecto principal 1.4 %, efecto condicional **14.4×** (draft: 35×).
- **El FIFO ya no está en el piso del espacio de genotipos**: 107 arquitecturas
  puntúan estrictamente peor. Rango #416–#469 de 576.

### `exp02` — la ley del umbral

Confirmada en las tres capacidades probadas. El cruce de dominancia cae **entre
r = 0.75 y r = 1.00** en C=10, C=20 y C=40, independientemente de la capacidad —
que es exactamente la predicción falsable.

| C | η² escritura en r≤0.5 | η² escritura en r≥1 |
|---|---|---|
| 10 | 0.61–0.77 | 0.01–0.02 |
| 20 | 0.64–0.81 | 0.01–0.02 |
| 40 | 0.67–0.87 | 0.01–0.03 |

**Diseño importante**: hay que mantener constantes las **visitas por prototipo**
al barrer el ratio. Con un largo de flujo fijo, subir K baja la recurrencia y el
ratio queda confundido con ella.

### `exp04` — el benchmark de arquitecturas

- **El 1.000 de retención de eventos raros de SDM sobrevive** a los contadores
  arreglados.
- **El circuito spiking puro sigue sin pasar el gate** (0.459). La motivación de
  la PoC-5 queda intacta.
- **T3 con señal diferenciada da SDM 0.875 / ENN 0.375 / FIFO 0.000**, que son
  exactamente los tres números que el draft predecía para esa versión.
- **El FIFO es el piso entre arquitecturas** (0.038). Es decir: "el incumbente
  está en el piso" es cierto entre arquitecturas y falso dentro del espacio de
  genotipos. Hay que separar las dos afirmaciones al reescribir.

## Lo siguiente

1. **Terminar `exp05`** — la ley sobre embeddings de CIFAR-100 con `K` estimado.
   El banco se extrae con `extract_cifar100_embeddings(Path('data/cache'))`.
2. **Escribir el texto del paper.** El esqueleto, las tablas generadas y las
   figuras están; falta la prosa. Todo número medido va con `\result{}`.
3. **Caracterizar las 107 arquitecturas que puntúan peor que el FIFO** — hace
   falta para decir si son combinaciones degeneradas o no.
4. **Confirmar la fecha límite de BIP2026.**

## Comandos

```bash
uv run pytest -q                                  # 239 tests
uv run pytest -q -m "not slow"
uv run ruff check . && uv run ruff format --check .

uv run python -m experiments.exp01_nas_full       # 16 s
uv run python -m experiments.exp02_threshold_grid # ~15 min
uv run python -m experiments.exp03_axis_liveness  # 15 s
uv run python -m experiments.exp04_arch_benchmark # ~6 min (el spiking es lento)
uv run python -m experiments.exp05_real_embeddings

uv run python -c "from ember.paper_sync import render_tables; print(render_tables())"
uv run python -c "from ember.paper_sync import verify_paper; print(verify_paper('paper/main.tex'))"
```

## Decisiones que no son obvias del código

- **`Radius` usa radio relativo al mejor match, no absoluto.** Con umbral
  absoluto 0.70 sobre vectores casi ortogonales en R³² el círculo de activación
  contiene siempre una sola traza y el eje de lectura muere por segunda vez.
- **`variance_share` en vez de eta cuadrado parcial para tabular.** El parcial
  satura en 1.0 cuando el modelo no deja varianza residual, que es lo que pasa
  en un espacio de diseño determinístico.
- **`rank_of` devuelve un intervalo, nunca un entero.** El "#415 de 576" del
  piloto era el extremo optimista de un empate de 162.
- **`PrototypeEstimate.has_structure`.** Sobre ruido isotrópico el argmax de la
  curva de silueta siempre devuelve algún k, pero no significa nada. `r` no está
  definido para un flujo sin estructura de prototipos.
- **El muestreo de `exp03` es por conjuntos de hermanos.** Un prefijo de la
  enumeración comparte el valor de los ejes que varían más lento y reportaría
  ejes vivos como muertos.
- **En `ember.envs` el error de predicción se calcula, no se etiqueta.** Un
  modelo de transición de un paso da la señal de sorpresa; ponerla a mano sería
  decidir a dedo la variable que gobierna todo el resultado.
- **`verify_paper` ignora los comentarios de LaTeX** pero conserva los números
  de línea, blanqueando en vez de borrar.
