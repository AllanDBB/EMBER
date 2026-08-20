# Estado del trabajo — EMBER

**Última actualización:** 2026-08-20
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
| 18 · `exp02` grilla del umbral | **corrido**, Figura 1 generada |
| 19 · `exp04` benchmark de arquitecturas | **corrido** |
| 20 · `exp05` embeddings reales | **corrido** con el banco real de CIFAR-100, Figura 2 generada |
| 21 · `ember.envs` (MiniGrid) | listo, verificado, y **corrido** (`exp06_minigrid`) |
| 22 · `CLAUDE.md`, skills, README | listo |
| 23 · `paper/` y `paper_sync` | **listo**, incluida la prosa completa (compila con `pdflatex`) |

**265 tests pasan.** `ruff` limpio.
CI tiene tres jobs: tests, auditoría de ejes y verificación de sincronía del
paper. `verify_paper` no reporta discrepancias.

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

### `exp02` — la ley del umbral, confirmada (~15 min)

24 celdas × 5 semillas × 576 genotipos, con IC bootstrap.

| Régimen | Rango | η² escritura | η² desalojo |
|---|---|---|---|
| Compresión | r ≤ 0.5 | **73.0 %** | 8.2 % |
| Transición | 0.5 < r < 1 | 52.4 % | 10.6 % |
| Selección | r ≥ 1 | **1.6 %** | 16.3 % |

El draft reportaba **72.9 %** y **1.6 %** sobre 14 celdas sin intervalos. La
réplica es casi exacta con un motor completamente reescrito.

El cruce de dominancia cae **entre r = 0.75 y r = 1.00 en las tres capacidades**
(C=10, 20, 40), independientemente de la capacidad — que es exactamente la
predicción falsable.

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

### `exp05` — la ley tiene una precondición que el borrador no enuncia

**Es el hallazgo que más afecta cómo se escribe el paper.** Salió al instrumentar
el experimento, antes incluso de tener los datos reales, y **se confirmó al
correrlo sobre el banco real de CIFAR-100** (5 ratios × 3 semillas × 576
genotipos, `results/exp05_real_embeddings/`, Figura 2 generada).

Sobre CIFAR-100 real la cosa es más extrema que la hipótesis de trabajo: **la
fusión no se dispara nunca**. La similitud coseno intra-prototipo medida sobre
el banco real es **0.401** de mediana, contra el umbral de fusión de **0.85**.
`fusion_alcanzable = False` en las cinco celdas del ratio. Con el eje de
escritura inoperante, η² de escritura queda en 0.000–0.004 en **las cinco
celdas**, sin importar el ratio, y **no hay cruce de régimen que localizar**
(`cruce r nominal = nan`, `cruce r efectivo = nan`). El dominio sintético, en
la misma corrida, sí cruza normalmente (r nominal 0.77, r efectivo 0.71),
confirmando que el motor y el resto de la grilla están bien.

Sobre un flujo sintético con varianza intra-prototipo realista (el diseño previo
a tener el banco real, que motivó investigar esto) el eje de escritura explica
~0 % de la varianza **incluso con r = 0.5**, donde la ley predice que debería
dominar. Ahí la causa no era que la fusión no se disparara (lo hacía en el 50 %
de las escrituras) sino que **la consolidación parcial dejaba varias trazas por
prototipo**:

| Dominio | K nominal | K efectivo | r efectivo | η² escritura |
|---|---|---|---|---|
| sintético | 10 | 10 | 0.50 | **0.818** |
| embeddings | 10 | **35** | **1.75** | 0.004 |

La ley no se rompe: se estaba aplicando a la variable equivocada. Con r efectivo
de 1.75, un η² de escritura de 0.004 es **exactamente** lo que la ley predice.

**La corrida real resuelve la decisión que quedaba abierta.** Sobre CIFAR-100,
`r_efectivo` no rescata la ley — no hay ninguna celda donde valga la pena
calcularlo, porque la fusión nunca se dispara y `K_efectivo` crece sin techo con
el largo del flujo (84 → 178 → 373 → 736 → 1435 según crece K nominal, siempre
muy por encima de C=20). Reescribir sobre `r_efectivo` habría sido correcto para
el caso de consolidación *parcial* (la hipótesis de trabajo), pero CIFAR-100 no
es ese caso: es consolidación **nula**, un régimen más simple y más binario.

Enunciado corregido, con las dos causas separadas:

1. **Precondición binaria**: si la similitud intra-prototipo del dominio no
   alcanza el umbral de fusión, el eje de escritura es inoperante y no existe
   régimen de compresión para ningún r — nominal o efectivo. Es lo que pasa en
   CIFAR-100 con este extractor (ResNet-18 + PCA a 32 dim).
2. **Si la fusión sí es alcanzable**, la variable de control del régimen
   es el número de prototipos efectivo tras consolidar, no el nominal — es el
   caso de consolidación parcial que motivó investigar esto.

Sobre flujos sintéticos con dispersión baja las tres cantidades (K nominal, K
efectivo, umbral de fusión alcanzable) coinciden, que es por qué la distinción
no aparecía. Le da al robot dos cantidades medibles en línea: `fusion_alcanzable`
se decide con una comparación de similitud contra el umbral, y `K_efectivo` se
cuenta mirando cuántas trazas tiene la memoria.

### `exp06` — MiniGrid expone una segunda precondición: la saliencia necesita señal, no solo sorpresa

Corrido el 2026-08-20 (`results/exp06_minigrid/`). Compara el genotipo de
frontera (`append + both + decay 1.0 + min_strength`) contra el FIFO sobre 40
rollouts de política aleatoria en `MiniGrid-MemoryS13-v0`, con
`t1_rare_retention` midiendo si un episodio recompensado (evento raro) se
retiene tras 1200 pasos en capacidad 20.

**Ambos quedan en el piso**: frontera 0.0 %, FIFO 2.6 %, sobre 38 episodios
recompensados. No es un bug de recuperación —sin presión de capacidad la
frontera recupera cualquier traza con similitud 1.0—, es que el error de
predicción de un paso (`OneStepPredictor`, un modelo lineal en línea) no
separa lo recompensado de lo meramente novedoso: **75.4 %** de las
experiencias comunes tiene error de predicción igual o mayor que el evento
recompensado menos sorpresivo. Los flujos sintéticos y CIFAR-100 no tienen
este problema porque el error de predicción se fija o se mide para separar
las dos poblaciones por construcción; una política de exploración aleatoria
no ofrece esa garantía.

Es la misma forma de hallazgo que `exp05`: un dominio real expone una
precondición que el diseño sintético no necesita enunciar. Acá la
precondición es distinta —no es sobre el eje de escritura sino sobre el eje
de fuerza/desalojo—: la señal de sorpresa tiene que correlacionar con
relevancia de tarea, no solo con novedad perceptual. Una política de
exploración dirigida, o una señal de sorpresa derivada de la recompensa,
son candidatas para resolverlo; queda como trabajo futuro explícito en el
paper (§ Limitaciones).

## Lo siguiente

1. **Confirmar la fecha límite de BIP2026.**
2. **Revisión editorial del texto**: tono, longitud por sección, y decidir si
   la promoción de `exp05` a sección propia (en vez de un párrafo de
   limitaciones) es la que el autor quiere para el envío.
3. Trabajo futuro ya explícito en el paper, no pendiente de esta sesión:
   barrer `exp05` en más capacidades y con otro extractor de embeddings;
   repetir `exp06` con una política de exploración dirigida o con una señal
   de sorpresa derivada de la recompensa; extender a POPGym.

Ya hecho: escribir la prosa completa del paper (`paper/main.tex` compila limpio
con `pdflatex`+`bibtex`, 7 páginas, cero discrepancias en `verify_paper`);
caracterizar las 107 arquitecturas peores que el FIFO
(`docs/notas/2026-08-11-numeros-movidos.md`, commit `dc271e1`); correr `exp05`
contra el banco real de CIFAR-100 (arriba); correr `exp06` contra MiniGrid
(arriba).

## Comandos

```bash
uv run pytest -q                                  # 265 tests
uv run pytest -q -m "not slow"
uv run ruff check . && uv run ruff format --check .

uv run python -m experiments.exp01_nas_full       # 16 s
uv run python -m experiments.exp02_threshold_grid # ~15 min
uv run python -m experiments.exp03_axis_liveness  # 15 s
uv run python -m experiments.exp04_arch_benchmark # ~6 min (el spiking es lento)
uv run python -m experiments.exp05_real_embeddings # ~3 min
uv run python -m experiments.exp06_minigrid       # ~8 s

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
