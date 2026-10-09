# Estado del trabajo — EMBER

**Última actualización:** 2026-10-08
**Rama:** `revision/bip2026-overleaf` (sobre `revision/bip2026-reviews`)

Retomar leyendo, en este orden:
1. `docs/superpowers/specs/2026-08-11-ember-repo-design.md` — el diseño y por qué.
2. `docs/superpowers/plans/2026-08-11-ember-foundation.md` — las 23 tareas.
3. `docs/notas/2026-08-11-numeros-movidos.md` — **qué resultados del draft no sobrevivieron.**
4. `docs/revision/respuesta-revisores.md` y las notas `docs/notas/2026-10-06-*` — la revisión de BIP2026.

## El paper, hoy

El paper está en inglés (traducido el 27 ago, commit `aea0fa1`), 8 páginas,
`verify_paper` limpio. Dos pases editoriales encima de esa traducción:

- `a9e5b2f` — contribuciones alineadas a cuatro ítems, conexión con repetición
  priorizada y Neural Episodic Control, sección de disponibilidad de código.
- `d8ac51b` (28 ago, Nick Florez) — dos afirmaciones sin sustento corregidas
  (el 14.41× era del régimen de compresión, no de selección; el FIFO no está
  "en el piso en toda condición", su reconstrucción es 0.907), abstract a 248
  palabras, nueva §III-E (arquitectura de frontera), `tab:two-knobs`,
  `fig:write`.
- **2026-09-01** — pase sobre comentarios de una revisión anotada
  (`paper/BIP2026_EMBER_revised.pdf`, con highlights y 2 comentarios de texto,
  no versionada — es material de revisión, no un artefacto del repo). El
  comentario central: siglas usadas antes de definirse. Corregido en
  `main.tex`: `e-MDB`, `FIFO`, `LIF`, `STDP`, `ENN` (Engram Neural Network),
  `PoC`, `LTP`, `PCA` ahora se expanden en su primer uso real, no donde
  aparecían por casualidad más definidos que usados. También se glosó
  `incumbent`/`frontier` en la pregunta falsificable de la intro, antes de su
  definición formal en §III-B/§III-E. Sigue en 8 páginas, `verify_paper`
  limpio, 256 tests pasan. `paper/EMBER-overleaf.zip` (export de Overleaf, no
  versionado) se comparó contra el repo: es idéntico, no traía cambios.
- **2026-10-08** — la revisión de BIP2026 se había hecho sobre el proyecto de
  Overleaf equivocado. El correcto (el de envío) era `main` más el pase de
  siglas del 09-01 y un título nuevo; la rama de revisión ya contenía ese pase,
  así que de ahí solo se portaron el título (sin el punto final y con
  "bioinspired", como en el resto del texto), la lectura de $r<1$ / $r>1$ y de
  $K_{proto}$ conocido o estimado en la intro, CIFAR-100 descrito como
  benchmark de imágenes naturales, y la lista de los cuatro sustratos al abrir
  Related work. El abstract de la revisión tenía 287 palabras; recortado a 249
  (límite de BIP: 250) sin tocar ninguna afirmación acotada por la auditoría.
  `verify_paper` limpio. El conteo de páginas se confirma en Overleaf.
  Aclaración posterior: el texto pegado es **la versión que se envió a la
  revista y leyeron los revisores**. La carta ya estaba escrita contra ella
  (su numeración de referencias, con [8] = MiniGrid, y las cuatro frases que
  declara corregidas aparecen literalmente en ese texto), así que no hubo que
  rehacerla. PR: AllanDBB/EMBER#2.

**Para reenviar** (al 2026-10-08):
1. Subir `dist/EMBER-overleaf-final.zip` (lo arma el comando de `CLAUDE.md`) al
   proyecto de Overleaf de envío, compilar `main.tex` y confirmar 8 páginas con
   referencias. `paper/main.pdf` en el repo es anterior al título y al abstract
   nuevos: reemplazarlo por el PDF de Overleaf.
2. El CI de GitHub no corre: la cuenta está bloqueada por facturación. Los
   checks en rojo de la PR no son fallas del código.
3. R1.10 (repositorio durante la revisión) queda como está: la carta promete
   publicarlo después.

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

### `exp06` — MiniGrid: la saliencia no transfiere con sorpresa perceptual, pero sí con sorpresa de recompensa

Corrido el 2026-08-20 (`results/exp06_minigrid/`), en tres condiciones que
cambian una variable a la vez. Compara el genotipo de frontera
(`append + both + decay 1.0 + min_strength`) contra el FIFO sobre 40 rollouts
en `MiniGrid-MemoryS13-v0`, con `t1_rare_retention` midiendo si un episodio
recompensado (evento raro) se retiene tras 1200 pasos en capacidad 20.

**1. Línea de base (política aleatoria + `OneStepPredictor`).** Ambos quedan
en el piso: frontera 0.0 %, FIFO 2.6 %, sobre 38 episodios recompensados. No
es un bug de recuperación —sin presión de capacidad la frontera recupera
cualquier traza con similitud 1.0—, es que el error de predicción de un paso
(un modelo lineal en línea sobre la observación) no separa lo recompensado de
lo meramente novedoso: **75.4 %** de las experiencias comunes tiene error de
predicción igual o mayor que el evento recompensado menos sorpresivo.

**2. Se descartó la explicación fácil (escasez de datos).**
`ForwardBiasedPolicy` (favorece avanzar sobre girar, sin tocar la sorpresa)
produce **4.1× más episodios recompensados (38 → 156)** con el mismo
presupuesto de pasos. La retención sigue en el piso y el solapamiento no
mejora (75.4 % → 73.5 %). Cuadruplicar los datos sin tocar la señal no
cambia nada.

**3. La señal, no la cantidad — y acá está el giro.** La compuerta de fuerza
está motivada por "la cuenta dopaminérgica", pero esa cuenta es sobre error
de predicción de **recompensa** (Schultz), no de percepción genérica.
Se agregó `RewardPredictionError` (mismo modelo lineal en línea, misma
política aleatoria que la línea de base, cambia solo qué se predice) y el
solapamiento cae de 75.4 % a **5.6 %**. La retención de la frontera sube de
**0.0 % a 52.6 %**, mientras el FIFO —que no lee la señal de fuerza— se queda
en 2.6 %, igual que en la línea de base. Que solo la arquitectura que
consulta la señal mejore descarta que sea un artefacto del cambio de
predictor: es la compuerta de saliencia funcionando, con una señal que
correlaciona con lo que importa.

**Conclusión:** el mecanismo de saliencia sí transfiere a un entorno real bajo
observabilidad parcial, bajo una precondición identificable y corregible —la
sorpresa tiene que anclarse en la recompensa, no en la novedad perceptual—.
Esto pasó de ser un párrafo de "trabajo futuro" en Limitaciones a su propia
sección del paper (§VI, con tabla y subsección de Discusión propia), porque
ya no es una limitación: es un resultado positivo completo.

## La revisión de BIP2026 (2026-10-06)

Los siete experimentos de la revisión (exp07–exp13) están corridos, fusionados
en `revision/bip2026-reviews` e integrados al paper. No queda ningún
`TODO(expNN)` en `main.tex` ni ningún `PENDIENTE` en la respuesta a los
revisores. `verify_paper` está limpio, pasan 404 tests y `ruff` está limpio.
Una auditoría independiente (`docs/revision/auditoria-2026-10-06.md`) encontró
3 afirmaciones falsas, 14 de alcance mal enunciado y 11 menores; todas están
corregidas. Después el paper se reestructuró de 12 a **8 páginas** con
referencias (~7 300 palabras en el PDF): Discusión y Limitaciones fusionadas
(§VIII), la comparación contra e-MDB como eje (§IV), Related Work comprimido y
la tabla de MiniGrid eliminada (sus números están en el texto).

Afirmaciones que cambiaron (cada una con su nota en `docs/notas/`):

- **exp07**: el umbral de fusión no mueve el cruce (r ≈ 0.87 entre 0.40 y
  0.85). La precondición de CIFAR-100 pasa a ser *separabilidad*, no alcanzar
  0.85: ningún umbral da fusiones frecuentes y puras.
- **exp08**: el 14.41× vale con una señal perfectamente separable y se
  desvanece al degradarla. La ventaja aguanta hasta un AUC de ≈ 0.91–0.96 y se
  pierde con retraso. SDM baja a 0.60 con AUC 0.95, por debajo de ENN.
- **exp09**: ningún baseline externo supera a la frontera, pero el desalojo
  voraz por sorpresa queda a 0.035. El buffer real de e-MDB, leído en forma
  secuencial, no pasa el gate (0.368). Ese resultado depende del criterio de
  acierto asumido (coseno ≥ 0.85).
- **exp10**: **r nominal no gobierna la transición; r_eff = K_efectivo/C sí**
  (23 familias). La hipótesis se enuncia ahora sobre r_eff.
- **exp11**: con el rollout como unidad, los intervalos de exp06 no cambian. La
  saliencia transfiere en 4 entornos cuando la señal está alineada con lo
  importante. La importancia demorada sigue abierta.
- **exp12**: la retención de SDM la decide el desalojo con compuerta y borrado
  exacto, no la superposición: la lista sola retiene todo, y los contadores
  también si restan exactamente lo desalojado (41.96× más bytes). A igual
  presupuesto, sin presión de capacidad, el FIFO le gana. Tabla III tiene una
  columna nueva de KiB.
- **exp13**: retener mejor no es actuar mejor. La frontera gana al reaparecer
  una tarea, pero pierde en el retorno acumulado frente al FIFO.

## Lo siguiente

1. Revisión de los autores del texto recortado a 8 páginas. Ninguna figura
   nueva entró al cuerpo (las de exp07, exp08, exp10 y exp13 quedan en
   `paper/figures/`).
2. **Pendiente de los autores (R1.10)**: el repositorio público y su URL en el paper.
3. Trabajo futuro ya explícito en el paper, no pendiente de esta sesión:
   barrer `exp05` en más capacidades y con otro extractor de embeddings;
   extender `exp06` a POPGym y a políticas entrenadas.

Ya hecho: escribir la prosa completa del paper (`paper/main.tex` compila limpio
con `pdflatex`+`bibtex`, 8 páginas, cuatro contribuciones, cero discrepancias
en `verify_paper`); caracterizar las 107 arquitecturas peores que el FIFO
(`docs/notas/2026-08-11-numeros-movidos.md`, commit `dc271e1`); correr `exp05`
contra el banco real de CIFAR-100 y `exp06` contra MiniGrid, ambos con el
hallazgo resuelto en resultado positivo (arriba).

## Comandos

```bash
uv run pytest -q                                  # 274 tests
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
