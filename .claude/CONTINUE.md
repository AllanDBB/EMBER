# Continue — handoff de sesión (2026-08-27)

Doc para retomar en otra computadora. Borrar cuando ya no sirva.

## Qué se hizo en esta sesión

**El paper pasó de español a inglés.** Traducción nativa, terminología técnica
intacta, todos los `\result{}` / `\cite{}` / `\ref{}` / math preservados verbatim.

Archivos tocados (todo en el working tree, ya commiteado — ver abajo):

- `paper/main.tex` — reescritura completa de la prosa (título, abstract,
  keywords, 9 secciones, la tabla `tab:axes` inline, caption de `fig:threshold`).
- `src/ember/paper_sync.py` — traducidos **solo los string literals que se
  emiten al LaTeX** (captions, headers, row labels de las 5 tablas generadas;
  `"inobservable"` → `"unobservable"`). Docstrings y comentarios **siguen en
  español** por la convención de `CLAUDE.md`.
- `src/ember/figures.py` — labels de ejes, títulos y leyendas de fig1 y fig2.
  Docstrings en español.
- `tests/test_paper_sync.py` — el único assert que dependía de `"inobservable"`.
- `paper/tables/*.tex` — regeneradas con `render_tables()`.
- `paper/figures/fig1_threshold.pdf`, `fig2_dominios.pdf` — regeneradas desde
  `results/exp02_*/data.json` y `results/exp05_*/data.json` (no hizo falta
  recorrer los experimentos; las figuras salen directo del JSON guardado).

### Estado de verificación (todo verde)

- `verify_paper('paper/main.tex')` → **CLEAN** (ningún número desincronizado).
- Build: `pdflatex ×3 + bibtex`, exit 0, **8 páginas**, sin refs/citas sin
  resolver. Único `Overfull \hbox` de 0.52 pt en `tab:axes` (`reinforce|` en la
  columna de raíz biológica) — cosmético, preexistente.
- `uv run pytest -q -m "not slow"` → 256 passed.
- `ruff check` + `ruff format --check` → limpio.
- `paper/main.pdf` regenerado pero **sin trackear** (nunca estuvo en git; solo
  se versionan los PDFs de figuras). Regenerable con pdflatex.

### Pendiente menor (no bloquea)

1. **Dos ajustes de wording** que ofrecí y no se aplicaron todavía:
   - Abstract: "performance variance" es impreciso — la métrica de `exp02` es
     `rare_retention`. Debería decir "variance in rare-event retention across
     the design space".
   - Enmarcar el `73.0%` / `1.6%` como η² del **eje `write`** (append↔merge),
     no de "merge consolidation" como si fuera una cosa con porcentaje. `merge`
     es una de las dos opciones del factor que carga el 73%.
2. **`docs/ESTADO.md` está desactualizado**: fechado 2026-08-20, no menciona ni
   el commit de revisión editorial (`a9e5b2f`) ni el cambio a inglés. Actualizar
   cuando se retome.

## De qué veníamos hablando (Q&A conceptual sobre la ley del umbral)

Larga conversación aclarando la sección de resultados. Puntos que quedaron
establecidos, por si hay que seguir:

- **Ley del umbral**: qué mecanismo de memoria importa no es fijo, es función de
  `r = K_proto/C`. `r < 1` → domina la consolidación (`merge`); `r > 1` → domina
  el desalojo (`min_strength`). Cruce reproducible en C = 10, 20, 40.
- **`K_proto`** = nº de situaciones recurrentes distintas en el flujo.
  **`C`** = nº de trazas que la memoria sostiene a la vez (presupuesto fijo).
- **"Varianza de desempeño"** = la dispersión del score `rare_retention` **entre
  las 576 arquitecturas** corridas sobre el mismo flujo en una celda `(C, r)`.
  No es ruido de muestreo; es varianza descriptiva de una población enumerada.
  Las 5 semillas son la única parte aleatoria (se promedian).
- **η² (eta²)** = `SS_between / SS_total` = qué fracción de esa dispersión la
  explica **un eje**. Es direction-blind: dice qué perilla mueve el resultado,
  **no** qué ajuste es mejor. "Qué guarda mejor los datos" lo responden los
  scores crudos (frontera 0.818 vs FIFO 0.106), la Tabla del benchmark
  (SDM 1.000 vs FIFO 0.038) y el efecto condicional (3.6% → 52.0%).
- **`append` vs `merge`** (en el contexto de una lista de trazas):
  `append` = cada escritura agrega una fila nueva. `merge` = si ya hay una traza
  con similitud coseno ≥ 0.85, refuerza **esa fila** (sube strength, no crea
  fila); si no, hace append. `merge` comprime lo rutinario en 1 slot.
- **strength**: NO empieza en 0. Valor inicial de la compuerta (`constant`→1.0;
  `novelty`→1+2·novelty; `pred_error`→1+2·pred_error; `both`→ suma). Después:
  `+= s` en cada merge, `+= reinforce` (0 o 0.5) en cada lectura, `×= decay`
  cada escritura. Únicos factores.
- **key vector** = la dirección de contenido (lo que se compara por similitud en
  lecturas y en el routing de merge). No es un ID de unicidad — claves casi
  iguales se colapsan a propósito.
- **`value`** = el payload que devuelve la lectura (el `Episode`). `age` /
  `utility` / `contribution` NO son metadata de usuario: son contabilidad de
  desalojo (`fifo`→age, `min_utility`→utility, `contribution`→resta exacta en
  sustratos distribuidos). Nota: `touch()` dice que "rejuvenece" pero el código
  solo incrementa utility — **las lecturas no resetean age**, solo los merges.
- **Señales de sorpresa** (alimentan la compuerta de strength):
  - `novelty` = `1 − max(similitud a lo guardado)`. Siempre la calcula la
    memoria.
  - `pred_error` = en flujos sintéticos lo pone a mano el generador (tag de
    importancia); en MiniGrid lo calcula un predictor lineal en línea. Antes de
    exp06 **no hay predicción de recompensa** en ningún lado; "prediction error"
    era solo una etiqueta.
- **`min_strength` no borra lo sorpresivo**: desaloja lo **más débil**. La
  sorpresa subió el strength → lo sorpresivo es lo último que se va. Gate +
  `min_strength` son un par. Falla conocida: `merge` acumula strength en
  progresión geométrica y un prototipo visto 30× puede ahogar a un evento raro
  visto 1× → por eso la **frontera usa `append`, no `merge`**.
- **rare ≠ surprise**: "rare event" es una etiqueta ground-truth del task
  (`t1_rare_retention`); "surprise" es el proxy que la memoria calcula al
  escribir. La pregunta del paper es si el proxy alcanza. Sobre datos reales
  rarity ≠ importance (un glitch es raro y basura). La sorpresa **perceptual**
  no separa importante-raro de trivial-raro (75.4% solapamiento, retención 0%);
  anclarla en **recompensa** sí (5.6% solapamiento, 52.6%). Esa es la
  contribución de §VI. Limitación abierta: entornos sin recompensa → de vuelta a
  novelty y su basura (trabajo futuro).
- **MiniGrid** = suite de gridworlds RL (Farama). `MemoryS13` fuerza uso de
  memoria: hay que recordar un objeto visto al inicio para elegir bien al final;
  observación parcial (ventana egocéntrica 7×7). El adapter NO entrena un
  agente: corre rollouts con política fija, proyecta la observación a un vector
  unitario (proyección aleatoria fija = Johnson-Lindenstrauss = lo que hace SDM)
  y lo pasa como `Stream` a las arquitecturas. Sirve para chequear que los
  hallazgos valen sobre datos que vienen de las acciones del agente, no de un
  generador.
- **Dónde está la mejor configuración en el PDF**: score y ranking en la
  **página 3** (§III-B, "Axis observability": frontera 0.818, ~8× sobre el
  incumbente). El genotipo explícito `{write: append, strength: both,
  decay: 1.0, evict: min_strength}` en la **página 6** (§VII-A, Discusión). No
  hay tabla dedicada al ganador ni gráfico de la frontera. **No todo supera al
  FIFO**: el FIFO queda en el cuartil inferior (rank ~416–469/576) pero
  ~107–160 genotipos puntúan peor.
- **FIFO no lee la señal**: genotipo con `strength: constant` + `evict: fifo` →
  la sorpresa se calcula y se descarta. Por eso su retención no se mueve entre
  condiciones de MiniGrid (2.6% → 2.6%): es el control negativo que permite
  atribuir el salto 0 → 52.6% de la frontera a la compuerta de saliencia y no a
  un artefacto del cambio de predictor.

### Preguntas del usuario que quedaron respondidas pero pueden reabrirse

- "¿La varianza mide qué tan bien funciona?" → No, mide qué perilla domina el
  spread. El desempeño lo dan los scores crudos y el benchmark.
- "Si tengo que darle la señal de importancia, ¿para qué sirve el paper?" →
  El paper no pretende importancia sin grounding (ningún sistema lo hace, los
  cerebros tampoco). Aporta: la ley del umbral (mecanismo, independiente de cómo
  se define importancia), que los efectos principales engañan en espacios
  compuertados, que el incumbente es cuantificablemente malo, y las dos
  precondiciones (fusión alcanzable; sorpresa anclada en recompensa). Vos
  elegís *la fuente* de la señal una vez, no evento por evento.

## Comandos útiles

```bash
uv run python -c "from ember.paper_sync import render_tables; render_tables()"
uv run python -c "from ember.paper_sync import verify_paper; print(verify_paper('paper/main.tex'))"
cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
uv run pytest -q -m "not slow"
```

Regenerar figuras sin correr experimentos:

```python
import json
from ember.figures import figura_umbral, figura_comparacion_dominios
d2 = json.loads(open('results/exp02_threshold_grid/data.json').read())
figura_umbral(d2['celdas'], 'paper/figures/fig1_threshold.pdf')
d5 = json.loads(open('results/exp05_real_embeddings/data.json').read())
figura_comparacion_dominios(d5['dominios']['synthetic']['celdas'],
                            d5['dominios']['cifar100']['celdas'],
                            'paper/figures/fig2_dominios.pdf')
```
