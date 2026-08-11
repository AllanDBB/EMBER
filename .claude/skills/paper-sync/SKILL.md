---
name: paper-sync
description: Use when writing or editing the EMBER paper under paper/, when putting a measured number into LaTeX, when regenerating tables or figures, or when checking that published numbers still match results/. Covers the \result macro, verify_paper, and what to do when a number moves.
---

# Mantener el paper sincronizado con lo medido

## El problema que esto resuelve

Un número entra al LaTeX. Después se arregla un bug o se rediseña una tarea. Los
resultados cambian. El número del paper se queda donde estaba, y nada avisa.

Ya pasó en este proyecto: el draft afirmaba cosas que el código no producía, y
se descubrió por una auditoría manual meses después.

## Todo número medido va con macro

```latex
El modo de escritura explica
\result{exp01_nas_full:main_effects.0.eta2|pct}{56.6}\% de la varianza.
```

El macro imprime el segundo argumento, así que el PDF compila sin correr nada.
`verify_paper` compara ese valor contra el JSON del experimento.

**Un número sin macro es un número que nadie está verificando.** Si escribís una
cifra medida a mano, la próxima corrida que la mueva no va a decir nada.

### Sintaxis de la clave

```
experimento:ruta.punteada[|formato]
```

`ruta.punteada` navega el `data.json`; los índices de lista van como números.
Formatos: `pct` (×100, un decimal), `pct2`, `f2`, `f3` (por defecto), `int`.

Para ver qué claves hay disponibles:

```bash
uv run python -c "
import json; from pathlib import Path
print(json.dumps(json.loads(Path('results/exp01_nas_full/data.json').read_text()), indent=1)[:2000])
"
```

## Flujo de trabajo

```bash
# 1. Regenerar tablas y el macro desde los resultados
uv run python -c "from ember.paper_sync import render_tables; print(render_tables())"

# 2. Verificar que nada se desincronizó
uv run python -c "
from ember.paper_sync import verify_paper
d = verify_paper('paper/main.tex')
print('\n'.join(str(x) for x in d) if d else 'todo sincronizado')
"
```

En `paper/main.tex` hay que incluir el macro una vez:

```latex
\input{tables/result_macro}
```

## Cuando un número se mueve

Esto es lo importante, y no es un problema técnico.

1. **No actualices el número y sigas.** Preguntate por qué se movió: ¿un bug
   arreglado, un cambio de protocolo, una tarea rediseñada?
2. **Verificá si la afirmación sigue siendo cierta**, no solo si el número
   cambió. A veces se mueve un decimal; a veces se cae un claim. En agosto de
   2026 la afirmación "el FIFO está en el piso de su propio espacio de diseño"
   dejó de ser cierta al arreglar dos tareas degeneradas, y hubo que
   reemplazarla por una más débil pero verdadera.
3. **Escribilo en `docs/notas/`.** Ver `2026-08-11-numeros-movidos.md` como
   modelo: separa lo que sobrevive, lo que no, y con qué reemplazar cada
   afirmación caída.
4. **Después** actualizá el LaTeX.

## Qué se genera solo y qué no

Generado desde `results/`, no editar a mano:
- `paper/tables/result_macro.tex`
- `paper/tables/tab_main_effects.tex` (desde `exp01`)
- `paper/tables/tab_regimes.tex` (desde `exp02`)
- `paper/tables/tab_benchmark.tex` (desde `exp04`)
- `paper/figures/fig1_threshold.pdf` (desde `exp02`)
- `paper/figures/fig2_dominios.pdf` (desde `exp05`)

Escrito a mano: el texto, y los `\result{}` que aparecen en él.

Para agregar una tabla generada, escribí la función en
`ember.paper_sync.tabla_*` y registrala en `GENERADORES`.

## Cosas que el paper afirma y hay que poder respaldar

- **"576 arquitecturas enumerables exhaustivamente"** solo se puede escribir si
  `exp03_axis_liveness` pasa. Si un eje está muerto, el espacio efectivo es más
  chico y la ventaja metodológica que vende la sección de método es falsa.
- **Un rango dentro del espacio** se reporta como intervalo. `rank_of` devuelve
  `(optimista, pesimista)`; publicar el optimista como si fuera la posición fue
  el error del piloto.
- **La ley del umbral sobre datos reales** se enuncia contra `K̂` con su
  intervalo, no contra el número de clases del generador — sobre datos reales
  `K_proto` no se conoce.

## Contexto

Vocabulario del dominio: skill `ember-domain`. Cómo se produce un resultado:
skill `ember-experiment`.
