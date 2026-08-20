---
name: ember-experiment
description: Use when adding, running, or modifying an EMBER experiment under experiments/, when a result needs to be reproducible, or when designing a sweep over the threshold law. Covers ExperimentRun, seed propagation, parallel evaluation, and the confounds that make a sweep meaningless.
---

# Escribir un experimento de EMBER

## La forma

```python
from ember.experiment import ExperimentRun


def main() -> int:
    with ExperimentRun("exp06_lo_que_sea") as run:
        run.set_seeds(SEMILLAS)
        run.note("algo que quien lea esto en tres meses necesita saber")

        resultado = hacer_el_trabajo()
        run.record("tabla_principal", resultado)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Escribe `results/exp06_lo_que_sea/data.json` más un `manifest.json` con SHA de
git, si el árbol estaba sucio, semillas, versiones y timestamp. **Si el
experimento aborta no escribe nada**: un resultado parcial es peor que ninguno
porque parece completo.

Reglas:

- **La lógica va en una función importable**, y `main()` solo la orquesta. Así
  se puede testear sin correr el experimento entero — ver
  `medir_liveness` en `exp03` o `barrer_grilla` en `exp02`.
- **Toda semilla se propaga explícitamente.** Nada de `np.random.*` global.
- **`run.note()` para lo que sorprenda.** Si un invariante no se cumplió, si un
  eje quedó al borde del umbral, si una celda no tenía estructura: al
  manifiesto. Es la diferencia entre un resultado y un resultado interpretable.
- **El experimento debe fallar ruidosamente**, no silenciosamente. `exp03`
  devuelve código ≠ 0 si un eje muere.

## Evaluar genotipos en paralelo

`run_search` paraleliza con procesos, así que el evaluador **tiene que ser
serializable**: una clase con `__call__`, no una closure ni un `lambda`. Por eso
`GenotypeEvaluator` y `RareRetentionEvaluator` viven en `experiments/_common.py`
a nivel de módulo.

```python
from ember.nas.engine import run_search
from experiments._common import EvalConfig, GenotypeEvaluator

resultados = run_search(GenotypeEvaluator(EvalConfig(seeds=(0, 1, 2))), n_jobs=-1)
```

El orden del resultado es determinista e independiente del paralelismo: se
desempata por la etiqueta del genotipo, no por orden de finalización. Verificalo
si tocás el motor.

## Los confundidos que arruinan un barrido

### La recurrencia por prototipo, al barrer el ratio

La ley se enuncia sobre `r = K_proto / C`, donde los prototipos son
**recurrentes**. Si dejás el largo del flujo fijo y subís K, cada prototipo se ve
menos veces, y en el extremo aparece una sola vez: ya no hay nada que fusionar.
La caída de la dominancia de la escritura no diría nada sobre el umbral, diría
que el flujo dejó de ser recurrente.

**Fijá las visitas por prototipo** y dejá que el largo del flujo crezca con K.
Ver `VISITAS_POR_PROTOTIPO` en `exp02`.

### Una tarea que no discrimina

Si una tarea devuelve el mismo valor para todo el espacio, su varianza es cero
pero **sigue entrando al promedio** y arrastra el puntaje agregado hacia una
constante. En el piloto T2 aportaba 0.256 a las 576 arquitecturas, y el 98 % del
puntaje del incumbente era esa constante.

Antes de agregar una tarea a una batería, verificá que su varianza sobre el
espacio no sea cero.

### Un rango dentro de un empate

`rank_of` devuelve `(optimista, pesimista)`. **Nunca reportes el optimista como
si fuera la posición.** Si el intervalo es ancho, el ranking puntual no
significa nada; usá `is_at_floor` si lo que querés afirmar es que nada puntúa
peor.

### Muestrear el espacio por prefijo

`enumerate_space()[:n]` comparte el valor de los ejes que varían más lento. Para
cualquier cosa que compare genotipos hermanos, muestreá **conjuntos de
hermanos** — ver `genotipos_a_auditar` en `exp03`.

## Cuánto cuesta cada cosa

Con 12 núcleos: el espacio completo sobre la batería de tres tareas es ~16 s. La
grilla del umbral es ~12 min porque las celdas de capacidad y ratio altos tienen
flujos de miles de escrituras. Si un barrido va a tardar más de unos minutos,
medí una celda primero.

## Los experimentos que hay

| Script | Qué produce |
|---|---|
| `exp01_nas_full` | Efectos principales, interacciones, dónde cae el incumbente |
| `exp02_threshold_grid` | La ley del umbral y la Figura 1 |
| `exp03_axis_liveness` | Auditoría de observabilidad; corre en CI |
| `exp04_arch_benchmark` | Gate de reconstrucción y batería, por arquitectura |
| `exp05_real_embeddings` | La ley sobre CIFAR-100, con K estimado |
| `exp06_minigrid` | Frontera vs. FIFO sobre un entorno real bajo observabilidad parcial |

## Cuando los números se mueven

Si un experimento cambia números que ya están en `paper/`, escribí qué se movió
y por qué causa en `docs/notas/`. Ver `2026-08-11-numeros-movidos.md` como
modelo: separa lo que sobrevive, lo que no, y con qué reemplazar cada afirmación
caída.
