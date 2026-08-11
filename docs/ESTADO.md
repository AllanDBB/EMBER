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
| 1–4 · Esqueleto, entorno uv, CI, núcleo (`ember.core`) | listo, commiteado |
| 5–8 · Arquitecturas (`ember.memories`) | listo, commiteado |
| 9–11 · Datos (`ember.data`) | listo, commiteado |
| 12–13 · Tareas R1–R4 y T1–T3 (`ember.tasks`) | listo, commiteado |
| 14–16 · NAS y `ExperimentRun` | listo, commiteado |
| 17 · `exp01` y `exp03` | **corridos**, resultados en `results/` |
| 18 · `exp02` grilla del umbral | escrito, **corriendo cuando se paró** |
| 19 · `exp04` benchmark de arquitecturas | falta |
| 20 · `exp05` embeddings reales | falta |
| 21 · `ember.envs` (MiniGrid) | falta |
| 22 · `CLAUDE.md`, skills de `.claude/`, README | falta |
| 23 · `paper/` y `paper_sync` | falta |

219 tests pasan. `ruff check` y `ruff format --check` limpios.

## Los cuatro defectos del piloto: arreglados y fijados con tests

1. **Dos motores de memoria con semánticas distintas.** Ahora hay uno solo:
   `ember.core.policies` define cada mecanismo una vez, `PolicyMemory` los
   compone, y `ARCHITECTURES["FIFO"]` es `FIFO_GENOTYPE`, no una clase aparte.
2. **Contadores de SDM corrompidos al desalojar.** `TraceStore.contribution`
   guarda el escalar exacto escrito. Test: desalojar todo deja `‖V‖ ≈ 0`.
3. **`SpikingMemory` sembrada con `id()`.** Toda aleatoriedad sale de la semilla.
   Test: misma semilla, pesos idénticos bit a bit.
4. **Tres ejes/tareas inobservables.** Los seis ejes viven ahora (ver abajo).
   `exp03` falla con código ≠ 0 si alguno vuelve a morir, y corre en CI.

## Resultados que ya están

**`exp03` — los seis ejes son observables.** El espacio tiene 576 arquitecturas
funcionalmente distintas, no 96. `reinforce` pasó de 0.000000 a 0.206 y `read`
de 0.000000 a 0.119.

**`exp01` — la búsqueda exhaustiva.** Corre en 16 s sobre 12 núcleos.
- Frontera: `append + both + decay 1.0 + min_strength`, puntaje 0.818.
  **Es el mismo genotipo que reportaba el draft.**
- Saliencia: efecto principal 1.4 %, efecto condicional **14.4×** (el draft
  decía 35×).
- **El FIFO ya no está en el piso**: 107 arquitecturas puntúan estrictamente
  peor. Rango #416–#469 de 576. Esa afirmación del draft y del addendum no
  sobrevive. Detalle en `docs/notas/2026-08-11-numeros-movidos.md`.

## Lo siguiente, en orden

1. **Terminar `exp02`** (`uv run python -m experiments.exp02_threshold_grid`).
   Tarda varios minutos: 4 capacidades × 8 ratios × 5 semillas × 576 genotipos.
   Verifica si el cruce de régimen sigue cayendo en r = 1 en toda capacidad, y
   genera la Figura 1.
2. **`exp04`** — benchmark de dos fases sobre las cinco arquitecturas. Es donde
   se sabrá si el 1.000 de retención de SDM sobrevive a los contadores
   arreglados.
3. **`exp05`** — la ley del umbral sobre embeddings de CIFAR-100, con `K_proto`
   estimado. Requiere bajar el dataset una vez.
4. **`.claude/` y docs** — `CLAUDE.md`, las cuatro skills, README.
5. **`paper/`** — LaTeX con el macro `\result{}` y `paper_sync` para que ningún
   número publicado pueda desincronizarse de `results/`.

## Comandos

```bash
uv run pytest -q                                  # 219 tests
uv run pytest -q -m "not slow"                    # rápido
uv run ruff check . && uv run ruff format --check .
uv run python -m experiments.exp03_axis_liveness  # auditoría, 15 s
uv run python -m experiments.exp01_nas_full       # búsqueda, 16 s
```

## Decisiones tomadas que no son obvias del código

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
