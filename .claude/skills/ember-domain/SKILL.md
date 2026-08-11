---
name: ember-domain
description: Use when working on EMBER and you need the vocabulary — SDM, engrama, STDP, e-MDB, LOLA, la ley del umbral r=K/C, genotipo, gate de reconstrucción — or when a term in the code or the paper is unfamiliar. Load before reading the memory architectures or the NAS.
---

# El dominio de EMBER

Glosario operativo. Cada entrada dice qué es el concepto **y qué forma toma en
este repo**, porque el nombre de la literatura y el nombre en el código no
siempre coinciden.

## El problema

Un robot que aprende durante toda su vida enfrenta un problema de memoria que va
más allá del almacenamiento: tiene que decidir **qué vale la pena guardar**,
mantenerlo accesible en proporción a su relevancia, y **recuperarlo desde una
clave degradada**. Esas tres exigencias corresponden a tres propiedades bien
estudiadas de la memoria biológica: codificación selectiva modulada por sorpresa,
consolidación y decaimiento regulados por uso, y recuperación por contenido
robusta al ruido.

## Sparse Distributed Memory (SDM)

Modelo de Kanerva (1988). La idea central: guardar información en un solo lugar
es frágil; hay que distribuirla sobre muchas ubicaciones solapadas, para que una
consulta ruidosa active suficientes de las mismas y reconstruya el original.

- **Hard locations**: direcciones fijas y aleatorias. En el original están en
  {0,1}ⁿ y se activan por radio de Hamming. En `ember.memories.sdm` son vectores
  unitarios y se activa la fracción más alineada por producto interno, lo que
  hace el radio autoadaptativo a la densidad local.
- **Degradación suave**: la propiedad que hace a SDM útil para un robot. No
  falla de golpe cuando la consulta se degrada.
- **Lectura por superposición**: se suman los contadores de las ubicaciones
  activas y se compara la reconstrucción contra lo guardado. No es una consulta
  a un solo lugar — confundir esto es lo que dejó muerto al eje de lectura en el
  piloto.

## Engrama y consolidación Hebbiana

El **engrama** es la traza física que una memoria deja en el tejido neural
(Semon, 1904; confirmado experimentalmente en Liu et al., 2012). El mecanismo
celular es la potenciación de largo plazo (**LTP**): neuronas que disparan juntas
repetidamente fortalecen su sinapsis.

En `ember.memories.enn` eso es una matriz de pesos rápidos, una fila por
episodio, actualizada en línea sin retropropagación. Cada escritura crea una
traza o **se fusiona** con una existente si ya hay algo suficientemente parecido.

**El modo de falla que hay que conocer**: la fusión acumula fuerza en progresión
geométrica, así que un prototipo visitado 30 veces supera a un evento raro
codificado con sorpresa máxima una sola vez. La compuerta de saliencia funciona
solo mientras la señal de fuerza no quede ahogada por la acumulación de otro
mecanismo.

## STDP y circuitos spiking

**Spike-Timing-Dependent Plasticity** (Bi y Poo, 1998) es la regla Hebbiana a
nivel de impulso: la sinapsis de A a B se fortalece si A dispara justo antes que
B (orden causal) y se debilita si es al revés. Agrupa neuronas en ensambles por
sí sola — un ensamble es precisamente un engrama.

En la PoC-3 esto se validó: pesos intra-ensamble ~0.21 contra inter-ensamble
~0.02, y un cue parcial del 40 % reclutaba el resto del grupo.

**El límite conocido**: opera nativamente sobre patrones de activación discretos.
Sobre vectores continuos, la selección de neuronas por producto interno es
sensible a perturbaciones chicas, así que una versión ruidosa activa un conjunto
parcialmente distinto. Cerrar esa brecha necesita una capa de codificación
estable de Rⁿ a ensambles consistentes — el objetivo de la PoC-5.

## Compuerta de saliencia (error de predicción)

La fuerza con que se codifica un episodio se modula por cuán sorpresivo es. En
biología esa señal la llevan proyecciones dopaminérgicas.

**El punto que más se malinterpreta de todo el proyecto**: el efecto principal de
esta compuerta sobre el espacio completo es de ~1 %, lo que invita a concluir que
el mecanismo más citado de la literatura casi no importa. Es falso. De las cuatro
políticas de desalojo, **tres son insensibles a la fuerza**: en tres cuartos del
espacio la señal se calcula y se descarta. Condicionada a la única que sí la lee,
multiplica la retención de eventos raros por más de diez.

La conclusión no es sobre el mecanismo, es sobre el instrumento: en un espacio
con mecanismos compuertados, los efectos principales engañan.

## e-MDB y el problema LOLA

**e-MDB** (epistemic Multilevel Darwinist Brain) es una arquitectura cognitiva
sobre ROS2 diseñada para el problema **LOLA** (Lifelong Open-ended Learning
Autonomy): un robot que adquiere y relaciona conocimiento de forma autónoma y
continua, sin objetivos predefinidos ni fronteras de tarea.

Su unidad de experiencia:

```
Episode = {old_perception, policy, action, perception, reward_list}
```

replicada en `ember.core.types.Episode`.

Su componente de memoria episódica, `EpisodicBuffer`, es un
`collections.deque` de tamaño fijo. Sin criterio de importancia, sin
recuperación por contenido, sin consolidación. Es una base de datos sin criterio.

**En EMBER no es un baseline externo: es un punto del espacio de diseño**,
`FIFO_GENOTYPE`. Esa decisión convierte una comparación potencialmente circular
en una pregunta falsable. El mapeo es deliberadamente generoso —le regala una
búsqueda por similitud que el `deque` real no tiene— y decirlo explícitamente
desarma la objeción de "compararon contra un hombre de paja".

## La ley del umbral

El claim central del programa:

> Qué mecanismo de memoria bioinspirada importa no es una propiedad del
> mecanismo: es función del cociente **r = K_proto / C** entre prototipos de
> experiencia recurrentes y capacidad de memoria. Por debajo de 1 domina la
> compresión (consolidación por fusión); por encima domina la selección (olvido
> por mínima fuerza, modulado por saliencia).

La intuición: fusionar solo paga cuando los prototipos recurrentes **caben** en
la memoria. Si hay más prototipos que ranuras, fusionar llena la memoria de
rutina y no queda espacio para lo raro.

Interpretación operativa para un robot: en un ambiente temprano y estructurado
(una bodega, un laboratorio) `r < 1` y consolidar es lo eficiente. En un ambiente
heterogéneo `r` crece por encima de 1 y lo que manda es el olvido selectivo. La
recomendación de diseño no es elegir un régimen: es usar la arquitectura de la
frontera, que nunca depende de la fusión y siempre explota la señal de
importancia, porque **el régimen cambia a lo largo de la vida del robot**.

**Cuidado al barrer el ratio**: hay que mantener constantes las visitas por
prototipo. Con un largo de flujo fijo, subir K baja automáticamente la
recurrencia, y el ratio queda confundido con ella.

**Sobre datos reales K_proto no se conoce** y hay que estimarlo
(`ember.data.prototypes`). Si la silueta es baja, no hay estructura de
prototipos y `r` simplemente no está definido para ese flujo.

### La precondición: prototipos efectivos, no nominales

Descubierta al instrumentar `exp05` (2026-08-11). **Afecta cómo se enuncia la
ley**, así que hay que conocerla antes de escribir sobre el tema.

`K_proto` es cuántos prototipos tiene el flujo. `K_efectivo` es **cuántas trazas
ocupa la experiencia rutinaria después de consolidar**, y no son lo mismo: si la
dispersión intra-prototipo deja parte de las visitas por debajo del umbral de
fusión, cada prototipo se fragmenta en varias trazas.

| Dominio | K nominal | K efectivo | r efectivo | η² escritura |
|---|---|---|---|---|
| sintético (ruido 0.05) | 10 | 10 | 0.50 | **0.818** |
| embeddings (variación real) | 10 | **35** | **1.75** | 0.004 |

Diez clases se vuelven 35 trazas, la rutina llena la memoria igual que sin
fusionar, y no queda lugar para lo raro. **La ley predice ese 0.004
correctamente** — sobre `r` efectivo, que es 1.75.

Enunciado corregido: *la variable de control del régimen es el número de
prototipos efectivo tras consolidar*. Sobre flujos sintéticos con dispersión baja
los dos coinciden, que es por qué la distinción no aparecía.

Se mide con `experiments.exp05_real_embeddings.prototipos_efectivos`, o
directamente con `PolicyMemory.n_merges / n_writes` más el conteo final de
trazas sin presión de capacidad.

**Queda abierta la decisión** de si el paper se reescribe sobre `r_efectivo` o si
la precondición se reporta como calificación de la ley nominal.

## Genotipo y espacio de diseño

Un **genotipo** es una arquitectura de memoria descrita por seis ejes:

| Eje | Opciones | Raíz biológica |
|---|---|---|
| `read` | nn / topk3 / radius | Recuperación episódica vs. círculo de activación |
| `write` | append / merge | Creación de traza vs. consolidación Hebbiana |
| `strength` | constant / novelty / pred_error / both | Compuerta de LTP por saliencia |
| `decay` | 1.0 / 0.995 / 0.98 | Debilitamiento sináptico sin refuerzo |
| `evict` | fifo / min_strength / min_utility / random | Olvido por edad vs. por importancia |
| `reinforce` | 0 / 0.5 | LTP por reactivación |

3 × 2 × 4 × 3 × 4 × 2 = **576**, enumerables de forma exhaustiva. La ventaja
metodológica es que ningún resultado se puede atribuir al comportamiento de un
optimizador aproximado.

Esa ventaja **hay que verificarla, no suponerla**: el piloto reportaba 576 pero
solo 96 eran funcionalmente distintas.

## Las dos fases de evaluación

**Fase 1, gate de reconstrucción** (R1–R4): sin presión de capacidad. Mide si la
arquitectura puede recuperar desde una clave degradada. Sin este gate, una
arquitectura puede puntuar bien en retención porque su lectura devuelve
cualquier cosa, o mal aunque su política de olvido sea correcta.

**Fase 2, batería bajo presión** (T1–T3): llegan más experiencias de las que
caben. T1 estresa saliencia × desalojo, T2 el modo de lectura con la memoria
comprimida, T3 la resistencia a interferencia secuencial.

## Referencias

Kanerva (1988) *Sparse Distributed Memory* · Hebb (1949) *The Organization of
Behavior* · Liu et al. (2012) Nature 484:381 · Bi y Poo (1998) J. Neurosci.
18:10464 · Elsken et al. (2019) JMLR 20:55 · Graves et al. (2014)
arXiv:1410.5401 · Kanerva (2009) Cogn. Comput. 1:139.
