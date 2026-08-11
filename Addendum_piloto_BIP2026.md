# Addendum a la propuesta BIP 2026 — auditoría del piloto

Reproduje la búsqueda completa (576 arquitecturas, 59 s, mismos números que reportaste)
y corrí dos experimentos nuevos. Tres de los hallazgos del piloto no sobreviven, y a
cambio aparece un resultado más fuerte. Esto reemplaza las secciones 5 y 6 de la propuesta.

---

## 1. Lo que no sobrevive

### El 63.5% de la consolidación por fusión es una propiedad de la tarea, no del mecanismo

La tarea de retención usa `n_common=300` experiencias comunes generadas alrededor de
`n_clusters=5` centros. Fusionar colapsa 300 escrituras en 5 entradas y libera 15 de las
20 ranuras para lo raro. El mecanismo no está descubriendo nada sobre memoria: está
descubriendo que el flujo tenía 5 cosas distintas y la memoria tenía espacio para 20.

Barrido sobre `n_clusters` (96 genotipos funcionalmente distintos × 5 semillas):

| prototipos distintos | η² escritura | η² desalojo | η² saliencia |
|---|---|---|---|
| 1  | 83.7% | 3.6% | 0.4% |
| 5  | **77.1%** | 5.2% | 0.6% |
| 20 | **1.3%** | 28.4% | 6.6% |
| 50 | 0.6% | 28.0% | 8.7% |
| 300 | 0.4% | 25.2% | 8.8% |

El efecto dominante desaparece al pasar de 5 a 20 prototipos — y 20 prototipos sobre 300
experiencias sigue siendo 93% de redundancia. No es que la tarea fuera "poco redundante":
es que el umbral está en otra parte.

### El "#415 de 576" está mal, y en la dirección que te perjudica

De las 576 arquitecturas, **cero** puntúan estrictamente peor que el FIFO, y **162 empatan
con él**. El FIFO obtiene el mínimo exacto del espacio; el #415 salió del orden de
desempate del `sort`, y es el mejor puesto posible que se le puede asignar. El rango
defendible es #415–#576, o sea: no reportar rango. Reportar que es el piso.

### El "0.261 vs 0.856" es casi todo la tarea degenerada

La tarea de ruido devuelve 0.7667 para las 576 arquitecturas (varianza exactamente 0).
Como entra al promedio, aporta 0.256 a todo el mundo. El 0.261 del FIFO es
0.017 + 0.767 + 0.000 dividido por 3: el 98% de su puntaje es la constante.
Sin esa tarea, la comparación real es **0.009 vs 0.900**. Los números actuales no se
pueden defender en revisión.

### El eje de refuerzo también está muerto, no solo el de lectura

Ya habías detectado `read_mode`. Verifiqué comparando genotipos hermanos: la diferencia
máxima es 0.000000 tanto para `read_mode` **como para `reinforce`**. La causa del segundo
es distinta y no es un bug de implementación: `reinforce` solo suma fuerza en la lectura,
y en las tres tareas todas las lecturas ocurren después de todas las escrituras, así que
la fuerza modificada nunca llega a influir en un desalojo. El eje es inobservable por
diseño de la tarea.

Consecuencia: **el espacio tiene 96 arquitecturas funcionalmente distintas, no 576.**
El "producto cartesiano de 576, enumerable exhaustivamente" es la ventaja metodológica
que vende la sección 3 de la propuesta, y hoy es 6× más chico de lo que dice.

### La saliencia no es un resultado negativo

Su efecto principal es 0.2% porque en 3 de las 4 políticas de desalojo la fuerza nunca se
lee. Condicionada a `evict=min_strength` + `write=append`, la retención de eventos raros va
de **0.017** (fuerza constante) a **0.594** (novedad + error de predicción). Publicar eso
como "el mecanismo más citado como bioinspirado apenas explica el 0.2%" es un autogol:
el revisor va a ver la interacción en la tabla y va a concluir que el ANOVA de efectos
principales era la herramienta equivocada. Que es exactamente el caso.

---

## 2. Lo que aparece a cambio (y es mejor paper)

### La ley: el régimen lo define prototipos / capacidad

Hipótesis: fusionar solo paga cuando los prototipos recurrentes **caben** en la memoria.
Si hay más prototipos que ranuras, fusionar llena la memoria de rutina y no queda espacio
para lo raro. Predicción falsable: el cruce debe ocurrir en ratio ≈ 1 y debe moverse al
mover la capacidad, no al mover `n_clusters`.

Cuadrícula de 14 celdas (capacidad 10/20/40 × prototipos 2–80):

| ratio prototipos/capacidad | η² escritura | η² desalojo |
|---|---|---|
| ≤ 0.5 | **72.9%** | 6.2% |
| ≥ 1.0 | **1.6%** | 29.4% |

El cruce cae en ratio = 1.00 en las tres capacidades probadas, de forma independiente.
La variable de control es el ratio, no la redundancia del flujo. Esto ya no es "nuestra
tarea resultó favorable a la fusión": es una condición cuantitativa y comprobable sobre
cuándo cada familia de mecanismos bioinspirados sirve.

### Los dos invariantes

1. **El incumbente está en el piso en las 14 celdas.** Retención normalizada por el techo
   alcanzable: FIFO entre 0.00 y 0.13; la frontera saturada en 1.00 en todas.
   El resultado que querías defender es el que mejor aguanta.
2. **La arquitectura de la frontera es la misma en las 14 celdas:**
   `append` + fuerza inicial modulada por error de predicción o novedad + desalojo por
   mínima fuerza. Ninguna usa fusión. La invariancia es el hallazgo: no es "buscamos y
   encontramos una buena", es "el óptimo no se mueve aunque el régimen sí".

### El claim del paper

> Qué mecanismo de memoria bioinspirada importa no es una propiedad del mecanismo: es una
> función del cociente entre prototipos de experiencia recurrentes y capacidad de memoria.
> Por debajo de 1 domina la compresión (consolidación por fusión); por encima domina la
> selección (olvido por mínima fuerza, modulado por saliencia). Dos cosas no dependen del
> régimen: el buffer FIFO de una arquitectura cognitiva desplegada está en el piso de su
> propio espacio de diseño, y la arquitectura de la frontera es la misma en todo el rango.

Tres contribuciones separables, que es lo que quiere un comité: una ley con un umbral,
un resultado sobre un sistema real, y una recomendación de diseño accionable.

Ventaja secundaria: el incumbente sale mejor tratado de lo que se trata a sí mismo. El
`EpisodicBuffer` real de e-MDB no tiene recuperación por similitud; mapearlo a
`read_mode=nn` le regala una capacidad que no tiene, y aun así queda en el piso. Decirlo
explícitamente desarma la objeción de "compararon contra un hombre de paja".

---

## 3. Plan de trabajo revisado

**B1 · Resucitar los tres ejes muertos.** (2–3 días)
Superposición real en la lectura distribuida; intercalar lecturas entre escrituras para que
el refuerzo pueda influir en desalojos — que además es lo que hace un robot, consultar
mientras opera; rediseñar o eliminar la tarea de ruido. Sin esto no se puede escribir
"576" ni "seis mecanismos".

**B2 · La cuadrícula como experimento principal.** (2–3 días)
Extender a capacidad 10–160 y prototipos 1–320, 10 semillas, con IC bootstrap. Localizar
el umbral con precisión en vez de decir "≈1". Esta es la Figura 1.

**B3 · Estadística acorde al diseño.** (2 días)
ANOVA factorial con interacciones, η² parcial y tamaños de efecto. La tesis metodológica
es que en un espacio con mecanismos *compuertados* los efectos principales engañan; la
saliencia es el caso de estudio. Reportar el contraste 0.2% vs el efecto condicional.

**B4 · Pareto.** (2 días)
Conjunto no dominado entre retención, interferencia y la tarea de ruido rediseñada,
por régimen. La pregunta interesante: si la frontera se mantiene invariante, ¿el conjunto
no dominado también?

**B5 · Sustrato spiking como eje.** (opcional, 3–4 días)
Tenés la PoC-4 corrida y con un resultado limpio para esto: la Spiking-SDM conserva la
curva de capacidad y paga ~8–9 puntos de piso de ruido que se cierran con ruido alto.
Entra como un eje más de direccionamiento, y le da al artículo el ángulo neuromórfico
sin depender de la validación en navegación.

**B6 · Navegación.** Fuera del alcance. El claim ya no la necesita: la ley del ratio se
sostiene en sintético y la validación en robot pertenece al paper siguiente.

Cronograma: B1–B4 son ~10 días de trabajo efectivo. Con la fecha límite probablemente a
fines de agosto, la versión sin B5 es entregable con holgura.

---

## 4. Lo bloqueante sigue siendo la fecha

Verifiqué el sitio hoy: la sección *Important dates* del call for papers sigue vacía.
Referencia histórica útil: BIP 2021 se realizó el 4–5 de noviembre con fecha límite el
20 de agosto. Para un evento del 11–13 de noviembre, esperá fines de agosto o principios
de setiembre — es decir, entre tres y seis semanas.

Atajo: Esteban aparece en el comité organizador de ediciones anteriores de BIP. Una llamada
resuelve en cinco minutos lo que el sitio no dice. Vale la pena confirmar también si su
participación en el comité genera un conflicto de interés declarable, dado que la revisión
es doble ciego.
