# Qué números se movieron respecto del draft, y por qué

**Fecha:** 2026-08-11
**Fuente:** `results/exp01_nas_full/`, `results/exp03_axis_liveness/`
**Comparación contra:** `docs/borradores/BIP2026_EMBER_draft.pdf`

El rediseño unificó el motor de memoria, corrigió tres defectos numéricos y
arregló las tres tareas degeneradas. Los números del draft se movieron. Esto
registra cuánto y por qué causa, que era el riesgo declarado en la §5 del spec.

## 1. Lo que sobrevive, y sale reforzado

### Los seis ejes son observables

| Eje | Piloto | Ahora |
|---|---|---|
| `evict` | vivo | 0.728 |
| `strength` | vivo | 0.711 |
| `decay` | vivo | 0.645 |
| `write` | vivo | 0.583 |
| `reinforce` | **0.000000** | **0.206** |
| `read` | **0.000000** | **0.119** |

El espacio tiene **576 arquitecturas funcionalmente distintas**, no 96. La
ventaja metodológica que vende la §III del paper —un espacio discreto
completamente enumerable— ahora es una afirmación verificada, y `exp03` corre en
CI para que siga siéndolo.

Las dos causas se arreglaron por separado: `read` porque los tres modos ahora
hacen cosas distintas (superposición real de Kanerva, con radio relativo al
mejor match en vez de absoluto), y `reinforce` porque T1 intercala lecturas
entre escrituras, de modo que la fuerza modificada llega a influir en desalojos.

### La arquitectura de la frontera no se movió

```
write=append · strength=both · decay=1.0 · evict=min_strength
```

Es exactamente el genotipo que el piloto reportaba como frontera. Puntaje 0.818.
La invariancia del óptimo es el hallazgo que mejor aguanta.

### La saliencia sigue siendo un efecto condicional, no un efecto principal

Efecto principal de `strength`: **1.4 %** (draft: 0.2 %).
Condicionado a `evict=min_strength, write=append`, la retención de eventos raros
va de 0.036 (fuerza constante) a 0.520 (novedad + error de predicción).

**Mejora condicional: 14.4×** (draft: 35×).

La tesis metodológica se sostiene —en un espacio con mecanismos compuertados los
efectos principales engañan— pero la magnitud es menos de la mitad. Hay que
reescribir el número en el abstract y en la conclusión.

## 2. Lo que NO sobrevive

### El incumbente ya no está en el piso de su espacio

| | Draft | Addendum | Ahora |
|---|---|---|---|
| Puntaje del FIFO | 0.261 | 0.009 (sin la tarea degenerada) | **0.106** |
| Puestos estrictamente peores | 0 | 0 | **107** |
| Empates | 162 | 162 | 54 |
| Rango | "#415 de 576" | "#415–#576" | **#416–#469 de 576** |
| ¿En el piso? | sí | sí | **no** |

Esto contradice al draft *y* al addendum. La afirmación
> "el buffer FIFO de una arquitectura cognitiva desplegada está en el piso de su
> propio espacio de diseño"

**es falsa con las tareas arregladas.** 107 arquitecturas puntúan estrictamente
peor que el FIFO.

La causa es directa: en el piloto T2 devolvía una constante para las 576
arquitecturas y T3 devolvía cero para todas, así que el puntaje agregado casi no
tenía resolución en la parte baja de la distribución y todo lo malo se apilaba
en un empate masivo. Con T2 discriminando y T3 con señal diferenciada, la cola
inferior se separa: genotipos con fusión más desalojo aleatorio más decaimiento
fuerte hacen peor que simplemente descartar lo más viejo.

**Qué se puede afirmar en su lugar**, y sigue siendo fuerte:
- El FIFO está en el **cuartil inferior** del espacio (#416–#469 de 576).
- Su puntaje es 0.106 contra 0.818 de la frontera: **un factor de 7.7**.
- Ninguna de las 107 que puntúan peor es una arquitectura que alguien
  propondría: todas combinan mecanismos que se anulan entre sí.

Esa última observación hay que verificarla antes de escribirla.

### El 63.5 % de la escritura bajó a 56.6 %

| Eje | Draft (η²) | Ahora (η²) |
|---|---|---|
| `write` | 63.5 % | 56.6 % |
| `evict` | 9.9 % | 17.0 % |
| `decay` | 2.5 % | 1.8 % |
| `strength` | 0.2 % | 1.4 % |
| `read` | 0.0 % | 0.0 % |
| `reinforce` | 0.0 % | 0.0 % |

`evict` casi se duplica porque T2 ahora la mide. La lectura del draft —que el
63.5 % es una propiedad de la tarea y no del mecanismo— se mantiene.

Nota sobre `read` y `reinforce`: tienen η² de 0.0 % pero observabilidad no nula.
Eso **no** es lo mismo que en el piloto. Explican ~0 % de la varianza promediando
sobre el espacio, pero sí cambian el resultado en puntos concretos. Un eje con
η²=0 y observabilidad=0 no está midiendo nada; uno con η²=0 y observabilidad>0
es un mecanismo cuyo efecto no es un efecto principal. Hay que decirlo así.

### La descomposición con interacciones

| Término | Varianza explicada |
|---|---|
| `write` | 56.6 % |
| `evict` | 17.0 % |
| `write×evict` | 7.8 % |
| `decay×evict` | 5.5 % |
| `strength×evict` | 4.3 % |
| `decay` | 1.8 % |
| `strength` | 1.4 % |

Las tres interacciones más grandes involucran a `evict`, que es la lectura
correcta: es la única política que lee la señal de fuerza, así que compuerta a
todo lo demás.

## 3. El benchmark de arquitecturas (`exp04`)

Corrido con los contadores arreglados. **La afirmación del draft sobre
arquitecturas sí sobrevive**, a diferencia de la afirmación sobre el espacio de
genotipos.

### Fase 1 · gate de reconstrucción

| Arq. | Draft | Ahora | Gate |
|---|---|---|---|
| SDM | 0.908 | 0.859 | ✓ |
| ENN | 0.902 | 0.907 | ✓ |
| Spiking-SDM | 0.623 | 0.619 | ✓ |
| FIFO | 0.902 | 0.907 | ✓ |
| Spiking (LIF+STDP) | 0.483 | **0.459** | ✗ |

El circuito spiking puro **sigue sin pasar el gate**. La motivación de la PoC-5
—que la brecha es el puente de vectores continuos a ensambles neuronales
consistentes, no el mecanismo de engrama en sí— se mantiene intacta.

### Fase 2 · batería bajo presión

| Arq. | T1 raros (draft) | T1 raros (ahora) | T3 interf. (draft) | T3 interf. (ahora) |
|---|---|---|---|---|
| SDM | 1.000 | **1.000** | 0.000 | **0.875** |
| ENN | 0.167 | **0.750** | 0.000 | **0.375** |
| Spiking-SDM | 0.167 | 0.675 | 0.000 | 0.719 |
| FIFO | 0.067 | **0.038** | 0.000 | **0.000** |

Tres cosas que vale la pena mirar:

**El 1.000 de SDM sobrevive.** Estaba medido sobre contadores contaminados por
~195 residuos de trazas desalojadas, y aun así el número era correcto: la
compuerta de error de predicción más el desalojo por mínima fuerza retienen los
20 eventos raros con contadores limpios igual que con contadores sucios. El
titular del draft aguanta.

**T3 reproduce la predicción del draft casi exactamente.** El draft dice que la
versión con señal diferenciada "da SDM = 0.875, ENN = 0.375, FIFO = 0.000, que
es la comparación pretendida". Medido: **0.875, 0.375, 0.000**. Coincidencia
exacta en los tres. Es la confirmación más limpia que salió de todo el
rediseño.

**El FIFO es el piso entre arquitecturas.** 0.038 de retención de eventos raros,
el mínimo de las cuatro admitidas. Es decir: la afirmación "el incumbente está
en el piso" **es cierta entre arquitecturas y falsa dentro del espacio de
genotipos**. Son dos afirmaciones distintas y el draft las mezcla. Hay que
separarlas en la reescritura.

**ENN mejoró mucho** (0.167 → 0.750). El modo de falla que el draft documenta
—la fusión acumula fuerza y ahoga la señal de saliencia— sigue existiendo y
tiene su test, pero es menos severo con los flujos nuevos, donde la variación
intra-prototipo hace que el umbral de fusión se dispare menos seguido.

## 4. Qué falta comprobar

- [ ] `exp02`: si el cruce de régimen sigue cayendo en r=1 en todas las
      capacidades. Al 2026-08-11 va confirmado en C=10, C=20 y C=40, siempre
      entre r=0.75 y r=1.00.
- [ ] `exp05`: si el umbral se mueve al pasar a embeddings de CIFAR-100.
- [ ] Caracterizar las 107 arquitecturas que puntúan peor que el FIFO, para
      poder decir si son combinaciones degeneradas o no.

---

## 5. Un hallazgo nuevo: la ley tiene una precondición que el borrador no enuncia

Salió al instrumentar `exp05`. **Afecta cómo hay que enunciar el claim central.**

### Lo que se observó

Sobre un flujo construido desde embeddings —con varianza intra-clase realista en
vez de un centro más ruido de 0.05— el eje de escritura explica ~0 % de la
varianza **incluso con r = 0.5**, donde la ley predice que debería dominar.

Primera hipótesis: el umbral de fusión (0.85) es inalcanzable en ese dominio, el
mismo modo de falla que mataba al eje de lectura. **Falsa**: la fusión se dispara
en el 50 % de las escrituras.

Segunda medición, directa: cuántas trazas ocupa la experiencia rutinaria después
de consolidar, sin presión de capacidad.

| Dominio | K nominal | K efectivo | r efectivo | η² escritura |
|---|---|---|---|---|
| sintético | 10 | 10 | 0.50 | **0.818** |
| sintético | 40 | 40 | 2.00 | 0.004 |
| embeddings | 10 | **35** | **1.75** | 0.004 |
| embeddings | 40 | **149** | **7.45** | 0.000 |

### Qué significa

Con consolidación **parcial**, cada prototipo ocupa varias ranuras. Diez clases
de un banco de embeddings se convierten en 35 trazas, porque la mitad de las
visitas queda por debajo del umbral de fusión y crea traza nueva. La rutina llena
la memoria igual que si no se hubiera fusionado nada, y no queda lugar para lo
raro.

**La ley no se rompe: se estaba aplicando a la variable equivocada.** El caso de
K=10 sobre embeddings tiene r efectivo = 1.75, y su η² de escritura es 0.004 —
exactamente lo que la ley predice para r > 1.

### Cómo hay que enunciarla

> La variable de control del régimen es el número de prototipos **efectivo tras
> consolidar**, no el nominal. Sobre flujos sintéticos con dispersión
> intra-prototipo baja los dos coinciden, que es por qué la distinción no
> aparecía.

Esto es **mejor** que el enunciado del borrador, no peor:

1. Explica por qué la ley podría no transferir a datos reales, y muestra que
   transfiere si se mide la variable correcta.
2. Le da al robot una cantidad medible en línea: `K_efectivo` se cuenta mirando
   cuántas trazas tiene la memoria, sin saber nada del ambiente.
3. Convierte la sección de limitaciones en un resultado.

**Decisión pendiente del autor**: si el paper se reescribe sobre `r_efectivo` o
si se reporta la precondición como una calificación de la ley nominal.
`exp05` localiza el cruce contra ambas variables para que la comparación esté
sobre la mesa.
