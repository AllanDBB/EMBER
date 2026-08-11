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

## 3. Qué falta comprobar

- [ ] `exp02`: si el cruce de régimen sigue cayendo en r=1 en todas las
      capacidades. El script lo verifica y anota si no.
- [ ] `exp04`: si el 1.000 de retención de SDM y el 0.908 de fidelidad
      sobreviven con los contadores no contaminados.
- [ ] `exp05`: si el umbral se mueve al pasar a embeddings de CIFAR-100.
- [ ] Caracterizar las 107 arquitecturas que puntúan peor que el FIFO, para
      poder decir si son combinaciones degeneradas o no.
