# exp09: qué afirmaciones del paper matiza

**Fecha:** 2026-10-06
**Fuente:** `results/exp09_external_baselines/` (5 semillas, celda de exp01 y
barrido de r)

## Lo que sobrevive

- **La frontera le gana al incumbente y a todo baseline externo** en el puntaje
  agregado. La frontera puntúa 0.824. LRU, LFU y la caché de utilidad quedan en
  ~0.10, al nivel del proxy FIFO-NN. Reservoir llega a 0.306, la cobertura a
  0.439 y el replay priorizado estocástico a 0.500.
- **El proxy FIFO-NN es optimista.** Ahora está medido: el buffer real queda
  todavía más abajo.

## Lo que se matiza

1. **"El espacio no contiene reglas de muestreo priorizado"** (§II-C). Era
   cierto como lista de etiquetas, pero no en cuanto a comportamiento. El
   desalojo priorizado greedy por sorpresa (`per_min`) da resultados idénticos,
   semilla por semilla, a la configuración del espacio `pred_error` +
   `min_strength`, que está en el rango 4–5 de 577. La frontera le gana solo por
   0.035 [0.028, 0.043], y toda esa diferencia está en T3. En T1 empatan, en la
   celda de exp01 y en las cuatro r del barrido.
   **Reemplazo:** la ventaja de la frontera viene de leer la sorpresa al
   desalojar, un principio que ya usa el replay priorizado, y no de un mecanismo
   exclusivo de la memoria bioinspirada. La hipótesis H1 del experimento lo
   anticipaba.
2. **"Su deficiencia es elegir, no recordar"** (conclusión). Vale para el
   proxy, pero no para el buffer desplegado. Con su lectura secuencial real (el
   primer episodio con coseno ≥ 0.85 en orden de inserción, que es un supuesto)
   el buffer reprueba el gate de reconstrucción: 0.368 contra 0.903. Su puntaje
   agregado es 0.010 y queda último entre 577.
   **Reemplazo:** el proxy reconstruye bien y falla al elegir; el buffer real
   falla en las dos cosas.
3. **Limitaciones, "el incumbente es un proxy y no el buffer desplegado".** Ahora
   la lectura del buffer real está evaluada, aunque reimplementada y sin correr
   el código de GII. El texto lo dice así.

## Cuidado

Con un umbral de coincidencia de 0.5, el gate del buffer real sube a 0.826,
pero el puntaje queda en 0.067. Que el buffer real repruebe el gate depende
entonces del criterio de coincidencia supuesto. Que quede en el piso del
puntaje no depende de ese criterio.
