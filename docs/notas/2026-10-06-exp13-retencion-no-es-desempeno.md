# exp13: retener mejor no es todavía actuar mejor

**Fecha:** 2026-10-06
**Fuente:** `results/exp13_downstream_learning/` (re-corrida con el `Reservoir`
del motor único, commit `fa39bde`)

Un agente de control episódico (MFEC) cuya única experiencia es la memoria,
sobre cuatro variantes de MiniGrid que se alternan y reaparecen. 10 semillas,
IC95 bootstrap pareado.

## Qué sale

| C=300 | frontera − FIFO |
|---|---|
| retorno al reaparecer la tarea | +0.038 [+0.005, +0.071], 7/10 |
| AUC del flujo completo | −0.075 [−0.098, −0.053], 0/10 |

En C=1000 pasa lo mismo: reaparición +0.109 y AUC −0.064.

- La frontera con decaimiento (0.995 y 0.98, del mismo espacio) sube el AUC
  respecto de la frontera sin decaimiento (+0.076 y +0.081) y queda al nivel
  del FIFO. A cambio, pierde la ventaja al reaparecer. **Ningún genotipo
  probado le gana al FIFO en las dos métricas.**
- La frontera supera al reservoir en AUC en ambas capacidades, y en
  reaparición con C=300.
- La memoria sin límite da AUC 0.566, contra 0.144 de la frontera.

## Qué afirmaciones del paper se matizan

- **§VII:** "el buffer tal como está implementado es inadecuado para el régimen
  LOLA". Sigue siendo cierto como almacén *de qué recordar*: es una afirmación
  sobre retención. Pero ya no puede leerse como una afirmación sobre el
  comportamiento. Se agregó la salvedad, con referencia a §VIII.
- **§VIII:** "si la mejor retención mejora los modelos de e-MDB no está
  probado". Ahora hay una prueba con un proxy, y da en contra en el retorno
  integrado. La frase quedó acotada a los modelos reales de e-MDB.
- **Conclusión:** se agregó que retener no es todavía actuar. Lo que dice la
  conclusión sobre la retención sigue en pie.

## Lectura

La retención de eventos raros que mide la batería tiene un costo de
plasticidad cuando el objetivo es el desempeño en la tarea en curso. Sin
decaimiento, `min_strength` congela la memoria en trazas viejas de fuerza alta.
Combinar las dos cosas no es posible dentro del espacio de 576 con las
condiciones probadas: queda como trabajo futuro.
