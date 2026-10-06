# La precondición de fusión era de separabilidad, no de calibración

**Fecha:** 2026-10-06
**Fuente:** `results/exp07_merge_sensitivity/` (umbral 0.30–0.97, C = 20, 5 semillas)
**Responde a:** R1.4, R2.2

## Lo que sobrevive

- **Sintético.** El cruce no depende del umbral entre 0.40 y 0.85: queda en
  r ≈ 0.87 en todos los casos, igual en cada semilla. Medido sobre
  K_effective, queda entre 0.86 y 1.02 para todo umbral en el que la fusión se
  dispara (0.30–0.90). La hipótesis de régimen sale reforzada si se enuncia
  sobre la razón efectiva.
- **CIFAR-100 sin régimen de compresión.** Se sostiene para todos los umbrales.
  El criterio de viabilidad fijado antes de correr (fusión ≥ 0.25, pureza ≥ 0.9
  y escritura dominante en r = 0.25) no lo cumple ningún umbral.

## Lo que no sobrevive, y con qué se reemplazó

- *"La consolidación requiere similitud intra-prototipo por encima del umbral
  de fusión"* (contribuciones, §VII, conclusión). El umbral es libre: a 0.30 la
  fusión se dispara en el 86 % de las escrituras de CIFAR-100. El problema es
  que intra (mediana 0.38) e inter (p95 0.42) se solapan, así que las fusiones
  son impuras (48 %) y absorben el 70 % de los raros.
  **Reemplazo:** la precondición es de separabilidad. Hace falta un umbral que
  los encuentros repetidos superen y los distintos no.
- *"A per-domain merge threshold could change it"* (§VIII). Falso con este
  extractor. **Reemplazo:** "a lower merge threshold does not". El extractor
  sigue siendo la variable no probada.
- *"One merge threshold (0.85)"* figuraba entre las condiciones no probadas de
  §III-D. Ya no está.

## Trampa a recordar

A 0.30 en CIFAR-100 reaparece un cruce (r nominal 0.61, efectivo 1.07): η²_write
supera a η²_evict en r bajo. **No es un régimen de compresión.** η² no tiene
signo: la escritura domina porque las fusiones destruyen información, no porque
la consoliden. La mejor configuración con fusión retiene 0.30 y la frontera
`append` retiene 1.00. Por eso exp07 registra el efecto con signo, la pureza y
la absorción de raros, no solo η².

## Pendiente

El abstract todavía dice "0.401 against 0.85". Es cierto como dato, pero sugiere
que el problema es el valor del umbral. Si se recupera espacio, conviene
reescribirlo en términos de separabilidad.
