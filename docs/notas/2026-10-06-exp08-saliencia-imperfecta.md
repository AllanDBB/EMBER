# exp08: qué afirmaciones del paper sobreviven a una saliencia imperfecta

**Fecha:** 2026-10-06
**Fuente:** `results/exp08_imperfect_salience/` (5 semillas, flujo T1 de exp01 a r = 0.25 y flujo r = 2 de exp02)
**Responde:** R1.3 y R1.11 (revisor 1, BIP2026)

## Lo que sobrevive

- La frontera le gana al FIFO en **todas** las condiciones a r = 0.25,
  incluidas AUC ≈ 0.5 y retraso de 20 pasos.
- La interacción saliencia × desalojo es real mientras la señal informa: con
  AUC ≥ 0.95 el cociente condicional sigue siendo grande.
- La ventaja de la frontera sobre la mejor configuración sin saliencia no
  depende de que los rangos no se solapen: aguanta hasta AUC empírico 0.95
  (solapamiento), 0.96 (ruido), 0.91 (10 % invertido); a r = 2 hasta 0.84.

## Lo que no sobrevive tal cual

| Afirmación | Dónde | Qué pasa | Reemplazo |
|---|---|---|---|
| "mejora la retención 14.41×" sin condición | contribuciones, conclusión | con solo `pred_error` el cociente va de 16.27× a 0.80× (IC 0.17–2.82) en AUC 0.5; `both` se sostiene por la novedad, que también es perfecta por construcción, y cae a 3.10× con distractores | "14.41× sobre una señal perfectamente separable, que se desvanece al degradarla" |
| "la frontera es regime-neutral, el robot no necesita conocer su r" | §III-F | a r = 0.25 con señal débil (AUC ≈ 0.5) una configuración con fusión sin saliencia retiene más (0.76 contra 0.60, IC pareado excluye 0); con la mitad de la sorpresa retrasada un paso, también | se agrega que por debajo de r = 1 una señal débil le devuelve la ventaja a la fusión |
| "SDM logra retención perfecta" | §IV, conclusión | es de señal limpia: 0.81 con AUC 0.99, 0.60 con AUC 0.95, por debajo de ENN (0.75, independiente de la señal) | "con la señal limpia"; en la conclusión, "ventaja que pierde frente a ENN cuando la separabilidad se degrada" |

## Lo que quedó fuera del paper por espacio

- El retraso es la degradación peor de lo que su AUC indica (H3 parcialmente
  refutada): `delay1_p050` (AUC 0.74) da 0.59 contra 0.69 de `auc700` (AUC 0.67).
- Precisión: en `main` es idéntica al recall de conjunto por aritmética; en
  `selection` está acotada en 0.5 y sigue al recall. No aporta una lectura
  separada.
- El abstract no se tocó: reporta 0.983 contra 0.050 sobre el benchmark limpio,
  que sigue siendo cierto; si hay espacio, conviene agregar "on a separable
  signal".
