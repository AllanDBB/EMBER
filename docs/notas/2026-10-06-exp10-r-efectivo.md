# exp10: la transición la gobierna r efectivo, no r nominal

Fuente: `results/exp10_law_robustness/` (Parte A: 23 familias de flujos; Parte
B: 14 familias held-out + semillas nuevas).

## Qué se cae

- "La transición la gobierna `r = K_proto/C`". El cruce en `r` nominal va de
  0.16 a 1.32 entre familias (sd de log 0.46). Deriva (30°, 60°) y ruido
  (0.065, 0.075) lo bajan muy por debajo de 1; con ruido 0.085 no hay cruce:
  el desalojo domina en todo `r`. La forma fuerte de la hipótesis A del
  experimento (cruce en [0.5, 1.5] en toda familia) queda refutada.
- "La generadora tiene deriva pero nunca se activa" (Limitaciones): ahora se
  activa en las familias de exp10.

## Qué sobrevive

- La transición abrupta cerca de 1, enunciada sobre
  `r_eff = K_efectivo / C` (trazas que deja la consolidación sin presión de
  capacidad). Cruces en [0.56, 1.28], sd de log 0.17; más colapsada que
  `r_nom` en el 99.8 % de las réplicas bootstrap y mejor que `r_hill`,
  `r_masa`, `r_carga_nom` y `r_carga_eff`. Ruido 0.085 es consistente: ya en
  `r_nom = 0.1` la fragmentación da `r_eff ≈ 0.95`.
- Dispersión residual: prevalencia de raros (0.56 con 20 %, 0.69 con 10 %) y
  C = 5 (0.71). Ninguna de las medidas de carga la absorbe.
- En el generador estándar `r_eff = r_nom`, así que la Fig. 1 y exp02 no
  cambian.
- La frontera de exp01 generaliza: intervalo de rango [1, 1] en 13 de las 14
  familias held-out (y con semillas nuevas), [15, 20] con 0.1 % de raros; Spearman con el ranking
  estándar ≥ 0.77 (mínimo en r = 4). Con T1 sola el ranking del resto del
  espacio se descorrelaciona en selección fuerte (ρ = −0.03 en r = 4), pero la
  frontera sigue en el 10 % superior salvo con 0.1 % de raros (pesimista 297,
  por empates).

## Cómo quedó en el paper

Abstract, contribuciones, §III-B (generador), §III-D, §VII-A, §VIII y
conclusión enuncian la hipótesis sobre `r_eff`; `r_nom` queda como el caso en
que los encuentros consolidan limpio.
