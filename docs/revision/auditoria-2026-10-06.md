# Auditoría de interpretación — `paper/main.tex` (rama `revision/bip2026-reviews`)

Alcance: cada afirmación empírica se cotejó contra `results/expNN_*/data.json` y, cuando
hacía falta, contra el script que la produce. Números que ya pasan `verify_paper` no se
re-chequean; lo que se audita es **qué dicen**. Las líneas son de `paper/main.tex` tal como
está hoy (1327 líneas).

Conteo: **A = 3**, **B = 14**, **C = 11**. Más 5 incoherencias con `respuesta-revisores.md`.

---

## A — afirmaciones falsas o no sostenidas

### A1. "bit-identical ... in every combination of the remaining axes" (l. 797–799)

> "Changing either alone leaves the architecture *bit-identical* to the incumbent, in every
> combination of the remaining axes (Table \ref{tab:two-knobs})."

**Dato** (`exp01_nas_full:all_records`, comparando `evict=min_strength, strength=constant`
contra `evict=fifo, strength=constant` en las 36 combinaciones de read × write × decay × reinforce):
- poner solo la compuerta de saliencia (`strength=both` con `fifo`) da puntajes idénticos en las **36/36**. Eso se sostiene.
- poner solo `min_strength` **cambia el puntaje en 20 de 36**: en toda combinación con `write=merge`
  y decay 1.0 o 0.995 el puntaje se duplica (p. ej. nn/merge/1.0: 0.340 → 0.673). Con
  `reinforce=0.5` también cambia: 0.106 → 0.090 en nn/append/1.0. 11 de las 107 configuraciones por
  debajo del incumbente son append+min_strength con refuerzo.

Esto también afecta la l. 598–600: "under a constant gate without reinforcement min_strength
reduces exactly to fifo". Es falso con `merge`, porque la fusión acumula fuerza. En las
contribuciones (l. 183–184) dice "neither is worth installing without the other", lo que vale
solo con los demás ejes del incumbente.

**Reemplazo (l. 797–799):** "Changing either alone leaves the incumbent bit-identical (Table
\ref{tab:two-knobs}); the salience gate alone is inert in every combination of the other axes."
**l. 598–600:** "...under a constant gate with \texttt{append} writes and no reinforcement
\texttt{min\_strength} reduces exactly to \texttt{fifo}..."
**l. 183–184:** "...on two axes only, and from the incumbent neither changes anything alone."

### A2. La retención de SDM "en su lista de trazas, no en sus contadores" (l. 899–905; conclusión l. 1302–1304)

> "An ablation locates SDM's retention in its trace list, not its counters" / "the retention comes
> from its gated trace list, not from its distributed counters"

**Dato** (`exp12_substrate_cost:ablacion_sdm`): la ablación tiene cuatro variantes, y el texto
omite la que contradice la atribución.
- `SDM-a+`, contadores como **único decodificador** con borrado exacto, retiene
  `variantes.SDM-a+.battery.rare_retention.mean` = **1.000** y da batería 0.735.
- Contra la SDM completa (0.741), la diferencia es `diferencias.SDM-SDM-a+` = +0.006 con IC
  [−0.001, 0.013], o sea que empata.
- Solo fallan los contadores que **no restan** lo desalojado (`SDM-a`: 0.005).

La SDM real resta exactamente (invariante 4), así que sus contadores también bastan para retener.
Lo que decide no es "lista frente a contadores": es el ciclo de vida con compuerta más un borrado
exacto. Lo que sí se sostiene es que la lista sola alcanza, y que lo hace con 1/42 de los bytes.

**Reemplazo (l. 899–905):** "An ablation shows retention needs gated eviction with exact erasure,
not superposition: the trace list alone keeps every rare event
(\result{...SDM-b.battery_mean|f3}{0.788} vs \result{...SDM.battery_mean|f3}{0.741}); counters alone
keep them only if an evicted trace is subtracted exactly
(\result{exp12_substrate_cost:ablacion_sdm.variantes.SDM-a+.battery.rare_retention.mean|f3}{1.000},
else \result{...SDM-a...}{0.005})."
**Conclusión (l. 1302–1304):** "...but the retention comes from gated eviction with exact erasure,
which its trace list provides at 1/42 of the bytes."

### A3. "Only because merging destroys" en CIFAR-100 con umbral 0.30 (l. 1026–1028)

> "At 0.30 the write axis regains dominance at low $r$ only because merging destroys"

**Dato**: el efecto con signo de la fusión, que exp07 registra justamente para responder esta
pregunta, es **positivo**. `exp07_merge_sensitivity:umbral_fusion.cifar100.umbrales.t030.efecto_merge_r_min`
= **+0.072**: en promedio sobre el espacio, fusionar mejora la retención a r = 0.25. La nota del
manifiesto ("la escritura domina porque fusionar DAÑA") solo se emite si ese efecto es menor que 0,
y **no se emitió** (el `manifest.json` no la contiene). Lo que el dato sí respalda:
- pureza 47.7 %;
- 70 % de raros absorbidos;
- la mejor configuración con fusión retiene 0.30, contra 1.00 de la frontera `append`.

No es un régimen de compresión útil, pero la dominancia del eje no viene "solo de destruir". La
misma afirmación aparece en `docs/notas/2026-10-06-umbral-de-fusion.md` ("Trampa a recordar").

**Reemplazo:** "At 0.30 the write axis regains dominance at low $r$, but not as useful compression:
only 47.7\,\% of merges join the same class, 70.0\,\% of rare events are absorbed on arrival, and
the best merging configuration retains 0.30 against 1.00 for the \texttt{append} frontier."

---

## B — contradicciones internas o alcance mal enunciado

### B1. El chequeo "online" de la precondición necesita etiquetas (l. 1213–1217; l. 1310–1311; l. 1062–1065)

> "a check it can run online: intra-prototype similarity, measured on the vectors the memory already
> receives and compared against inter-prototype similarity ... with no external instrumentation" /
> "an online-measurable check"

**Dato:** tanto `similitud_intra_prototipo` (exp05, l. 115–119) como `distribucion_similitudes`
(exp07, l. 316–323) agrupan los pares por `it.value`, es decir, por la identidad del prototipo o
la clase. Un robot no sabe qué encuentros son "el mismo tipo". El chequeo, tal como se midió, usa
la verdad de terreno. Ningún experimento midió una versión sin etiquetas.
(Era A por contenido; queda en B porque el paper no reporta un número que esté mal: lo que hace es
prometer una propiedad del método que no se probó.)

**Reemplazo (l. 1213–1217):** "...gives a diagnostic: whether intra-prototype similarity separates
from inter-prototype similarity decides whether consolidation is worth attempting. Here it is
measured with class labels; a label-free online estimate is untested."
**Conclusión:** borrar "online-measurable" y poner "a measurable check".

### B2. "The same $r$ for three capacities" frente a exp10 (l. 49, 664–666, 686, caption l. 742–745)

> "The switch falls at the same $r$ for three capacities" (abstract) / "lands at the same $r$ for
> three capacities" / caption: "if absolute capacity governed the transition the three curves
> would cross at three different points ... Neither happens."

**Dato:**
- **exp02.** Los tres cruces caen en la misma celda de la grilla, con `r_below` entre 0.75 y 0.8
  y `r_above` = 1.0 (`umbrales`). La igualdad está cuantizada por la resolución de la grilla.
- **exp10.** Las familias de capacidad, todas en el generador estándar con r_eff = r_nom, muestran
  una **tendencia monótona con C**. Según `parte_a.colapso.r_nom.cruces`: C005 0.71, C010 0.89,
  C020 0.91, C040 0.94, C080 0.95, C160 0.98.
- **exp10, recurrencia.** También mueve el cruce: visitas_05 da 0.78 y visitas_40 da 0.95.

El paper atribuye esa dispersión solo a "C = 5", y la caption dice que ni la capacidad ni la
recurrencia lo mueven.

**Reemplazo (l. 686):** "...is abrupt and falls in the same grid interval for $C=10$--40; over
$C=5$--160 it drifts from 0.71 to 0.98."
**Abstract:** "The switch falls near $r=1$ for capacities 5 to 160, but across 23 stream
families..."
**Caption:** reemplazar "Neither happens" por "Neither happens at this resolution".

### B3. "Every threshold at which merging fires" (l. 719–723)

> "Stated over $K_{effective}$ instead, the crossover stays between 0.86 and 1.02 for every
> threshold at which merging fires"

**Dato:** con umbral 0.95 la fusión se dispara en el 36 % de las escrituras
(`umbral_fusion.synthetic.umbrales.t095.tasa_fusion_r_min` = 0.36), pero **no hay cruce**:
`cruce_r_efectivo` = null y el desalojo domina en todo r. El rango 0.86–1.02 vale solo para los
umbrales de 0.30 a 0.90. La misma línea del paper ya dice que desde 0.95 "eviction dominates at
every $r$", así que la frase se contradice a sí misma.

**Reemplazo:** "...stays between 0.86 and 1.02 for every threshold from 0.30 to 0.90..."

### B4. 3.10× atribuido solo a los distractores (l. 779–782)

> "the combined gate ... falls to 3.10× once a quarter of routine inputs are novel distractors"

**Dato:** la clave citada es `auc500_distr025`, que combina distractores **y** error de predicción
al azar (AUC 0.51).
- Con distractores y señal limpia (`settings.main.conditions.distr025.search.salience_interaction.ratio_vs_constant.both.mean`),
  el factor es **11.80** (IC 6.64–35.9).
- Con distractores y AUC 0.95, es 9.15.

**Reemplazo:** "...and falls to 3.10× when a quarter of routine inputs are novel distractors *and*
prediction error is at chance."

### B5. "Surprise must come from reward, not perceptual novelty" (abstract l. 65; contribuciones l. 194–195; l. 1272–1273)

**Dato (dos problemas):**
1. **La señal de exp06 no es novedad perceptual.** Es el error de predicción a un paso de un
   modelo lineal de transición (l. 1088–1090).
2. **exp11 muestra que lo que importa es el alineamiento con la definición de importancia, no la
   recompensa en sí.** Según `resumen`:
   - cuando la importancia es `sorpresa_transicion`, la señal perceptual le gana al control nulo
     en 15/16 celdas;
   - cuando es `novedad_estado`, la novedad por conteo gana en 16/16;
   - el error de predicción de recompensa solo funciona cuando la importancia es la recompensa
     inmediata.

La conclusión de §VI (l. 1156–1157) ya lo dice bien. El abstract y las contribuciones lo
contradicen.

**Reemplazo (abstract):** "Surprise must be aligned with what defines importance: with reward as
importance, reward prediction error raises retention from 0.0\,\% to 52.6\,\%, perceptual
prediction error does not."
**l. 1273:** "perceptual prediction error" en lugar de "perceptual novelty".

### B6. "Identifiable and correctable" (l. 1154–1157; contribuciones l. 195–196 "we identify and correct")

**Dato:** la precondición se corrige solo cuando la importancia es inmediata.
- Para `previa_k`, ninguna señal causal le gana al nulo de forma consistente.
- Para `evento_clave`, **ninguna señal le gana, ni siquiera el retorno no causal**: 0/8, y
  0–1/8 para el resto (`resumen.evento_clave.*.gana_vs_azar`). El paper no reporta este caso.

**Reemplazo (l. 1154–1157):** "...transfers when surprise is anchored in the variable that defines
importance (reward or visitation counts); for delayed or key-event importance no causal signal we
tried does." En las contribuciones: "this one we identify, and correct when importance is
immediate."

### B7. Precondición de CIFAR-100 enunciada de tres maneras (abstract l. 60–64; §V l. 1002–1011 y 1062–1065; contribuciones y conclusión)

- **Abstract:** "Merging needs repeated encounters to look alike ... (0.401 against 0.85)". Es un
  marco de calibración del umbral.
- **§V:** "Consolidation is possible only when it clears the merge threshold" (l. 1004–1005), y
  "merge reachability ... as the robust diagnostic" (l. 1063). A 0.30 la fusión es alcanzable en
  el 86 % de las escrituras (`t030.tasa_fusion_r_min`).
- **Contribuciones (l. 190–193), §V (l. 1036–1038) y conclusión:** la precondición es la
  **separabilidad**.

La versión que respalda exp07 es la de separabilidad: intra p50 0.38 frente a inter p95 0.42.

**Reemplazo:**
- **Abstract:** "Merging needs repeated encounters to be separable from distinct ones; on
  natural-image embeddings they are not (intra-class median 0.38, inter-class 95th percentile 0.42)."
- **l. 1004:** "...only when a threshold separates repeated encounters from distinct ones".
- **l. 1063:** "merge separability" en lugar de "merge reachability".

### B8. El abstract presenta r nominal como el determinante (l. 44–45)

> "First, which mechanism matters depends on $r = K_{proto}/C$"

**Dato:** exp10 refuta r nominal como variable que gobierna: el cruce va de 0.16 a 1.32, sd log
0.46. El cuerpo, las contribuciones y la conclusión enuncian la hipótesis sobre r_eff. El abstract
la corrige recién dos frases más adelante.

**Reemplazo:** "First, which mechanism matters depends on how many traces recurring experience
occupies after consolidation relative to capacity, $r_{eff}$; on clean streams this is $r =
K_{proto}/C$."

### B9. "Works in both [regimes]" sin la salvedad de la señal (Discusión l. 1171–1173)

> "the frontier architecture works in both, never depending on merging ... and always exploiting
> the importance signal."

**Dato:** con r = 0.25 y una señal débil, la mejor configuración sin saliencia (que fusiona) le
gana a la frontera (exp08 `settings.main`):
- `auc700`: −0.07 [−0.12, −0.03];
- `auc500`: −0.16 [−0.21, −0.12];
- `delay1_p050`: −0.17 [−0.28, −0.08].

§III-F (l. 861–862) lo dice; la Discusión no.

**Reemplazo:** "...the frontier works in both given a surprise AUC of about 0.95 or better; below
$r=1$ a weaker signal favours merging."

### B10. "Inadequate for the LOLA regime" (l. 1208–1211)

**Dato:** lo medido es retención de raros en flujos sintéticos con señal separable, más MiniGrid.
El régimen LOLA, sin recompensa y abierto, no se probó: §VIII lo reconoce como la pregunta
abierta. En exp13 el FIFO integra **más** retorno que la frontera: AUC −0.075 [−0.098, −0.053].

**Reemplazo:** "...as a store of what to remember, the buffer as implemented is the floor at
rare-event retention under capacity pressure, with a quantified gap even for its most favourable
proxy..."

### B11. "Erases its advantage on returning tasks" (l. 1261–1263)

> "Adding decay ... lifts the frontier's area to the FIFO level but erases its advantage on
> returning tasks, so no tested configuration beats FIFO on both"

**Dato** (`exp13:resumen.C300.diferencias_pareadas`):
- Con decay 0.98 a C = 300, la ventaja de reaparición contra la frontera sin decay no cae de forma
  detectable: `frontera_decay0.98_menos_frontera.reaparicion` = −0.013 [−0.048, 0.022].
- Contra el FIFO, las dos medias son positivas: AUC +0.007 [−0.023, 0.035] y reaparición +0.024
  [−0.011, 0.060]. Ninguna de las dos es significativa.
- La pérdida sí es significativa a C = 1000: −0.11 [−0.17, −0.06].

El texto concede más de lo que el dato obliga.

**Reemplazo:** "Adding decay lifts the frontier's area to the FIFO level, and its return-task
advantage shrinks to within noise of FIFO (lost at $C=1000$): no tested configuration is
significantly better than FIFO on both."

### B12. "A lead it loses to ENN once they are not [separable]" (conclusión l. 1301–1302)

**Dato:** en exp08, SDM queda por debajo de ENN solo en el flujo r = 0.25, donde ENN retiene 0.75
gracias a la fusión. Y ahí solo desde AUC ≈ 0.95: con AUC 0.99, SDM 0.81 contra ENN 0.75.
- **En el flujo r = 2:** ENN cae a 0.04–0.10, y SDM ≥ ENN en todas las condiciones con AUC ≥ 0.64.
- **Con un 25 % de distractores novedosos y r = 0.25:** ENN cae a 0.16–0.17 y SDM retiene 0.65
  (AUC 0.96).

De paso, la l. 952 ("ENN ... holds 0.75 regardless of the signal") es falsa con distractores.

**Reemplazo (conclusión):** "(a lead it loses to ENN at $r=0.25$ once surprise AUC falls to 0.95)".
**l. 952:** "...holds 0.75 under every prediction-error degradation".

### B13. "Rises from the floor to more than half" (conclusión l. 1312–1313)

**Dato:** 52.6 % sale de 40 rollouts, con IC [37.9, 69.0]. Con 200 rollouts, que es la estimación
más precisa del propio paper:
- `exp11:exp06_ampliado.aleatoria_recompensa.resultados.frontera.tasa` = **47.2 %** [39.0, 56.3];
- la media por rollout es 53.8 % [44.6, 62.7].

Ningún IC excluye el 50 %, así que "más de la mitad" no está establecido.

**Reemplazo:** "...retention rises from the floor to about half (47.2\,\% over 200 rollouts)..."
En el abstract conviene usar también el 47.2 % de 200 rollouts, en lugar de 52.6 %.

### B14. Conclusión: "the buffer's actual sequential read fails reconstruction too" (l. 1306–1307)

**Dato:** depende del criterio de coincidencia supuesto.
- Con 0.85, el gate da 0.368.
- Con 0.5, el gate da 0.826 y pasa (`exp09:emdb_threshold_sensitivity.tau_050.gate_mean.mean`).

Lo que no depende del criterio es el puntaje: 0.010 con 0.85 y 0.067 con 0.5. §VII-C lo dice
bien; la conclusión lo da por incondicional.

**Reemplazo:** "...while the buffer's sequential read, under our assumed match rule, scores lower
still (0.010--0.067)."

---

## C — menores

| # | Línea | Cita | Dato | Reemplazo |
|---|---|---|---|---|
| C1 | 46–48 | "Below $r=1$ the write axis explains 73.0 %" | 73.0 % es el régimen `compression` = **r ≤ 0.5** (`exp02:regimenes.compression.rango`); entre 0.5 y 1 da 52.4 % | "At $r\le0.5$ the write axis explains…" |
| C2 | 52–57 | "retains 0.983 ... against 0.050" | Es con la señal separable por construcción. Con AUC 0.46 la frontera retiene 0.60 contra 0.03 del FIFO (exp08), así que la dirección se sostiene y la magnitud no | agregar "on a separable surprise signal" |
| C3 | 183 | "deployed incumbent" | Es el proxy FIFO-NN, no el buffer desplegado (R1.6 dice que se renombró en todas partes) | "the FIFO-NN proxy" |
| C4 | 648–650 | "...or \texttt{topk3} reads over compressed memory" | Las 107 configuraciones por debajo del incumbente son **todas** `append` (no hay memoria comprimida): 72 min_utility, 24 fifo+topk3, 11 min_strength+reinforce | "...with minimum-utility eviction or \texttt{topk3} reads" |
| C5 | 788–789 | "is lost once half the surprise is credited one step late" | Se invierte: `delay1_p050` da −0.17 [−0.28, −0.08]; con el 25 % de retraso se sostiene (+0.04 [0.01, 0.07]) | "is reversed once half…" |
| C6 | 894–898 | FIFO con 921 trazas "outscores it" | Con 921 ranuras el flujo de T1 (320 escrituras) entra entero y no hay presión de capacidad. A 16 KiB (C = 102) la lista de SDM y ENN dan 0.924 contra 0.685 del FIFO | agregar "(with no capacity pressure left)" |
| C7 | 909–911 | "only a 0.5 fraction, or an admission threshold above 0.5, drops it below the gate" | Con una fracción de 0.5 caen la retención (0.395) y el gate. Con un umbral de admisión > 0.5 cae solo el gate (0.42): la retención sigue en 1.000 | "...only a 0.5 fraction lowers rare retention (0.395); it, or an admission threshold above 0.5, fails the gate" |
| C8 | 1018 | "crosses normally ($r\approx0.77$)" | El 0.77 de exp05 es sobre $\hat K/C$ (celdas r = 0.6 y 1.0), no sobre r nominal. Además, el `r_umbral_ci` del JSON, [0.45, 0.6], ni siquiera contiene el 0.77 (posible bug del bootstrap, no se publica) | "($\hat r\approx0.77$)" |
| C9 | 1025–1026 | "from 0.40 up eviction dominates at every $r$" | Con 0.40, 3 de 5 semillas tienen cruce (`t040.n_semillas_con_cruce` = 3); solo en la media domina el desalojo | "from 0.40 up eviction dominates on average" |
| C10 | 1148–1152 | "no causal signal retains the steps that precede a reward" | La novedad por conteo, que es causal, le gana al nulo en **6/15** celdas de `previa_k`, con una ganancia media de solo +1.8 puntos. El texto cita solo reward-PE (0/15) y TD (2/15) | "...TD error in 2 and count novelty in 6, with gains under 2 points..." |
| C11 | 1251 | "An e-MDB-like proxy" | exp13 es un controlador episódico (MFEC). e-MDB entrena modelos de mundo y de utilidad, no actúa por vecinos | "An episodic-control proxy" |

Otras observaciones menores, sin propuesta porque no cambian el significado:
- **l. 137–139:** "preconditions no synthetic design reveals". La familia sintética `ruido_085`
  de exp10 ya reproduce un dominio sin régimen de compresión.
- **l. 1057:** "$0.04$--$0.07$" sobre ruido isotrópico es un número escrito a mano, sin
  `\result{}` (invariante 7).

---

## Coherencia con `docs/revision/respuesta-revisores.md`

1. **R1.5** dice "§IV" para estos tres puntos, pero el §IV solo da el tiempo de lectura y el
   resultado a 144 KiB:
   - la escritura, 57 contra 31 µs;
   - la complejidad O(M·d + C·d);
   - el escalado de hard locations al presupuesto ("de 16 KiB a 512 KiB queda por debajo del
     FIFO; a 4 KiB lo supera sin pasar el gate").

   El dato existe (`igual_presupuesto.sustrato_escalado`) y es correcto, pero no está en el paper.
2. **R1.5** dice "La retención de raros de la SDM la sostiene su lista de trazas con compuerta, no
   la superposición distribuida". Debe acompañar la corrección de A2: (a+) retiene 1.000.
3. **R1.2** lista "40 rollouts en MiniGrid" entre las semillas del §III-B. El §III-B (l. 575–579)
   no menciona MiniGrid.
4. **R1.6** dice que se renombró a "FIFO-NN proxy" en todo el texto. Las contribuciones (l. 183)
   siguen diciendo "deployed incumbent".
5. **R2.2** dice que la precondición se corrigió a separabilidad en "contribuciones, §V, §VII,
   conclusión y limitaciones". En el §V quedan dos formulaciones de alcanzabilidad o calibración
   (l. 1004–1005 y l. 1063), y en el abstract sigue "0.401 against 0.85" (ver B7).
   **R1.4** dice que el §III-D agrega "que caiga en el mismo r para tres capacidades" como lo
   nuevo, y eso queda tensionado por exp10 (ver B2).

---

## Qué sigue sostenido frente a e-MDB y qué no

**Sostenido, con este alcance.** En la batería sintética (T1–T3, C = 20), la frontera supera al
proxy FIFO-NN del buffer de e-MDB:
- 0.818 contra 0.106 en el puntaje agregado, y 0.983 contra 0.050 en la retención de raros;
- con rango [1, 1] en 13 de 14 familias held-out.

La ventaja sobre el proxy en retención de raros se sostiene en **todas** las degradaciones de la
sorpresa a r = 0.25: incluso con AUC 0.46 queda 0.60 contra 0.03. A r = 2 llega hasta AUC ≈ 0.74.
El buffer con su lectura secuencial real (reimplementada) queda todavía más abajo, así que el
rango del proxy es una cota superior.

Ningún baseline externo supera a la frontera. La ventaja viene de leer la sorpresa al desalojar:
el desalojo greedy priorizado por sorpresa queda a 0.035, toda en T3. No es un mecanismo
exclusivamente bioinspirado.

En MiniGrid, con una señal alineada con la importancia, la frontera retiene alrededor de la mitad
de los eventos importantes (47.2 % [39.0, 56.3] en 200 rollouts), contra 0.7 % del FIFO y 0.7 % de
la misma frontera con una señal nula.

**No sostenido:**
- que retener mejor mejore el comportamiento: el FIFO integra más retorno en exp13;
- que la frontera le gane a la mejor configuración que fusiona cuando la señal es débil, por
  debajo de r = 1;
- que algo de esto valga en el régimen LOLA sin recompensa, o con importancia demorada o de evento
  clave, donde ninguna señal causal funciona;
- que haya ventaja a igual presupuesto de bytes sin presión de capacidad;
- que los dos ejes sean inertes por separado fuera del contexto del incumbente;
- que la precondición de fusión sea un chequeo online, porque se midió con etiquetas;
- que el buffer sea "inadecuado para LOLA" más allá de la retención bajo presión.
