# Respuesta a los revisores — BIP2026, EMBER (borrador)

Estado: borrador de la rama `revision/bip2026-reviews`. Las secciones citadas
son las del `paper/main.tex` recortado a 8 páginas (numeración romana del PDF):
I Introducción, II Background and related work, III Method (A espacio de
diseño, B protocolo), IV The e-MDB proxy and the frontier (A dónde cae el
proxy, B los dos ejes, C hasta dónde aguanta la ventaja), V The regime
transition, VI Memory substrates, VII Real data (A CIFAR-100, B MiniGrid),
VIII Discussion and limitations, IX Conclusion. Todos los experimentos de la
revisión están corridos e integrados; no queda ningún `PENDIENTE`.

Experimentos de la revisión:

| exp | Qué responde |
|---|---|
| exp07 | Sensibilidad al umbral de fusión (0.85), sintético y CIFAR-100 |
| exp08 | Saliencia imperfecta: rangos solapados, ruido, retraso, correlación débil o engañosa; precisión y recall |
| exp09 | Baselines externos (reservoir, LRU/LFU, replay priorizado, cachés por utilidad) y el buffer real de e-MDB con recuperación secuencial |
| exp10 | Robustez de la hipótesis de régimen y familias de streams held-out |
| exp11 | MiniGrid ampliado, con el rollout como unidad estadística |
| exp12 | Costo de los sustratos (bytes, complejidad, tiempo) y sensibilidad de SDM |
| exp13 | Efecto de la memoria en el aprendizaje o el desempeño posterior |

---

## Revisor 1

### R1.1 — Referencias [1] y [5] no localizables; [6] no sostiene la afirmación; [8] con título inexacto

Tenía razón, y le agradecemos haberlo detectado. Fue un error nuestro: [1] y [5]
no correspondían a publicaciones reales. **No era una autocita anonimizada**:
los autores no pertenecen al grupo que desarrolla e-MDB. Rehicimos la
bibliografía completa y verificamos cada entrada contra una fuente primaria
(DOI vía Crossref, PubMed, página del editor o proceedings oficiales). También
revisamos que cada cita sostenga lo que se le atribuye.

- **[1] `emdb2021` → eliminada.** e-MDB se cita ahora con sus fuentes reales,
  del Integrated Group for Engineering Research (GII) de la Universidade da
  Coruña:
  - Romero, tesis doctoral, UDC 2022, que presenta e-MDB (hdl 2183/31717).
  - Romero, Meden, Bellas y Duro, *Integrated Computer-Aided Engineering* 30(3),
    2023 (DOI 10.3233/ICA-230707): e-MDB, P-nodes y LOLA.
  - Romero, Bellas y Duro, *Sensors* 23(3):1611, 2023: definición de LOLA.
  - Becerra, Romero, Bellas y Duro, *Neurocomputing* 452, 2021: motor
    motivacional acoplado a la memoria de largo plazo.
  - Duro, Becerra, Monroy y Bellas, *IJNS* 29(6), 2019: la LTM de tipo network
    memory y la activación por percepción.
  - Bellas et al., *IEEE TAMD* 2(4), 2010: MDB, el predecesor de e-MDB.

  La descripción del `EpisodicBuffer` (un `deque` acotado, leído por índice o
  por lotes de entrenamiento) no aparece en ninguna publicación. Sale del
  repositorio público `GII/emdb_cognitive_nodes_gii` (archivo
  `episodic_buffer.py`, commit d15f96a), que citamos como software. El texto de
  §II lo dice explícitamente.
- **[5] `cole2015` → eliminada.** La compuerta por error de predicción ahora se
  sostiene con literatura primaria:
  - §I: Lisman y Grace 2005 (*Neuron*) y Shohamy y Adcock 2010 (*TiCS*), sobre
    la modulación dopaminérgica de la codificación hipocampal.
  - §I: Rouhani, Norman y Niv 2018 (*JEP:LMC*), donde errores de predicción de
    recompensa más grandes mejoran la memoria episódica.
  - §VII-B: Schultz, Dayan y Montague 1997 (*Science*) sostiene la afirmación
    "la cuenta dopaminérgica es sobre error de predicción de recompensa", junto
    con Rouhani et al.
- **[6] `murdock1962` → eliminada.** Trata el efecto de posición serial y no se
  usaba en ningún otro lugar. Reescribimos la frase de §I y ahora la sostienen:
  - Anderson y Schooler 1991 (*Psychological Science*): la disponibilidad de la
    memoria sigue la probabilidad de necesitar el ítem, es decir, un criterio de
    utilidad.
  - Richards y Frankland 2017 (*Neuron*): el olvido es un proceso regulado al
    servicio de la decisión.
- **[8] `minigrid`:** la entrada ahora es exacta. Chevalier-Boisvert et al.,
  "Minigrid & Miniworld: Modular & Customizable Reinforcement Learning
  Environments for Goal-Oriented Tasks", NeurIPS 36 (Datasets and Benchmarks
  Track), 2023, pp. 73383–73394, con la lista completa de autores.
- Resto de las entradas: autores completos, título exacto, venue, volumen,
  páginas y DOI donde existe. Hubo dos correcciones de sustancia:
  - Semon 1904 lleva ahora el título completo.
  - Chaudhry et al. 2019 figura como arXiv/workshop, porque nunca salió en
    proceedings.

### R1.2 — Protocolo experimental no autocontenido

Agregamos la nueva **§III-B "Experimental protocol"**, antes de cualquier
resultado. Cada detalle está sacado del código que produjo los resultados:

- el generador de streams: centros, ruido σ=0.05, recurrencia, posición e
  independencia de los raros, y las opciones desactivadas;
- los rangos de error de predicción: comunes N(0.10,0.05) recortada a
  [0,0.35], raros N(0.90,0.05) recortada a [0.65,1];
- la novedad y las fórmulas de la compuerta;
- la etiqueta de importancia, que solo se usa para puntuar;
- R1–R4 y el umbral 0.50;
- T1–T3, con la generación de consultas y el criterio de acierto;
- el puntaje agregado, que es la media sin pesos de T1–T3;
- las semillas por experimento: 3 en la búsqueda, 5 por celda en la grilla,
  4 en el benchmark, 3 en CIFAR-100 y 40 rollouts en MiniGrid;
- el cálculo de η², de las interacciones y del bootstrap (ver R1.8).

### R1.3 — Resultados que salen de la construcción del benchmark

Estamos de acuerdo y lo decimos explícitamente en el texto:

- En **§IV-B**, el 14.41× muestra que la señal de saliencia no tiene efecto
  hasta que una regla de desalojo la lee. Con rangos de sorpresa no solapados,
  retener los raros sale casi directamente de la construcción; el factor no
  demuestra que seleccionar desde una señal realista esté resuelto.
- En **§VI**, la retención perfecta confirma el cableado. Aclaramos además que
  la grilla de umbral saca el techo de "20 raros en 20 ranuras", pero mantiene
  la separabilidad.

Además corrimos la versión con señales imperfectas (exp08, 5 semillas): la
sorpresa se degrada por solapamiento de rangos, ruido aditivo, retraso en la
asignación de crédito e inversión parcial (señal engañosa), y en un eje aparte
se agregan distractores novedosos, porque la novedad también separa
perfectamente lo raro por construcción. El resultado matiza la afirmación y la
corregimos donde aparece (abstract, contribuciones, §IV-B, §IV-C, §VI, §VIII y conclusión):

- **El 14.41× depende de la señal.** Con solo error de predicción como
  compuerta, el cociente condicional cae de 16.27× (señal limpia) a 0.80×
  (IC 95 % 0.17–2.82) con rangos totalmente solapados. La compuerta combinada
  conserva más solo por la novedad; con un 25 % de distractores novedosos *y*
  error de predicción al azar cae a 3.10× (con distractores y señal limpia
  queda en 11.80×).
- **La ventaja de la frontera tiene un umbral de calidad de señal.** Contra la
  mejor configuración sin saliencia (que a r = 0.25 ya retiene 0.76
  fusionando), la diferencia pareada excluye el 0 hasta un AUC empírico de la
  sorpresa de 0.95 (solapamiento), 0.96 (ruido) y 0.91 (10 % de señal
  invertida); se invierte cuando la mitad de la sorpresa llega un paso tarde, y
  en AUC ≈ 0.5 se vuelve desventaja (0.60 contra 0.76). En el régimen r = 2,
  donde fusionar no libera espacio, la ventaja aguanta hasta AUC 0.84. Contra
  el FIFO gana en todas las condiciones de r = 0.25.
- **La retención perfecta de SDM en §VI es de señal limpia.** SDM baja a 0.81
  con AUC 0.99 y a 0.60 con AUC 0.95, por debajo de ENN, que se queda en 0.75
  bajo toda degradación del error de predicción porque su retención viene de
  la fusión. Eso vale solo en el flujo r = 0.25: en r = 2 la ENN cae a
  0.04–0.10 y la SDM la iguala o supera, y con un 25 % de distractores
  novedosos la ENN cae a 0.16–0.17.
- **Precisión y recall.** En el flujo de §VI (20 raros en 20 ranuras, memoria
  llena) la precisión de retención es igual al recall por aritmética; en el
  flujo r = 2 (10 raros en 20 ranuras) la precisión está acotada en 0.5 y
  sigue al recall. SDM supera al FIFO ahí hasta AUC 0.74.

El retraso es la degradación más dañina: con la sorpresa corrida un paso, la
frontera retiene 0.21, peor de lo que su AUC sugiere. Todos los números están
con `\result{exp08_imperfect_salience:...}` en el LaTeX; la figura por eje está
en `paper/figures/fig_exp08_imperfect_salience.pdf` (no entra al cuerpo por
espacio). Nota de los números movidos: `docs/notas/2026-10-06-exp08-saliencia-imperfecta.md`.

### R1.4 — "Threshold law" prematura

Tenía razón. Reformulamos todo el paper, incluidos el abstract, las
contribuciones, §V, §VIII y la conclusión, como una **regime hypothesis**
sostenida bajo las condiciones evaluadas. §V agrega dos cosas:

- que la transición es casi una reformulación de la definición de capacidad;
  lo nuevo es que el cambio sea abrupto, no que exista. En la grilla original
  los tres cruces caen en la misma celda (C = 10–40), pero esa igualdad está
  cuantizada por la resolución de la grilla: en exp10 el cruce nominal se
  desplaza de 0.71 (C = 5) a 0.98 (C = 160), y la recurrencia también lo mueve;
- la prueba de robustez que pedía (exp10), cuyo resultado cambia el
  enunciado.

Repetimos el barrido en 23 familias de flujos que mueven una perilla por vez
(frecuencias de Zipf, deriva, ruido intra-prototipo, similitud entre centros,
prevalencia de raros, recurrencia y C de 5 a 160; §III-B). **El revisor tenía
razón en sospechar de r nominal: no es la variable que gobierna.** Su cruce va
de 0.16 a 1.32 entre familias (desvío de su logaritmo 0.46): la deriva y el
ruido lo bajan muy por debajo de 1, y con el ruido más alto el desalojo domina
en todo r. En cambio, con r_eff = K_efectivo/C —las trazas que deja la
consolidación— los cruces se agrupan en [0.56, 1.28] (desvío 0.17), más
colapsados que con r nominal en el 99.8 % de las réplicas bootstrap y mejor que
otras cuatro medidas de presión declaradas antes de correr. La dispersión
residual viene de la prevalencia de raros y de C = 5. En el generador original
r_eff = r, así que la Figura 3 no cambia. El paper enuncia ahora la hipótesis
sobre r_eff (abstract, contribuciones, §V, §VIII, conclusión) y lo dice
explícitamente en §V.

El umbral de fusión ya no figura entre las condiciones no probadas (exp07,
§V). Barrimos el umbral de 0.30 a 0.97 (C = 20, cinco semillas, los 576
genotipos). Entre 0.40 y 0.85 el cruce queda exactamente en r ≈ 0.87 para todos
los umbrales. A 0.90 la fusión pierde parte de las visitas y el cruce nominal
baja a 0.61. A 0.30 la fusión absorbe el 38 % de los raros. Desde 0.95 la
fusión fragmenta los prototipos y el desalojo domina en todo r. Medido sobre
K_effective, el cruce queda entre 0.86 y 1.02 para todo umbral entre 0.30 y
0.90 (a 0.95 la fusión se dispara en el 36 % de las escrituras, pero no hay
cruce), así que la hipótesis se enuncia mejor sobre la razón
efectiva. Un barrido 1D de la ganancia de la compuerta de fuerza (0.5–8) no
cambia el patrón de régimen.

### R1.5 — Sustratos con presupuestos de almacenamiento y cómputo no equivalentes

En §VI reconocemos que la capacidad cuenta trazas y no bytes. SDM además
mantiene 512 hard locations con contadores, y el benchmark no separa el aporte
de los contadores del de la lista de trazas.

Lo medimos (exp12, 10 semillas, IC bootstrap) y el resultado matiza el paper:

- **Bytes.** A C = 20 y d = 32, la SDM ocupa 131.1 KiB contra 3.1 KiB del FIFO
  y del ENN (×41.96); Spiking-SDM 133.0 y Spiking 105.1. La contabilidad real
  (`nbytes`) coincide con la fórmula analítica en todas las configuraciones.
  Agregamos la columna **KiB** a la Tabla III (generada desde `results/`).
- **Complejidad y tiempo.** Medido a C = 20, la lectura cuesta 28 µs en SDM
  contra 12 µs en FIFO; el paper (§VI) da este número. Por espacio, el resto
  queda en esta respuesta y en `results/exp12_substrate_cost/`: lectura y
  escritura son O(C·d) en FIFO/ENN y O(M·d + C·d) en SDM, y la escritura cuesta
  57 contra 31 µs.
- **Igual presupuesto.** Con los 144 KiB que ocupa la SDM, un FIFO guarda 921
  trazas y la supera en la batería T1–T3 (0.924 contra 0.745); el paper (§VI)
  aclara que a ese presupuesto el flujo de T1 entra entero y ya no hay presión
  de capacidad. Escalar las hard locations de la SDM al presupuesto tampoco lo
  revierte: de 16 KiB a 512 KiB queda por debajo del FIFO, y a 4 KiB lo supera
  solo sin pasar el gate. Este barrido está en `results/exp12_substrate_cost/`
  (`igual_presupuesto.sustrato_escalado`) y, por espacio, no en el paper.
- **Contadores frente a lista de trazas.** La ablación tiene cuatro variantes.
  La lista sola, leída por vecino más cercano, retiene todos los raros y da
  0.788 en la batería contra 0.741 de la SDM completa (diferencia pareada
  −0.047, IC [−0.053, −0.041]). Los contadores solos retienen 0.005 de los raros
  si no pueden restar una traza desalojada, pero **1.000 si se les da borrado
  exacto** (batería 0.735, empatada con la SDM completa: +0.006, IC
  [−0.001, 0.013]). La SDM real resta exactamente, así que lo que sostiene la
  retención no es "lista frente a contadores": es el desalojo con compuerta más
  un borrado exacto, que la lista sola provee con 1/42 de los bytes. Corregimos
  en consecuencia la frase de §VI que atribuía la recuperación a la
  superposición y la afirmación de la conclusión de que la SDM era "el único
  sustrato" que combina reconstrucción y retención.
- **Sensibilidad.** La media de la batería varía 0.091 entre 64 y 2048 hard
  locations; la retención de raros es perfecta para fracciones de activación
  entre 0.005 y 0.2 y solo cae a fracción 0.5 (que además no pasa el gate). Un
  umbral de admisión externo no cambia nada hasta 0.5 y por encima bloquea la
  reconstrucción. El umbral de fusión lo barre exp07. §VI.

### R1.6 — El "e-MDB FIFO" es un proxy optimista

Renombramos el baseline a **FIFO-NN proxy of the e-MDB buffer** en §I, §II,
§IV-A, la Tabla II, §VI, §IV-C, §VIII y la conclusión. Decimos explícitamente que el
buffer real recupera por posición o por lotes, sin búsqueda por contenido, y
por qué elegimos NN: para aislar el desalojo y la ponderación.

Además evaluamos el buffer real (exp09). Reimplementamos su lectura tal como
está en el código público de GII (`episodic_buffer.py`): un barrido en orden de
inserción que devuelve el primer episodio con coseno ≥ 0.85. Esa traducción de
una consulta es un supuesto, porque el buffer no responde consultas. Con el
mismo desalojo retiene exactamente las mismas trazas que el proxy, pero
**reprueba el gate de reconstrucción**: 0.368 contra 0.903 del proxy. Su puntaje
agregado es 0.010 y queda último entre 577. Con un umbral de coincidencia de 0.5
el gate se recupera (0.826), pero el puntaje no (0.067). El ranking del proxy es,
entonces, una cota superior para el buffer desplegado. Esto quedó en §IV-C,
§VIII y la conclusión. Que el buffer real repruebe también la reconstrucción
depende del criterio de coincidencia supuesto; lo que no depende de él es el
puntaje (0.010–0.067), y así lo dice la conclusión.

### R1.7 — MiniGrid limitado; importancia definida por la tarea; Wilson sobre eventos

Lo resolvimos con un experimento nuevo (exp11), reportado en §VII-B y, para
la importancia demorada, en §VIII.

- **Unidad estadística.** Re-analizamos exp06 con el rollout como unidad
  (bootstrap por clúster sobre rollouts). Con los mismos 40 rollouts, la
  frontera con error de predicción de recompensa retiene 52.6 % con IC 95 %
  [37.9 %, 69.0 %] (Wilson daba [37.3 %, 67.5 %]) y el FIFO 2.6 % [0.0 %,
  8.6 %]. El efecto de diseño es 0.92: en la práctica, los eventos de un mismo
  rollout no estaban correlacionados y la conclusión no cambia. §VII-B ahora
  reporta los intervalos por rollout en lugar de los de Wilson.
- **Más rollouts y semillas.** Con 200 rollouts en 5 bloques de semillas
  independientes, la frontera retiene 47.2 % [39.0 %, 56.3 %] contra 0.7 % del
  FIFO (diferencia 46.5 puntos, IC [38.3, 55.7]). Es la estimación más
  precisa, y la que usan ahora el abstract y la conclusión: "alrededor de la
  mitad", no "más de la mitad", porque ningún IC excluye el 50 %.
- **Control con señal nula.** Ganarle al FIFO no basta, porque el FIFO desaloja
  por edad. La misma frontera alimentada con una señal uniforme al azar retiene
  0.7 % en esa condición: la ganancia sale de la señal, no del desalojo por
  fuerza.
- **Importancia no definida por la tarea.** Cruzamos 4 entornos (MemoryS13,
  DoorKey-6x6, FourRooms, KeyCorridorS3R2), 4 políticas (uniforme, sesgada,
  planificador BFS con ruido, Q-learning tabular), 5 definiciones de
  importancia (recompensa inmediata, los k pasos previos a una recompensa, un
  evento clave del entorno, estado novedoso, transición sorpresiva) y 5 señales
  candidatas (más el control nulo): 350 celdas. La ganancia sobre el control
  nulo sigue a la separación AUC de la señal (Spearman ρ = 0.67): +37.3 puntos
  en las celdas con AUC ≥ 0.97 y +1.8 cerca de AUC 0.5.
- **Objetivos intrínsecos.** La novedad por conteo le gana al control nulo en
  16 de 16 celdas cuando lo importante es el estado novedoso.
- **Recompensa demorada.** Ninguna señal causal retiene los pasos que preceden
  a una recompensa: el error de predicción de recompensa le gana al control
  nulo en 0 de 15 celdas, el error TD en 2 de 15 y la novedad por conteo en 6
  de 15, con una ganancia media de solo 1.8 puntos; solo lo logra de forma
  consistente el retorno descontado, que no es causal (15 de 15). Con
  importancia de evento clave ninguna señal le gana al control nulo, ni
  siquiera el retorno (0 de 8). §VIII lo declara como el problema
  abierto, ahora con evidencia.

### R1.8 — Estadística insuficientemente explicada

En §III-B definimos:

- **η²** = SS_between/SS_total sobre las 576 configuraciones. Es una
  descomposición marginal de una vía. Con un factorial completo y balanceado,
  los efectos principales son ortogonales.
- **Interacciones:** son residuos de medias de celda de dos vías, reportados
  aparte y nunca sumados al η² del eje.
- **Sin término de error:** el resultado es determinista dada la semilla, así
  que no hay término de error ni tests de significancia.
- **Bootstrap:** 2000 réplicas percentil, con la semilla como unidad. El
  intervalo del cruce es un bootstrap sobre tres capacidades y es solo
  descriptivo.

Un párrafo nuevo de §III-B, "Structural dependence", explica que la fuerza y
el refuerzo son inertes bajo tres de cuatro desalojos. Por eso los efectos
principales se leen como atribución sobre todo el espacio, y los efectos
condicionales se reportan donde la dependencia se conoce. Las fracciones de
interacción cuantifican la dependencia: strength×evict 4.3 % y write×evict
7.8 %, contra 1.4 % del efecto principal de strength.

### R1.9 — Retención ≠ efecto sobre el aprendizaje; no integrado en e-MDB; literatura no biológica

- Ampliamos §VIII: la integración con e-MDB no está hecha, y la Figura 1 es un
  diseño, no una implementación.
- Medimos el efecto sobre el desempeño (exp13, §VIII, con una frase en el abstract y
  en la Conclusión). Un agente de control episódico cuya única experiencia es
  la memoria enfrenta cuatro variantes de MiniGrid que se alternan y
  reaparecen (10 semillas, IC95 bootstrap pareado). Con C=300 la frontera
  rinde más que el FIFO al reaparecer una tarea (+0.038, IC [+0.005, +0.071]),
  pero integra menos retorno sobre todo el flujo (AUC −0.075, IC
  [−0.098, −0.053], ninguna semilla a favor). En C=1000 pasa lo mismo
  (reaparición +0.109, AUC −0.064). Sin decaimiento, retener cuesta
  plasticidad. Con decaimiento 0.995 o 0.98 (del mismo espacio), la frontera
  sube su AUC al nivel del FIFO y su ventaja al reaparecer queda dentro del
  ruido (a C=300, con decaimiento 0.98: +0.024, IC [−0.011, +0.060]); se pierde
  de forma significativa a C=1000. Ningún genotipo probado es
  significativamente mejor que el FIFO en las dos métricas. La frontera sí
  supera al reservoir en AUC en ambas capacidades. La memoria sin límite
  (AUC 0.566 contra 0.144) muestra cuánto falta. Conclusión honesta:
  retener mejor no es todavía actuar mejor. Lo dice el paper, y matizamos en
  §VIII la afirmación sobre el buffer de e-MDB. El efecto sobre los modelos que
  e-MDB entrena desde su buffer sigue sin probarse.
- Agregamos en §II un párrafo de gestión de memoria no biológica:
  - LRU-K (O'Neil et al. 1993) y ARC (Megiddo y Modha 2003);
  - reservoir sampling (Vitter 1985);
  - Chaudhry et al. 2019;
  - iCaRL (Rebuffi et al. 2017);
  - GSS (Aljundi et al. 2019);
  - selective experience replay (Isele y Cosgun 2018);
  - experience replay para aprendizaje continuo (Rolnick et al. 2019).
- Ese párrafo ahora también compara contra esos métodos de forma empírica
  (exp09; ver R2.3). Ninguno supera a la frontera. El más cercano, el desalojo
  priorizado greedy por sorpresa, coincide con un punto del espacio.

### R1.10 — Liberar código y configuraciones durante la revisión

La sección "Code and data availability" dice ahora que el código, las
configuraciones, los resultados y los manifiestos de procedencia se publicarán
en un repositorio público con licencia abierta. No se promete un repositorio
anonimizado durante la revisión.

**Pendiente de los autores:** crear el repositorio público (y limpiarlo) antes
de la versión final, y agregar su URL al paper.

### R1.11 — Mejoras recomendadas (resumen)

| Recomendación | Respuesta |
|---|---|
| Protocolo consolidado | §III-B (hecho) |
| Hipótesis o más generadores | Reformulado como hipótesis (hecho); exp10 en 23 familias: el cruce lo ordena r_eff ([0.56, 1.28]), no r nominal ([0.16, 1.32]); §V (hecho) |
| Señales imperfectas, precisión y recall | §IV-B, §IV-C y §VI (exp08, hecho): la ventaja aguanta hasta AUC ≈ 0.91–0.96 (0.84 a r = 2) y se pierde con retraso; afirmaciones corregidas (ver R1.3) |
| Frontera en familias held-out; reservoir, LRU/LFU, replay priorizado | exp10: rango [1, 1] en 13 de las 14 familias held-out, [15, 20] con 0.1 % de raros; Spearman ≥ 0.77; §V (hecho). §IV-C (hecho, exp09: ningún baseline externo supera a la frontera; el replay priorizado greedy queda a 0.035) |
| Sustratos bajo restricciones equivalentes; sensibilidad de hiperparámetros | §VI y columna KiB de la Tabla III: bytes, tiempos, igual presupuesto, ablación contadores/lista y sensibilidad de SDM (exp12, hecho); umbral de fusión y ganancia de fuerza: hecho (exp07, §V y §VII-A) |
| MiniGrid con rollout como unidad, más entornos; efecto en aprendizaje | §VII-B y §VIII (exp11, hecho): IC por rollout, 200 rollouts, 4 entornos × 4 políticas × 5 importancias × 5 señales, control nulo; efecto en aprendizaje medido en §VIII (exp13): la frontera retiene mejor la tarea que reaparece pero integra menos retorno que el FIFO, y ningún genotipo probado gana en las dos métricas |
| Integración en e-MDB real frente a su buffer secuencial | Figura 1 y §VIII (hecho); buffer real con lectura secuencial en §IV-C (hecho, exp09: reprueba el gate, 0.368); la integración completa queda como trabajo futuro declarado |

---

## Revisor 2

### R2.1 — Alcance de la validación: un dataset, un extractor, un entorno, sin robot físico

- §VIII declara explícitamente estos límites y agrega los que ningún
  experimento de esta revisión resuelve: el robot físico, la integración en el
  e-MDB real y un único extractor visual.
- MiniGrid se amplió a 4 entornos (MemoryS13, DoorKey-6x6, FourRooms,
  KeyCorridorS3R2) y 4 políticas, dos de ellas dirigidas (planificador BFS con
  ruido y Q-learning tabular), con 200 rollouts y el rollout como unidad
  (§VII-B, exp11; detalle en R1.7). La saliencia transfiere cuando la señal está
  alineada con lo importante y falla para importancia demorada; §VIII lo dice.
  Siguen siendo mundos de grilla discretos, sin robot físico.

### R2.2 — Dependencia del umbral de fusión 0.85

Tenía razón en que había que probarlo. Lo barrimos de 0.30 a 0.97 en ambos
dominios (exp07). El resultado de CIFAR-100 no depende de la calibración de
0.85:

- **CIFAR-100 (§VII-A).** La similitud intra-clase (mediana 0.38) se solapa con la
  inter-clase (percentil 95: 0.42). Por eso ningún umbral de la grilla da
  fusiones a la vez frecuentes y puras: a 0.85 se fusiona el 6 % de las
  escrituras, y desde 0.40 el desalojo domina en todo r. A 0.30 el eje de
  escritura recupera la dominancia en r bajo, pero no como compresión útil
  (el efecto medio de fusionar es incluso positivo, +0.072): solo el 48 % de
  las fusiones une la misma clase, el 70 % de los raros se absorbe al llegar, y
  la mejor configuración con fusión retiene 0.30 frente a 1.00 de la frontera
  `append`. El criterio de viabilidad que fijamos antes de correr
  (tasa de fusión ≥ 0.25, pureza ≥ 0.9 y escritura dominante en r bajo) no lo
  cumple ningún umbral.
- **Corrección del paper.** La precondición deja de enunciarse como "similitud
  intra-prototipo por encima del umbral". Ahora es de *separabilidad*: hace
  falta un umbral que superen los encuentros repetidos y no los distintos. Lo
  corregimos en el abstract, las contribuciones, §VII-A, §VIII, la conclusión y las
  limitaciones. El abstract ya no compara la mediana 0.401 con 0.85, sino la
  mediana intra-clase (0.38) con el percentil 95 inter-clase (0.42). La
  separabilidad se midió con las etiquetas de clase; un estimador en línea, sin
  etiquetas, queda sin probar y el paper lo dice. Retiramos la frase de §VIII que sugería que "un umbral por
  dominio podría cambiarlo"; un extractor ajustado sí podría.
- **Sintético (§V).** El cruce es robusto al umbral entre 0.40 y 0.85
  (r ≈ 0.87). Sobre K_effective queda entre 0.86 y 1.02 para todo umbral
  entre 0.30 y 0.90. Ver R1.4.

### R2.3 — Más baselines de gestión de memoria

- §II incluye ahora la literatura no biológica: caché, reservoir y
  selección para replay.
- En la misma §IV-C reportamos la comparación empírica (exp09). Evaluamos siete
  políticas externas al espacio, cada una cambiando solo el desalojo del proxy
  FIFO-NN, con T1–T3 y cinco semillas. La frontera puntúa 0.824. LRU, LFU y una
  caché de utilidad (frecuencia × recencia) quedan al nivel del proxy: 0.106,
  0.096 y 0.106. Reservoir llega a 0.306, la selección por cobertura (Isele y
  Cosgun 2018) a 0.439 y el replay priorizado estocástico a 0.500.
- El desalojo priorizado greedy por sorpresa llega a 0.789, en el rango 4–5 de
  577. Coincide exactamente con una configuración que ya está en el espacio
  (compuerta por error de predicción con `min_strength`). La frontera le gana
  por 0.035 [0.028, 0.043], y toda esa diferencia está en T3.
- Lo decimos así en el paper: la ventaja de la frontera viene de leer la
  sorpresa al desalojar, un principio que el replay priorizado ya usa, y no de
  un mecanismo exclusivo de la memoria bioinspirada.

---

## Revisor 3

### R3.1 — Alcance de la validación y generalización a robots

Lo tratamos igual que en R2.1: §VIII ampliada, y además la hipótesis de régimen
ya no se enuncia como ley (R1.4).

Para MiniGrid, ver R2.1 y R1.7: 4 entornos, 4 políticas y 5 definiciones de
importancia, con el rollout como unidad (§VII-B, exp11).

Sobre la generalización fuera del generador (exp10, §V): la transición de
régimen se sostiene en 23 familias de flujos (frecuencias desiguales, deriva,
ruido, prevalencia de raros, C de 5 a 160) siempre que se enuncie sobre
r_eff = K_efectivo/C, no sobre r nominal (ver R1.4). La frontera elegida en el
flujo estándar, re-evaluada en 14 familias held-out con semillas nuevas,
conserva el intervalo de rango [1, 1] en todas salvo una ([15, 20] de 576 con
0.1 % de raros), y el ranking de las 576 configuraciones correlaciona con el
estándar con Spearman ≥ 0.77 (el mínimo, en selección fuerte, r = 4). Con la
tarea de retención de raros sola el resto del ranking sí se reordena en
selección fuerte (ρ = −0.03 en r = 4), aunque la frontera sigue en el 10 %
superior salvo con 0.1 % de raros, donde los empates estiran su intervalo
pesimista hasta 297; lo dejamos registrado en `docs/notas/`.

### R3.2 — Esquema de la integración EMBER/e-MDB

Agregamos la **Figura 1** en §II, en TikZ. Muestra:

- la memoria de largo plazo de e-MDB, con sus nodos cognitivos (P-nodes,
  C-nodes, objetivos, modelos de mundo y de utilidad, políticas) y el motor
  motivacional (necesidades y drives);
- el camino de cada `Episode` hacia EMBER, que ocupa el lugar del
  `EpisodicBuffer` FIFO;
- el camino write, con la compuerta por la sorpresa que aportan los propios
  modelos de e-MDB, y luego evict por mínima fuerza;
- el camino read por contenido, que devuelve episodios similares para el
  entrenamiento de modelos y la deliberación.

El esquema se basa en las fuentes de R1.1 y en el código público. La leyenda
aclara que es un diseño y no una integración implementada.
