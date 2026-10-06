# Respuesta a los revisores — BIP2026, EMBER (borrador)

Estado: borrador de la rama `rev/text-refs`. Las secciones citadas son las del
`paper/main.tex` revisado (numeración romana del PDF). Los puntos que dependen
de experimentos nuevos llevan `PENDIENTE(expNN)`; en el LaTeX hay un
`% TODO(expNN)` en el lugar exacto donde va el resultado.

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
  §II-D lo dice explícitamente.
- **[5] `cole2015` → eliminada.** La compuerta por error de predicción ahora se
  sostiene con literatura primaria:
  - §I: Lisman y Grace 2005 (*Neuron*) y Shohamy y Adcock 2010 (*TiCS*), sobre
    la modulación dopaminérgica de la codificación hipocampal.
  - §I: Rouhani, Norman y Niv 2018 (*JEP:LMC*), donde errores de predicción de
    recompensa más grandes mejoran la memoria episódica.
  - §VI: Schultz, Dayan y Montague 1997 (*Science*) sostiene la afirmación
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

- En **§III-E**, el 14.41× muestra que la señal de saliencia no tiene efecto
  hasta que una regla de desalojo la lee. Con rangos de sorpresa no solapados,
  retener los raros sale casi directamente de la construcción; el factor no
  demuestra que seleccionar desde una señal realista esté resuelto.
- En **§IV**, la retención perfecta confirma el cableado. Aclaramos además que
  la grilla de umbral saca el techo de "20 raros en 20 ranuras", pero mantiene
  la separabilidad.

Queda PENDIENTE(exp08): la versión con señales imperfectas, que reportará
precisión y recall por calidad de señal.

### R1.4 — "Threshold law" prematura

Tenía razón. Reformulamos todo el paper, incluidos el abstract, las
contribuciones, §III-D, §VII-A y la conclusión, como una **regime hypothesis**
sostenida bajo las condiciones evaluadas. §III-D agrega dos cosas:

- que la transición es casi una reformulación de la definición de capacidad;
  lo nuevo es que el cambio sea abrupto y caiga en el mismo r para tres
  capacidades, no que exista;
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
sobre r_eff (abstract, contribuciones, §III-D, §VII-A, conclusión) y lo dice
explícitamente en §III-D.

Queda PENDIENTE(exp07), el umbral de fusión.

### R1.5 — Sustratos con presupuestos de almacenamiento y cómputo no equivalentes

En §IV reconocemos que la capacidad cuenta trazas y no bytes. SDM además
mantiene 512 hard locations con contadores, y el benchmark no separa el aporte
de los contadores del de la lista de trazas.

Queda PENDIENTE(exp12): bytes totales, complejidad de lectura y escritura,
tiempo, aporte separado de contadores y trazas, y sensibilidad a hard
locations, fracción de activación y umbrales.

### R1.6 — El "e-MDB FIFO" es un proxy optimista

Renombramos el baseline a **FIFO-NN proxy of the e-MDB buffer** en §I, §II-D,
§III-C, la Tabla II, §IV, §VII-C y la conclusión. Decimos explícitamente que el
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
entonces, una cota superior para el buffer desplegado. Esto quedó en §VII-C,
§VIII y la conclusión, que ahora aclara que el buffer real falla también en
reconstrucción, no solo en elegir.

### R1.7 — MiniGrid limitado; importancia definida por la tarea; Wilson sobre eventos

Lo resolvimos con un experimento nuevo (exp11), reportado en §VI (párrafo
"Beyond one environment") y en §VIII.

- **Unidad estadística.** Re-analizamos exp06 con el rollout como unidad
  (bootstrap por clúster sobre rollouts). Con los mismos 40 rollouts, la
  frontera con error de predicción de recompensa retiene 52.6 % con IC 95 %
  [37.9 %, 69.0 %] (Wilson daba [37.3 %, 67.5 %]) y el FIFO 2.6 % [0.0 %,
  8.6 %]. El efecto de diseño es 0.92: en la práctica, los eventos de un mismo
  rollout no estaban correlacionados y la conclusión no cambia. §VI ahora
  reporta los intervalos por rollout en lugar de los de Wilson.
- **Más rollouts y semillas.** Con 200 rollouts en 5 bloques de semillas
  independientes, la frontera retiene 47.2 % [39.0 %, 56.3 %] contra 0.7 % del
  FIFO (diferencia 46.5 puntos, IC [38.3, 55.7]).
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
  nulo en 0 de 15 celdas y el error TD en 2 de 15; solo lo logra el retorno
  descontado, que no es causal (15 de 15). §VIII lo declara como el problema
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

Un párrafo nuevo, "Structural dependence between axes", explica que la fuerza y
el refuerzo son inertes bajo tres de cuatro desalojos. Por eso los efectos
principales se leen como atribución sobre todo el espacio, y los efectos
condicionales se reportan donde la dependencia se conoce. Las fracciones de
interacción cuantifican la dependencia: strength×evict 4.3 % y write×evict
7.8 %, contra 1.4 % del efecto principal de strength.

### R1.9 — Retención ≠ efecto sobre el aprendizaje; no integrado en e-MDB; literatura no biológica

- Ampliamos §VIII: la integración con e-MDB no está hecha, y la Figura 1 es un
  diseño, no una implementación.
- Medimos el efecto sobre el desempeño (exp13, §VIII, con una frase en §VII y
  en la Conclusión). Un agente de control episódico cuya única experiencia es
  la memoria enfrenta cuatro variantes de MiniGrid que se alternan y
  reaparecen (10 semillas, IC95 bootstrap pareado). Con C=300 la frontera
  rinde más que el FIFO al reaparecer una tarea (+0.038, IC [+0.005, +0.071]),
  pero integra menos retorno sobre todo el flujo (AUC −0.075, IC
  [−0.098, −0.053], ninguna semilla a favor). En C=1000 pasa lo mismo
  (reaparición +0.109, AUC −0.064). Sin decaimiento, retener cuesta
  plasticidad. Con decaimiento 0.995 o 0.98 (del mismo espacio), la frontera
  sube su AUC al nivel del FIFO pero pierde la ventaja al reaparecer:
  ningún genotipo probado le gana al FIFO en las dos métricas. La frontera sí
  supera al reservoir en AUC en ambas capacidades. La memoria sin límite
  (AUC 0.566 contra 0.144) muestra cuánto falta. Conclusión honesta:
  retener mejor no es todavía actuar mejor. Lo dice el paper, y matizamos en
  §VII la afirmación sobre el buffer de e-MDB. El efecto sobre los modelos que
  e-MDB entrena desde su buffer sigue sin probarse.
- Agregamos en §II-C un párrafo de gestión de memoria no biológica:
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
| Hipótesis o más generadores | Reformulado como hipótesis (hecho); exp10 en 23 familias: el cruce lo ordena r_eff ([0.56, 1.28]), no r nominal ([0.16, 1.32]); §III-D (hecho) |
| Señales imperfectas, precisión y recall | PENDIENTE(exp08) |
| Frontera en familias held-out; reservoir, LRU/LFU, replay priorizado | exp10: rango [1, 1] en 13 de las 14 familias held-out, [15, 20] con 0.1 % de raros; Spearman ≥ 0.77; §III-D (hecho). §II-C (hecho, exp09: ningún baseline externo supera a la frontera; el replay priorizado greedy queda a 0.035) |
| Sustratos bajo restricciones equivalentes; sensibilidad de hiperparámetros | PENDIENTE(exp12), PENDIENTE(exp07) |
| MiniGrid con rollout como unidad, más entornos; efecto en aprendizaje | §VI y §VIII (exp11, hecho): IC por rollout, 200 rollouts, 4 entornos × 4 políticas × 5 importancias × 5 señales, control nulo; efecto en aprendizaje medido en §VIII (exp13): la frontera retiene mejor la tarea que reaparece pero integra menos retorno que el FIFO, y ningún genotipo probado gana en las dos métricas |
| Integración en e-MDB real frente a su buffer secuencial | Figura 1 y §VIII (hecho); buffer real con lectura secuencial en §VII-C (hecho, exp09: reprueba el gate, 0.368); la integración completa queda como trabajo futuro declarado |

---

## Revisor 2

### R2.1 — Alcance de la validación: un dataset, un extractor, un entorno, sin robot físico

- §VIII declara explícitamente estos límites y agrega los que ningún
  experimento de esta revisión resuelve: el robot físico, la integración en el
  e-MDB real y un único extractor visual.
- MiniGrid se amplió a 4 entornos (MemoryS13, DoorKey-6x6, FourRooms,
  KeyCorridorS3R2) y 4 políticas, dos de ellas dirigidas (planificador BFS con
  ruido y Q-learning tabular), con 200 rollouts y el rollout como unidad
  (§VI, exp11; detalle en R1.7). La saliencia transfiere cuando la señal está
  alineada con lo importante y falla para importancia demorada; §VIII lo dice.
  Siguen siendo mundos de grilla discretos, sin robot físico.

### R2.2 — Dependencia del umbral de fusión 0.85

- En §V dejamos un marcador para el barrido del umbral sobre CIFAR-100.
- En §III-D, el umbral figura entre las condiciones no probadas de la hipótesis.
- Queda PENDIENTE(exp07).

### R2.3 — Más baselines de gestión de memoria

- §II-C incluye ahora la literatura no biológica: caché, reservoir y
  selección para replay.
- En la misma §II-C reportamos la comparación empírica (exp09). Evaluamos siete
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
importancia, con el rollout como unidad (§VI, exp11).

Sobre la generalización fuera del generador (exp10, §III-D): la transición de
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

Queda PENDIENTE(exp11).

### R3.2 — Esquema de la integración EMBER/e-MDB

Agregamos la **Figura 1** en §II-D, en TikZ. Muestra:

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
