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
- qué queda sin probar: frecuencias desiguales, deriva, similitud ruidosa,
  prevalencia de raros, umbral de fusión, largo del flujo y rango de
  capacidades.

Quedan PENDIENTE(exp10), la robustez y las familias held-out, y
PENDIENTE(exp07), el umbral de fusión.

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

Queda PENDIENTE(exp09): el buffer real con recuperación secuencial como
baseline adicional.

### R1.7 — MiniGrid limitado; importancia definida por la tarea; Wilson sobre eventos

En §VI aclaramos que el intervalo de Wilson trata cada evento raro como
independiente y que la unidad defendible es el rollout.

En §VIII queda explícito que la saliencia con importancia demorada, ambigua o
sin recompensa externa sigue abierta.

Queda PENDIENTE(exp11): el rollout o la semilla como unidad, y más entornos,
políticas y definiciones de relevancia (recompensa demorada, objetivos
intrínsecos).

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
- Queda PENDIENTE(exp13): el efecto de la memoria sobre aprendizaje y
  desempeño.
- Agregamos en §II-C un párrafo de gestión de memoria no biológica:
  - LRU-K (O'Neil et al. 1993) y ARC (Megiddo y Modha 2003);
  - reservoir sampling (Vitter 1985);
  - Chaudhry et al. 2019;
  - iCaRL (Rebuffi et al. 2017);
  - GSS (Aljundi et al. 2019);
  - selective experience replay (Isele y Cosgun 2018);
  - experience replay para aprendizaje continuo (Rolnick et al. 2019).

### R1.10 — Liberar código y configuraciones durante la revisión

La sección "Code and data availability" dice ahora que el código, las
configuraciones, los resultados y los manifiestos de procedencia están
disponibles para los revisores en un repositorio anonimizado.

**Pendiente de los autores:** crear ese repositorio anonimizado y poner la URL
en `main.tex` (marcado `% TODO(autores)`). El texto ya afirma que existe.

### R1.11 — Mejoras recomendadas (resumen)

| Recomendación | Respuesta |
|---|---|
| Protocolo consolidado | §III-B (hecho) |
| Hipótesis o más generadores | Reformulado como hipótesis (hecho) + PENDIENTE(exp10) |
| Señales imperfectas, precisión y recall | PENDIENTE(exp08) |
| Frontera en familias held-out; reservoir, LRU/LFU, replay priorizado | PENDIENTE(exp10), PENDIENTE(exp09) |
| Sustratos bajo restricciones equivalentes; sensibilidad de hiperparámetros | PENDIENTE(exp12), PENDIENTE(exp07) |
| MiniGrid con rollout como unidad, más entornos; efecto en aprendizaje | PENDIENTE(exp11), PENDIENTE(exp13) |
| Integración en e-MDB real frente a su buffer secuencial | Figura 1 y §VIII (hecho); PENDIENTE(exp09) para el buffer real; la integración completa queda como trabajo futuro declarado |

---

## Revisor 2

### R2.1 — Alcance de la validación: un dataset, un extractor, un entorno, sin robot físico

- §VIII declara explícitamente estos límites y agrega los que ningún
  experimento de esta revisión resuelve: el robot físico, la integración en el
  e-MDB real y un único extractor visual.
- Queda PENDIENTE(exp11) para más entornos y políticas en MiniGrid.

### R2.2 — Dependencia del umbral de fusión 0.85

- En §V dejamos un marcador para el barrido del umbral sobre CIFAR-100.
- En §III-D, el umbral figura entre las condiciones no probadas de la hipótesis.
- Queda PENDIENTE(exp07).

### R2.3 — Más baselines de gestión de memoria

- §II-C incluye ahora la literatura no biológica: caché, reservoir y
  selección para replay.
- Queda PENDIENTE(exp09) para la comparación empírica.

---

## Revisor 3

### R3.1 — Alcance de la validación y generalización a robots

Lo tratamos igual que en R2.1: §VIII ampliada, y además la hipótesis de régimen
ya no se enuncia como ley (R1.4).

Quedan PENDIENTE(exp10) y PENDIENTE(exp11).

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
