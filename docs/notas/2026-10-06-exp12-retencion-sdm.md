# La retención de raros de la SDM es de su lista de trazas, no de sus contadores

**Fecha:** 2026-10-06
**Fuente:** `results/exp12_substrate_cost/` (re-corrida sobre el motor integrado, sha 090f1d9)
**Afecta a:** §IV (benchmark) y la conclusión de `paper/main.tex`

## Qué se midió

`exp12` separa las dos estructuras de estado de la SDM: (a) solo contadores,
(a+) contadores con borrado exacto, (b) solo la lista de trazas con vecino más
cercano, (c) completa. También mide bytes, tiempos e igual presupuesto de bytes.

| Variante | Batería T1–T3 | T1 (raros) | KiB a C = 20 |
|---|---|---|---|
| (c) SDM completa | 0.741 | 1.000 | 131.1 |
| (b) solo lista de trazas | 0.788 | 1.000 | 3.1 |
| (a) solo contadores | 0.388 | 0.005 | 195.1 |
| (a+) contadores + borrado exacto | 0.735 | 1.000 | 197.6 |

Diferencia pareada (c) − (b): −0.047, IC [−0.053, −0.041]. La hipótesis H2 del
experimento (la lista empata con la completa) se refuta, pero en la dirección
contraria a la que la debilitaría: la lista sola es *mejor*.

A igual presupuesto (144 KiB, la huella de la SDM a C = 20), el FIFO guarda 921
trazas y da 0.924 en la batería contra 0.745 de la SDM (H1 confirmada).

## Lo que sobrevive

- La SDM retiene todos los raros (1.000) y pasa el gate. El número no cambia.
- La explicación de §IV —ganancia inicial por error de predicción más desalojo
  por mínima fuerza— es correcta: es justamente el ciclo de vida de la lista.
- SDM vs ENN sigue siendo la comparación informativa (misma compuerta y mismo
  desalojo).
- Sensibilidad: la batería se mueve 0.091 entre 64 y 2048 hard locations; la
  retención de raros es perfecta entre fracción 0.005 y 0.2.

## Lo que no sobrevive, y con qué se reemplazó

- §IV decía que en la fase 2 cada traza "debe recuperarse desde la
  superposición distribuida". La variante (b) no tiene superposición y rinde
  igual o mejor: lo que decide la recuperación es la lista. Reemplazado por
  "must still be read back at similarity ≥ 0.85, which for SDM is decided by its
  trace list".
- La conclusión decía que la SDM con compuerta es "the only substrate pairing
  reconstruction fidelity with perfect retention". Un punto del espacio de
  genotipos (la lista de la SDM sin contadores) hace lo mismo con 1/42 de los
  bytes. Reemplazado por: la SDM lo logra, pero la retención viene de su lista
  de trazas con compuerta, no de los contadores, que dominan su huella.

## Matiz que conviene no perder

Los contadores solos fallan (0.005) porque no pueden restar el aporte de una
traza desalojada; con borrado exacto (a+) vuelven a 1.000. Es el invariante 4
del repo visto desde el otro lado: sin contabilidad exacta de lo sumado, el
residuo se come a los raros.
