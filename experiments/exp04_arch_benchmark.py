"""exp04 · Benchmark de arquitecturas en dos fases.

El NAS identifica qué combinación de mecanismos rinde mejor, pero no evalúa las
arquitecturas como sistemas de memoria completos. En particular no prueba si
cada una puede hacer la operación fundamental que cualquier memoria asociativa
tiene que soportar: reconstruir un patrón completo desde una clave ruidosa o
parcial.

**Fase 1** es un gate: sin presión de capacidad, mide reconstrucción. Una
arquitectura que no llega al umbral no pasa a la fase 2, porque medirle
retención bajo presión no diría nada — no se sabría si falla por su política de
olvido o porque su lectura es mala de entrada.

**Fase 2** es la batería bajo presión, la misma que corre el NAS.

Esta separación es la que permite interpretar el resultado del circuito spiking
puro: forma engramas distinguibles pero no transfiere al dominio continuo, y eso
es un resultado sobre el espacio de diseño, no un defecto del mecanismo.
"""

from __future__ import annotations

import time

from ember.experiment import ExperimentRun
from ember.memories import ARCHITECTURES
from ember.tasks.battery import run_battery
from ember.tasks.reconstruction import RECON_THRESHOLD, reconstruction_gate

SEMILLAS = (0, 1, 2, 3)
DIM = 32
CAPACITY = 20

LENTAS = ("Spiking",)
"""Cada escritura simula 30 presentaciones de ráfagas sobre 128 neuronas LIF."""


def _fabrica(nombre: str):
    cls = ARCHITECTURES[nombre]

    def factory(capacity: int, seed: int):
        return cls(dim=DIM, capacity=capacity, seed=seed)

    return factory


def correr_benchmark(
    *, seeds: tuple[int, ...] = SEMILLAS, incluir_lentas: bool = True, verbose: bool = True
) -> dict[str, dict]:
    """Gate de reconstrucción, y batería para las que lo pasan."""
    nombres = [n for n in ARCHITECTURES if incluir_lentas or n not in LENTAS]
    resultados: dict[str, dict] = {}

    if verbose:
        print(f"\n── Fase 1 · reconstrucción (umbral {RECON_THRESHOLD:.2f}) ──")
        print(
            f"{'arquitectura':<14}{'completado':>11}{'ruido':>9}"
            f"{'A→B':>8}{'capacidad':>11}{'media':>8}  gate"
        )
        print("-" * 68)

    for nombre in nombres:
        t0 = time.time()
        gate = reconstruction_gate(_fabrica(nombre), seeds=seeds, dim=DIM, name=nombre)
        resultados[nombre] = {"gate": gate.to_dict(), "gate_s": round(time.time() - t0, 1)}
        if verbose:
            p = gate.per_task
            print(
                f"{nombre:<14}{p['pattern_completion']:>11.3f}{p['noise_robustness']:>9.3f}"
                f"{p['ab_interference']:>8.3f}{p['capacity_profile']:>11.3f}"
                f"{gate.mean:>8.3f}  {'✓' if gate.passes else '✗'}"
                f"  ({resultados[nombre]['gate_s']}s)",
                flush=True,
            )

    admitidas = [n for n in nombres if resultados[n]["gate"]["passes"]]
    if verbose:
        print(f"\nAdmitidas a fase 2: {', '.join(admitidas) or '(ninguna)'}")
        print("\n── Fase 2 · batería bajo presión de capacidad ──")
        print(f"{'arquitectura':<14}{'raros':>9}{'ruido':>9}{'interf.':>10}{'media':>8}")
        print("-" * 52)

    for nombre in admitidas:
        t0 = time.time()
        bateria = run_battery(
            _fabrica(nombre), seeds=seeds, dim=DIM, capacity=CAPACITY, name=nombre
        )
        resultados[nombre]["battery"] = bateria.to_dict()
        resultados[nombre]["battery_s"] = round(time.time() - t0, 1)
        if verbose:
            p = bateria.per_task
            print(
                f"{nombre:<14}{p['rare_retention']:>9.3f}{p['noise_under_pressure']:>9.3f}"
                f"{p['sequential_interference']:>10.3f}{bateria.mean:>8.3f}"
                f"  ({resultados[nombre]['battery_s']}s)",
                flush=True,
            )

    return resultados


def main() -> int:
    with ExperimentRun("exp04_arch_benchmark") as run:
        run.set_seeds(SEMILLAS)
        run.note(
            "Los contadores de SDM y Spiking-SDM ahora conservan lo escrito: el "
            "piloto restaba 1x al desalojar lo que había sumado strength x, así "
            "que sus lecturas se hacían contra residuos acumulados."
        )

        resultados = correr_benchmark(seeds=SEMILLAS)
        run.record("architectures", resultados)
        run.record("gate_threshold", RECON_THRESHOLD)
        run.record("capacity", CAPACITY)

        admitidas = {n: d for n, d in resultados.items() if d["gate"]["passes"]}
        rechazadas = [n for n, d in resultados.items() if not d["gate"]["passes"]]
        run.record("admitidas", sorted(admitidas))
        run.record("rechazadas", rechazadas)

        if admitidas:
            raras = {n: d["battery"]["per_task"]["rare_retention"] for n, d in admitidas.items()}
            mejor = max(raras, key=lambda k: raras[k])
            peor = min(raras, key=lambda k: raras[k])
            run.record("mejor_retencion_raros", {"arch": mejor, "score": raras[mejor]})
            run.record("peor_retencion_raros", {"arch": peor, "score": raras[peor]})
            run.record("fifo_en_el_piso", peor == "FIFO")

            print(f"\nMejor retención de eventos raros: {mejor} ({raras[mejor]:.3f})")
            print(f"Peor:                             {peor} ({raras[peor]:.3f})")
            if peor != "FIFO":
                run.note(
                    f"El FIFO NO es el peor en retención de eventos raros: lo es {peor}. "
                    "La afirmación del draft no se sostiene entre arquitecturas."
                )
                print(
                    f"\n  ATENCIÓN: el draft afirma que el FIFO está en el piso, "
                    f"pero {peor} retiene menos."
                )

        for nombre in rechazadas:
            run.note(f"{nombre} no pasó el gate de reconstrucción")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
