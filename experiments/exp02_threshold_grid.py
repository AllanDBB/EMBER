"""exp02 · La ley del umbral: qué mecanismo importa es función de r = K_proto / C.

La hipótesis: consolidar por fusión solo paga cuando los prototipos recurrentes
del flujo **caben** en la memoria. Si hay más prototipos que ranuras, fusionar
llena la memoria de rutina y no queda espacio para lo raro; la pregunta se
reduce entonces a qué política de olvido conserva mejor lo importante.

La predicción falsable es precisa: el cruce entre "domina la escritura" y
"domina el desalojo" tiene que caer en r ≈ 1, y tiene que moverse al mover la
capacidad, no al mover el número de prototipos. Si el cruce dependiera de la
redundancia del flujo en vez del ratio, la ley sería falsa.

Sobre el piloto: la cuadrícula tenía 14 celdas, 3 semillas y ningún intervalo de
confianza. Acá se barre un rango más ancho de capacidades y prototipos, con
suficientes semillas para poder poner una banda alrededor de cada punto.
"""

from __future__ import annotations

import numpy as np

from ember.experiment import ExperimentRun
from ember.nas.engine import run_search
from ember.nas.stats import bootstrap_ci, eta_squared
from experiments._common import EvalConfig, RareRetentionEvaluator

CAPACIDADES = (10, 20, 40, 80)
RATIOS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
"""Ratios objetivo. Los prototipos de cada celda salen de r × capacidad."""

SEMILLAS = (0, 1, 2, 3, 4)


def _eta_por_semilla(capacity: int, n_prototypes: int, seed: int) -> tuple[float, float]:
    """eta² de escritura y de desalojo sobre el espacio completo, para una semilla."""
    config = EvalConfig(
        capacity=capacity,
        n_prototypes=n_prototypes,
        seeds=(seed,),
        n_common=300,
        n_rare=20,
    )
    r = run_search(RareRetentionEvaluator(config), n_jobs=-1, progress=False)
    return (
        eta_squared(r.records, "write", metric="rare_retention"),
        eta_squared(r.records, "evict", metric="rare_retention"),
    )


def barrer_grilla(
    *,
    capacities: tuple[int, ...] = CAPACIDADES,
    ratios: tuple[float, ...] = RATIOS,
    seeds: tuple[int, ...] = SEMILLAS,
    verbose: bool = True,
) -> list[dict]:
    """Una celda por (capacidad, ratio), con eta² de ambos ejes e intervalos."""
    celdas = []
    for capacity in capacities:
        for ratio in ratios:
            n_prototypes = max(1, int(round(ratio * capacity)))
            por_semilla = [_eta_por_semilla(capacity, n_prototypes, s) for s in seeds]
            escritura = [w for w, _ in por_semilla]
            desalojo = [e for _, e in por_semilla]

            celda = {
                "capacity": capacity,
                "n_prototypes": n_prototypes,
                "r": n_prototypes / capacity,
                "eta2_write": float(np.mean(escritura)),
                "eta2_write_ci": list(bootstrap_ci(escritura, seed=0)),
                "eta2_evict": float(np.mean(desalojo)),
                "eta2_evict_ci": list(bootstrap_ci(desalojo, seed=0)),
                "dominante": "write" if np.mean(escritura) > np.mean(desalojo) else "evict",
            }
            celdas.append(celda)
            if verbose:
                print(
                    f"  C={capacity:<4} K={n_prototypes:<4} r={celda['r']:<6.2f}"
                    f" η²write={celda['eta2_write']:.3f}"
                    f" η²evict={celda['eta2_evict']:.3f}"
                    f"  → {celda['dominante']}",
                    flush=True,
                )
    return celdas


def localizar_umbral(celdas: list[dict], capacity: int) -> dict | None:
    """El r donde una capacidad dada cruza de dominancia de escritura a de desalojo."""
    de_esta = sorted((c for c in celdas if c["capacity"] == capacity), key=lambda c: c["r"])
    for anterior, actual in zip(de_esta, de_esta[1:], strict=False):
        if anterior["dominante"] == "write" and actual["dominante"] == "evict":
            return {
                "capacity": capacity,
                "r_below": anterior["r"],
                "r_above": actual["r"],
                "r_cruce": float(np.sqrt(anterior["r"] * actual["r"])),
            }
    return None


def main() -> int:
    with ExperimentRun("exp02_threshold_grid") as run:
        run.set_seeds(SEMILLAS)
        run.log(
            f"{len(CAPACIDADES)}×{len(RATIOS)} celdas × {len(SEMILLAS)} semillas × 576 genotipos"
        )

        celdas = barrer_grilla()
        run.record("celdas", celdas)
        run.record("capacidades", list(CAPACIDADES))
        run.record("ratios", list(RATIOS))

        # ── resumen por régimen ─────────────────────────────────────────────
        regimenes = {
            "compresion (r <= 0.5)": [c for c in celdas if c["r"] <= 0.5],
            "transicion (0.5 < r < 1)": [c for c in celdas if 0.5 < c["r"] < 1.0],
            "seleccion (r >= 1)": [c for c in celdas if c["r"] >= 1.0],
        }
        resumen = {
            nombre: {
                "n_celdas": len(cs),
                "eta2_write": float(np.mean([c["eta2_write"] for c in cs])) if cs else None,
                "eta2_evict": float(np.mean([c["eta2_evict"] for c in cs])) if cs else None,
            }
            for nombre, cs in regimenes.items()
        }
        run.record("regimenes", resumen)

        print(f"\n{'régimen':<28}{'η² escritura':>14}{'η² desalojo':>14}")
        print("-" * 58)
        for nombre, datos in resumen.items():
            if datos["eta2_write"] is None:
                continue
            print(f"{nombre:<28}{datos['eta2_write']:>14.3f}{datos['eta2_evict']:>14.3f}")

        # ── dónde cae el cruce en cada capacidad ────────────────────────────
        umbrales = [u for c in CAPACIDADES if (u := localizar_umbral(celdas, c))]
        run.record("umbrales", umbrales)

        print(f"\n{'capacidad':<12}{'cruce entre':>22}")
        print("-" * 36)
        for u in umbrales:
            print(f"{u['capacity']:<12}r ∈ [{u['r_below']:.2f}, {u['r_above']:.2f}]")

        if umbrales:
            cruces = [u["r_cruce"] for u in umbrales]
            run.record("r_umbral_medio", float(np.mean(cruces)))
            run.record("r_umbral_ci", list(bootstrap_ci(cruces, seed=0)))
            print(f"\nCruce medio: r = {np.mean(cruces):.2f}")
            invariante = all(u["r_below"] < 1.0 <= u["r_above"] for u in umbrales)
            run.record("umbral_en_1_para_toda_capacidad", invariante)
            print(f"¿El cruce cae en r=1 en todas las capacidades?: {'sí' if invariante else 'no'}")
            if not invariante:
                run.note(
                    "El cruce no cae en r=1 en toda capacidad. La ley, tal como está "
                    "enunciada, no se sostiene sin calificar."
                )

        try:
            from ember.figures import figura_umbral

            destino = figura_umbral(celdas, "paper/figures/fig1_threshold.pdf")
            run.record("figura", str(destino))
            print(f"\nFigura: {destino}")
        except ImportError:
            run.note("matplotlib no disponible; figura no generada")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
