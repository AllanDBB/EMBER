"""exp01 · Búsqueda exhaustiva sobre las 576 arquitecturas.

Evalúa el espacio completo sobre la batería de tres tareas y reporta:

  1. Efectos principales por eje, junto con su observabilidad. Un eje con
     efecto 0.000 y observabilidad 0.000 no dice nada sobre el mecanismo: dice
     que el mecanismo no se está midiendo.
  2. La descomposición factorial con interacciones, que es donde aparece el
     efecto real de la compuerta de saliencia.
  3. El efecto condicional de la saliencia restringido al subespacio donde la
     señal de fuerza llega a leerse.
  4. Dónde cae el `EpisodicBuffer` FIFO de e-MDB dentro de su propio espacio de
     diseño, reportado como intervalo de rango y no como puesto puntual.
"""

from __future__ import annotations

from ember.core.genotype import FIFO_GENOTYPE
from ember.experiment import ExperimentRun
from ember.nas.engine import run_search
from ember.nas.space import AXES, BIOLOGICAL_ROOT, space_size
from ember.nas.stats import conditional_effect, main_effects_table, variance_share
from experiments._common import SEEDS, EvalConfig, GenotypeEvaluator, fmt_pct

TAREAS = ("rare_retention", "noise_under_pressure", "sequential_interference")


def main() -> int:
    config = EvalConfig(seeds=SEEDS)

    with ExperimentRun("exp01_nas_full") as run:
        run.set_seeds(config.seeds)
        run.log(f"evaluando {space_size()} arquitecturas × {len(config.seeds)} semillas")

        resultados = run_search(GenotypeEvaluator(config), n_jobs=-1, progress=True)

        # ── 1. efectos principales, con observabilidad ──────────────────────
        tabla = main_effects_table(resultados.records)
        run.record("main_effects", tabla)

        print(f"\n{'eje':<12}{'eta²':>9}{'observab.':>12}   raíz biológica")
        print("-" * 88)
        for fila in tabla:
            print(
                f"{fila['axis']:<12}{fmt_pct(fila['eta2']):>9}"
                f"{fila['liveness']:>12.4f}   {BIOLOGICAL_ROOT[fila['axis']]}"
            )

        # ── 2. descomposición con interacciones ─────────────────────────────
        share = variance_share(resultados.records)
        run.record("variance_share", share)
        print("\nTérminos que más varianza explican (con interacciones):")
        for termino, v in sorted(share.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {termino:<28}{fmt_pct(v)}")

        # ── 3. el efecto condicional de la saliencia ────────────────────────
        cond = conditional_effect(
            resultados.records,
            "strength",
            given={"evict": "min_strength", "write": "append"},
            metric="rare_retention",
        )
        marginal = conditional_effect(
            resultados.records, "strength", given={}, metric="rare_retention"
        )
        run.record("salience_conditional", cond)
        run.record("salience_marginal", marginal)

        print("\nRetención de eventos raros por compuerta de saliencia:")
        print(f"{'compuerta':<16}{'marginal':>12}{'| evict=min_strength, write=append':>36}")
        for opcion in sorted(cond):
            print(f"  {opcion:<14}{marginal[opcion]:>12.3f}{cond[opcion]:>36.3f}")

        base = cond.get("constant", 0.0)
        mejor = max(cond.values())
        if base > 0:
            print(f"\n  mejora condicional: {mejor / base:.1f}×  ({base:.3f} → {mejor:.3f})")
            run.record("salience_conditional_ratio", mejor / base)

        # ── 4. dónde cae el incumbente ──────────────────────────────────────
        fifo_score = resultados.score_of(FIFO_GENOTYPE)
        optimista, pesimista = resultados.rank_of(FIFO_GENOTYPE)
        en_el_piso = resultados.is_at_floor(FIFO_GENOTYPE)
        mejor_registro = resultados.records[0]

        run.record(
            "incumbent",
            {
                "genotype": FIFO_GENOTYPE.as_dict(),
                "score": fifo_score,
                "rank_optimistic": optimista,
                "rank_pessimistic": pesimista,
                "n_tied": pesimista - optimista + 1,
                "at_floor": en_el_piso,
                "scores": next(r.scores for r in resultados.records if r.genotype == FIFO_GENOTYPE),
            },
        )
        run.record(
            "frontier",
            {
                "genotype": mejor_registro.genotype.as_dict(),
                "score": mejor_registro.mean,
                "scores": mejor_registro.scores,
            },
        )
        run.record("top8", [r.to_dict() for r in resultados.best(8)])
        run.record("all_records", [r.to_dict() for r in resultados.records])

        print(f"\nEpisodicBuffer FIFO de e-MDB: {fifo_score:.3f}")
        print(f"  rango: #{optimista}–#{pesimista} de {len(resultados)}", end="")
        print(f"  ({pesimista - optimista + 1} arquitecturas empatan)")
        print(f"  ¿en el piso del espacio?: {'sí' if en_el_piso else 'no'}")
        print(f"\nFrontera: {mejor_registro.mean:.3f}  {mejor_registro.genotype.label()}")

        if pesimista > optimista:
            run.note(
                f"El rango del incumbente es un intervalo de {pesimista - optimista + 1} "
                "puestos por empate. Reportar el extremo optimista como si fuera la "
                "posición sería el error del piloto."
            )
        for eje in AXES:
            fila = next(f for f in tabla if f["axis"] == eje)
            if fila["liveness"] == 0.0:
                run.note(f"eje inobservable: {eje}")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
