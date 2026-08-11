"""
Bateria de tareas + busqueda exhaustiva sobre el espacio de arquitecturas.

Tareas (todas bajo presion de capacidad, que es la condicion realista de un
robot con memoria acotada operando de por vida):
  T1 Retencion de eventos raros:  flujo dominado por experiencias comunes y
     repetitivas con unos pocos eventos raros de alto error de prediccion.
     Mide si la arquitectura conserva lo raro-pero-importante.
  T2 Robustez al ruido: recuperar la identidad correcta desde una clave
     degradada.
  T3 Interferencia / olvido: bloques secuenciales de patrones; mide cuanto
     del primer bloque sobrevive al final (olvido catastrofico).
"""

import numpy as np
from memory_space import ConfigurableMemory, enumerate_space, FIFO_GENOTYPE

DIM = 32


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


# ------------------------------------------------------------------ tareas


def task_rare_retention(
    genotype, seed=0, capacity=20, n_common=300, n_rare=20, n_clusters=5, noise=0.05
):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(0, 1, size=(n_clusters, DIM)))
    rare = _unit(rng.normal(0, 1, size=(n_rare, DIM)))

    stream = []
    for _ in range(n_common):
        c = centers[rng.integers(n_clusters)]
        stream.append(
            (
                _unit(c + rng.normal(0, noise, DIM)),
                "common",
                float(np.clip(rng.normal(0.10, 0.05), 0, 1)),
            )
        )
    positions = sorted(rng.choice(n_common + n_rare, size=n_rare, replace=False))
    for i, p in enumerate(positions):
        stream.insert(p, (rare[i], f"rare{i}", float(np.clip(rng.normal(0.90, 0.05), 0, 1))))

    mem = ConfigurableMemory(genotype, capacity, DIM, seed=seed)
    for k, v, pe in stream:
        mem.write(k, v, pe)

    hits = 0
    for i, pat in enumerate(rare):
        val, sim = mem.read(pat)
        if val == f"rare{i}" and sim >= 0.90:
            hits += 1
    return hits / n_rare


def task_noise_robustness(genotype, seed=0, capacity=20, n_items=20, noise=0.35, n_queries=60):
    rng = np.random.default_rng(seed + 1000)
    items = _unit(rng.normal(0, 1, size=(n_items, DIM)))

    mem = ConfigurableMemory(genotype, capacity, DIM, seed=seed)
    for i, k in enumerate(items):
        mem.write(k, f"item{i}", 0.5)

    hits = 0
    for _ in range(n_queries):
        i = int(rng.integers(n_items))
        q = _unit(items[i] + rng.normal(0, noise, DIM))
        val, _ = mem.read(q)
        hits += val == f"item{i}"
    return hits / n_queries


def task_interference(genotype, seed=0, capacity=20, n_blocks=4, per_block=10):
    rng = np.random.default_rng(seed + 2000)
    blocks = [_unit(rng.normal(0, 1, size=(per_block, DIM))) for _ in range(n_blocks)]

    mem = ConfigurableMemory(genotype, capacity, DIM, seed=seed)
    for b, block in enumerate(blocks):
        for j, k in enumerate(block):
            # cada patron se repite 3 veces dentro de su bloque (aprendizaje)
            for _ in range(3):
                mem.write(k, f"b{b}_{j}", 0.5)

    hits = 0
    for j, k in enumerate(blocks[0]):
        val, sim = mem.read(k)
        if val == f"b0_{j}" and sim >= 0.90:
            hits += 1
    return hits / per_block


TASKS = [
    ("rare_retention", task_rare_retention),
    ("noise_robustness", task_noise_robustness),
    ("interference", task_interference),
]


def evaluate(genotype, seeds=(0, 1, 2)):
    """Evalua un genotipo promediando sobre varias semillas."""
    scores = {}
    for name, fn in TASKS:
        vals = [fn(genotype, seed=s) for s in seeds]
        scores[name] = float(np.mean(vals))
    scores["mean"] = float(np.mean([scores[n] for n, _ in TASKS]))
    return scores


# --------------------------------------------------------------- busqueda


def run_search(seeds=(0, 1, 2), verbose=True):
    results = []
    space = list(enumerate_space())
    for i, g in enumerate(space):
        sc = evaluate(g, seeds=seeds)
        results.append((g, sc))
        if verbose and (i + 1) % 100 == 0:
            print(f"  evaluadas {i + 1}/{len(space)} arquitecturas...")
    results.sort(key=lambda r: -r[1]["mean"])
    return results


if __name__ == "__main__":
    import time, json

    t0 = time.time()
    print(f"Espacio de busqueda: {len(list(enumerate_space()))} arquitecturas")
    print("Evaluando (3 tareas x 3 semillas cada una)...\n")
    results = run_search()
    elapsed = time.time() - t0

    fifo_score = evaluate(FIFO_GENOTYPE)
    print(f"\n{'=' * 78}")
    print(f"BASELINE — EpisodicBuffer FIFO de e-MDB (un punto del espacio):")
    print(
        f"  raras={fifo_score['rare_retention']:.3f}  "
        f"ruido={fifo_score['noise_robustness']:.3f}  "
        f"interferencia={fifo_score['interference']:.3f}  "
        f"MEDIA={fifo_score['mean']:.3f}"
    )

    ranks = [i for i, (g, s) in enumerate(results) if g == FIFO_GENOTYPE]
    print(f"  ranking del FIFO dentro del espacio: #{ranks[0] + 1} de {len(results)}")

    print(f"\n{'=' * 78}")
    print("TOP 8 ARQUITECTURAS ENCONTRADAS:\n")
    print(
        f"{'#':<3}{'read':<8}{'write':<8}{'init':<10}{'decay':<8}"
        f"{'evict':<14}{'reinf':<7}{'raras':<8}{'ruido':<8}{'interf':<8}{'MEDIA'}"
    )
    print("-" * 96)
    for i, (g, s) in enumerate(results[:8]):
        print(
            f"{i + 1:<3}{g['read_mode']:<8}{g['write_mode']:<8}{g['init_str']:<10}"
            f"{g['decay']:<8}{g['evict']:<14}{g['reinforce']:<7}"
            f"{s['rare_retention']:<8.3f}{s['noise_robustness']:<8.3f}"
            f"{s['interference']:<8.3f}{s['mean']:.3f}"
        )

    print(f"\nTiempo total: {elapsed:.1f}s")

    with open("search_results.json", "w") as f:
        json.dump([{"genotype": g, "scores": s} for g, s in results], f, indent=1)
    print("Resultados guardados en search_results.json")
