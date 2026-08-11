"""
benchmark.py
------------
Dos fases de evaluación:

FASE 1 — Reconstrucción (gate de admisión al grid)
  R1. Pattern completion:     recuperar key completa desde un cue degradado.
  R2. Noise robustness:       recuperar key correcta con ruido aditivo.
  R3. Interference (2-item):  después de escribir B, ¿se puede leer A?
  R4. Capacity profile:       accuracy vs. n_items guardados (curva completa).

  Una arquitectura "pasa" si su score de reconstrucción medio >= RECON_THRESHOLD.

FASE 2 — Batería completa (equivalente a search.py del NAS, pero sobre arquitecturas)
  T1. Rare-event retention
  T2. Noise robustness bajo presión de capacidad
  T3. Sequential interference

Todas las tareas usan vectores continuos float32 de dimensión DIM.
"""

import numpy as np
from architectures import ARCHITECTURES, FIFOMemory

DIM = 32
RECON_THRESHOLD = 0.50  # gate de admisión a la fase 2
SEEDS = (0, 1, 2, 3)


# ─── utilidades ──────────────────────────────────────────────────────────────


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _make_mem(arch_cls, capacity=20, seed=0):
    return arch_cls(dim=DIM, capacity=capacity, seed=seed)


# ════════════════════════════════════════════════════════════════════
# FASE 1 — RECONSTRUCCIÓN
# ════════════════════════════════════════════════════════════════════


def r1_pattern_completion(arch_cls, seed=0, capacity=20, n_patterns=10, cue_fracs=(0.3, 0.5, 0.7)):
    """
    Escribe n_patterns. Para cada uno, presenta cues con distintas fracciones
    de dimensiones no enmascaradas y mide similitud coseno de lo recuperado.
    """
    rng = np.random.default_rng(seed)
    mem = _make_mem(arch_cls, capacity, seed)
    patterns = [_unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(n_patterns)]

    for i, p in enumerate(patterns):
        mem.write(p, i, pred_error=0.8)

    scores = []
    for i, p in enumerate(patterns):
        for frac in cue_fracs:
            # cue: poner a cero 1-frac de las dimensiones
            cue = p.copy()
            n_mask = int(DIM * (1 - frac))
            mask_idx = rng.choice(DIM, n_mask, replace=False)
            cue[mask_idx] = 0.0
            val, sim = mem.read(cue)
            # correcto si devuelve el mismo índice
            correct = 1.0 if val == i else 0.0
            scores.append(correct * sim + (1 - correct) * 0.0)
    return float(np.mean(scores))


def r2_noise_robustness(
    arch_cls, seed=0, capacity=20, n_patterns=15, noise_levels=(0.1, 0.2, 0.3, 0.4)
):
    """Recuperar el ítem correcto desde una clave con ruido gaussiano."""
    rng = np.random.default_rng(seed + 100)
    mem = _make_mem(arch_cls, capacity, seed)
    patterns = [_unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(n_patterns)]

    for i, p in enumerate(patterns):
        mem.write(p, i, pred_error=0.5)

    scores = []
    for i, p in enumerate(patterns):
        for sigma in noise_levels:
            noisy = _unit(p + rng.standard_normal(DIM).astype(np.float32) * sigma)
            val, _ = mem.read(noisy)
            scores.append(1.0 if val == i else 0.0)
    return float(np.mean(scores))


def r3_ab_interference(arch_cls, seed=0, capacity=20, n_pairs=10):
    """
    Escribe par (A, B). Después de B, ¿se puede leer A correctamente?
    Mide si la escritura de B destruye el recuerdo de A.
    """
    rng = np.random.default_rng(seed + 200)
    scores = []
    for _ in range(n_pairs):
        mem = _make_mem(arch_cls, capacity, seed + _)
        a = _unit(rng.standard_normal(DIM).astype(np.float32))
        b = _unit(rng.standard_normal(DIM).astype(np.float32))
        mem.write(a, "A", pred_error=0.9)
        mem.write(b, "B", pred_error=0.9)
        val, sim = mem.read(a)
        scores.append(sim if val == "A" else 0.0)
    return float(np.mean(scores))


def r4_capacity_profile(arch_cls, seed=0, noise=0.15, loads=(2, 5, 10, 15, 20, 30)):
    """
    Curva accuracy vs. n_items: escribe n_items y recupera cada uno con ruido.
    Devuelve dict {n_items: accuracy}.
    """
    rng = np.random.default_rng(seed + 300)
    profile = {}
    for n in loads:
        mem = _make_mem(arch_cls, capacity=n, seed=seed)
        patterns = [_unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(n)]
        for i, p in enumerate(patterns):
            mem.write(p, i, pred_error=0.5)
        hits = 0
        for i, p in enumerate(patterns):
            noisy = _unit(p + rng.standard_normal(DIM).astype(np.float32) * noise)
            val, _ = mem.read(noisy)
            hits += val == i
        profile[n] = hits / n
    return profile


def evaluate_reconstruction(arch_cls, seeds=SEEDS, verbose=False):
    """
    Corre las 4 subtareas de reconstrucción y devuelve dict con scores.
    """
    r1 = np.mean([r1_pattern_completion(arch_cls, seed=s) for s in seeds])
    r2 = np.mean([r2_noise_robustness(arch_cls, seed=s) for s in seeds])
    r3 = np.mean([r3_ab_interference(arch_cls, seed=s) for s in seeds])
    # r4: promedio de la curva de capacidad sobre semillas
    profiles = [r4_capacity_profile(arch_cls, seed=s) for s in seeds]
    avg_profile = {k: np.mean([p[k] for p in profiles]) for k in profiles[0]}
    r4 = float(np.mean(list(avg_profile.values())))

    mean = float(np.mean([r1, r2, r3, r4]))
    result = {
        "pattern_completion": round(r1, 3),
        "noise_robustness": round(r2, 3),
        "ab_interference": round(r3, 3),
        "capacity_profile": round(r4, 3),
        "mean": round(mean, 3),
        "capacity_curve": {k: round(v, 3) for k, v in avg_profile.items()},
        "passes_gate": mean >= RECON_THRESHOLD,
    }
    if verbose:
        print(
            f"  pattern_completion={r1:.3f}  noise={r2:.3f}  "
            f"ab_interf={r3:.3f}  capacity={r4:.3f}  → MEAN={mean:.3f} "
            f"{'✓ GATE' if result['passes_gate'] else '✗ FAIL'}"
        )
    return result


# ════════════════════════════════════════════════════════════════════
# FASE 2 — BATERÍA COMPLETA (solo para arquitecturas que pasan el gate)
# ════════════════════════════════════════════════════════════════════


def t1_rare_retention(
    arch_cls, seed=0, capacity=20, n_common=200, n_rare=15, n_clusters=5, noise=0.12
):
    """Flujo con eventos comunes repetitivos + eventos raros de alta sorpresa."""
    rng = np.random.default_rng(seed + 1000)
    centers = np.array(
        [_unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(n_clusters)]
    )
    rares = np.array([_unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(n_rare)])

    stream = []
    for _ in range(n_common):
        c = centers[rng.integers(n_clusters)]
        k = _unit(c + rng.standard_normal(DIM).astype(np.float32) * noise)
        stream.append((k, "common", float(np.clip(rng.normal(0.1, 0.05), 0, 1))))
    positions = sorted(rng.choice(n_common + n_rare, n_rare, replace=False))
    for i, pos in enumerate(positions):
        stream.insert(pos, (rares[i], f"rare{i}", float(np.clip(rng.normal(0.9, 0.05), 0, 1))))

    mem = _make_mem(arch_cls, capacity, seed)
    for k, v, pe in stream:
        mem.write(k, v, pe)

    hits = 0
    for i, r in enumerate(rares):
        val, sim = mem.read(r)
        if val == f"rare{i}" and sim >= 0.75:
            hits += 1
    return hits / n_rare


def t2_noise_capacity(arch_cls, seed=0, capacity=20, n_items=20, noise=0.30, n_queries=50):
    """Ruido fuerte bajo presión de capacidad (a diferencia de la versión NAS que no presionaba)."""
    rng = np.random.default_rng(seed + 2000)
    items = [_unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(n_items)]
    mem = _make_mem(arch_cls, capacity, seed)
    for i, k in enumerate(items):
        mem.write(k, i, pred_error=0.5)
    hits = 0
    for _ in range(n_queries):
        i = int(rng.integers(n_items))
        noisy = _unit(items[i] + rng.standard_normal(DIM).astype(np.float32) * noise)
        val, _ = mem.read(noisy)
        hits += val == i
    return hits / n_queries


def t3_sequential_interference(arch_cls, seed=0, capacity=20, n_blocks=4, per_block=8):
    """4 bloques secuenciales; mide cuánto del bloque-1 sobrevive al final."""
    rng = np.random.default_rng(seed + 3000)
    blocks = [
        [_unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(per_block)]
        for _ in range(n_blocks)
    ]
    mem = _make_mem(arch_cls, capacity, seed)
    for b, block in enumerate(blocks):
        for j, k in enumerate(block):
            for _ in range(3):  # 3 repeticiones por ítem dentro del bloque
                mem.write(k, f"b{b}_{j}", pred_error=0.5)
    hits = 0
    for j, k in enumerate(blocks[0]):
        val, sim = mem.read(k)
        if val == f"b0_{j}" and sim >= 0.75:
            hits += 1
    return hits / per_block


def evaluate_battery(arch_cls, seeds=SEEDS, verbose=False):
    t1 = np.mean([t1_rare_retention(arch_cls, seed=s) for s in seeds])
    t2 = np.mean([t2_noise_capacity(arch_cls, seed=s) for s in seeds])
    t3 = np.mean([t3_sequential_interference(arch_cls, seed=s) for s in seeds])
    mean = float(np.mean([t1, t2, t3]))
    result = {
        "rare_retention": round(float(t1), 3),
        "noise_capacity": round(float(t2), 3),
        "seq_interference": round(float(t3), 3),
        "mean": round(mean, 3),
    }
    if verbose:
        print(f"  rare={t1:.3f}  noise={t2:.3f}  interf={t3:.3f}  → MEAN={mean:.3f}")
    return result


# ════════════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, time

    print("=" * 72)
    print("EMBER EVALUATION HARNESS")
    print(f"dim={DIM}  seeds={SEEDS}  gate_threshold={RECON_THRESHOLD}")
    print("=" * 72)

    all_results = {}

    # ── FASE 1: reconstrucción ──────────────────────────────────────
    print("\n── FASE 1: Reconstrucción ──────────────────────────────────────")
    print(
        f"{'Arq.':<14} {'completion':>10} {'noise':>7} {'A→B':>7} {'capacity':>9} {'MEAN':>7}  gate"
    )
    print("-" * 65)

    admitted = []
    for name, cls in ARCHITECTURES.items():
        t0 = time.time()
        r = evaluate_reconstruction(cls, verbose=False)
        elapsed = time.time() - t0
        all_results[name] = {"recon": r}
        mark = "✓" if r["passes_gate"] else "✗"
        print(
            f"{name:<14} {r['pattern_completion']:>10.3f} {r['noise_robustness']:>7.3f} "
            f"{r['ab_interference']:>7.3f} {r['capacity_profile']:>9.3f} "
            f"{r['mean']:>7.3f}  {mark}  ({elapsed:.1f}s)"
        )
        if r["passes_gate"]:
            admitted.append((name, cls))

    print(f"\nAdmitidas para fase 2: {[n for n, _ in admitted]}")

    # ── FASE 2: batería completa ────────────────────────────────────
    print("\n── FASE 2: Batería completa ────────────────────────────────────")
    print(f"{'Arq.':<14} {'rare':>7} {'noise':>7} {'interf':>8} {'MEAN':>7}")
    print("-" * 50)

    for name, cls in admitted:
        t0 = time.time()
        b = evaluate_battery(cls, verbose=False)
        elapsed = time.time() - t0
        all_results[name]["battery"] = b
        print(
            f"{name:<14} {b['rare_retention']:>7.3f} {b['noise_capacity']:>7.3f} "
            f"{b['seq_interference']:>8.3f} {b['mean']:>7.3f}  ({elapsed:.1f}s)"
        )

    # ── curvas de capacidad ─────────────────────────────────────────
    print("\n── Curva de capacidad (accuracy@noise=0.15 vs n_items) ─────────")
    print(f"{'Arq.':<14}", end="")
    sample_profile = list(all_results.values())[0]["recon"]["capacity_curve"]
    for k in sample_profile:
        print(f"  n={k}", end="")
    print()
    for name, data in all_results.items():
        print(f"{name:<14}", end="")
        for k, v in data["recon"]["capacity_curve"].items():
            print(f"  {v:.3f}", end="")
        print()

    # guardar
    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nGuardado eval_results.json")
