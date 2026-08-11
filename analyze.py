"""Analisis de importancia de componentes sobre los resultados de la busqueda."""

import json
import numpy as np
import matplotlib.pyplot as plt
from memory_space import SEARCH_SPACE, FIFO_GENOTYPE

with open("search_results.json") as f:
    data = json.load(f)

genos = [d["genotype"] for d in data]
means = np.array([d["scores"]["mean"] for d in data])
rare = np.array([d["scores"]["rare_retention"] for d in data])
interf = np.array([d["scores"]["interference"] for d in data])
noise = np.array([d["scores"]["noise_robustness"] for d in data])

print("=" * 78)
print("IMPORTANCIA DE COMPONENTES (efecto marginal sobre la media)\n")

importance = {}
for comp, options in SEARCH_SPACE.items():
    print(f"{comp}:")
    opt_means = {}
    for opt in options:
        mask = np.array([g[comp] == opt for g in genos])
        opt_means[opt] = means[mask].mean()
        print(
            f"    {str(opt):<14} media={opt_means[opt]:.3f}   "
            f"(raras={rare[mask].mean():.3f}  ruido={noise[mask].mean():.3f}  "
            f"interf={interf[mask].mean():.3f})"
        )
    spread = max(opt_means.values()) - min(opt_means.values())
    importance[comp] = spread
    print(f"    --> rango de efecto: {spread:.3f}\n")

print("=" * 78)
print("RANKING DE IMPORTANCIA (cuanto mueve la aguja cada mecanismo):\n")
for comp, spread in sorted(importance.items(), key=lambda x: -x[1]):
    bar = "#" * int(spread * 100)
    print(f"  {comp:<12} {spread:.3f}  {bar}")

# ---- varianza explicada por componente (ANOVA de una via, eta cuadrado) ----
print("\n" + "=" * 78)
print("VARIANZA EXPLICADA (eta^2) POR COMPONENTE:\n")
grand_mean = means.mean()
ss_total = ((means - grand_mean) ** 2).sum()
eta = {}
for comp, options in SEARCH_SPACE.items():
    ss_between = 0.0
    for opt in options:
        mask = np.array([g[comp] == opt for g in genos])
        ss_between += mask.sum() * (means[mask].mean() - grand_mean) ** 2
    eta[comp] = ss_between / ss_total
    print(f"  {comp:<12} eta^2 = {eta[comp]:.3f}  ({eta[comp] * 100:.1f}% de la varianza)")

# ------------------------------------------------------------------ figura
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# panel A: distribucion de scores con el FIFO marcado
fifo_idx = [i for i, g in enumerate(genos) if g == FIFO_GENOTYPE][0]
fifo_score = means[fifo_idx]
axes[0].hist(means, bins=40, color="#5B6470", alpha=0.75, edgecolor="white", linewidth=0.4)
axes[0].axvline(
    fifo_score, color="#C0392B", linewidth=2.2, label=f"FIFO de e-MDB ({fifo_score:.3f})"
)
axes[0].axvline(
    means.max(),
    color="#C76B2C",
    linewidth=2.2,
    linestyle="--",
    label=f"mejor hallada ({means.max():.3f})",
)
axes[0].set_xlabel("Puntaje medio en la bateria de tareas")
axes[0].set_ylabel("Numero de arquitecturas")
axes[0].set_title("Distribucion del espacio de busqueda (576 arquitecturas)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.25)

# panel B: varianza explicada por componente
comps = sorted(eta.keys(), key=lambda c: -eta[c])
vals = [eta[c] * 100 for c in comps]
bars = axes[1].barh(range(len(comps)), vals, color="#C76B2C")
axes[1].set_yticks(range(len(comps)))
axes[1].set_yticklabels(comps)
axes[1].invert_yaxis()
axes[1].set_xlabel("Varianza explicada del puntaje (%)")
axes[1].set_title("Que mecanismo importa realmente")
for b, v in zip(bars, vals):
    axes[1].text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
axes[1].grid(alpha=0.25, axis="x")

plt.tight_layout()
plt.savefig("nas_pilot_analysis.png", dpi=145)
print("\nGuardado nas_pilot_analysis.png")

# ---------------- frontera de Pareto entre retencion e interferencia -------
print("\n" + "=" * 78)
print("TENSION ENTRE OBJETIVOS (mejores por tarea individual):\n")
for label, arr in [
    ("retencion de eventos raros", rare),
    ("robustez al ruido", noise),
    ("resistencia a interferencia", interf),
]:
    best = int(np.argmax(arr))
    g = genos[best]
    print(
        f"  {label:<30} {arr[best]:.3f}  <- write={g['write_mode']}, "
        f"init={g['init_str']}, evict={g['evict']}"
    )
