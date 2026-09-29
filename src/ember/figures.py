"""Figuras del paper, generadas desde los resultados de los experimentos.

matplotlib se importa dentro de cada función: este módulo se puede importar
desde cualquier lado sin arrastrar el stack de laboratorio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Paleta sobria y con contraste suficiente en escala de grises, por si el
# artículo se imprime en blanco y negro.
COLORES = ("#1B4965", "#C76B2C", "#5B8C5A", "#8B5A8C")
GRIS = "#5B6470"


def _preparar(destino: str | Path) -> Path:
    ruta = Path(destino)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return ruta


def figura_umbral(celdas: list[dict[str, Any]], destino: str | Path) -> Path:
    """Figura 1: la transición de régimen en r = K_proto / C.

    Un panel por eje (escritura y desalojo), una serie por capacidad, bandas de
    intervalo de confianza, y la línea vertical en r = 1. Si la ley se sostiene,
    las curvas de capacidades distintas tienen que superponerse al graficarlas
    contra r — que es la afirmación de que la variable de control es el ratio y
    no la redundancia del flujo.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ruta = _preparar(destino)
    capacidades = sorted({c["capacity"] for c in celdas})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, eje, titulo in (
        (axes[0], "write", "Write mode (merge consolidation)"),
        (axes[1], "evict", "Eviction policy (selective forgetting)"),
    ):
        for i, cap in enumerate(capacidades):
            de_esta = sorted((c for c in celdas if c["capacity"] == cap), key=lambda c: c["r"])
            r = [c["r"] for c in de_esta]
            y = [c[f"eta2_{eje}"] for c in de_esta]
            lo = [c[f"eta2_{eje}_ci"][0] for c in de_esta]
            hi = [c[f"eta2_{eje}_ci"][1] for c in de_esta]

            color = COLORES[i % len(COLORES)]
            ax.plot(r, y, "o-", color=color, label=f"C = {cap}", linewidth=1.6, markersize=4)
            ax.fill_between(r, lo, hi, color=color, alpha=0.15, linewidth=0)

        ax.axvline(1.0, color=GRIS, linestyle="--", linewidth=1.2)
        ax.text(1.05, 0.94, "r = 1", color=GRIS, fontsize=9, transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_xlabel("prototypes per memory slot,  $r = K_{proto}/C$")
        ax.set_title(titulo, fontsize=10)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("score variance explained  ($\\eta^2$)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


def figura_comparacion_dominios(
    sintetico: list[dict[str, Any]], real: list[dict[str, Any]], destino: str | Path
) -> Path:
    """Figura 2: la ley del umbral en dominio sintético contra embeddings reales.

    Si el umbral se mueve al pasar a datos con estructura correlacionada y
    densidad no uniforme, la ley es una propiedad del generador sintético y no
    de la memoria. Es la objeción que el paper deja abierta en limitaciones.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ruta = _preparar(destino)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for celdas, etiqueta, color, marcador in (
        (sintetico, "synthetic (structured Gaussian)", COLORES[0], "o"),
        (real, "CIFAR-100 (ResNet-18)", COLORES[1], "s"),
    ):
        de_estas = sorted(celdas, key=lambda c: c["r"])
        ax.plot(
            [c["r"] for c in de_estas],
            [c["eta2_write"] - c["eta2_evict"] for c in de_estas],
            marcador + "-",
            color=color,
            label=etiqueta,
            linewidth=1.6,
            markersize=5,
        )

    ax.axhline(0.0, color=GRIS, linewidth=1.0)
    ax.axvline(1.0, color=GRIS, linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("$r = \\hat{K}_{proto}/C$")
    ax.set_ylabel("$\\eta^2$ write $-$ $\\eta^2$ evict")
    ax.set_title("Where the regime transition falls in each domain", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


def figura_saliencia_imperfecta(datos: dict[str, Any], destino: str | Path) -> Path:
    """Figura de `exp08`: retención contra la calidad de la señal de saliencia.

    El eje x es el AUC empírico de la sorpresa como clasificador de importancia,
    que pone a todos los modos de degradación en una escala común. `datos` es
    `settings` del JSON: un bloque por régimen de flujo (`main`, r = 0.25, y
    `selection`, r = 2). Las curvas con banda (IC 95 % sobre semillas) son el
    eje de solapamiento; los marcadores sueltos del primer panel son la frontera
    en los otros ejes, para ver si a igual AUC dan lo mismo.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ruta = _preparar(destino)
    series = (
        ("frontier", "frontier (surprise + novelty)", COLORES[0], "o", "-"),
        ("frontier_pe", "frontier, surprise only", COLORES[0], "o", "--"),
        ("no_salience_best", "best without salience", COLORES[2], "s", "-"),
        ("fifo", "FIFO proxy", COLORES[1], "v", "-"),
        ("oracle", "oracle (labels)", GRIS, "", ":"),
    )
    otros_ejes = (("noise", "D"), ("delay_partial", "^"), ("delay_lag", ">"), ("misleading", "p"))

    paneles = [
        ("main", "overlap", "recall", "r = 0.25 · retention recall"),
        ("main", "overlap_distractors", "recall", "r = 0.25, novel distractors · recall"),
        ("selection", "overlap", "recall", "r = 2 · retention recall"),
        ("selection", "overlap", "precision", "r = 2 · retention precision"),
    ]
    paneles = [p for p in paneles if p[0] in datos and p[1] in datos[p[0]]["axes"]]

    fig, axes = plt.subplots(1, len(paneles), figsize=(3.6 * len(paneles), 3.9), sharey=True)
    axes = list(np.atleast_1d(axes))
    for ax, (reg, eje, metrica, titulo) in zip(axes, paneles, strict=True):
        conds = datos[reg]["conditions"]
        labels = datos[reg]["axes"][eje]
        x = [conds[lab]["auc"]["mean"] for lab in labels]
        for clave, nombre, color, marcador, estilo in series:
            fila = [conds[lab]["policies"][clave][metrica] for lab in labels]
            ax.plot(
                x,
                [f["mean"] for f in fila],
                marcador + estilo,
                color=color,
                label=nombre,
                lw=1.5,
                ms=3.5,
            )
            ax.fill_between(
                x,
                [f["ci"][0] for f in fila],
                [f["ci"][1] for f in fila],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        if metrica == "precision":
            ax.plot(
                x,
                [conds[lab]["prevalence"] for lab in labels],
                color=GRIS,
                lw=1.0,
                ls="-.",
                label="prevalence",
            )
        if reg == "main" and eje == "overlap":
            for otro, marcador in otros_ejes:
                if otro not in datos[reg]["axes"]:
                    continue
                ls = datos[reg]["axes"][otro][1:]
                ax.scatter(
                    [conds[lab]["auc"]["mean"] for lab in ls],
                    [conds[lab]["policies"]["frontier"]["recall"]["mean"] for lab in ls],
                    marker=marcador,
                    facecolors="none",
                    edgecolors=COLORES[3],
                    s=20,
                    label=f"frontier, {otro.replace('_', ' ')}",
                )
        ax.set_xlim(0.42, 1.02)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("empirical AUC of surprise")
        ax.set_title(titulo, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("fraction of important (recall) / retained (precision)", fontsize=8)
    manejadores, nombres = axes[0].get_legend_handles_labels()
    fig.legend(
        manejadores,
        nombres,
        fontsize=7,
        frameon=False,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta
