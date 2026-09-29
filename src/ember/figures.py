"""Figuras del paper, generadas desde los resultados de los experimentos.

matplotlib se importa dentro de cada función: este módulo se puede importar
desde cualquier lado sin arrastrar el stack de laboratorio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def figura_aprendizaje_downstream(
    resumen: dict[str, Any], destino: str | Path, *, capacidad: int, n_tareas: int = 4
) -> Path:
    """Figura de exp13: desempeño a lo largo del flujo de por vida, por memoria.

    Panel izquierdo: retorno medio por fase (IC95 sobre semillas) a la capacidad
    principal; las líneas verticales separan los ciclos, así que lo que pasa
    justo después de cada línea es la reaparición de una tarea ya vista.
    Panel derecho: retorno en los primeros episodios de una tarea que reaparece,
    por capacidad. Es la pregunta de si retener mejor ayuda a actuar mejor.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ruta = _preparar(destino)
    estilos = {
        "frontera": ("EMBER frontier", COLORES[0], "-"),
        "FIFO": ("FIFO (e-MDB buffer)", COLORES[1], "-"),
        "reservorio": ("reservoir sampling", COLORES[2], "-"),
        "sin_saliencia": ("best without salience", COLORES[3], "-"),
        "sin_limite": ("unbounded (ceiling)", GRIS, "--"),
    }
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})

    bloque = resumen[f"C{capacidad}"]
    for cond, (etiqueta, color, linea) in estilos.items():
        if cond not in bloque:
            continue
        c = bloque[cond]["curva_por_fase"]
        x = np.arange(1, len(c["media"]) + 1)
        ax.plot(
            x,
            c["media"],
            linea,
            color=color,
            label=etiqueta,
            linewidth=1.6,
            marker="o",
            markersize=3,
        )
        ax.fill_between(x, c["ci_low"], c["ci_high"], color=color, alpha=0.15, linewidth=0)
    n_fases = len(bloque["FIFO"]["curva_por_fase"]["media"])
    for borde in range(n_tareas, n_fases, n_tareas):
        ax.axvline(borde + 0.5, color=GRIS, linestyle=":", linewidth=1.0)
    ax.set_xlabel("phase (tasks A B C D, repeated)")
    ax.set_ylabel("mean episode return")
    ax.set_title(f"Lifelong stream, capacity C = {capacidad}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)

    capacidades = sorted(int(k[1:]) for k in resumen)
    ancho = 0.8 / len(estilos)
    for i, (cond, (etiqueta, color, _)) in enumerate(estilos.items()):
        medias, errs = [], [[], []]
        for cap in capacidades:
            d = resumen[f"C{cap}"][cond]["reaparicion"]
            medias.append(d["media"])
            errs[0].append(d["media"] - d["ci_low"])
            errs[1].append(d["ci_high"] - d["media"])
        pos = np.arange(len(capacidades)) + (i - (len(estilos) - 1) / 2) * ancho
        bx.bar(
            pos,
            medias,
            ancho,
            color=color,
            yerr=errs,
            capsize=2,
            label=etiqueta,
            error_kw={"linewidth": 0.8},
        )
    bx.set_xticks(np.arange(len(capacidades)), [f"C = {c}" for c in capacidades])
    bx.set_ylabel("return, first episodes of a reappearing task")
    bx.set_title("Functional retention", fontsize=10)
    bx.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta
