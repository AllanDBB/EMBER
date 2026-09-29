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


def figura_sensibilidad_fusion(
    resultado: dict[str, Any], destino: str | Path, *, r_diagnostico: float = 1.0
) -> Path:
    """Figura de `exp07`: η² del eje de escritura contra el umbral de fusión.

    Fila superior: un panel por dominio, una curva por ratio `r` (tono único de
    claro a oscuro, porque `r` es una magnitud ordenada) con su banda de IC
    bootstrap sobre semillas. La línea vertical marca el umbral por defecto.

    Fila inferior: qué hace la fusión a ese umbral en `r = r_diagnostico` —
    tasa de fusión, pureza (fracción de fusiones que unen el mismo prototipo o
    clase) y fracción de eventos raros absorbidos. Es lo que distingue "la
    escritura no importa porque la fusión no se dispara" de "no importa porque
    fusiona lo que no debe".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ruta = _preparar(destino)
    dominios = [d for d in ("synthetic", "cifar100") if d in resultado]
    titulos = {"synthetic": "Synthetic (Gaussian prototypes)", "cifar100": "CIFAR-100 (ResNet-18)"}

    fig, axes = plt.subplots(
        2, len(dominios), figsize=(5.2 * len(dominios), 6.4), sharex=True, squeeze=False
    )
    for j, d in enumerate(dominios):
        umbrales = sorted(resultado[d]["umbrales"].values(), key=lambda u: u["umbral"])
        ts = [u["umbral"] for u in umbrales]
        claves = list(umbrales[0]["celdas"])
        ratios = [umbrales[0]["celdas"][k]["ratio"] for k in claves]
        cmap = plt.get_cmap("Blues")
        ax = axes[0, j]
        for i, (k, r) in enumerate(zip(claves, ratios, strict=True)):
            color = cmap(0.35 + 0.6 * i / max(len(claves) - 1, 1))
            y = [u["celdas"][k]["eta2_write"] for u in umbrales]
            lo = [u["celdas"][k]["eta2_write_ci"][0] for u in umbrales]
            hi = [u["celdas"][k]["eta2_write_ci"][1] for u in umbrales]
            ax.plot(ts, y, "o-", color=color, label=f"r = {r:g}", linewidth=1.6, markersize=4)
            ax.fill_between(ts, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.set_title(titulos.get(d, d), fontsize=10)
        ax.set_ylim(-0.02, 1.0)
        ax.grid(alpha=0.25)

        ax2 = axes[1, j]
        k_diag = min(claves, key=lambda k: abs(umbrales[0]["celdas"][k]["ratio"] - r_diagnostico))
        celdas = [u["celdas"][k_diag] for u in umbrales]
        ax2.plot(
            ts,
            [c["tasa_fusion"]["media"] for c in celdas],
            "o-",
            color=COLORES[0],
            label="merge rate",
            linewidth=1.6,
            markersize=4,
        )
        ax2.plot(
            ts,
            [
                c["pureza_fusion"] if c["pureza_fusion"] is not None else float("nan")
                for c in celdas
            ],
            "s--",
            color=COLORES[1],
            label="merge purity",
            linewidth=1.6,
            markersize=4,
        )
        ax2.plot(
            ts,
            [c["raros_absorbidos"]["media"] for c in celdas],
            "^:",
            color=COLORES[2],
            label="rare events absorbed",
            linewidth=1.6,
            markersize=4,
        )
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_xlabel("merge threshold (cosine)")
        ax2.grid(alpha=0.25)
        ax2.text(
            0.02,
            0.04,
            f"r = {celdas[0]['ratio']:g}",
            transform=ax2.transAxes,
            fontsize=8,
            color=GRIS,
        )

        for a in (ax, ax2):
            a.axvline(0.85, color=GRIS, linestyle="--", linewidth=1.0)

    axes[0, 0].set_ylabel("write-axis $\\eta^2$")
    axes[1, 0].set_ylabel("fraction")
    axes[0, -1].legend(fontsize=7, frameon=False, ncol=2)
    axes[1, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta
