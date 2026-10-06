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
        "frontera_decay0.995": ("frontier, decay 0.995", COLORES[0], "--"),
        "frontera_decay0.98": ("frontier, decay 0.98", COLORES[0], ":"),
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
    for i, (cond, (etiqueta, color, linea)) in enumerate(estilos.items()):
        if cond not in resumen[f"C{capacidades[0]}"]:
            continue
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
            hatch={"-": None, "--": "//", ":": ".."}[linea],
            edgecolor="white" if linea != "-" else None,
        )
    bx.set_xticks(np.arange(len(capacidades)), [f"C = {c}" for c in capacidades])
    bx.set_ylabel("return, first episodes of a reappearing task")
    bx.set_title("Functional retention", fontsize=10)
    bx.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


def figura_cruces_por_familia(
    parte_a: dict[str, Any], destino: str | Path, *, alternativa: str | None = None
) -> Path:
    """Figura de `exp10`: dónde cruza la dominancia en cada familia de flujos.

    Dos paneles con las mismas familias: el cruce expresado en `r` nominal y en
    la medida de presión alternativa que mejor lo colapsa. Si `r` nominal fuera
    la variable que gobierna, los puntos del panel izquierdo tendrían que caer
    todos sobre una misma vertical; las familias sin cruce se marcan en el
    borde con una flecha.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    ruta = _preparar(destino)
    colapso = parte_a["colapso"]
    if alternativa is None:
        candidatas = [
            v for v in parte_a["variables"] if v != "r_nom" and colapso[v]["sd_log"] is not None
        ]
        alternativa = min(candidatas, key=lambda v: colapso[v]["sd_log"])

    etiquetas_var = {
        "r_nom": "$K/C$  (nominal)",
        "r_eff": "$K_{eff}/C$",
        "r_hill": "$e^{H}/C$",
        "r_masa": "$e^{H}\\,(K_{eff}/K_{seen})/C$",
        "r_carga_nom": "$(K + n_{rare})/C$",
        "r_carga_eff": "$(K_{eff} + n_{rare})/C$",
    }
    grupos = []
    for nombre in parte_a["orden"]:
        g = parte_a["familias"][nombre]["grupo"]
        if g not in grupos:
            grupos.append(g)
    color_de = {
        g: (COLORES + (GRIS, "#A23B72", "#3B8EA5", "#6D6875"))[i] for i, g in enumerate(grupos)
    }

    nombres = list(parte_a["orden"])
    y = list(range(len(nombres)))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.28 * len(nombres) + 1.6), sharey=True)
    for ax, var in ((axes[0], "r_nom"), (axes[1], alternativa)):
        xs = [
            c for n in nombres if (c := parte_a["familias"][n]["cruce"][var]["x_cruce"]) is not None
        ]
        lo_ax = min(xs) / 2 if xs else 0.1
        hi_ax = max(xs) * 2 if xs else 10
        for yi, n in zip(y, nombres, strict=True):
            fam = parte_a["familias"][n]
            cr = fam["cruce"][var]
            color = color_de[fam["grupo"]]
            if cr["x_cruce"] is None:
                borde = lo_ax if cr["estado"] == "siempre_desalojo" else hi_ax
                ax.plot([borde], [yi], marker="<" if borde == lo_ax else ">", color=color)
                continue
            if cr.get("ci"):
                ax.plot(cr["ci"], [yi, yi], color=color, linewidth=1.4, alpha=0.7)
            ax.plot([cr["x_cruce"]], [yi], "o", color=color, markersize=4.5)
        med = colapso[var]["cruce_mediano"]
        if med:
            ax.axvline(med, color=GRIS, linestyle=":", linewidth=1.0)
        ax.axvline(1.0, color=GRIS, linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlim(lo_ax / 1.2, hi_ax * 1.2)
        marcas = [
            m for m in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0) if lo_ax / 1.2 <= m <= hi_ax * 1.2
        ]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(marcas))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{m:g}" for m in marcas]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        sd = colapso[var]["sd_log"]
        ax.set_title(f"{etiquetas_var.get(var, var)}   (sd log = {sd:.2f})", fontsize=10)
        ax.set_xlabel("write/evict dominance crossing")
        ax.grid(alpha=0.25, axis="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(nombres, fontsize=7)
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
