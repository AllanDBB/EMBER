"""Sincronización entre el paper y los resultados medidos.

El problema que resuelve: un número entra al LaTeX, después se corrige un bug o
se rediseña una tarea, los resultados cambian, y el número del paper se queda
donde estaba. No hay nada que avise. Eso ya pasó una vez en este proyecto — el
draft afirmaba cosas que el código no producía.

La solución es un macro que hace explícita la procedencia de cada número:

    El modo de escritura explica \\result{exp01_nas_full:main_effects.write.eta2|pct}{56.6}\\%

El macro renderiza el segundo argumento —así el PDF compila sin correr nada— y
`verify_paper` compara ese valor contra el JSON del experimento. Si no coinciden,
falla y dice cuál.

Sintaxis de la clave
--------------------
    experimento:ruta.punteada[|formato]

`ruta.punteada` navega el `data.json`; los índices de lista se escriben como
números. Formatos: `pct` (×100, un decimal), `pct2`, `f2`, `f3` (el que se usa
por defecto), `int`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MACRO = re.compile(r"\\result\{([^}]+)\}\{([^}]*)\}")

FORMATOS = {
    "pct": lambda v: f"{100 * v:.1f}",
    "pct2": lambda v: f"{100 * v:.2f}",
    "f2": lambda v: f"{v:.2f}",
    "f3": lambda v: f"{v:.3f}",
    "int": lambda v: f"{int(round(v))}",
}
FORMATO_POR_DEFECTO = "f3"

PREAMBULO = r"""% Generado por ember.paper_sync. No editar a mano.
%
% \result{clave}{valor} imprime el valor y deja registrada su procedencia.
% `ember.paper_sync.verify_paper` verifica que el valor coincida con lo medido
% en results/. Un número sin macro es un número que nadie está verificando.
\newcommand{\result}[2]{#2}
"""


@dataclass(frozen=True, slots=True)
class Discrepancia:
    """Un número publicado que ya no coincide con lo medido."""

    clave: str
    publicado: float | None
    medido: float | None
    motivo: str
    linea: int

    def __str__(self) -> str:
        return (
            f"línea {self.linea}: {self.clave}\n"
            f"    publicado: {self.publicado}\n"
            f"    medido:    {self.medido}\n"
            f"    {self.motivo}"
        )


def _navegar(datos: Any, ruta: str) -> Any:
    actual = datos
    for parte in ruta.split("."):
        if isinstance(actual, list):
            actual = actual[int(parte)]
        elif isinstance(actual, dict):
            if parte not in actual:
                raise KeyError(parte)
            actual = actual[parte]
        else:
            raise KeyError(parte)
    return actual


def resolver(clave: str, results_dir: Path | str = "results") -> tuple[float, str]:
    """Devuelve `(valor_medido, formato)` para una clave de `\\result`."""
    cuerpo, _, formato = clave.partition("|")
    formato = formato or FORMATO_POR_DEFECTO
    if formato not in FORMATOS:
        raise ValueError(f"formato desconocido '{formato}'; hay {sorted(FORMATOS)}")

    experimento, _, ruta = cuerpo.partition(":")
    archivo = Path(results_dir) / experimento / "data.json"
    if not archivo.exists():
        raise FileNotFoundError(f"{experimento} no fue corrido: falta {archivo}")

    valor = _navegar(json.loads(archivo.read_text(encoding="utf-8")), ruta)
    return float(valor), formato


def _sin_comentarios(texto: str) -> str:
    """Blanquea los comentarios de LaTeX conservando el número de línea.

    Un `\\result` de ejemplo dentro de un comentario no es un número publicado, y
    verificarlo daría un falso positivo. Se reemplaza por espacios en vez de
    borrarse para que los offsets —y por tanto las líneas que se reportan— no se
    corran.
    """
    salida = []
    for linea in texto.split("\n"):
        corte, i, escapado = len(linea), 0, False
        while i < len(linea):
            if escapado:
                escapado = False
            elif linea[i] == "\\":
                escapado = True
            elif linea[i] == "%":
                corte = i
                break
            i += 1
        salida.append(linea[:corte] + " " * (len(linea) - corte))
    return "\n".join(salida)


def verify_paper(tex_path: Path | str, results_dir: Path | str = "results") -> list[Discrepancia]:
    """Compara cada `\\result` del LaTeX contra el JSON del experimento.

    Devuelve la lista de discrepancias; vacía significa que todo número
    publicado coincide con lo que el código produce hoy.
    """
    texto = _sin_comentarios(Path(tex_path).read_text(encoding="utf-8"))
    discrepancias: list[Discrepancia] = []

    for m in MACRO.finditer(texto):
        clave, publicado_txt = m.group(1), m.group(2).strip()
        linea = texto.count("\n", 0, m.start()) + 1

        try:
            medido, formato = resolver(clave, results_dir)
        except (FileNotFoundError, KeyError, ValueError, IndexError) as e:
            discrepancias.append(
                Discrepancia(clave, None, None, f"no se pudo resolver: {e}", linea)
            )
            continue

        esperado = FORMATOS[formato](medido)
        if publicado_txt != esperado:
            try:
                publicado = float(publicado_txt)
            except ValueError:
                publicado = None
            discrepancias.append(
                Discrepancia(
                    clave,
                    publicado,
                    float(esperado),
                    f"el paper dice '{publicado_txt}' y la medición da '{esperado}'",
                    linea,
                )
            )

    return discrepancias


# ═════════════════════════════════════════════════════ generación de tablas


def _tabular(
    encabezados: list[str], filas: list[list[str]], alineacion: str, caption: str, label: str
) -> str:
    lineas = [
        "% Generado por ember.paper_sync. No editar a mano.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alineacion}}}",
        "\\hline",
        " & ".join(encabezados) + " \\\\",
        "\\hline",
    ]
    lineas += [" & ".join(f) + " \\\\" for f in filas]
    lineas += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lineas)


def tabla_efectos_principales(datos: dict[str, Any]) -> str:
    """Efectos principales por eje, con su observabilidad."""
    filas = []
    for f in datos["main_effects"]:
        nota = "inobservable" if f["liveness"] == 0.0 else ""
        filas.append(
            [
                f["axis"].replace("_", "\\_"),
                f"{100 * f['eta2']:.1f}\\,\\%",
                f"{f['liveness']:.3f}",
                nota,
            ]
        )
    return _tabular(
        ["Eje", "$\\eta^2$", "Observabilidad", "Nota"],
        filas,
        "lrrl",
        "Efectos principales sobre el espacio completo. La columna de "
        "observabilidad es la máxima diferencia entre genotipos hermanos: un eje "
        "con $\\eta^2$ nulo y observabilidad nula no está siendo medido.",
        "tab:main-effects",
    )


def tabla_regimenes(datos: dict[str, Any]) -> str:
    """Varianza explicada por régimen del ratio."""
    filas = [
        [
            nombre.replace("_", "\\_"),
            str(d["n_celdas"]),
            f"{100 * d['eta2_write']:.1f}\\,\\%",
            f"{100 * d['eta2_evict']:.1f}\\,\\%",
        ]
        for nombre, d in datos["regimenes"].items()
        if d["eta2_write"] is not None
    ]
    return _tabular(
        ["Régimen", "Celdas", "$\\eta^2$ escritura", "$\\eta^2$ desalojo"],
        filas,
        "lrrr",
        "Transición de régimen según el cociente prototipos-a-capacidad $r = K_{proto}/C$.",
        "tab:regimes",
    )


def tabla_benchmark(datos: dict[str, Any]) -> str:
    """Gate de reconstrucción y batería, por arquitectura."""
    filas = []
    for nombre, d in datos["architectures"].items():
        g = d["gate"]["per_task"]
        fila = [
            nombre,
            f"{g['pattern_completion']:.3f}",
            f"{g['noise_robustness']:.3f}",
            f"{g['ab_interference']:.3f}",
            f"{g['capacity_profile']:.3f}",
            f"{d['gate']['mean']:.3f}",
            "\\checkmark" if d["gate"]["passes"] else "$\\times$",
        ]
        if "battery" in d:
            b = d["battery"]["per_task"]
            fila += [
                f"{b['rare_retention']:.3f}",
                f"{b['noise_under_pressure']:.3f}",
                f"{b['sequential_interference']:.3f}",
            ]
        else:
            fila += ["--", "--", "--"]
        filas.append(fila)

    return _tabular(
        ["Arq.", "R1", "R2", "R3", "R4", "Media", "Gate", "T1", "T2", "T3"],
        filas,
        "lrrrrrcrrr",
        "Evaluación en dos fases. R1--R4 miden reconstrucción sin presión de "
        "capacidad y deciden la admisión; T1--T3 miden retención bajo presión.",
        "tab:benchmark",
    )


GENERADORES = {
    "exp01_nas_full": [("tab_main_effects", tabla_efectos_principales)],
    "exp02_threshold_grid": [("tab_regimes", tabla_regimenes)],
    "exp04_arch_benchmark": [("tab_benchmark", tabla_benchmark)],
}


def render_tables(
    results_dir: Path | str = "results", out_dir: Path | str = "paper/tables"
) -> list[Path]:
    """Genera los `.tex` de las tablas desde los resultados disponibles.

    Los experimentos que todavía no se corrieron se saltan en silencio: es
    normal tener el paper a medias mientras se produce la evidencia.
    """
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result_macro.tex").write_text(PREAMBULO, encoding="utf-8")

    generadas: list[Path] = []
    for experimento, tablas in GENERADORES.items():
        archivo = results_dir / experimento / "data.json"
        if not archivo.exists():
            continue
        datos = json.loads(archivo.read_text(encoding="utf-8"))
        for nombre, generador in tablas:
            destino = out_dir / f"{nombre}.tex"
            destino.write_text(generador(datos), encoding="utf-8")
            generadas.append(destino)

    return generadas
