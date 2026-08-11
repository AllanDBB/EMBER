"""Infraestructura de experimentos reproducibles.

Todo número que llega al paper sale de un `ExperimentRun`, y todo
`ExperimentRun` deja junto a sus datos un manifiesto con el SHA de git, si el
árbol estaba sucio, las semillas, las versiones de las dependencias y el momento
de la corrida. Sin eso, un número en una tabla no se puede volver a producir.

El manifiesto se escribe **al final y solo si el experimento terminó**: un
resultado a medias es peor que ninguno, porque parece completo.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import Any

import ember

PAQUETES_DE_INTERES = ("numpy", "torch", "scipy", "matplotlib", "gymnasium", "minigrid")


def _git(*args: str) -> str | None:
    try:
        salida = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return salida.stdout.strip() if salida.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versiones() -> dict[str, str]:
    encontradas = {}
    for nombre in PAQUETES_DE_INTERES:
        try:
            encontradas[nombre] = metadata.version(nombre)
        except metadata.PackageNotFoundError:
            continue
    return encontradas


class ExperimentRun:
    """Contexto que recoge resultados y escribe datos más manifiesto de procedencia.

    with ExperimentRun("exp01_nas_full") as run:
        run.set_seeds([0, 1, 2])
        run.record("main_effects", tabla)
    """

    def __init__(self, name: str, *, results_dir: Path | str = "results") -> None:
        self.name = name
        self.dir = Path(results_dir) / name
        self.data: dict[str, Any] = {}
        self.seeds: list[int] = []
        self.notes: list[str] = []
        self._t0 = 0.0

    # ------------------------------------------------------------ recolección

    def record(self, key: str, value: Any) -> None:
        """Guarda un resultado bajo una clave."""
        self.data[key] = value

    def set_seeds(self, seeds: object) -> None:
        """Registra las semillas usadas. Van al manifiesto, no a los datos."""
        self.seeds = [int(s) for s in seeds]  # type: ignore[union-attr]

    def note(self, texto: str) -> None:
        """Anota algo que quien lea los resultados después necesita saber."""
        self.notes.append(texto)

    def log(self, mensaje: str) -> None:
        print(f"[{self.name}] {mensaje}", flush=True)

    # -------------------------------------------------------- ciclo de vida

    def __enter__(self) -> ExperimentRun:
        self._t0 = time.time()
        self.log("iniciando")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            # Un resultado parcial es peor que ninguno: parece completo.
            self.log(f"abortado por {exc_type.__name__}; no se escribe nada")
            return False

        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "data.json").write_text(
            json.dumps(self.data, indent=1, default=str), encoding="utf-8"
        )
        (self.dir / "manifest.json").write_text(
            json.dumps(self.manifest(), indent=1), encoding="utf-8"
        )
        self.log(f"listo en {time.time() - self._t0:.1f} s → {self.dir}")
        return False

    def manifest(self) -> dict[str, Any]:
        sucio = _git("status", "--porcelain")
        return {
            "experiment": self.name,
            "ember_version": ember.__version__,
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(sucio) if sucio is not None else None,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "duration_s": round(time.time() - self._t0, 2),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _versiones(),
            "seeds": self.seeds,
            "notes": self.notes,
        }


def load_results(name: str, *, results_dir: Path | str = "results") -> dict[str, Any]:
    """Lee los datos de un experimento ya corrido."""
    ruta = Path(results_dir) / name / "data.json"
    if not ruta.exists():
        raise FileNotFoundError(f"{name} no fue corrido todavía: falta {ruta}")
    return json.loads(ruta.read_text(encoding="utf-8"))
