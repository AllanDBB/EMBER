"""El genotipo: una arquitectura de memoria como tupla de políticas.

Un genotipo es un punto del espacio de diseño del NAS. La afirmación central del
paper —que el `EpisodicBuffer` FIFO de e-MDB no es un baseline externo sino un
punto del mismo espacio que se busca— es literal aquí: `FIFO_GENOTYPE` es un
`Genotype` como cualquier otro.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ember.core.policies import (
    FIFO,
    Append,
    Constant,
    DecayPolicy,
    EvictPolicy,
    NearestNeighbour,
    NoDecay,
    ReadPolicy,
    StrengthPolicy,
    WritePolicy,
)


@dataclass(frozen=True, slots=True)
class Genotype:
    """Una arquitectura de memoria descrita por sus seis ejes de diseño."""

    read: ReadPolicy
    write: WritePolicy
    strength: StrengthPolicy
    decay: DecayPolicy
    evict: EvictPolicy
    reinforce: float

    AXES = ("read", "write", "strength", "decay", "evict", "reinforce")

    def as_dict(self) -> dict[str, str]:
        """Etiquetas cortas por eje, para agrupar y tabular en el análisis."""
        return {
            "read": self.read.label,
            "write": self.write.label,
            "strength": self.strength.label,
            "decay": self.decay.label,
            "evict": self.evict.label,
            "reinforce": str(self.reinforce),
        }

    def label(self) -> str:
        """Identificador estable y ordenable del genotipo."""
        return "|".join(f"{k}={v}" for k, v in self.as_dict().items())

    def axis(self, name: str) -> str:
        """Etiqueta de un eje por nombre. Útil para el análisis de varianza."""
        return self.as_dict()[name]

    def with_axis(self, name: str, value: object) -> Genotype:
        """Copia con un eje reemplazado. Sirve para construir genotipos hermanos."""
        return replace(self, **{name: value})

    def __str__(self) -> str:
        return self.label()


FIFO_GENOTYPE = Genotype(
    read=NearestNeighbour(),
    write=Append(),
    strength=Constant(),
    decay=NoDecay(),
    evict=FIFO(),
    reinforce=0.0,
)
"""El `EpisodicBuffer` de e-MDB expresado como genotipo.

El mapeo es deliberadamente generoso: el `EpisodicBuffer` real es un
`collections.deque` sin recuperación por contenido de ningún tipo, y
`read=nn` le regala una búsqueda por similitud que no tiene. Aun con esa
ventaja queda en el piso de su propio espacio de diseño, lo que desarma la
objeción de que se lo comparó contra un hombre de paja.
"""
