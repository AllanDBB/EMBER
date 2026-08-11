"""Embeddings perceptuales reales como fuente de flujo de experiencia.

La sección de limitaciones del paper dice que todos los experimentos corren sobre
vectores gaussianos aleatorios, que en R^32 son casi ortogonales, y que eso
probablemente sobreestima la precisión de recuperación frente a embeddings
perceptuales reales — los cuales exhiben estructura correlacionada, densidad de
clúster no uniforme y desplazamiento distribucional.

Este módulo cierra esa objeción sin depender del robot: extrae embeddings de
CIFAR-100 con un ResNet-18 preentrenado, los reduce por PCA a la dimensión de
trabajo y los normaliza a la esfera. Un flujo se construye tomando algunas clases
como experiencia común recurrente y reservando otras como eventos raros.

torch se importa **dentro** de la función de extracción, nunca a nivel de módulo:
`embedding_stream` opera sobre un banco ya cacheado y solo necesita numpy, de
modo que el resto del laboratorio no arrastra el stack pesado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ember.core.types import unit_rows
from ember.data.streams import Stream, StreamItem, StreamSpec
from ember.data.synthetic import PE_COMUN, PE_RARO, _pe, _pesos_de_visita


@dataclass(frozen=True, slots=True)
class EmbeddingBank:
    """Embeddings etiquetados de los que se pueden muestrear flujos."""

    vectors: NDArray[np.float32]
    labels: NDArray[np.int64]
    label_names: list[str]
    source: str = "cifar100"

    def __post_init__(self) -> None:
        if len(self.vectors) != len(self.labels):
            raise ValueError("vectors y labels deben tener el mismo largo")

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    def indices_por_clase(self) -> dict[int, NDArray[np.intp]]:
        return {c: np.where(self.labels == c)[0] for c in np.unique(self.labels)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            labels=self.labels,
            label_names=np.array(self.label_names, dtype=object),
            source=self.source,
        )

    @classmethod
    def load(cls, path: Path) -> EmbeddingBank:
        d = np.load(path, allow_pickle=True)
        return cls(
            vectors=d["vectors"].astype(np.float32),
            labels=d["labels"].astype(np.int64),
            label_names=list(d["label_names"]),
            source=str(d["source"]),
        )


def _pca(X: NDArray[np.float32], dim: int, seed: int = 0) -> NDArray[np.float32]:
    """Reduce a `dim` componentes principales. numpy puro."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD sobre la matriz centrada: las componentes principales son V.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:dim].T).astype(np.float32)


def extract_cifar100_embeddings(
    cache_dir: Path,
    *,
    dim: int = 32,
    device: str = "auto",
    batch_size: int = 256,
    seed: int = 0,
) -> EmbeddingBank:
    """Extrae embeddings de CIFAR-100 con un ResNet-18 preentrenado.

    Requiere el extra `lab`. El resultado se cachea: la segunda llamada no
    descarga ni recalcula nada.
    """
    destino = cache_dir / f"cifar100_resnet18_d{dim}.npz"
    if destino.exists():
        return EmbeddingBank.load(destino)

    import torch
    import torchvision
    from torch.utils.data import DataLoader

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    pesos = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    modelo = torchvision.models.resnet18(weights=pesos)
    modelo.fc = torch.nn.Identity()  # nos quedamos con la penúltima capa
    modelo.eval().to(device)

    dataset = torchvision.datasets.CIFAR100(
        root=str(cache_dir / "torchvision"),
        train=True,
        download=True,
        transform=pesos.transforms(),
    )
    cargador = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    trozos, etiquetas = [], []
    with torch.no_grad():
        for x, y in cargador:
            trozos.append(modelo(x.to(device)).cpu().numpy())
            etiquetas.append(y.numpy())

    crudos = np.concatenate(trozos).astype(np.float32)
    banco = EmbeddingBank(
        vectors=unit_rows(_pca(crudos, dim, seed=seed)),
        labels=np.concatenate(etiquetas).astype(np.int64),
        label_names=list(dataset.classes),
    )
    banco.save(destino)
    return banco


def embedding_stream(
    bank: EmbeddingBank,
    n_prototypes: int,
    capacity: int,
    *,
    n_common: int = 300,
    n_rare: int = 20,
    density_skew: float = 0.0,
    seed: int = 0,
) -> Stream:
    """Construye un flujo de experiencia a partir de embeddings reales.

    `n_prototypes` clases hacen de experiencia común recurrente; otras clases,
    disjuntas de esas, aportan los eventos raros. Cada visita a un prototipo
    muestrea una imagen distinta de esa clase, lo que da la variación intra-clase
    que un generador sintético tiene que simular con ruido.
    """
    rng = np.random.default_rng(seed)
    por_clase = bank.indices_por_clase()
    clases = np.array(sorted(por_clase))

    if n_prototypes > len(clases):
        raise ValueError(
            f"se pidieron {n_prototypes} prototipos pero el banco tiene {len(clases)} clases"
        )

    orden = rng.permutation(clases)
    comunes = orden[:n_prototypes]
    disponibles_raras = orden[n_prototypes:]
    if n_rare > 0 and len(disponibles_raras) == 0:
        raise ValueError("no quedan clases disjuntas para los eventos raros")

    pesos = _pesos_de_visita(n_prototypes, density_skew)

    items: list[StreamItem] = []
    for _ in range(n_common):
        c = int(comunes[rng.choice(n_prototypes, p=pesos)])
        i = int(rng.choice(por_clase[c]))
        items.append(
            StreamItem(
                key=bank.vectors[i],
                value=int(c),
                pred_error=_pe(rng, PE_COMUN),
                is_rare=False,
            )
        )

    raros: list[StreamItem] = []
    if n_rare > 0:
        posiciones = sorted(rng.choice(n_common + n_rare, size=n_rare, replace=False))
        for j, pos in enumerate(posiciones):
            c = int(disponibles_raras[j % len(disponibles_raras)])
            i = int(rng.choice(por_clase[c]))
            it = StreamItem(
                key=bank.vectors[i],
                value=f"rare{j}",
                pred_error=_pe(rng, PE_RARO),
                is_rare=True,
            )
            items.insert(int(pos), it)
            raros.append(it)

    return Stream(
        items=items,
        spec=StreamSpec(
            n_prototypes=n_prototypes,
            capacity=capacity,
            dim=bank.dim,
            source=bank.source,
        ),
        rare_items=raros,
    )
