"""Fuentes de flujo de experiencia para evaluar memorias.

Dos dominios, la misma interfaz `Stream`:

- `synthetic` — control exacto sobre `n_prototypes`, que es la variable
  independiente de la ley del umbral, más perillas de correlación, sesgo de
  densidad y deriva.
- `embeddings` — embeddings perceptuales reales de CIFAR-100, donde
  `n_prototypes` deja de conocerse y hay que estimarlo con `prototypes`.
"""

from ember.data.embeddings import EmbeddingBank, embedding_stream, extract_cifar100_embeddings
from ember.data.prototypes import PrototypeEstimate, estimate_n_prototypes
from ember.data.streams import Stream, StreamItem, StreamSpec
from ember.data.synthetic import block_stream, clustered_stream

__all__ = [
    "EmbeddingBank",
    "PrototypeEstimate",
    "Stream",
    "StreamItem",
    "StreamSpec",
    "block_stream",
    "clustered_stream",
    "embedding_stream",
    "estimate_n_prototypes",
    "extract_cifar100_embeddings",
]
