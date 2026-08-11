"""Experimentos reproducibles de EMBER.

Cada script se corre con `uv run python -m experiments.expNN_nombre` y escribe a
`results/expNN_nombre/` un `data.json` con los resultados y un `manifest.json`
con la procedencia (SHA de git, semillas, versiones, timestamp).

| Script                  | Qué produce                                        |
|-------------------------|----------------------------------------------------|
| `exp01_nas_full`        | Búsqueda exhaustiva; efectos principales           |
| `exp02_threshold_grid`  | La ley del umbral y su figura                      |
| `exp03_axis_liveness`   | Auditoría de observabilidad de ejes; corre en CI   |
| `exp04_arch_benchmark`  | Benchmark de arquitecturas en dos fases            |
| `exp05_real_embeddings` | La ley del umbral sobre embeddings de CIFAR-100    |
"""
