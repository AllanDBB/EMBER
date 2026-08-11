"""EMBER — Emergent Memory-Based Encoding and Reactivation.

Memoria episódica de largo plazo bioinspirada para robots cognitivos.

El paquete se divide en dos mitades con dependencias distintas:

- El **núcleo** (`ember.core`, `ember.memories`) depende solo de numpy. Es el
  código que termina corriendo en el robot como un `Cognitive Node` de e-MDB.
- El **laboratorio** (`ember.data`, `ember.tasks`, `ember.nas`, `ember.envs`,
  `ember.figures`) puede depender de torch, scipy, matplotlib y gymnasium.
  Se instala con los extras `lab` y `envs`.

Esa frontera está verificada por `tests/test_smoke.py`.
"""

__version__ = "0.1.0"
