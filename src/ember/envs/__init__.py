"""Adaptadores de entornos secuenciales a flujos de experiencia.

Requiere el extra `envs`. La diferencia con `ember.data` es de dónde sale el
error de predicción: en un flujo sintético lo pone el generador, y en un entorno
hay que calcularlo, porque poner una etiqueta a mano sería decidir a dedo la
variable que gobierna todo el resultado.
"""

from ember.envs.minigrid import MiniGridStreamAdapter, OneStepPredictor

__all__ = ["MiniGridStreamAdapter", "OneStepPredictor"]
