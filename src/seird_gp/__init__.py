"""seird-gp: genetic programming per β(t) di un modello SEIRD."""

from .model import SEIRDParams, simulate

__all__ = ["SEIRDParams", "simulate"]
