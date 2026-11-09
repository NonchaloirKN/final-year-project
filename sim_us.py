"""
Top-level export for the acoustic simulation engine.
Enables 'from sim_us import AcousticSimulator'.
"""

from src.simulator.sim_us import (
    DEFAULT_ALPHA_WM,
    DEFAULT_C,
    DEFAULT_F0,
    DEFAULT_SIGMA_RAYLEIGH,
    AcousticSimulator,
)

__all__ = [
    "AcousticSimulator",
    "DEFAULT_F0",
    "DEFAULT_C",
    "DEFAULT_ALPHA_WM",
    "DEFAULT_SIGMA_RAYLEIGH",
]
