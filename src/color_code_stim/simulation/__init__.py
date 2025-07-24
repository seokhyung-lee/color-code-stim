"""
Simulation module for color code quantum error correction.

This module provides simulation, sampling, and statistical analysis functionality
for color code circuits. It implements the extraction of simulation logic from
the monolithic ColorCode class into a modular architecture.

Classes
-------
Simulator : Main simulation and sampling interface
"""

from .simulator import Simulator

__all__ = ["Simulator"]