"""
simulation_engine.py
====================
Compatibility shim — re-exports HostileRetriever from simulation.py
so that `from simulation_engine import HostileRetriever` works as expected
by main.py.
"""

from simulation import HostileRetriever  # noqa: F401

__all__ = ["HostileRetriever"]
