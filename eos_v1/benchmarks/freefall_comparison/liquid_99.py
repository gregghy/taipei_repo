"""Backward-compatibility shim — delegates to mpm99_materials.

Prefer importing from ``benchmarks.freefall_comparison.mpm99_materials``
directly.  This module exists so older code that imports
``apply_liquid_99_properties`` / ``liquid_99_properties`` /
``MPM99LiquidSolver2D`` continues to work.
"""

from benchmarks.freefall_comparison.mpm99_materials import (
    MPM99LiquidSolver2D,
    apply_mpm99_properties as apply_liquid_99_properties,
    mpm99_properties as liquid_99_properties,
)
