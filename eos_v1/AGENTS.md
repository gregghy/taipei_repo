# eos_v1 notes

- Python env: use `.venv` in this directory (`.venv/bin/python`); it has taichi 1.7.4 + numpy.
- Quick checks for the adaptive (quadtree) MPM:
  - `.venv/bin/python -m benchmarks.smoke_adaptive_mpm` — construction + 3 steps sanity check (CPU).
  - `.venv/bin/python verify_penalty_boundary.py` — checks quadrature mass assembly, directional switches, and moving-wall momentum (CPU).
  - `.venv/bin/python verify_dynamic_refinement.py` — checks moving-platform refinement geometry, level reassignment, and one adaptive step (CPU; set `TAICHI_ARCH=gpu` for GPU).
  - `.venv/bin/python verify_adaptive.py` — hydrostatic settling regression: checks mass conservation, hydrostatic pressure profile, and pressure consistency across the fine/coarse interface (CPU, ~1 min).
- Full simulation: `.venv/bin/python main.py` (GPU). Scenario is selected via `ACTIVE_SCENARIO` in `config.py`; the adaptive solver runs for `ACTIVE_SCENARIO = "ADAPTIVE_MPM"` or when `USE_ADAPTIVE_MPM = True` with `DAM_BREAK`/`IMMERSED` in 2D. Set `AMR_DYNAMIC_REFINEMENT = True` for an `IMMERSED` moving refinement window.
