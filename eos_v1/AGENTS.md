# eos_v1 notes

- Python env: use `.venv` in this directory (`.venv/bin/python`); it has taichi 1.7.4 + numpy.
- Quick checks for the adaptive (quadtree) MPM:
  - `.venv/bin/python -m benchmarks.smoke_adaptive_mpm` — construction + 3 steps sanity check (CPU).
  - `.venv/bin/python verify_penalty_boundary.py` — checks quadrature mass assembly, directional switches, and moving-wall momentum (CPU).
  - `.venv/bin/python verify_dynamic_refinement.py` — checks moving-platform refinement geometry, supported transition levels, particle mass ratios, and one adaptive step (CPU; set `TAICHI_ARCH=gpu` for GPU).
  - `.venv/bin/python verify_particle_merge.py` — checks complete 4-way splits, rejects partial/mixed merges, and verifies mass/momentum through level 0/1 and 1/2 round trips (CPU).
  - `.venv/bin/python verify_gradient_refinement.py` — checks the `three_blocks` gradient probe refines level 0 particles through level 2, then conservatively coarsens them (CPU).
  - `.venv/bin/python verify_adaptive.py` — hydrostatic settling regression: checks mass conservation, hydrostatic pressure profile, and pressure consistency across the fine/coarse interface (CPU, ~1 min).
  - `.venv/bin/python -m benchmarks.freefall.run_experiment --smoke --arch cpu` — freefall benchmark construction and short-step check.
  - `.venv/bin/python -m benchmarks.freefall_old.run_experiment --smoke --arch cpu --particles 100` (or `--particles 1500`) — legacy fixed-resolution freefall construction and short-step check.
  - `.venv/bin/python -m benchmarks.dynamic_block.run_experiment --smoke --arch cpu` — moving-block dynamic-grid construction and VTK-series check.
- Full freefall cycle: `.venv/bin/python -m benchmarks.freefall.run_experiment --arch gpu` — undamped gravity-driven impact, gradient refinement, and natural local coarsening.
- Fixed/adaptive freefall comparison: `.venv/bin/python -m benchmarks.freefall_comparison.run_comparison --arch gpu` — runs coarse fixed-100, uniformly fine fixed-1500, and gradient-adaptive cases for 0.6 s, exports VTK by default, and writes comparison JSON; pass `--no-vtk` for timing-only output.
- Dynamic-grid presentation: `.venv/bin/python -m benchmarks.dynamic_block.run_experiment --arch gpu` — exports synchronized particle, composite-grid, and patch-outline VTK series.
- Full simulation: `.venv/bin/python main.py` (GPU). Scenario is selected via `ACTIVE_SCENARIO` in `config.py`; the adaptive solver runs for `ACTIVE_SCENARIO = "ADAPTIVE_MPM"` or when `USE_ADAPTIVE_MPM = True` with `DAM_BREAK`/`IMMERSED` in 2D. Set `AMR_DYNAMIC_REFINEMENT = True` for an `IMMERSED` moving refinement window.
