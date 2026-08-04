import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

import taichi as ti
import config

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
config.ACTIVE_SCENARIO = "IMMERSED"
config.USE_ADAPTIVE_MPM = True
config.DIM = 2
config.AMR_MAX_LEVEL = 3
config.AMR_DYNAMIC_REFINEMENT = True
config.AMR_PARTICLE_CAPACITY_FACTOR = 50.0

# Refinement criterion: "platform", "velocity", "pressure", "deformation", "combined"
#   - "platform":    finest patch follows the immersed platform (default)
#   - "velocity":    follows the mass-weighted centroid of fast particles
#   - "pressure":    follows the mass-weighted centroid of high-pressure particles
#   - "deformation": follows the mass-weighted centroid of deformed particles
#   - "combined":    weighted union of velocity + pressure + deformation
config.AMR_REFINEMENT_CRITERION = "platform"
config.AMR_REFINEMENT_MARGIN = 0.02       # half-size of finest box around criterion center

# Make the immersed platform ~2/3 of the domain width and center it.
config.PLATFORM_WIDTH = 0.4 * config.GRID_WIDTH
config.FLUID_CENTER_X = (config.PADDING * config.DX) + (config.GRID_WIDTH / 2.0)
config.INT_MOVINGRECT_XMIN = config.FLUID_CENTER_X - (config.PLATFORM_WIDTH / 2.0)
config.INT_MOVINGRECT_XMAX = config.FLUID_CENTER_X + (config.PLATFORM_WIDTH / 2.0)

# Build the initial refinement box from the criterion center + margin.
# For "platform" criterion the center is the platform midpoint.
# For physics criteria the initial center is computed from the initial particle
# state (which is at rest, so it falls back to the platform center).
# The margin is asymmetric in y: the platform only moves down, but the fluid
# surface rises above the platform, so we need more headroom above than below.
_margin = float(getattr(config, "AMR_REFINEMENT_MARGIN", 0.02))
_margin_below = float(getattr(config, "AMR_REFINEMENT_MARGIN_BELOW", _margin))
_margin_above = float(getattr(config, "AMR_REFINEMENT_MARGIN_ABOVE", _margin + 0.5 * config.MP_HEIGHT))
_platform_cx = 0.5 * (config.INT_MOVINGRECT_XMIN + config.INT_MOVINGRECT_XMAX)
_platform_hw = 0.5 * (config.INT_MOVINGRECT_XMAX - config.INT_MOVINGRECT_XMIN) + _margin
_platform_ymin = config.INT_MOVINGRECT_YMIN - _margin_below
_platform_ymax = config.INT_MOVINGRECT_YMAX + _margin_above
refinement_box = (
    (_platform_cx - _platform_hw, _platform_ymin),
    (_platform_cx + _platform_hw, _platform_ymax),
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
RELAXATION_FRAMES = 5
RELAXATION_DAMPING = 0.98
TOTAL_FRAMES = 300
EXPORT_EVERY = 3          # export every N frames
REPORT_EVERY = 30         # print particle-level summary every N frames

# ---------------------------------------------------------------------------
# Imports (after config overrides so they pick up the right values)
# ---------------------------------------------------------------------------
import physics.boundary as bnd
from utils.exporter import write_vtk, write_boundary_vtk, WriteInteriorMoving
from solver.adaptive_engine import AdaptiveMPMSolver2D

output_directory = os.path.dirname(__file__)
os.makedirs(output_directory, exist_ok=True)


def get_particle_arrays(solver):
    pos = solver.particles.x.to_numpy()
    pressure = solver.particles.pressure.to_numpy()
    velocity = solver.particles.v.to_numpy()
    n_active = solver.particles.n_active()
    return pos[:n_active], pressure[:n_active], velocity[:n_active]


def export_frame(solver, frame, t):
    """Export particle VTK and the moving platform box for a single frame."""
    pos, pressure, velocity = get_particle_arrays(solver)
    write_vtk(frame, pos, pressure, velocity, output_dir=output_directory)
    displacement, _ = solver.grid._platform_motion_numpy(t)
    WriteInteriorMoving(
        frame,
        config.INT_MOVINGRECT_XMIN + displacement[0],
        config.INT_MOVINGRECT_YMIN + displacement[1],
        config.INT_MOVINGRECT_XMAX + displacement[0],
        config.INT_MOVINGRECT_YMAX + displacement[1],
        output_dir=output_directory,
    )


def report_particle_levels(solver, counts, tag):
    solver.count_particles_by_level(counts)
    print(
        f"{tag}: active={solver.particles.n_active()} "
        f"per_level={counts.to_numpy().tolist()} "
        f"shift={solver.grid.level_refinement_shift_np[-1].tolist()} "
        f"box={solver.grid.region_min_np[-1].tolist()}..{solver.grid.region_max_np[-1].tolist()}"
    )


def main():
    ti.init(arch=ti.gpu, default_fp=ti.f64)
    bnd.init_boundary_fields()

    print("Initializing immersed adaptive MPM solver...")
    print(f"Refinement criterion: {config.AMR_REFINEMENT_CRITERION}")
    print(f"Refinement box: x=[{refinement_box[0][0]:.4f}, {refinement_box[1][0]:.4f}], "
          f"y=[{refinement_box[0][1]:.4f}, {refinement_box[1][1]:.4f}]")
    solver = AdaptiveMPMSolver2D(refinement_box=refinement_box)

    # Pre-flight check: verify the finest patch can shift (platform criterion only)
    if config.AMR_REFINEMENT_CRITERION == "platform":
        planned_shift = solver.grid._level_dynamic_shifts(config.PLATFORM_STOP_TIME)[-1]
        if all(abs(c) < 1e-12 for c in planned_shift):
            raise RuntimeError("The finest AMR patch is pinned and cannot follow the moving platform")

    print(f"Grid levels: {solver.grid.num_levels}")
    print(f"Finest dx: {solver.grid.dx[-1]:.6e}")
    print(f"DT: {config.DT:.6e}  Substeps/frame: {config.SUBSTEPS}")
    counts = ti.field(dtype=ti.i32, shape=solver.grid.num_levels)

    # Static domain boundary VTK (written once).
    write_boundary_vtk(
        config.PADDING * config.DX,
        config.PADDING * config.DY,
        config.PADDING * config.DX + config.GRID_WIDTH,
        config.PADDING * config.DY + config.GRID_HEIGHT,
        output_dir=output_directory,
    )

    # --- Phase 1: export initial state, then relax -------------------------
    report_particle_levels(solver, counts, "Initial")
    export_frame(solver, 0, t=0.0)

    print(f"Relaxation ({RELAXATION_FRAMES} frames, damping={RELAXATION_DAMPING})...")
    for _ in range(RELAXATION_FRAMES):
        for _ in range(config.SUBSTEPS):
            solver.step(damping=RELAXATION_DAMPING, current_time=0.0)
    report_particle_levels(solver, counts, "After relaxation")
    export_frame(solver, 0, t=0.0)  # overwrite frame 0 with settled state

    # --- Phase 2: main simulation loop -------------------------------------
    print(f"Main loop ({TOTAL_FRAMES} frames, export every {EXPORT_EVERY})...")
    t = 0.0
    for frame in range(1, TOTAL_FRAMES + 1):
        for _ in range(config.SUBSTEPS):
            solver.step(damping=1.0, current_time=t)
            t += config.DT

        if frame % EXPORT_EVERY == 0:
            export_frame(solver, frame, t)

        if frame % REPORT_EVERY == 0 or frame == 1:
            report_particle_levels(solver, counts, f"Frame {frame}")

        print(f"Frame {frame}/{TOTAL_FRAMES}  t={t:.4e}s")

    print("Done.")


if __name__ == "__main__":
    main()
