"""Dam-break benchmark for the adaptive B-spline MPM solver.

A classic dam-break: a tall column of fluid is held on the left side of the
domain and released at t=0.  It collapses under gravity, hits the right wall,
and splashes back.  The adaptive solver refines around the collapsing fluid
front, with the PoU checker running every frame to verify the B-spline
partition of unity.

Geometry (in the 1.0 x 1.0 domain):
    +----------------------+
    |                      |
    |        air           |
    |                      |
    |######|               |
    |######|               |
    |######|         ->    |
    |######|               |
    |######+---------------+
    |                      |
    +----------------------+
     ^ fluid column  ^ right wall
     width = 0.25    (full domain)
     height = 0.50

Usage:
    .venv/bin/python benchmarks/dam_break/run_experiment.py
"""
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

import taichi as ti
import config

# ---------------------------------------------------------------------------
# Scenario: dam-break with adaptive refinement
# ---------------------------------------------------------------------------
config.ACTIVE_SCENARIO = "DAM_BREAK"
config.USE_ADAPTIVE_MPM = True
config.DIM = 2
config.AMR_MAX_LEVEL = 3
config.AMR_DYNAMIC_REFINEMENT = True
config.AMR_PARTICLE_CAPACITY_FACTOR = 50.0

# Dam-break geometry: narrow tall column on the left
DAM_WIDTH = 0.25 * config.GRID_WIDTH       # fluid column width
DAM_HEIGHT = 0.50 * config.GRID_HEIGHT     # fluid column height

# Override the particle generation geometry
config.MP_WIDTH = DAM_WIDTH
config.MP_HEIGHT = DAM_HEIGHT
config.POS_MP_LEFT_BOTTOM = [config.PADDING * config.DX, config.PADDING * config.DY]

# Recompute derived particle counts/volumes after geometry override
config.NUM_MP_WIDTH = int((config.MP_WIDTH / config.DX) * config.P_PER_CELL_AXIS)
config.NUM_MP_HEIGHT = int((config.MP_HEIGHT / config.DY) * config.P_PER_CELL_AXIS)
config.TOTAL_NUM_MP = config.NUM_MP_WIDTH * config.NUM_MP_HEIGHT
config.P_VOL = (config.MP_WIDTH * config.MP_HEIGHT) / config.TOTAL_NUM_MP
config.P_MASS = config.P_VOL * config.RHO_0

# Recompute CFL/dt with the new fluid height (taller column -> faster waves)
import math
config.H = config.MP_HEIGHT
config.C_0 = 10.0 * math.sqrt(2.0 * config.G_MAG * config.H)
config.V_MAX_ESTIMATE = math.sqrt(2.0 * config.G_MAG * config.H)
config.MAX_WAVE_SPEED = config.C_0 + config.V_MAX_ESTIMATE
config.DT_ACOUSTIC = config.CFL * (config.DX / config.MAX_WAVE_SPEED)
config.DT_RAW = config.DT_ACOUSTIC
config.SUBSTEPS = int(math.ceil(config.FRAME_DT / config.DT_RAW))
config.DT = config.FRAME_DT / config.SUBSTEPS

# Refinement criterion: follow the fluid front (velocity-based)
config.AMR_REFINEMENT_CRITERION = "velocity"
config.AMR_REFINEMENT_MARGIN = 0.15        # wide box to capture the collapse
config.AMR_REFINEMENT_VELOCITY_FRACTION = 0.02  # low threshold catches the front

# Initial refinement box: covers the dam column + some run-up space
domain_min_x = config.PADDING * config.DX
domain_min_y = config.PADDING * config.DY
domain_max_x = domain_min_x + config.GRID_WIDTH
domain_max_y = domain_min_y + config.GRID_HEIGHT

refinement_box = (
    (domain_min_x, domain_min_y),
    (domain_min_x + DAM_WIDTH + config.AMR_REFINEMENT_MARGIN, domain_max_y),
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
RELAXATION_FRAMES = 0          # no relaxation — dam break starts immediately
TOTAL_FRAMES = 300
EXPORT_EVERY = 3               # export every N frames
REPORT_EVERY = 30              # print particle-level summary every N frames

# ---------------------------------------------------------------------------
# Imports (after config overrides so they pick up the right values)
# ---------------------------------------------------------------------------
import physics.boundary as bnd
from utils.exporter import write_vtk, write_boundary_vtk
from solver.adaptive_engine import AdaptiveMPMSolver2D

output_directory = os.path.dirname(__file__)
os.makedirs(output_directory, exist_ok=True)


def get_particle_arrays(solver):
    pos = solver.particles.x.to_numpy()
    pressure = solver.particles.pressure.to_numpy()
    velocity = solver.particles.v.to_numpy()
    n_active = solver.particles.n_active()
    return pos[:n_active], pressure[:n_active], velocity[:n_active]


def export_frame(solver, frame):
    pos, pressure, velocity = get_particle_arrays(solver)
    write_vtk(frame, pos, pressure, velocity, output_dir=output_directory)


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

    print("Initializing dam-break adaptive MPM solver...")
    print(f"Dam geometry: width={DAM_WIDTH:.4f}, height={DAM_HEIGHT:.4f}")
    print(f"Refinement criterion: {config.AMR_REFINEMENT_CRITERION}")
    print(f"Refinement box: x=[{refinement_box[0][0]:.4f}, {refinement_box[1][0]:.4f}], "
          f"y=[{refinement_box[0][1]:.4f}, {refinement_box[1][1]:.4f}]")
    solver = AdaptiveMPMSolver2D(refinement_box=refinement_box)

    print(f"Grid levels: {solver.grid.num_levels}")
    print(f"Finest dx: {solver.grid.dx[-1]:.6e}")
    print(f"DT: {config.DT:.6e}  Substeps/frame: {config.SUBSTEPS}")
    print(f"Particles: {solver.particles.n_active()}")
    counts = ti.field(dtype=ti.i32, shape=solver.grid.num_levels)

    # Static domain boundary VTK (written once)
    write_boundary_vtk(
        config.PADDING * config.DX,
        config.PADDING * config.DY,
        config.PADDING * config.DX + config.GRID_WIDTH,
        config.PADDING * config.DY + config.GRID_HEIGHT,
        output_dir=output_directory,
    )

    # --- Phase 1: export initial state (no relaxation for dam break) -------
    report_particle_levels(solver, counts, "Initial")
    w_min, w_max, w_mean, g_max, n_violated = solver.check_partition_of_unity()
    print(f"Initial PoU: w=[{w_min:.12f},{w_max:.12f}] mean={w_mean:.12f}  "
          f"g_max={g_max:.6e}  violated={n_violated}")
    export_frame(solver, 0)

    # --- Phase 2: main simulation loop -------------------------------------
    print(f"Main loop ({TOTAL_FRAMES} frames, export every {EXPORT_EVERY})...")
    t = 0.0
    for frame in range(1, TOTAL_FRAMES + 1):
        for _ in range(config.SUBSTEPS):
            solver.step(damping=1.0, current_time=t)
            t += config.DT

        if frame % EXPORT_EVERY == 0:
            export_frame(solver, frame)

        if frame % REPORT_EVERY == 0 or frame == 1:
            report_particle_levels(solver, counts, f"Frame {frame}")

        # Partition-of-unity check: weight sum should be 1.0, grad sum 0.0
        w_min, w_max, w_mean, g_max, n_violated = solver.check_partition_of_unity()
        pou_tag = "OK" if n_violated == 0 else f"VIOLATED({n_violated})"
        print(f"Frame {frame}/{TOTAL_FRAMES}  t={t:.4e}s  "
              f"PoU: {pou_tag}  w=[{w_min:.12f},{w_max:.12f}] mean={w_mean:.12f}  g_max={g_max:.6e}")

    print("Done.")


if __name__ == "__main__":
    main()
