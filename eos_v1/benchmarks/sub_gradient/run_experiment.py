"""Immersed-platform benchmark with velocity-gradient refinement.

Same physical setup as sub_001 (a wide platform descends into a fluid pool
under gravity), but refinement is driven by the per-particle velocity
gradient ``deform = |C| * dx`` instead of a moving grid patch that follows
the platform.  The refinement box is static and covers the full domain;
particles split and merge locally based on the gradient criterion.
"""
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
config.AMR_PARTICLE_CAPACITY_FACTOR = 50.0

# Use the mpm99 liquid constitutive model (SVD-based, no J clamp).
from benchmarks.freefall_comparison.mpm99_materials import apply_mpm99_properties
apply_mpm99_properties("liquid")

# --- Refinement: velocity-gradient (static grid, no moving patch) -----------
# Measured deform = |C| * dx peaks around 0.07 in this scenario (platform
# squeezing the fluid), so the split threshold must sit below that.  0.02
# keeps the fluid coarse while at rest (deform < 0.007) and splits only the
# fastest ~10% of particles once the platform starts compressing the pool.
config.AMR_DYNAMIC_REFINEMENT = False   # grid stays fixed
config.AMR_GRADIENT_REFINE = True       # split/merge by |C|*dx
config.AMR_GRADIENT_REFINE_THRESHOLD = 0.02
# Gradient refinement is capped at level 2.  With the full depth-3 cascade the
# split noise amplifies at each level (|C| grows ~20x per generation near the
# platform contact), which drives every particle to the finest level and
# exhausts the particle capacity within ~10 frames.  Depth 2 refines the
# squeezing region stably (measured: n 6000 -> ~89k, v_max < 0.65 m/s).
config.AMR_GRADIENT_MAX_LEVEL = 2
config.AMR_SPLIT_PARTICLES = True
config.AMR_MERGE_PARTICLES = True
config.AMR_MERGE_MIN_PARTICLES = 4
config.AMR_MERGE_SPEED_LIMIT = 0.5      # only slow particles merge back
config.AMR_INITIAL_PARTICLE_LEVEL = 0   # start coarse; split on gradient

# Make the immersed platform ~2/3 of the domain width and center it.
config.PLATFORM_WIDTH = 0.4 * config.GRID_WIDTH
config.FLUID_CENTER_X = (config.PADDING * config.DX) + (config.GRID_WIDTH / 2.0)
config.INT_MOVINGRECT_XMIN = config.FLUID_CENTER_X - (config.PLATFORM_WIDTH / 2.0)
config.INT_MOVINGRECT_XMAX = config.FLUID_CENTER_X + (config.PLATFORM_WIDTH / 2.0)

# Static refinement box covering the full fluid domain.  With gradient
# refinement the box just defines where fine grid levels exist; particles
# anywhere inside can split up to AMR_MAX_LEVEL.
_domain_xmin = config.PADDING * config.DX
_domain_ymin = config.PADDING * config.DY
_domain_xmax = _domain_xmin + config.GRID_WIDTH
_domain_ymax = _domain_ymin + config.GRID_HEIGHT
refinement_box = (
    (_domain_xmin, _domain_ymin),
    (_domain_xmax, _domain_ymax),
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
RELAXATION_FRAMES = 5
RELAXATION_DAMPING = 0.98
TOTAL_FRAMES = 200
EXPORT_EVERY = 2          # export every N frames
REPORT_EVERY = 20         # print particle-level summary every N frames

# ---------------------------------------------------------------------------
# Imports (after config overrides so they pick up the right values)
# ---------------------------------------------------------------------------
import physics.boundary as bnd
from utils.exporter import write_vtk, write_boundary_vtk, WriteInteriorMoving
from solver.adaptive_engine import AdaptiveMPMSolver2D  # noqa: F401 (base class)
from benchmarks.freefall_comparison.mpm99_materials import MPM99LiquidSolver2D


@ti.data_oriented
class GradientHysteresisLiquidSolver2D(MPM99LiquidSolver2D):
    """Liquid solver with hysteresis in the gradient refinement criterion.

    deform = |C| * dx is level-dependent: after a split, dx halves so deform
    halves, and without hysteresis the particle would immediately fall below
    the next level's threshold and merge back (split/merge pulsing).  With
    hysteresis, a particle at level k only demotes once deform drops below
    10% of the threshold that promoted it.
    """

    @ti.kernel
    def compute_gradient_levels(self):
        grad_threshold = ti.cast(config.AMR_GRADIENT_REFINE_THRESHOLD, ti.f64)
        grad_max = ti.static(getattr(config, 'AMR_GRADIENT_MAX_LEVEL', self.grid.num_levels - 1))
        hysteresis = ti.cast(0.1, ti.f64)  # demote at 10% of promote threshold
        for p in range(self.particles.active_count[None]):
            lvl = self.particles.level[p]
            dx = self.grid.level_dx[lvl]
            C_norm = self.particles.C[p].norm()
            deform = C_norm * dx
            promote_target = 0
            for k in ti.static(range(self.grid.num_levels)):
                if k <= ti.static(grad_max):
                    if deform > grad_threshold * (2.0 ** k):
                        promote_target = k
            target = promote_target
            if lvl > promote_target:
                demote_threshold = hysteresis * grad_threshold * (2.0 ** lvl)
                if deform >= demote_threshold:
                    target = lvl  # stay at current level
            self.particles.gradient_level[p] = target

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
        f"box={solver.grid.region_min_np[-1].tolist()}..{solver.grid.region_max_np[-1].tolist()}"
    )


def main():
    ti.init(arch=ti.gpu, default_fp=ti.f64)
    bnd.init_boundary_fields()

    print("Initializing immersed adaptive MPM solver (gradient refinement)...")
    print(f"Refinement: gradient (threshold={config.AMR_GRADIENT_REFINE_THRESHOLD}, "
          f"max_level={config.AMR_GRADIENT_MAX_LEVEL})")
    print(f"Refinement box: x=[{refinement_box[0][0]:.4f}, {refinement_box[1][0]:.4f}], "
          f"y=[{refinement_box[0][1]:.4f}, {refinement_box[1][1]:.4f}]")
    solver = GradientHysteresisLiquidSolver2D(refinement_box=refinement_box)

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
    w_min, w_max, w_mean, g_max, n_violated = solver.check_partition_of_unity()
    print(f"Initial PoU: w=[{w_min:.12f},{w_max:.12f}] mean={w_mean:.12f}  "
          f"g_max={g_max:.6e}  violated={n_violated}")
    export_frame(solver, 0, t=0.0)

    print(f"Relaxation ({RELAXATION_FRAMES} frames, damping={RELAXATION_DAMPING})...")
    for _ in range(RELAXATION_FRAMES):
        for _ in range(config.SUBSTEPS):
            solver.step(damping=RELAXATION_DAMPING, current_time=0.0)
    report_particle_levels(solver, counts, "After relaxation")
    w_min, w_max, w_mean, g_max, n_violated = solver.check_partition_of_unity()
    print(f"Post-relax PoU: w=[{w_min:.12f},{w_max:.12f}] mean={w_mean:.12f}  "
          f"g_max={g_max:.6e}  violated={n_violated}")
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

        # Partition-of-unity check: weight sum should be 1.0, grad sum should be 0.0
        w_min, w_max, w_mean, g_max, n_violated = solver.check_partition_of_unity()
        pou_tag = "OK" if n_violated == 0 else f"VIOLATED({n_violated})"
        print(f"Frame {frame}/{TOTAL_FRAMES}  t={t:.4e}s  "
              f"PoU: {pou_tag}  w=[{w_min:.12f},{w_max:.12f}] mean={w_mean:.12f}  g_max={g_max:.6e}")

    print("Done.")


if __name__ == "__main__":
    main()
