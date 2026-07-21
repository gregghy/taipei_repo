import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

import taichi as ti
import config

# Pure fluid experiment.
config.USE_ADAPTIVE_MPM = True
config.DIM = 2
config.AMR_MAX_LEVEL = 3

# Retrieve the native pre-computed grid boundaries
domain_min_x = config.PADDING * config.DX
domain_max_x = domain_min_x + config.GRID_WIDTH
domain_min_y = config.PADDING * config.DY
domain_max_y = domain_min_y + config.GRID_HEIGHT

# Shift the refinement box to a thin strip on the left boundary (10% of width).
# The AMR engine will now grade from fine on the left to coarse on the right.
fine_width = config.GRID_WIDTH * 0.10
fine_xmin = domain_min_x
fine_xmax = domain_min_x + fine_width
fine_ymin = domain_min_y
fine_ymax = domain_max_y

refinement_box = ((fine_xmin, fine_ymin), (fine_xmax, fine_ymax))

import utils.visualization as ut
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

def main():
    ti.init(arch=ti.gpu, default_fp=ti.f64)
    bnd.init_boundary_fields()

    print("Initializing fluid-only adaptive MPM solver...")
    print(f"Refinement box on left edge: x=[{fine_xmin:.4f}, {fine_xmax:.4f}], y=[{fine_ymin:.4f}, {fine_ymax:.4f}]")
    solver = AdaptiveMPMSolver2D(refinement_box=refinement_box)
    print(f"Grid levels: {solver.grid.num_levels}")
    print(f"Finest dx: {solver.grid.dx[-1]:.6e}")
    
    counts = ti.field(dtype=ti.i32, shape=solver.grid.num_levels)

    def report_particle_levels(tag):
        solver.count_particles_by_level(counts)
        print(f"{tag}: active={solver.particles.n_active()} particles per level={counts.to_numpy().tolist()}")

    report_particle_levels("Initial")

    write_boundary_vtk(domain_min_x, domain_min_y, domain_max_x, domain_max_y, output_dir=output_directory)

    print("Exporting initial state (t=0)...")
    pos, pressure, velocity = get_particle_arrays(solver)
    write_vtk(0, pos, pressure, velocity, output_dir=output_directory)

    # Dynamic relaxation.
    RELAXATION_FRAMES = 5
    DAMPING_FACTOR = 0.98
    relaxation_time = 0.0
    print(f"Starting dynamic relaxation ({RELAXATION_FRAMES} frames)...")
    for frame in range(1, RELAXATION_FRAMES + 1):
        for _ in range(config.SUBSTEPS):
            solver.step(damping=DAMPING_FACTOR, current_time=relaxation_time)
            relaxation_time += config.DT
    print("Dynamic relaxation complete.")
    report_particle_levels("After relaxation")

    pos, pressure, velocity = get_particle_arrays(solver)
    write_vtk(-1, pos, pressure, velocity, output_dir=output_directory)

    # Main simulation loop.
    TOTAL_FRAMES = 30
    current_time = 0.0
    print(f"Starting main loop ({TOTAL_FRAMES} frames)...")
    for frame in range(1, TOTAL_FRAMES + 1):
        for _ in range(config.SUBSTEPS):
            solver.step(damping=1.0, current_time=current_time)
            current_time += config.DT

        pos, pressure, velocity = get_particle_arrays(solver)
        write_vtk(frame, pos, pressure, velocity, output_dir=output_directory)

        if frame == 1 or frame % 10 == 0:
            report_particle_levels(f"Frame {frame}")
        print(f"Frame {frame}/{TOTAL_FRAMES} exported.")

    print("Done.")

if __name__ == "__main__":
    main()
