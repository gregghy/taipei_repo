import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

import taichi as ti
import config

# Override config for this experiment.
config.ACTIVE_SCENARIO = "IMMERSED"
config.USE_ADAPTIVE_MPM = True
config.DIM = 2
config.AMR_MAX_LEVEL = 3

# Make the immersed platform ~2/3 of the domain width and center it.
config.PLATFORM_WIDTH = 0.96 * config.GRID_WIDTH
config.FLUID_CENTER_X = (config.PADDING * config.DX) + (config.GRID_WIDTH / 2.0)
config.INT_MOVINGRECT_XMIN = config.FLUID_CENTER_X - (config.PLATFORM_WIDTH / 2.0)
config.INT_MOVINGRECT_XMAX = config.FLUID_CENTER_X + (config.PLATFORM_WIDTH / 2.0)

# Build a custom refinement box that is half as tall as the default immersed fine
# region and shifted upward, so the bottom of the domain stays coarse like the sides.
platform_cx = 0.5 * (config.INT_MOVINGRECT_XMIN + config.INT_MOVINGRECT_XMAX)
platform_width = config.INT_MOVINGRECT_XMAX - config.INT_MOVINGRECT_XMIN
margin = getattr(config, "AMR_PROCESS_MARGIN", 0.05)
fine_xmin = platform_cx - (platform_width / 2.0 + margin)
fine_xmax = platform_cx + (platform_width / 2.0 + margin)
# Default immersed fine region is roughly AMR_DOMAIN_MIN_Y .. INT_MOVINGRECT_YMAX.
default_fine_height = config.INT_MOVINGRECT_YMAX - config.AMR_DOMAIN_MIN_Y
fine_height = 0.5 * default_fine_height
fine_ymax = config.INT_MOVINGRECT_YMAX
fine_ymin = fine_ymax - fine_height
refinement_box = ((fine_xmin, fine_ymin), (fine_xmax, fine_ymax))

# Scenario geometry and boundary fields are already computed for DIM=2 in config.py.
import utils.visualization as ut
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


def main():
    ti.init(arch=ti.gpu)
    bnd.init_boundary_fields()

    print("Initializing immersed adaptive MPM solver...")
    print(f"Custom refinement box: x=[{fine_xmin:.4f}, {fine_xmax:.4f}], y=[{fine_ymin:.4f}, {fine_ymax:.4f}]")
    solver = AdaptiveMPMSolver2D(refinement_box=refinement_box)
    print(f"Grid levels: {solver.grid.num_levels}")
    print(f"Finest dx: {solver.grid.dx[-1]:.6e}")
    print(f"DT after AMR adjustment: {config.DT:.6e}")
    print(f"Substeps per frame: {config.SUBSTEPS}")

    min_x = config.PADDING * config.DX
    min_y = config.PADDING * config.DY
    max_x = min_x + config.GRID_WIDTH
    max_y = min_y + config.GRID_HEIGHT
    write_boundary_vtk(min_x, min_y, max_x, max_y, output_dir=output_directory)

    print("Exporting initial state (t=0)...")
    pos, pressure, velocity = get_particle_arrays(solver)
    write_vtk(0, pos, pressure, velocity, output_dir=output_directory)
    WriteInteriorMoving(
        0,
        config.INT_MOVINGRECT_XMIN,
        config.INT_MOVINGRECT_YMIN,
        config.INT_MOVINGRECT_XMAX,
        config.INT_MOVINGRECT_YMAX,
        output_dir=output_directory,
    )

    # Dynamic relaxation.
    RELAXATION_FRAMES = 10
    DAMPING_FACTOR = 0.98
    relaxation_time = 0.0
    print(f"Starting dynamic relaxation ({RELAXATION_FRAMES} frames)...")
    for frame in range(1, RELAXATION_FRAMES + 1):
        for _ in range(config.SUBSTEPS):
            solver.step(damping=DAMPING_FACTOR, current_time=relaxation_time)
            relaxation_time += config.DT
    print("Dynamic relaxation complete.")

    pos, pressure, velocity = get_particle_arrays(solver)
    write_vtk(0, pos, pressure, velocity, output_dir=output_directory)

    # Main simulation loop.
    TOTAL_FRAMES = 300
    current_time = 0.0
    print(f"Starting main loop ({TOTAL_FRAMES} frames)...")
    for frame in range(1, TOTAL_FRAMES + 1):
        for _ in range(config.SUBSTEPS):
            solver.step(damping=1.0, current_time=current_time)
            current_time += config.DT

        pos, pressure, velocity = get_particle_arrays(solver)
        write_vtk(frame, pos, pressure, velocity, output_dir=output_directory)

        # Platform displacement for VTK output.
        if current_time < config.PLATFORM_STOP_TIME:
            displacement_y = config.PLATFORM_VELOCITY_Y * current_time
        elif current_time < config.PLATFORM_STOP_TIME + config.PLATFORM_DECEL_TIME:
            time_in_decel = current_time - config.PLATFORM_STOP_TIME
            dist_before_stop = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
            dist_during_decel = (
                config.PLATFORM_VELOCITY_Y * time_in_decel
                - 0.5
                * (config.PLATFORM_VELOCITY_Y / config.PLATFORM_DECEL_TIME)
                * (time_in_decel ** 2)
            )
            displacement_y = dist_before_stop + dist_during_decel
        else:
            dist_before_stop = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
            total_decel_dist = 0.5 * config.PLATFORM_VELOCITY_Y * config.PLATFORM_DECEL_TIME
            displacement_y = dist_before_stop + total_decel_dist

        current_y_min = config.INT_MOVINGRECT_YMIN + displacement_y
        current_y_max = config.INT_MOVINGRECT_YMAX + displacement_y
        WriteInteriorMoving(
            frame,
            config.INT_MOVINGRECT_XMIN,
            current_y_min,
            config.INT_MOVINGRECT_XMAX,
            current_y_max,
            output_dir=output_directory,
        )

        print(f"Frame {frame}/{TOTAL_FRAMES} exported.")

    print("Done.")


if __name__ == "__main__":
    main()
