import argparse
import contextlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import taichi as ti
import config

RELAXATION_FRAMES = 30
DAMPING_FACTOR = 0.98
TOTAL_FRAMES = 300
REFINEMENT_BOX = ((0.38, 0.03), (0.68, 0.13))
MAX_LEVEL = 2


def get_particle_arrays(solver):
    pos = solver.particles.x.to_numpy()
    pressure = solver.particles.pressure.to_numpy()
    velocity = solver.particles.v.to_numpy()
    if callable(getattr(solver.particles, 'n_active', None)):
        n_active = solver.particles.n_active()
        pos, pressure, velocity = pos[:n_active], pressure[:n_active], velocity[:n_active]
    return pos, pressure, velocity


def write_leaf_cells_vtk(grid, output_dir):
    filename = os.path.join(output_dir, 'leaf_cells.vtk')
    n = grid.leaf_count
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\nQuadtree leaf cells\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {4 * n} float\n")
        for origin, size in zip(grid.leaf_origin, grid.leaf_size):
            x0, y0 = float(origin[0]), float(origin[1])
            s = float(size)
            f.write(f"{x0} {y0} 0.0\n{x0 + s} {y0} 0.0\n{x0 + s} {y0 + s} 0.0\n{x0} {y0 + s} 0.0\n")
        f.write(f"POLYGONS {n} {5 * n}\n")
        for i in range(n):
            f.write(f"4 {4 * i} {4 * i + 1} {4 * i + 2} {4 * i + 3}\n")
        f.write(f"CELL_DATA {n}\nSCALARS level int 1\nLOOKUP_TABLE default\n")
        for level in grid.leaf_level:
            f.write(f"{int(level)}\n")
    print(f"Exported leaf cells to {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', choices=['standard', 'adaptive'], required=True)
    parser.add_argument('--frames', type=int, default=TOTAL_FRAMES)
    parser.add_argument('--relax-frames', type=int, default=RELAXATION_FRAMES)
    args = parser.parse_args()
    total_frames = args.frames
    relaxation_frames = args.relax_frames

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_name = 'standard_2D' if args.solver == 'standard' else 'adaptive_quadtree_2D'
    output_directory = os.path.join(base_dir, out_name)
    os.makedirs(output_directory, exist_ok=True)

    ti.init(arch=ti.gpu, kernel_profiler=True)

    import physics.boundary as bnd
    bnd.init_boundary_fields()
    from utils.exporter import write_vtk, write_boundary_vtk

    t_start = time.perf_counter()
    if args.solver == 'adaptive':
        # Oscillating flow keeps exchanging particles across the refinement
        # interface (no merging yet), so give the split pool extra headroom.
        config.AMR_PARTICLE_CAPACITY_FACTOR = 4.0
        from solver.adaptive_engine import AdaptiveMPMSolver2D
        solver = AdaptiveMPMSolver2D(refinement_box=REFINEMENT_BOX, max_level=MAX_LEVEL)
    else:
        from solver.standard_engine import StandardSolver
        solver = StandardSolver()
    ti.sync()
    setup_seconds = time.perf_counter() - t_start

    if args.solver == 'adaptive':
        n_particles = solver.particles.n_active()
        print(f"Adaptive solver: levels = {solver.grid.num_levels}, "
              f"refinement box = {REFINEMENT_BOX}, grid shapes = {solver.grid.res}")
    else:
        n_particles = solver.particles.n_particles
    print(f"Solver: {args.solver} | particles = {n_particles} | "
          f"DT = {config.DT:.4e} | SUBSTEPS/frame = {config.SUBSTEPS}")

    min_x = config.PADDING * config.DX
    min_y = config.PADDING * config.DY
    write_boundary_vtk(min_x, min_y, min_x + config.GRID_WIDTH, min_y + config.GRID_HEIGHT,
                       output_dir=output_directory)
    if args.solver == 'adaptive':
        write_leaf_cells_vtk(solver.grid, output_directory)

    print(f"Starting Dynamic Relaxation Phase ({relaxation_frames} frames)...")
    t_relax_start = time.perf_counter()
    relaxation_time = 0.0
    for frame in range(1, relaxation_frames + 1):
        for _ in range(config.SUBSTEPS):
            solver.step(damping=DAMPING_FACTOR, current_time=relaxation_time)
            relaxation_time += config.DT
    ti.sync()
    relax_seconds = time.perf_counter() - t_relax_start
    print("Dynamic Relaxation Complete.")

    pos, pressure, velocity = get_particle_arrays(solver)
    write_vtk(0, pos, pressure, velocity, output_dir=output_directory)

    # Profile only the steady-state main loop (excludes JIT warmup + relaxation).
    ti.profiler.clear_kernel_profiler_info()

    sim_seconds = 0.0
    export_seconds = 0.0
    current_time = 0.0
    for frame in range(1, total_frames + 1):
        t_a = time.perf_counter()
        for _ in range(config.SUBSTEPS):
            solver.step(damping=1.0, current_time=current_time)
            current_time += config.DT
        ti.sync()
        t_b = time.perf_counter()
        sim_seconds += t_b - t_a
        pos, pressure, velocity = get_particle_arrays(solver)
        write_vtk(frame, pos, pressure, velocity, output_dir=output_directory)
        export_seconds += time.perf_counter() - t_b

    profiler_buffer = io.StringIO()
    with contextlib.redirect_stdout(profiler_buffer):
        ti.profiler.print_kernel_profiler_info('count')
    profiler_text = profiler_buffer.getvalue()

    total_steps = total_frames * config.SUBSTEPS
    summary_lines = [
        f"solver              : {args.solver}",
        f"backend             : {ti.lang.impl.current_cfg().arch}",
        f"frames              : {total_frames} (+{relaxation_frames} relaxation)",
        f"DT                  : {config.DT:.6e} s",
        f"substeps per frame  : {config.SUBSTEPS}",
        f"total steps (main)  : {total_steps}",
        f"particles (initial) : {n_particles}",
    ]
    if args.solver == 'adaptive':
        counts = ti.field(dtype=ti.i32, shape=solver.grid.num_levels)
        solver.count_particles_by_level(counts)
        summary_lines += [
            f"particles (final)   : {solver.particles.n_active()}",
            f"particles per level : {counts.to_numpy().tolist()}",
            f"level dx            : {[f'{d:.4e}' for d in solver.grid.dx]}",
            f"split overflow      : {int(solver.particles.split_overflow[None])}",
        ]
    summary_lines += [
        f"setup wall time     : {setup_seconds:.2f} s",
        f"relaxation wall time: {relax_seconds:.2f} s ({relaxation_frames * config.SUBSTEPS} steps)",
        f"main loop wall time : {sim_seconds:.2f} s",
        f"steps per second    : {total_steps / sim_seconds:.1f}",
        f"ms per step         : {1e3 * sim_seconds / total_steps:.3f}",
        f"ms per frame (sim)  : {1e3 * sim_seconds / total_frames:.1f}",
        f"vtk export wall time: {export_seconds:.2f} s",
    ]
    summary = "\n".join(summary_lines)
    print("=========================================")
    print(summary)
    print("=========================================")
    print(profiler_text)

    with open(os.path.join(output_directory, 'benchmark_summary.txt'), 'w') as f:
        f.write(summary + "\n")
    with open(os.path.join(output_directory, 'kernel_profile.txt'), 'w') as f:
        f.write(profiler_text)
    print(f"Summary and kernel profile written to {output_directory}")


if __name__ == "__main__":
    main()
