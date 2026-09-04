"""Three-material collision benchmark using adaptive MPM with gradient refinement.

Reproduces the mpm99.py three-block liquid/jelly/snow collision using the
repository's adaptive quadtree MPM solver. Refinement is gradient-driven and
capped to keep the simulation stable.

Usage:
    .venv/bin/python -m benchmarks.materials.run_experiment --arch gpu
    .venv/bin/python -m benchmarks.materials.run_experiment --smoke --arch cpu
    .venv/bin/python -m benchmarks.materials.run_experiment --max-level 1
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

import taichi as ti
import config
from benchmarks.freefall_comparison.mpm99_materials import (
    apply_mpm99_properties,
    mpm99_properties,
    MPM99MultiMaterialSolver2D,
)
import shutil

# --- mpm99 reference parameters ----------------------------------------------
DOMAIN_WIDTH = 1.0
DOMAIN_HEIGHT = 1.0
GRAVITY = (0.0, -50.0)

# Block layout (matches mpm99 group offsets)
BLOCK_SIZE = 0.2
BLOCKS = [
    {"name": "liquid", "material_id": 0, "xmin": 0.30, "ymin": 0.05},
    {"name": "jelly",  "material_id": 1, "xmin": 0.40, "ymin": 0.37},
    {"name": "snow",   "material_id": 2, "xmin": 0.50, "ymin": 0.69},
]

TOTAL_TIME = 2.0


def configure(max_level=1, base_cells=64, gradient_threshold=0.5):
    config.ACTIVE_SCENARIO = "ADAPTIVE_MPM"
    config.USE_ADAPTIVE_MPM = True
    config.DIM = 2
    apply_mpm99_properties("multi")
    config.GRAVITY = list(GRAVITY)
    config.V_MAX_ESTIMATE = 10.0
    config.MAX_WAVE_SPEED = config.C_0 + config.V_MAX_ESTIMATE
    config.CFL = 0.1
    finest_dx = DOMAIN_WIDTH / (base_cells * (2 ** max_level))
    config.DT = 0.1 * finest_dx / config.MAX_WAVE_SPEED
    config.FRAME_DT = config.DT
    config.AMR_DOMAIN_MIN_X = 0.0
    config.AMR_DOMAIN_MIN_Y = 0.0
    config.AMR_DOMAIN_WIDTH = DOMAIN_WIDTH
    config.AMR_DOMAIN_HEIGHT = DOMAIN_HEIGHT
    config.AMR_BASE_CELLS_X = base_cells
    config.AMR_BASE_CELLS_Y = base_cells
    config.AMR_BASE_DX = DOMAIN_WIDTH / base_cells
    config.AMR_MAX_LEVEL = max_level
    config.AMR_GRID_PADDING = 3
    config.AMR_REFINEMENT_BUFFER_CELLS = 1
    config.AMR_GHOST_BAND_CELLS = 2
    config.AMR_PARTICLES_PER_CELL_AXIS = 2
    config.AMR_SPLIT_PARTICLES = True
    config.AMR_MERGE_PARTICLES = True
    config.AMR_MERGE_MIN_PARTICLES = 4
    config.AMR_MERGE_SPEED_LIMIT = 0.2
    config.AMR_MATERIAL_COUNT = 3
    config.AMR_PARTICLE_CAPACITY_FACTOR = 20.0
    config.AMR_INITIAL_PARTICLE_LEVEL = 0
    config.AMR_INITIAL_FLUID_XMIN = 0.30
    config.AMR_INITIAL_FLUID_XMAX = 0.70
    config.AMR_INITIAL_FLUID_YMIN = 0.05
    config.AMR_INITIAL_FLUID_YMAX = 0.89
    config.AMR_DYNAMIC_REFINEMENT = False
    config.AMR_GRADIENT_REFINE = True
    config.AMR_GRADIENT_REFINE_THRESHOLD = gradient_threshold
    config.AMR_GRADIENT_MAX_LEVEL = max_level
    return ((0.0, 0.0), (DOMAIN_WIDTH, DOMAIN_HEIGHT))


def block_particle_positions(block, particles_per_axis):
    xmin = block["xmin"]
    ymin = block["ymin"]
    spacing = BLOCK_SIZE / particles_per_axis
    positions = []
    for i in range(particles_per_axis):
        for j in range(particles_per_axis):
            x = xmin + (i + 0.5) * spacing
            y = ymin + (j + 0.5) * spacing
            positions.append((x, y))
    return np.asarray(positions, dtype=np.float64)


def initialize_materials_state(solver, particles_per_axis=None):
    """Initialize three blocks. If particles_per_axis is None, compute it from
    the solver's native level-0 mass so that split/merge mass conditions hold."""
    if particles_per_axis is None:
        native_mass_0 = float(solver.particles.native_mass[0])
        native_vol_0 = native_mass_0 / config.RHO_0
        particles_per_axis = int(round(math.sqrt(BLOCK_SIZE ** 2 / native_vol_0)))

    all_positions = []
    all_materials = []
    for block in BLOCKS:
        pos = block_particle_positions(block, particles_per_axis)
        all_positions.append(pos)
        all_materials.append(np.full(len(pos), block["material_id"], dtype=np.int32))
    positions = np.concatenate(all_positions, axis=0)
    materials = np.concatenate(all_materials, axis=0)
    n = len(positions)
    if n > solver.particles.capacity:
        raise RuntimeError(f"particle count {n} exceeds capacity {solver.particles.capacity}")
    capacity = solver.particles.capacity
    # Use the solver's native level-0 mass so merge conditions are satisfied.
    p_mass = float(solver.particles.native_mass[0])
    p_vol = p_mass / config.RHO_0

    def pad_1d(values, dtype):
        return np.concatenate([values, np.zeros(capacity - len(values), dtype=dtype)])
    def pad_2d(values, dtype):
        return np.concatenate([values, np.zeros((capacity - len(values),) + values.shape[1:], dtype=dtype)])

    solver.particles.x.from_numpy(pad_2d(positions, np.float64))
    solver.particles.v.from_numpy(np.zeros((capacity, 2), dtype=np.float64))
    solver.particles.C.from_numpy(np.zeros((capacity, 2, 2), dtype=np.float64))
    solver.particles.F.from_numpy(np.tile(np.eye(2, dtype=np.float64), (capacity, 1, 1)))
    solver.particles.stress.from_numpy(np.zeros((capacity, 2, 2), dtype=np.float64))
    solver.particles.pressure.from_numpy(np.zeros(capacity, dtype=np.float64))
    solver.particles.Jp.from_numpy(np.ones(capacity, dtype=np.float64))
    solver.particles.material.from_numpy(pad_1d(materials, np.int32))
    solver.particles.level.from_numpy(np.zeros(capacity, dtype=np.int32))
    solver.particles.mass.from_numpy(pad_1d(np.full(n, p_mass, dtype=np.float64), np.float64))
    solver.particles.volume0.from_numpy(pad_1d(np.full(n, p_vol, dtype=np.float64), np.float64))
    solver.particles.gradient_level.from_numpy(np.zeros(capacity, dtype=np.int32))
    solver.particles.active_count[None] = n
    solver.particles.split_overflow[None] = 0
    return n


def diagnostics(solver, frame, t):
    n = solver.particles.n_active()
    position = solver.particles.x.to_numpy()[:n]
    velocity = solver.particles.v.to_numpy()[:n]
    mass = solver.particles.mass.to_numpy()[:n]
    level = solver.particles.level.to_numpy()[:n]
    material = solver.particles.material.to_numpy()[:n]
    pressure = solver.particles.pressure.to_numpy()[:n]
    speed = np.linalg.norm(velocity, axis=1)
    total_mass = float(mass.sum())
    summary = {
        "frame": int(frame),
        "time": float(t),
        "particles": int(n),
        "particles_by_level": [int((level == lv).sum()) for lv in range(solver.grid.num_levels)],
        "particles_by_material": [int((material == m).sum()) for m in range(3)],
        "total_mass": total_mass,
        "max_speed": float(speed.max()) if n > 0 else 0.0,
        "max_pressure": float(np.abs(pressure).max()) if n > 0 else 0.0,
        "height_range": [float(position[:, 1].min()), float(position[:, 1].max())] if n > 0 else [0, 0],
    }
    return position, pressure, velocity, material, summary


def write_vtk_simple(frame, pos, vel, mat, output_dir):
    """Write a minimal, bulletproof VTK file — one value per line, no blanks."""
    n = len(pos)
    filename = os.path.join(output_dir, f"mpm_fluid_{frame:06d}.vtk")
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM99 Materials\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for i in range(n):
            f.write(f"{pos[i,0]:.6f} {pos[i,1]:.6f} 0.0\n")
        f.write(f"VERTICES {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {n}\n")
        f.write("SCALARS Material int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(n):
            f.write(f"{int(mat[i])}\n")
        f.write("VECTORS Velocity float\n")
        for i in range(n):
            f.write(f"{vel[i,0]:.6f} {vel[i,1]:.6f} 0.0\n")
    print(f"Exported frame {frame} to {filename}")


def export_frame(solver, frame, output_directory, t):
    position, pressure, velocity, material, summary = diagnostics(solver, frame, t)
    write_vtk_simple(frame, position, velocity, material, output_directory)
    return summary


def run_case(steps=None, export_every=200, output_directory=None, export_vtk=True,
             validate_run=True, particles_per_axis=None, max_level=1, base_cells=64,
             gradient_threshold=0.5):
    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    refinement_box = configure(max_level=max_level, base_cells=base_cells,
                               gradient_threshold=gradient_threshold)
    solver = MPM99MultiMaterialSolver2D(refinement_box=refinement_box)
    n_initial = initialize_materials_state(solver, particles_per_axis)
    initial_positions = solver.particles.x.to_numpy()
    ti.sync()
    setup_seconds = time.perf_counter() - setup_start

    # JIT warmup
    warmup_start = time.perf_counter()
    solver.step(damping=1.0, current_time=0.0)
    ti.sync()
    jit_warmup_seconds = time.perf_counter() - warmup_start
    solver.particles.x.from_numpy(initial_positions)
    initialize_materials_state(solver, particles_per_axis)
    solver._step_count = 0
    ti.sync()

    if output_directory is None:
        output_directory = os.path.join(os.path.dirname(__file__), "output")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    if steps is None:
        steps = int(math.ceil(TOTAL_TIME / config.DT))

    initial = diagnostics(solver, 0, 0.0)[-1]
    history = []
    export_seconds = 0.0

    if export_vtk:
        export_start = time.perf_counter()
        history.append(export_frame(solver, 0, output_directory, 0.0))
        export_seconds += time.perf_counter() - export_start
    else:
        history.append(initial)

    peak_particles = initial["particles"]
    simulation_seconds = 0.0
    segment_start = time.perf_counter()
    t = 0.0
    for step in range(1, steps + 1):
        solver.step(damping=1.0, current_time=t)
        t += config.DT
        n = solver.particles.n_active()
        if n > peak_particles:
            peak_particles = n
        if export_vtk and step % export_every == 0:
            ti.sync()
            simulation_seconds += time.perf_counter() - segment_start
            export_start = time.perf_counter()
            history.append(export_frame(solver, step, output_directory, t))
            export_seconds += time.perf_counter() - export_start
            segment_start = time.perf_counter()
            print(f"step {step}: t={t:.4f} n={n} levels={history[-1]['particles_by_level']} max_speed={history[-1]['max_speed']:.4f}")
    ti.sync()
    simulation_seconds += time.perf_counter() - segment_start

    final = diagnostics(solver, steps, t)[-1]
    if export_vtk:
        export_start = time.perf_counter()
        if steps % export_every != 0 or history[-1]["frame"] != steps:
            history.append(export_frame(solver, steps, output_directory, t))
        export_seconds += time.perf_counter() - export_start
    else:
        history.append(final)

    history_path = os.path.join(output_directory, "materials_history.json")
    with open(history_path, "w", encoding="utf-8") as output:
        json.dump(history, output, indent=2)

    if not np.isclose(final["total_mass"], initial["total_mass"], rtol=1e-10, atol=1e-14):
        raise AssertionError(f"mass changed: {initial['total_mass']} -> {final['total_mass']}")
    if solver.particles.split_overflow[None] != 0:
        raise AssertionError(f"split overflow {solver.particles.split_overflow[None]} times")

    timing = {
        "setup_seconds": setup_seconds,
        "jit_warmup_seconds": jit_warmup_seconds,
        "simulation_seconds": simulation_seconds,
        "export_seconds": export_seconds,
        "total_measured_seconds": time.perf_counter() - total_start,
        "steps_per_second": steps / simulation_seconds,
        "milliseconds_per_step": 1e3 * simulation_seconds / steps,
    }
    result = {
        "solver": "adaptive_mpm",
        "precision": "float64",
        "constitutive_update": "mpm99_multi_material",
        "material_properties": mpm99_properties("multi"),
        "refinement": f"velocity_gradient_levels_0_to_{max_level}",
        "blocks": [
            {"name": b["name"], "material_id": b["material_id"],
             "xmin": b["xmin"], "ymin": b["ymin"], "size": BLOCK_SIZE}
            for b in BLOCKS
        ],
        "particles_per_axis_per_block": particles_per_axis,
        "initial_particles": n_initial,
        "steps": int(steps),
        "time": float(t),
        "dt": float(config.DT),
        "base_dx": float(config.AMR_BASE_DX),
        "finest_dx": float(DOMAIN_WIDTH / (base_cells * (2 ** max_level))),
        "grid_cells": [base_cells, base_cells],
        "max_level": max_level,
        "gradient_threshold": gradient_threshold,
        "gravity": list(GRAVITY),
        "initial": initial,
        "peak_particles": int(peak_particles),
        "final": final,
        "timing": timing,
        "history_file": history_path,
        "history_entries": len(history),
        "output_directory": output_directory,
    }
    result_path = os.path.join(output_directory, "materials_results.json")
    with open(result_path, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Three-material collision benchmark (adaptive MPM)")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--export-every", type=int, default=200)
    parser.add_argument("--particles-per-axis", type=int, default=None, help="auto-computed from native mass if not set")
    parser.add_argument("--max-level", type=int, default=1, help="max refinement level (default 1 = 1 split)")
    parser.add_argument("--base-cells", type=int, default=64, help="base grid resolution")
    parser.add_argument("--gradient-threshold", type=float, default=0.5, help="higher = less refinement")
    parser.add_argument("--output")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    if args.export_every < 1:
        raise ValueError("--export-every must be positive")
    ti.init(arch=ti.cpu if args.arch == "cpu" else ti.gpu, default_fp=ti.f64)
    steps = 24 if args.smoke else args.steps
    result = run_case(
        steps=steps,
        export_every=args.export_every,
        output_directory=args.output,
        export_vtk=not args.no_export,
        validate_run=not args.smoke,
        particles_per_axis=args.particles_per_axis,
        max_level=args.max_level,
        base_cells=args.base_cells,
        gradient_threshold=args.gradient_threshold,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
