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
    MATERIALS,
    TARGET_FINEST_DX,
    apply_mpm99_properties,
    cfl_stable_dt,
    create_mpm99_solver,
    mpm99_properties,
)
from utils.exporter import write_boundary_vtk, write_mpm_grid_vtk, write_vtk

BLOCK_BOUNDS = (0.045, 0.070, 0.045, 0.070)
TOTAL_TIME = 0.60


def configure(particle_count, material="liquid"):
    config.ACTIVE_SCENARIO = "ADAPTIVE_MPM"
    config.USE_ADAPTIVE_MPM = True
    config.DIM = 2
    apply_mpm99_properties(material)
    config.GRAVITY = [0.0, -9.81]
    config.V_MAX_ESTIMATE = 2.0
    config.MAX_WAVE_SPEED = config.C_0 + config.V_MAX_ESTIMATE
    config.CFL = 0.1
    config.DT = 5e-6
    dt_cfl = cfl_stable_dt(TARGET_FINEST_DX, material)
    config.DT = min(config.DT, dt_cfl)
    config.FRAME_DT = config.DT
    config.AMR_DOMAIN_MIN_X = 0.0
    config.AMR_DOMAIN_MIN_Y = 0.0
    config.AMR_DOMAIN_WIDTH = 0.12
    config.AMR_DOMAIN_HEIGHT = 0.08
    if particle_count == 100:
        config.AMR_BASE_CELLS_X = 24
        config.AMR_BASE_CELLS_Y = 16
    elif particle_count == 1500:
        config.AMR_BASE_CELLS_X = 96
        config.AMR_BASE_CELLS_Y = 64
    else:
        raise ValueError(f"unsupported fixed particle count {particle_count}")
    config.AMR_BASE_DX = config.AMR_DOMAIN_WIDTH / config.AMR_BASE_CELLS_X
    config.AMR_MAX_LEVEL = 0
    config.AMR_GRID_PADDING = 3
    config.AMR_GHOST_BAND_CELLS = 2
    config.AMR_PARTICLES_PER_CELL_AXIS = 2
    config.AMR_SPLIT_PARTICLES = False
    config.AMR_MERGE_PARTICLES = False
    config.AMR_MATERIAL_COUNT = 1
    config.AMR_INITIAL_FLUID_XMIN = 0.0
    config.AMR_INITIAL_FLUID_XMAX = config.AMR_DOMAIN_WIDTH
    config.AMR_INITIAL_FLUID_YMIN = 0.0
    config.AMR_INITIAL_FLUID_YMAX = config.AMR_DOMAIN_HEIGHT
    config.AMR_INITIAL_PARTICLE_LEVEL = 0
    config.AMR_DYNAMIC_REFINEMENT = False
    config.AMR_GRADIENT_REFINE = False
    return ((0.0, 0.0), (config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT))


def particle_positions(particle_count):
    width = BLOCK_BOUNDS[1] - BLOCK_BOUNDS[0]
    height = BLOCK_BOUNDS[3] - BLOCK_BOUNDS[2]
    positions = []
    if particle_count == 100:
        row_counts = [10] * 10
    elif particle_count == 1500:
        row_counts = [38 if i % 4 in (0, 3) else 37 for i in range(40)]
    else:
        raise ValueError(f"unsupported fixed particle count {particle_count}")
    for i, ny in enumerate(row_counts):
        x = BLOCK_BOUNDS[0] + (i + 0.5) * width / len(row_counts)
        for j in range(ny):
            y = BLOCK_BOUNDS[2] + (j + 0.5) * height / ny
            positions.append((x, y))
    return np.asarray(positions, dtype=np.float64)


def initialize_freefall_state(solver, particle_count):
    positions = particle_positions(particle_count)
    if len(positions) > solver.particles.capacity:
        raise RuntimeError(f"fixed particle count {len(positions)} exceeds capacity {solver.particles.capacity}")
    capacity = solver.particles.capacity
    volume = (BLOCK_BOUNDS[1] - BLOCK_BOUNDS[0]) * (BLOCK_BOUNDS[3] - BLOCK_BOUNDS[2]) / particle_count
    mass = config.RHO_0 * volume
    def pad(values):
        return np.concatenate([values, np.zeros((capacity - particle_count,) + values.shape[1:], dtype=values.dtype)])
    solver.particles.x.from_numpy(pad(positions))
    solver.particles.v.from_numpy(np.zeros((capacity, 2), dtype=np.float64))
    solver.particles.C.from_numpy(np.zeros((capacity, 2, 2), dtype=np.float64))
    solver.particles.F.from_numpy(np.tile(np.eye(2, dtype=np.float64), (capacity, 1, 1)))
    solver.particles.stress.from_numpy(np.zeros((capacity, 2, 2), dtype=np.float64))
    solver.particles.pressure.from_numpy(np.zeros(capacity, dtype=np.float64))
    solver.particles.Jp.from_numpy(np.ones(capacity, dtype=np.float64))
    solver.particles.material.from_numpy(np.zeros(capacity, dtype=np.int32))
    solver.particles.level.from_numpy(np.zeros(capacity, dtype=np.int32))
    solver.particles.mass.from_numpy(pad(np.full(particle_count, mass, dtype=np.float64)))
    solver.particles.volume0.from_numpy(pad(np.full(particle_count, volume, dtype=np.float64)))
    solver.particles.gradient_level.from_numpy(np.zeros(capacity, dtype=np.int32))
    solver.particles.active_count[None] = particle_count


def diagnostics(solver, frame, time_value):
    n = solver.particles.n_active()
    position = solver.particles.x.to_numpy()[:n]
    velocity = solver.particles.v.to_numpy()[:n]
    pressure = solver.particles.pressure.to_numpy()[:n]
    affine = solver.particles.C.to_numpy()[:n]
    mass = solver.particles.mass.to_numpy()[:n]
    linear_momentum = mass[:, None] * velocity
    total_mass = float(mass.sum())
    center_of_mass = (mass[:, None] * position).sum(axis=0) / total_mass
    speed = np.linalg.norm(velocity, axis=1)
    gradient_indicator = np.linalg.norm(affine, axis=(1, 2)) * config.AMR_BASE_DX
    if not all(np.isfinite(values).all() for values in (position, velocity, pressure, affine)):
        raise AssertionError("freefall_old produced non-finite particle state")
    point_scalars = {
        "Mass": mass,
        "ParticleLevel": np.zeros(n, dtype=np.int32),
        "GradientIndicator": gradient_indicator,
        "KineticEnergy": 0.5 * mass * speed**2,
        "LinearMomentumMagnitude": np.linalg.norm(linear_momentum, axis=1),
    }
    point_vectors = {"LinearMomentum": linear_momentum}
    summary = {
        "frame": int(frame),
        "time": float(time_value),
        "particles": int(n),
        "particles_by_level": [int(n)],
        "total_mass": total_mass,
        "center_of_mass": center_of_mass.tolist(),
        "total_linear_momentum": linear_momentum.sum(axis=0).tolist(),
        "kinetic_energy": 0.5 * float(np.sum(mass * speed**2)),
        "max_speed": float(speed.max()),
        "max_pressure": float(np.abs(pressure).max()),
        "height_range": [float(position[:, 1].min()), float(position[:, 1].max())],
        "max_gradient_indicator": float(gradient_indicator.max()),
    }
    return position, pressure, velocity, point_scalars, point_vectors, summary


def export_frame(solver, frame, output_directory, time_value):
    position, pressure, velocity, point_scalars, point_vectors, summary = diagnostics(solver, frame, time_value)
    write_vtk(
        frame,
        position,
        pressure,
        velocity,
        output_dir=output_directory,
        material=np.zeros(len(position), dtype=np.int32),
        point_scalars=point_scalars,
        point_vectors=point_vectors,
    )
    return summary


def run_case(particle_count=100, steps=None, export_every=500, output_directory=None, export_vtk=True, validate_run=True, material="liquid"):
    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    refinement_box = configure(particle_count, material)
    solver = create_mpm99_solver(material, refinement_box=refinement_box, max_level=0)
    initialize_freefall_state(solver, particle_count)
    initial_positions = solver.particles.x.to_numpy()
    ti.sync()
    setup_seconds = time.perf_counter() - setup_start
    warmup_start = time.perf_counter()
    solver.step(damping=1.0, current_time=0.0, adapt_particles=False)
    ti.sync()
    jit_warmup_seconds = time.perf_counter() - warmup_start
    solver.particles.x.from_numpy(initial_positions)
    initialize_freefall_state(solver, particle_count)
    solver._step_count = 0
    ti.sync()
    if output_directory is None:
        output_directory = os.path.join(os.path.dirname(__file__), "output", f"particles_{particle_count}")
    os.makedirs(output_directory, exist_ok=True)
    if steps is None:
        steps = int(math.ceil(TOTAL_TIME / config.DT))
    initial = diagnostics(solver, 0, 0.0)[-1]
    history = []
    export_seconds = 0.0
    if export_vtk:
        export_start = time.perf_counter()
        write_boundary_vtk(0.0, 0.0, config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT, output_dir=output_directory)
        write_mpm_grid_vtk(solver.grid, output_dir=output_directory)
        history.append(export_frame(solver, 0, output_directory, 0.0))
        export_seconds += time.perf_counter() - export_start
    else:
        history.append(initial)
    simulation_seconds = 0.0
    segment_start = time.perf_counter()
    t = 0.0
    for step in range(1, steps + 1):
        solver.step(damping=1.0, current_time=t, adapt_particles=False)
        t += config.DT
        if export_vtk and step % export_every == 0 and step != steps:
            ti.sync()
            simulation_seconds += time.perf_counter() - segment_start
            export_start = time.perf_counter()
            history.append(export_frame(solver, step, output_directory, t))
            export_seconds += time.perf_counter() - export_start
            segment_start = time.perf_counter()
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
    if not np.isclose(final["total_mass"], initial["total_mass"], rtol=1e-12, atol=1e-14):
        raise AssertionError(f"freefall_old mass changed: {initial['total_mass']} -> {final['total_mass']}")
    if final["particles"] != particle_count:
        raise AssertionError(f"freefall_old particle count changed: {particle_count} -> {final['particles']}")
    if validate_run and final["center_of_mass"][1] >= initial["center_of_mass"][1]:
        raise AssertionError("freefall_old block did not fall")
    timing = {
        "setup_seconds": setup_seconds,
        "jit_warmup_seconds": jit_warmup_seconds,
        "simulation_seconds": simulation_seconds,
        "export_seconds": export_seconds,
        "total_measured_seconds": time.perf_counter() - total_start,
        "steps_per_second": steps / simulation_seconds,
        "milliseconds_per_step": 1e3 * simulation_seconds / steps,
    }
    history_path = os.path.join(output_directory, "freefall_old_history.json")
    with open(history_path, "w", encoding="utf-8") as output:
        json.dump(history, output, indent=2)
    result = {
        "solver": "fixed_single_level_mpm",
        "precision": "float64",
        "boundary_method": "weak_penalty_mass_and_particle_clamp",
        "constitutive_update": f"mpm99_{material}",
        "liquid_properties": mpm99_properties(material),
        "refinement": "none",
        "requested_particles": int(particle_count),
        "initial_layout": "10x10" if particle_count == 100 else "40 columns with symmetric alternating 38 and 37 rows",
        "steps": int(steps),
        "time": float(t),
        "dt": float(config.DT),
        "base_dx": float(config.AMR_BASE_DX),
        "grid_cells": [int(config.AMR_BASE_CELLS_X), int(config.AMR_BASE_CELLS_Y)],
        "grid_damping": 1.0,
        "initial": initial,
        "final": final,
        "timing": timing,
        "history_file": history_path,
        "history_entries": len(history),
        "output_directory": output_directory,
    }
    result_path = os.path.join(output_directory, "freefall_old_results.json")
    with open(result_path, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int)
    parser.add_argument("--particles", type=int, choices=(100, 1500), default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--material", choices=MATERIALS, default="liquid")
    parser.add_argument("--export-every", type=int, default=500)
    parser.add_argument("--output")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    if args.export_every < 1:
        raise ValueError("--export-every must be positive")
    ti.init(arch=ti.cpu if args.arch == "cpu" else ti.gpu, default_fp=ti.f64)
    steps = 24 if args.smoke else args.steps
    result = run_case(
        particle_count=args.particles,
        steps=steps,
        export_every=args.export_every,
        output_directory=args.output,
        export_vtk=not args.no_export,
        validate_run=not args.smoke and args.steps is None,
        material=args.material,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
