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
from utils.exporter import (
    write_boundary_vtk,
    write_mpm_grid_levels_vtk,
    write_mpm_grid_vtk,
    write_quadtree_grid_vtk,
    write_vtk,
)

BLOCK_BOUNDS = (0.045, 0.070, 0.045, 0.070)
TOTAL_TIME = 0.60


def configure(material="liquid"):
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
    config.AMR_BASE_CELLS_X = 24
    config.AMR_BASE_CELLS_Y = 16
    config.AMR_BASE_DX = config.AMR_DOMAIN_WIDTH / config.AMR_BASE_CELLS_X
    config.AMR_MAX_LEVEL = 2
    config.AMR_GRID_PADDING = 3
    config.AMR_REFINEMENT_BUFFER_CELLS = 1
    config.AMR_GHOST_BAND_CELLS = 2
    config.AMR_PARTICLES_PER_CELL_AXIS = 2
    config.AMR_SPLIT_PARTICLES = True
    config.AMR_MERGE_PARTICLES = True
    config.AMR_MERGE_MIN_PARTICLES = 4
    config.AMR_MATERIAL_COUNT = 1
    config.AMR_PARTICLE_CAPACITY_FACTOR = 20.0
    config.AMR_INITIAL_FLUID_XMIN = BLOCK_BOUNDS[0]
    config.AMR_INITIAL_FLUID_XMAX = BLOCK_BOUNDS[1]
    config.AMR_INITIAL_FLUID_YMIN = BLOCK_BOUNDS[2]
    config.AMR_INITIAL_FLUID_YMAX = BLOCK_BOUNDS[3]
    config.AMR_INITIAL_PARTICLE_LEVEL = 0
    config.AMR_DYNAMIC_REFINEMENT = False
    config.AMR_GRADIENT_REFINE = True
    config.AMR_GRADIENT_REFINE_THRESHOLD = 0.01
    config.AMR_GRADIENT_MAX_LEVEL = 2
    return ((0.0, 0.0), (config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT))


def initialize_freefall_state(solver):
    n = solver.particles.n_active()
    capacity = solver.particles.capacity
    solver.particles.v.from_numpy(np.zeros((capacity, 2), dtype=np.float64))
    solver.particles.C.from_numpy(np.zeros((capacity, 2, 2), dtype=np.float64))
    solver.particles.F.from_numpy(np.tile(np.eye(2, dtype=np.float64), (capacity, 1, 1)))
    solver.particles.stress.from_numpy(np.zeros((capacity, 2, 2), dtype=np.float64))
    solver.particles.pressure.from_numpy(np.zeros(capacity, dtype=np.float64))
    solver.particles.Jp.from_numpy(np.ones(capacity, dtype=np.float64))
    solver.particles.material.from_numpy(np.zeros(capacity, dtype=np.int32))
    solver.particles.gradient_level.from_numpy(np.zeros(capacity, dtype=np.int32))
    solver.particles.active_count[None] = n


def diagnostics(solver, frame, time):
    n = solver.particles.n_active()
    position = solver.particles.x.to_numpy()[:n]
    velocity = solver.particles.v.to_numpy()[:n]
    mass = solver.particles.mass.to_numpy()[:n]
    level = solver.particles.level.to_numpy()[:n]
    gradient_level = solver.particles.gradient_level.to_numpy()[:n]
    pressure = solver.particles.pressure.to_numpy()[:n]
    affine = solver.particles.C.to_numpy()[:n]
    total_mass = float(mass.sum())
    linear_momentum = mass[:, None] * velocity
    center_of_mass = (mass[:, None] * position).sum(axis=0) / total_mass
    kinetic_energy = 0.5 * float(np.sum(mass * np.sum(velocity * velocity, axis=1)))
    speed = np.linalg.norm(velocity, axis=1)
    gradient_indicator = np.linalg.norm(affine, axis=(1, 2)) * np.asarray(solver.grid.dx)[level]
    if not all(np.isfinite(values).all() for values in (position, velocity, mass, pressure, affine)):
        raise AssertionError("freefall produced non-finite particle state")
    point_scalars = {
        "Mass": mass,
        "ParticleLevel": level,
        "GradientTargetLevel": gradient_level,
        "GradientIndicator": gradient_indicator,
        "KineticEnergy": 0.5 * mass * speed**2,
        "LinearMomentumMagnitude": np.linalg.norm(linear_momentum, axis=1),
    }
    point_vectors = {"LinearMomentum": linear_momentum}
    summary = {
        "frame": int(frame),
        "time": float(time),
        "particles": int(n),
        "particles_by_level": [int((level == grid_level).sum()) for grid_level in range(solver.grid.num_levels)],
        "gradient_targets_by_level": [
            int((gradient_level == grid_level).sum()) for grid_level in range(solver.grid.num_levels)
        ],
        "total_mass": total_mass,
        "center_of_mass": center_of_mass.tolist(),
        "total_linear_momentum": linear_momentum.sum(axis=0).tolist(),
        "kinetic_energy": kinetic_energy,
        "max_speed": float(speed.max()),
        "max_pressure": float(np.abs(pressure).max()),
        "height_range": [float(position[:, 1].min()), float(position[:, 1].max())],
        "max_gradient_indicator": float(gradient_indicator.max()),
    }
    return position, pressure, velocity, point_scalars, point_vectors, summary


def export_frame(solver, frame, output_directory, time):
    position, pressure, velocity, point_scalars, point_vectors, summary = diagnostics(solver, frame, time)
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


def run_case(steps=None, export_every=500, output_directory=None, export_vtk=True, validate_run=True, material="liquid"):
    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    refinement_box = configure(material)
    solver = create_mpm99_solver(material, refinement_box=refinement_box)
    initialize_freefall_state(solver)
    initial_positions = solver.particles.x.to_numpy()
    ti.sync()
    setup_seconds = time.perf_counter() - setup_start
    warmup_start = time.perf_counter()
    solver.step(damping=1.0, current_time=0.0)
    ti.sync()
    jit_warmup_seconds = time.perf_counter() - warmup_start
    solver.particles.x.from_numpy(initial_positions)
    initialize_freefall_state(solver)
    solver.particles.split_overflow[None] = 0
    solver._step_count = 0
    ti.sync()
    if output_directory is None:
        output_directory = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_directory, exist_ok=True)
    if steps is None:
        steps = int(math.ceil(TOTAL_TIME / config.DT))
    initial = diagnostics(solver, 0, 0.0)[-1]
    history = []
    export_seconds = 0.0
    if export_vtk:
        export_start = time.perf_counter()
        write_boundary_vtk(0.0, 0.0, config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT, output_dir=output_directory)
        write_quadtree_grid_vtk(solver.grid, output_dir=output_directory)
        write_mpm_grid_vtk(solver.grid, output_dir=output_directory)
        write_mpm_grid_levels_vtk(solver.grid, output_dir=output_directory)
        history.append(export_frame(solver, 0, output_directory, 0.0))
        export_seconds += time.perf_counter() - export_start
    else:
        history.append(initial)
    peak_particles = initial["particles"]
    peak_levels = initial["particles_by_level"][:]
    simulation_seconds = 0.0
    segment_start = time.perf_counter()
    t = 0.0
    for step in range(1, steps + 1):
        solver.step(damping=1.0, current_time=t)
        t += config.DT
        n = solver.particles.n_active()
        if n > peak_particles:
            peak_particles = n
            level = solver.particles.level.to_numpy()[:n]
            peak_levels = [int((level == grid_level).sum()) for grid_level in range(solver.grid.num_levels)]
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
    history_path = os.path.join(output_directory, "freefall_history.json")
    with open(history_path, "w", encoding="utf-8") as output:
        json.dump(history, output, indent=2)
    if not np.isclose(final["total_mass"], initial["total_mass"], rtol=1e-12, atol=1e-14):
        raise AssertionError(f"freefall mass changed: {initial['total_mass']} -> {final['total_mass']}")
    if solver.particles.split_overflow[None] != 0:
        raise AssertionError(f"freefall exhausted particle capacity {solver.particles.split_overflow[None]} times")
    if validate_run:
        if final["center_of_mass"][1] >= initial["center_of_mass"][1]:
            raise AssertionError("freefall block did not fall")
        if material == "liquid":
            if peak_particles <= initial["particles"]:
                raise AssertionError("freefall impact did not trigger gradient refinement")
            if peak_levels[-1] == 0:
                raise AssertionError("freefall impact did not reach the finest particle level")
            if final["particles"] >= peak_particles:
                raise AssertionError("freefall did not naturally coarsen after peak refinement")
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
        "boundary_method": "weak_penalty_mass_and_particle_clamp",
        "constitutive_update": f"mpm99_{material}",
        "liquid_properties": mpm99_properties(material),
        "refinement": "velocity_gradient_levels_0_to_2",
        "steps": int(steps),
        "time": float(t),
        "dt": float(config.DT),
        "base_dx": float(config.AMR_BASE_DX),
        "grid_cells": [int(config.AMR_BASE_CELLS_X), int(config.AMR_BASE_CELLS_Y)],
        "gradient_threshold": float(config.AMR_GRADIENT_REFINE_THRESHOLD),
        "grid_damping": 1.0,
        "initial": initial,
        "peak_particles": int(peak_particles),
        "peak_particles_by_level": peak_levels,
        "final": final,
        "timing": timing,
        "history_file": history_path,
        "history_entries": len(history),
        "output_directory": output_directory,
    }
    result_path = os.path.join(output_directory, "freefall_results.json")
    with open(result_path, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int)
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
