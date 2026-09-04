import argparse
import json
import math
import os
import sys

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

import taichi as ti
import config
from benchmarks.three_materials.run_experiment import ThreeMaterialAdaptiveMPMSolver2D
from utils.exporter import (
    write_boundary_vtk,
    write_dynamic_mpm_grid_vtk,
    write_dynamic_quadtree_grid_vtk,
    write_vtk,
)

BLOCK_BOUNDS = (0.010, 0.034, 0.018, 0.042)
BLOCK_CENTER_Y = 0.5 * (BLOCK_BOUNDS[2] + BLOCK_BOUNDS[3])
ACCEL_TIME = 0.006
CRUISE_TIME = 0.018
DECEL_TIME = 0.006
MAX_SPEED = 3.0
TOTAL_TIME = ACCEL_TIME + CRUISE_TIME + DECEL_TIME
LEVEL_1_PARTICLE_X = 0.021
LEVEL_2_PARTICLE_X = 0.026
REFINEMENT_BOX = ((0.025, 0.015), (0.040, 0.045))


def configure():
    config.ACTIVE_SCENARIO = "ADAPTIVE_MPM"
    config.USE_ADAPTIVE_MPM = True
    config.DIM = 2
    config.RHO_0 = 1.0
    config.GRAVITY = [0.0, 0.0]
    config.C_0 = 32.0
    config.V_MAX_ESTIMATE = MAX_SPEED
    config.MAX_WAVE_SPEED = config.C_0 + config.V_MAX_ESTIMATE
    config.CFL = 0.1
    config.DT = 4e-6
    config.FRAME_DT = 4e-6
    config.AMR_DOMAIN_MIN_X = 0.0
    config.AMR_DOMAIN_MIN_Y = 0.0
    config.AMR_DOMAIN_WIDTH = 0.12
    config.AMR_DOMAIN_HEIGHT = 0.06
    config.AMR_BASE_CELLS_X = 24
    config.AMR_BASE_CELLS_Y = 12
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
    config.AMR_PARTICLE_CAPACITY_FACTOR = 22.0
    config.AMR_INITIAL_FLUID_XMIN = BLOCK_BOUNDS[0]
    config.AMR_INITIAL_FLUID_XMAX = BLOCK_BOUNDS[1]
    config.AMR_INITIAL_FLUID_YMIN = BLOCK_BOUNDS[2]
    config.AMR_INITIAL_FLUID_YMAX = BLOCK_BOUNDS[3]
    config.AMR_INITIAL_PARTICLE_LEVEL = 0
    config.AMR_DYNAMIC_REFINEMENT = True
    config.AMR_DYNAMIC_REGRID_INTERVAL = 1
    config.AMR_REFINEMENT_CRITERION = "velocity"
    config.AMR_REFINEMENT_VELOCITY_FRACTION = 1e-6
    config.AMR_GRADIENT_REFINE = False
    config.INT_MOVINGRECT_XMIN = BLOCK_BOUNDS[0]
    config.INT_MOVINGRECT_XMAX = BLOCK_BOUNDS[1]
    config.INT_MOVINGRECT_YMIN = BLOCK_BOUNDS[2]
    config.INT_MOVINGRECT_YMAX = BLOCK_BOUNDS[3]
    config.PLATFORM_VELOCITY_X = MAX_SPEED
    config.PLATFORM_VELOCITY_Y = 0.0
    config.PLATFORM_STOP_TIME = CRUISE_TIME
    config.PLATFORM_DECEL_TIME = ACCEL_TIME + DECEL_TIME


def contains(x):
    return BLOCK_BOUNDS[0] <= x[0] < BLOCK_BOUNDS[1] and BLOCK_BOUNDS[2] <= x[1] < BLOCK_BOUNDS[3]


def initialize_case(solver):
    grid = solver.grid
    ppc = grid.ppc_axis
    base_positions = []
    for i in range(grid.base_cells_x):
        for j in range(grid.base_cells_y):
            cell_origin = grid.domain_min + np.array([i * grid.base_dx, j * grid.base_dx])
            for pi in range(ppc):
                for pj in range(ppc):
                    x = cell_origin + np.array([(pi + 0.5) * grid.base_dx / ppc, (pj + 0.5) * grid.base_dx / ppc])
                    if contains(x):
                        base_positions.append(x)
    base_volume = (grid.base_dx / ppc) ** 2
    positions = []
    levels = []
    masses = []
    volumes = []
    for x in base_positions:
        target = 0
        if x[0] >= LEVEL_1_PARTICLE_X:
            target = 1
        if x[0] >= LEVEL_2_PARTICLE_X:
            target = 2
        particles = [(x, base_volume)]
        for _ in range(target):
            children = []
            for parent_x, parent_volume in particles:
                offset = 0.25 * math.sqrt(parent_volume)
                child_volume = 0.25 * parent_volume
                for sx in (-1.0, 1.0):
                    for sy in (-1.0, 1.0):
                        children.append((parent_x + np.array([sx * offset, sy * offset]), child_volume))
            particles = children
        for particle_x, particle_volume in particles:
            positions.append(particle_x)
            levels.append(target)
            masses.append(config.RHO_0 * particle_volume)
            volumes.append(particle_volume)
    x = np.asarray(positions, dtype=np.float64)
    level = np.asarray(levels, dtype=np.int32)
    mass = np.asarray(masses, dtype=np.float64)
    volume = np.asarray(volumes, dtype=np.float64)
    n = len(x)
    if n == 0 or n > solver.particles.capacity:
        raise RuntimeError(f"invalid dynamic block particle count {n} for capacity {solver.particles.capacity}")
    def pad(values):
        return np.concatenate([values, np.zeros((solver.particles.capacity - n,) + values.shape[1:], dtype=values.dtype)])
    solver.particles.x.from_numpy(pad(x))
    solver.particles.v.from_numpy(np.zeros((solver.particles.capacity, 2), dtype=np.float64))
    solver.particles.C.from_numpy(np.zeros((solver.particles.capacity, 2, 2), dtype=np.float64))
    solver.particles.F.from_numpy(np.tile(np.eye(2, dtype=np.float64), (solver.particles.capacity, 1, 1)))
    solver.particles.stress.from_numpy(np.zeros((solver.particles.capacity, 2, 2), dtype=np.float64))
    solver.particles.pressure.from_numpy(np.zeros(solver.particles.capacity, dtype=np.float64))
    solver.particles.Jp.from_numpy(np.ones(solver.particles.capacity, dtype=np.float64))
    solver.particles.level.from_numpy(pad(level))
    solver.particles.mass.from_numpy(pad(mass))
    solver.particles.volume0.from_numpy(pad(volume))
    solver.particles.material.from_numpy(np.zeros(solver.particles.capacity, dtype=np.int32))
    solver.particles.gradient_level.from_numpy(np.zeros(solver.particles.capacity, dtype=np.int32))
    solver.particles.active_count[None] = n
    return len(base_positions)


def motion(t):
    if t < ACCEL_TIME:
        return MAX_SPEED * t / ACCEL_TIME
    if t < ACCEL_TIME + CRUISE_TIME:
        return MAX_SPEED
    if t < TOTAL_TIME:
        return MAX_SPEED * (TOTAL_TIME - t) / DECEL_TIME
    return 0.0


def displacement(t):
    if t < ACCEL_TIME:
        return 0.5 * MAX_SPEED * t * t / ACCEL_TIME
    acceleration_distance = 0.5 * MAX_SPEED * ACCEL_TIME
    cruise_elapsed = t - ACCEL_TIME
    if cruise_elapsed < CRUISE_TIME:
        return acceleration_distance + MAX_SPEED * cruise_elapsed
    deceleration_elapsed = cruise_elapsed - CRUISE_TIME
    if deceleration_elapsed < DECEL_TIME:
        return acceleration_distance + MAX_SPEED * CRUISE_TIME + MAX_SPEED * deceleration_elapsed - 0.5 * MAX_SPEED * deceleration_elapsed**2 / DECEL_TIME
    return MAX_SPEED * (CRUISE_TIME + 0.5 * (ACCEL_TIME + DECEL_TIME))


def summary(solver, frame, step, time):
    n = solver.particles.n_active()
    position = solver.particles.x.to_numpy()[:n]
    velocity = solver.particles.v.to_numpy()[:n]
    mass = solver.particles.mass.to_numpy()[:n]
    level = solver.particles.level.to_numpy()[:n]
    total_mass = float(mass.sum())
    center_of_mass = (mass[:, None] * position).sum(axis=0) / total_mass
    linear_momentum = mass[:, None] * velocity
    return {
        "frame": int(frame),
        "step": int(step),
        "time": float(time),
        "particles": int(n),
        "particles_by_level": [int((level == grid_level).sum()) for grid_level in range(solver.grid.num_levels)],
        "total_mass": total_mass,
        "center_of_mass": center_of_mass.tolist(),
        "total_linear_momentum": linear_momentum.sum(axis=0).tolist(),
        "max_speed": float(np.linalg.norm(velocity, axis=1).max()),
        "level_shifts": solver.grid.level_refinement_shift_np.tolist(),
        "region_min": solver.grid.region_min_np.tolist(),
        "region_max": solver.grid.region_max_np.tolist(),
    }


def export_frame(solver, frame, step, time, output_directory):
    n = solver.particles.n_active()
    position = solver.particles.x.to_numpy()[:n]
    velocity = solver.particles.v.to_numpy()[:n]
    mass = solver.particles.mass.to_numpy()[:n]
    level = solver.particles.level.to_numpy()[:n]
    linear_momentum = mass[:, None] * velocity
    write_vtk(
        frame,
        position,
        solver.particles.pressure.to_numpy()[:n],
        velocity,
        output_dir=output_directory,
        material=np.zeros(n, dtype=np.int32),
        point_scalars={
            "Mass": mass,
            "ParticleLevel": level,
            "LinearMomentumMagnitude": np.linalg.norm(linear_momentum, axis=1),
        },
        point_vectors={"LinearMomentum": linear_momentum},
    )
    write_dynamic_mpm_grid_vtk(solver.grid, frame, output_dir=output_directory)
    write_dynamic_quadtree_grid_vtk(solver.grid, frame, output_dir=output_directory)
    return summary(solver, frame, step, time)


def run_case(steps=None, export_every=50, output_directory=None, validate_run=True):
    configure()
    solver = ThreeMaterialAdaptiveMPMSolver2D(refinement_box=REFINEMENT_BOX, drive_block=False)
    source_particles = initialize_case(solver)
    if output_directory is None:
        output_directory = os.path.join(os.path.dirname(__file__), "output")
    write_boundary_vtk(0.0, 0.0, config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT, output_dir=output_directory)

    @ti.kernel
    def prescribe_rigid_state(speed: ti.f64):
        for p in range(solver.particles.active_count[None]):
            solver.particles.v[p] = ti.Vector([speed, 0.0])
            solver.particles.C[p] = ti.Matrix.zero(ti.f64, 2, 2)
            solver.particles.F[p] = ti.Matrix.identity(ti.f64, 2)
            solver.particles.stress[p] = ti.Matrix.zero(ti.f64, 2, 2)
            solver.particles.pressure[p] = 0.0
            solver.particles.Jp[p] = 1.0

    if steps is None:
        steps = int(math.ceil(TOTAL_TIME / config.DT))
    history = [export_frame(solver, 0, 0, 0.0, output_directory)]
    initial = history[0]
    initial_levels = solver.particles.level.to_numpy()[:solver.particles.n_active()].copy()
    distinct_shifts = {tuple(np.asarray(solver.grid.level_refinement_shift_np[-1]).round(12))}
    t = 0.0
    frame = 0
    for step in range(1, steps + 1):
        prescribe_rigid_state(motion(t))
        solver.grid.update_dynamic_refinement(t, solver.particles)
        solver.step(current_time=t, adapt_particles=False)
        t += config.DT
        prescribe_rigid_state(motion(t))
        distinct_shifts.add(tuple(np.asarray(solver.grid.level_refinement_shift_np[-1]).round(12)))
        if step % export_every == 0 and step != steps:
            frame += 1
            history.append(export_frame(solver, frame, step, t, output_directory))
    solver.grid.update_dynamic_refinement(t, solver.particles)
    if steps % export_every != 0 or history[-1]["step"] != steps:
        frame += 1
        history.append(export_frame(solver, frame, steps, t, output_directory))
    final = history[-1]
    final_levels = solver.particles.level.to_numpy()[:solver.particles.n_active()]
    w_min, w_max, _, g_max, n_violated = solver.check_partition_of_unity()
    history_path = os.path.join(output_directory, "dynamic_block_history.json")
    with open(history_path, "w", encoding="utf-8") as output:
        json.dump(history, output, indent=2)
    if not np.isclose(final["total_mass"], initial["total_mass"], rtol=1e-12, atol=1e-14):
        raise AssertionError(f"dynamic block mass changed: {initial['total_mass']} -> {final['total_mass']}")
    if solver.particles.split_overflow[None] != 0:
        raise AssertionError(f"dynamic block exhausted particle capacity {solver.particles.split_overflow[None]} times")
    if not np.array_equal(final_levels, initial_levels):
        raise AssertionError("dynamic block changed particle refinement assignments")
    if n_violated != 0:
        raise AssertionError(f"dynamic block has {n_violated} incomplete grid stencils")
    if validate_run:
        if final["level_shifts"][-1][0] <= 0.05:
            raise AssertionError("dynamic refinement window did not follow the block")
        if len(distinct_shifts) < 10:
            raise AssertionError(f"dynamic refinement produced only {len(distinct_shifts)} distinct finest-level positions")
        if any(count == 0 for count in initial["particles_by_level"]):
            raise AssertionError(f"dynamic block does not span all grid levels: {initial['particles_by_level']}")
        center_x = final["center_of_mass"][0]
        if not final["region_min"][1][0] < center_x < final["region_min"][2][0]:
            raise AssertionError("dynamic grid does not place coarser cells left and finer cells right of the block center")
    result = {
        "steps": int(steps),
        "frames": len(history),
        "time": float(t),
        "dt": float(config.DT),
        "criterion": config.AMR_REFINEMENT_CRITERION,
        "source_level_0_particles": source_particles,
        "particle_levels_fixed": True,
        "partition_of_unity": {"min": w_min, "max": w_max, "max_gradient_sum": g_max, "violations": n_violated},
        "distinct_finest_patch_positions": len(distinct_shifts),
        "initial": initial,
        "final": final,
        "history_file": history_path,
        "output_directory": output_directory,
    }
    result_path = os.path.join(output_directory, "dynamic_block_results.json")
    with open(result_path, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--export-every", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.export_every < 1:
        raise ValueError("--export-every must be positive")
    ti.init(arch=ti.cpu if args.arch == "cpu" else ti.gpu, default_fp=ti.f64)
    steps = 24 if args.smoke else args.steps
    result = run_case(
        steps=steps,
        export_every=args.export_every,
        output_directory=args.output,
        validate_run=not args.smoke and args.steps is None,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
