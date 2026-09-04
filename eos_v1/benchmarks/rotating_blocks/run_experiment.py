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
from utils.exporter import write_boundary_vtk, write_mpm_grid_levels_vtk, write_mpm_grid_vtk, write_quadtree_grid_vtk, write_vtk

MATERIAL_IDS = {"fluid": 0, "jelly": 1, "snow": 2}
BLOCK_BOUNDS = (0.010, 0.034, 0.018, 0.042)
BLOCK_CENTER_X0 = 0.5 * (BLOCK_BOUNDS[0] + BLOCK_BOUNDS[1])
BLOCK_CENTER_Y = 0.5 * (BLOCK_BOUNDS[2] + BLOCK_BOUNDS[3])
BLOCK_HALF_WIDTH = 0.5 * (BLOCK_BOUNDS[1] - BLOCK_BOUNDS[0])
BLOCK_HALF_HEIGHT = 0.5 * (BLOCK_BOUNDS[3] - BLOCK_BOUNDS[2])
ACCEL_TIME = 0.006
CRUISE_TIME = 0.018
DECEL_TIME = 0.006
MAX_SPEED = 3.0
MAX_ANGULAR_SPEED = 40.0
TOTAL_TIME = ACCEL_TIME + CRUISE_TIME + DECEL_TIME


def configure(mode):
    config.ACTIVE_SCENARIO = "ADAPTIVE_MPM"
    config.USE_ADAPTIVE_MPM = True
    config.DIM = 2
    config.RHO_0 = 1.0
    config.GRAVITY = [0.0, 0.0]
    config.C_0 = 32.0
    config.V_MAX_ESTIMATE = MAX_SPEED + MAX_ANGULAR_SPEED * max(BLOCK_HALF_WIDTH, BLOCK_HALF_HEIGHT)
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
    config.AMR_MATERIAL_COUNT = 3
    config.AMR_PARTICLE_CAPACITY_FACTOR = 12.0
    config.AMR_INITIAL_FLUID_XMIN = 0.0
    config.AMR_INITIAL_FLUID_XMAX = config.AMR_DOMAIN_WIDTH
    config.AMR_INITIAL_FLUID_YMIN = 0.0
    config.AMR_INITIAL_FLUID_YMAX = config.AMR_DOMAIN_HEIGHT
    config.AMR_INITIAL_PARTICLE_LEVEL = 0
    config.AMR_DYNAMIC_REFINEMENT = False
    config.AMR_DYNAMIC_REGRID_INTERVAL = 1
    config.AMR_GRADIENT_REFINE = mode == "gradient"
    config.AMR_GRADIENT_REFINE_THRESHOLD = 0.003
    config.AMR_GRADIENT_MAX_LEVEL = 2
    if mode == "quadtree":
        return ((0.040, 0.014), (0.070, 0.046))
    return ((0.0, 0.0), (config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT))


def contains(x):
    return BLOCK_BOUNDS[0] <= x[0] < BLOCK_BOUNDS[1] and BLOCK_BOUNDS[2] <= x[1] < BLOCK_BOUNDS[3]


def build_state(solver, material):
    grid = solver.grid
    ppc = grid.ppc_axis
    positions = []
    levels = []
    masses = []
    volumes = []
    materials = []
    if config.AMR_GRADIENT_REFINE:
        cells = [(0, grid.domain_min, grid.base_dx, grid.base_cells_x, grid.base_cells_y)]
    else:
        cells = [(int(level), origin, float(size), 1, 1) for level, origin, size in zip(grid.leaf_level, grid.leaf_origin, grid.leaf_size)]
    for level, origin, dx, nx, ny in cells:
        for i in range(nx):
            for j in range(ny):
                cell_origin = origin + np.array([i * dx, j * dx])
                for pi in range(ppc):
                    for pj in range(ppc):
                        x = cell_origin + np.array([(pi + 0.5) * dx / ppc, (pj + 0.5) * dx / ppc])
                        if not contains(x):
                            continue
                        positions.append(x)
                        levels.append(level)
                        volume = (dx / ppc) ** 2
                        volumes.append(volume)
            export_frame(solver, step, output_directory)
                        masses.append(config.RHO_0 * volume)
                        materials.append(material)
    return tuple(np.asarray(values, dtype=dtype) for values, dtype in (
        (positions, np.float64), (levels, np.int32), (masses, np.float64),
        (volumes, np.float64), (materials, np.int32),
    ))


def initialize_case(solver, material):
    x, level, mass, volume, material_ids = build_state(solver, material)
    n = len(x)
    if n == 0 or n > solver.particles.capacity:
        raise RuntimeError(f"invalid block particle count {n} for capacity {solver.particles.capacity}")
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
    solver.particles.material.from_numpy(pad(material_ids))
    solver.particles.gradient_level.from_numpy(np.zeros(solver.particles.capacity, dtype=np.int32))
    solver.particles.active_count[None] = n
    solver._update_gradient_levels()
    solver._adapt_particles(complete=True)


def motion(t):
    if t < ACCEL_TIME:
        return MAX_SPEED * t / ACCEL_TIME
    if t < ACCEL_TIME + CRUISE_TIME:
        return MAX_SPEED
    if t < TOTAL_TIME:
        return MAX_SPEED * (TOTAL_TIME - t) / DECEL_TIME
    return 0.0


def angular_motion(t):
    if t < ACCEL_TIME:
        return MAX_ANGULAR_SPEED * t / ACCEL_TIME
    if t < ACCEL_TIME + CRUISE_TIME:
        return MAX_ANGULAR_SPEED
    if t < TOTAL_TIME:
        return MAX_ANGULAR_SPEED * (TOTAL_TIME - t) / DECEL_TIME
    return 0.0


def displacement(t):
    if t < ACCEL_TIME:
        return 0.5 * MAX_SPEED * t * t / ACCEL_TIME
    d1 = 0.5 * MAX_SPEED * ACCEL_TIME
    t2 = t - ACCEL_TIME
    if t2 < CRUISE_TIME:
        return d1 + MAX_SPEED * t2
    d2 = MAX_SPEED * CRUISE_TIME
    t3 = t2 - CRUISE_TIME
    if t3 < DECEL_TIME:
        return d1 + d2 + MAX_SPEED * t3 - 0.5 * MAX_SPEED * t3 * t3 / DECEL_TIME
    return d1 + d2 + 0.5 * MAX_SPEED * DECEL_TIME


def material_summary(solver):
    n = solver.particles.n_active()
    mass = solver.particles.mass.to_numpy()[:n]
    material = solver.particles.material.to_numpy()[:n]
    level = solver.particles.level.to_numpy()[:n]
    return {
        name: {
            "mass": float(mass[material == code].sum()),
            "particles": int((material == code).sum()),
            "levels": [int(((material == code) & (level == l)).sum()) for l in range(solver.grid.num_levels)],
        }
        for name, code in MATERIAL_IDS.items()
    }


def momentum_diagnostics(solver, frame, time):
    n = solver.particles.n_active()
    position = solver.particles.x.to_numpy()[:n]
    velocity = solver.particles.v.to_numpy()[:n]
    mass = solver.particles.mass.to_numpy()[:n]
    level = solver.particles.level.to_numpy()[:n]
    affine = solver.particles.C.to_numpy()[:n]
    total_mass = float(mass.sum())
    if total_mass <= 0.0:
        raise AssertionError("momentum diagnostics require positive total mass")
    linear_momentum = mass[:, None] * velocity
    total_linear_momentum = linear_momentum.sum(axis=0)
    center_of_mass = (mass[:, None] * position).sum(axis=0) / total_mass
    center_of_mass_velocity = total_linear_momentum / total_mass
    relative_position = position - center_of_mass
    relative_momentum = mass[:, None] * (velocity - center_of_mass_velocity)
    orbital_angular_momentum = (
        relative_position[:, 0] * relative_momentum[:, 1]
        - relative_position[:, 1] * relative_momentum[:, 0]
    )
    level_dx = np.asarray(solver.grid.dx, dtype=np.float64)[level]
    affine_angular_momentum = 0.25 * mass * level_dx**2 * (affine[:, 1, 0] - affine[:, 0, 1])
    angular_momentum = orbital_angular_momentum + affine_angular_momentum
    linear_momentum_magnitude = np.linalg.norm(linear_momentum, axis=1)
    relative_momentum_magnitude = np.linalg.norm(relative_momentum, axis=1)
    arrays = (
        position, velocity, mass, linear_momentum, relative_position, relative_momentum,
        orbital_angular_momentum, affine_angular_momentum, angular_momentum,
    )
    if not all(np.isfinite(values).all() for values in arrays):
        raise AssertionError("non-finite particle momentum diagnostic")
    relative_residual = relative_momentum.sum(axis=0)
    if not np.allclose(relative_residual, 0.0, rtol=1e-12, atol=1e-14):
        raise AssertionError(f"center-of-mass momentum residual is {relative_residual.tolist()}")
    point_scalars = {
        "Mass": mass,
        "ParticleLevel": level,
        "LinearMomentumMagnitude": linear_momentum_magnitude,
        "COMRelativeMomentumMagnitude": relative_momentum_magnitude,
        "OrbitalAngularMomentumCOM": orbital_angular_momentum,
        "AffineAngularMomentum": affine_angular_momentum,
        "AngularMomentumCOM": angular_momentum,
        "AngularMomentumMagnitude": np.abs(angular_momentum),
    }
    point_vectors = {
        "LinearMomentum": linear_momentum,
        "COMRelativeMomentum": relative_momentum,
        "PositionRelativeToCOM": relative_position,
    }
    summary = {
        "frame": int(frame),
        "time": float(time),
        "particles": int(n),
        "particles_by_level": [int((level == grid_level).sum()) for grid_level in range(solver.grid.num_levels)],
        "total_mass": total_mass,
        "center_of_mass": center_of_mass.tolist(),
        "center_of_mass_velocity": center_of_mass_velocity.tolist(),
        "total_linear_momentum": total_linear_momentum.tolist(),
        "total_linear_momentum_magnitude": float(np.linalg.norm(total_linear_momentum)),
        "total_orbital_angular_momentum_com": float(orbital_angular_momentum.sum()),
        "total_affine_angular_momentum": float(affine_angular_momentum.sum()),
        "total_angular_momentum_com": float(angular_momentum.sum()),
        "sum_absolute_angular_momentum": float(np.abs(angular_momentum).sum()),
        "max_particle_linear_momentum": float(linear_momentum_magnitude.max()),
        "max_particle_angular_momentum": float(np.abs(angular_momentum).max()),
        "relative_linear_momentum_residual": relative_residual.tolist(),
    }
    return point_scalars, point_vectors, summary


def export_frame(solver, frame, output_directory, time):
    n = solver.particles.n_active()
    point_scalars, point_vectors, summary = momentum_diagnostics(solver, frame, time)
    write_vtk(frame, solver.particles.x.to_numpy()[:n], solver.particles.pressure.to_numpy()[:n],
              solver.particles.v.to_numpy()[:n], output_dir=output_directory,
              material=solver.particles.material.to_numpy()[:n],
              point_scalars=point_scalars, point_vectors=point_vectors)
    return summary


def run_case(mode, material_name, steps, export_every):
    refinement_box = configure(mode)
    solver = ThreeMaterialAdaptiveMPMSolver2D(refinement_box=refinement_box, drive_block=False)
    initialize_case(solver, MATERIAL_IDS[material_name])
    output_directory = os.path.join(os.path.dirname(__file__), "output", f"{mode}_{material_name}")
    write_boundary_vtk(0.0, 0.0, config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT, output_dir=output_directory)
    write_quadtree_grid_vtk(solver.grid, output_dir=output_directory)
    write_mpm_grid_vtk(solver.grid, output_dir=output_directory)
    write_mpm_grid_levels_vtk(solver.grid, output_dir=output_directory)
    momentum_history = [export_frame(solver, 0, output_directory, 0.0)]
    initial = material_summary(solver)

    @ti.kernel
    def prescribe_rigid_state(speed: ti.f64, omega: ti.f64, cx: ti.f64, cy: ti.f64):
        for p in range(solver.particles.active_count[None]):
            dx_ = solver.particles.x[p][0] - cx
            dy_ = solver.particles.x[p][1] - cy
            solver.particles.v[p] = ti.Vector([speed - omega * dy_, omega * dx_])
            solver.particles.C[p] = ti.Matrix([[0.0, -omega], [omega, 0.0]])
            solver.particles.F[p] = ti.Matrix.identity(ti.f64, 2)
            solver.particles.stress[p] = ti.Matrix.zero(ti.f64, 2, 2)
            solver.particles.pressure[p] = 0.0
            solver.particles.Jp[p] = 1.0

    t = 0.0
    for step in range(1, steps + 1):
        speed = motion(t)
        omega = angular_motion(t)
        cx = BLOCK_CENTER_X0 + displacement(t)
        cy = BLOCK_CENTER_Y
        prescribe_rigid_state(speed, omega, cx, cy)
        solver.step(current_time=t)
        t += config.DT
        speed_next = motion(t)
        omega_next = angular_motion(t)
        cx_next = BLOCK_CENTER_X0 + displacement(t)
        prescribe_rigid_state(speed_next, omega_next, cx_next, BLOCK_CENTER_Y)
        if step % export_every == 0 and step != steps:
            momentum_history.append(export_frame(solver, step, output_directory, t))
    prescribe_rigid_state(0.0, 0.0, BLOCK_CENTER_X0 + displacement(TOTAL_TIME), BLOCK_CENTER_Y)
    momentum_history.append(export_frame(solver, steps, output_directory, t))
    momentum_history_path = os.path.join(output_directory, "momentum_history.json")
    with open(momentum_history_path, "w", encoding="utf-8") as output:
        json.dump(momentum_history, output, indent=2)
    peak_particle_count = max(entry["particles"] for entry in momentum_history)
    final = material_summary(solver)
    n = solver.particles.n_active()
    positions = solver.particles.x.to_numpy()[:n]
    velocity = solver.particles.v.to_numpy()[:n]
    pressure = solver.particles.pressure.to_numpy()[:n]
    stress = solver.particles.stress.to_numpy()[:n]
    deformation = solver.particles.F.to_numpy()[:n]
    if not np.isfinite(positions).all() or not np.isfinite(velocity).all():
        raise AssertionError(f"{mode} produced non-finite particle state")
    if not np.isclose(initial[material_name]["mass"], final[material_name]["mass"], rtol=1e-12, atol=1e-14):
        raise AssertionError(f"{mode}/{material_name} mass changed: {initial[material_name]['mass']} -> {final[material_name]['mass']}")
    if mode == "quadtree":
        if final[material_name]["levels"][-1] != 0:
            raise AssertionError(f"{mode}/{material_name} retained finest-level particles after leaving the corridor")
        if (peak_particle_count > initial[material_name]["particles"]
                and final[material_name]["particles"] >= peak_particle_count):
            raise AssertionError(f"{mode}/{material_name} did not reduce its refined particle count")
    clearance = 0.1 * solver.grid.dx[0]
    if np.any(positions[:, 0] <= clearance) or np.any(positions[:, 0] >= config.AMR_DOMAIN_WIDTH - clearance):
        raise AssertionError(f"{mode} block reached a horizontal boundary")
    max_pressure = float(np.abs(pressure).max())
    max_stress_norm = float(np.linalg.norm(stress, axis=(1, 2)).max())
    max_deformation_error = float(np.linalg.norm(deformation - np.eye(2), axis=(1, 2)).max())
    if max_pressure > 1e-12 or max_stress_norm > 1e-12 or max_deformation_error > 1e-12:
        raise AssertionError(f"{mode}/{material_name} was not traction-free at the final state")
    total_rotation = 0.5 * MAX_ANGULAR_SPEED * ACCEL_TIME + MAX_ANGULAR_SPEED * CRUISE_TIME + 0.5 * MAX_ANGULAR_SPEED * DECEL_TIME
    return {
        "mode": mode,
        "material": material_name,
        "steps": steps,
        "time": t,
        "prescribed_speed": motion(t),
        "prescribed_angular_speed": angular_motion(t),
        "total_rotation_rad": total_rotation,
        "total_rotation_deg": math.degrees(total_rotation),
        "peak_particle_count": peak_particle_count,
        "final_to_peak_particle_ratio": final[material_name]["particles"] / peak_particle_count,
        "max_particle_speed": float(np.linalg.norm(velocity, axis=1).max()),
        "max_pressure": max_pressure,
        "max_stress_norm": max_stress_norm,
        "max_deformation_error": max_deformation_error,
        "initial": initial,
        "final": final,
        "initial_momentum": momentum_history[0],
        "final_momentum": momentum_history[-1],
        "momentum_history_file": momentum_history_path,
        "momentum_history_entries": len(momentum_history),
        "output_directory": output_directory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--mode", choices=("quadtree", "gradient"))
    parser.add_argument("--material", choices=tuple(MATERIAL_IDS))
    parser.add_argument("--export-every", type=int, default=100)
    args = parser.parse_args()
    if args.export_every < 1:
        raise ValueError("--export-every must be positive")
    ti.init(arch=ti.cpu if args.arch == "cpu" else ti.gpu, default_fp=ti.f64)
    steps = 24 if args.smoke else args.steps
    modes = (args.mode,) if args.mode else ("quadtree", "gradient")
    materials = (args.material,) if args.material else tuple(MATERIAL_IDS)
    results = [run_case(mode, material, steps, args.export_every) for mode in modes for material in materials]
    for result in results:
        print(json.dumps(result, sort_keys=True))
    output_path = os.path.join(os.path.dirname(__file__), "rotating_blocks_results.json")
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
