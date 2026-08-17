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
from utils.exporter import write_boundary_vtk, write_vtk

MATERIAL_IDS = {"fluid": 0, "jelly": 1, "snow": 2}
BLOCK_BOUNDS = (0.010, 0.034, 0.018, 0.042)
BLOCK_CENTER_Y = 0.030
BLOCK_HALF_HEIGHT = 0.012
ACCEL_TIME = 0.006
CRUISE_TIME = 0.018
DECEL_TIME = 0.006
MAX_SPEED = 3.0
SHEAR_FRACTION = 0.08
TOTAL_TIME = ACCEL_TIME + CRUISE_TIME + DECEL_TIME


def configure(mode):
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


def export_frame(solver, frame, output_directory):
    n = solver.particles.n_active()
    write_vtk(frame, solver.particles.x.to_numpy()[:n], solver.particles.pressure.to_numpy()[:n],
              solver.particles.v.to_numpy()[:n], output_dir=output_directory,
              material=solver.particles.material.to_numpy()[:n])


def run_case(mode, material_name, steps, export_every):
    refinement_box = configure(mode)
    solver = ThreeMaterialAdaptiveMPMSolver2D(refinement_box=refinement_box, drive_block=False)
    initialize_case(solver, MATERIAL_IDS[material_name])
    output_directory = os.path.join(os.path.dirname(__file__), "output", f"{mode}_{material_name}")
    write_boundary_vtk(0.0, 0.0, config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT, output_dir=output_directory)
    export_frame(solver, 0, output_directory)
    initial = material_summary(solver)

    @ti.kernel
    def prescribe_motion(speed: ti.f64, shear: ti.f64):
        for p in range(solver.particles.active_count[None]):
            y_offset = solver.particles.x[p][1] - ti.cast(BLOCK_CENTER_Y, ti.f64)
            solver.particles.v[p] = ti.Vector([speed + shear * y_offset, 0.0])
            solver.particles.C[p] = ti.Matrix([[0.0, shear], [0.0, 0.0]])

    t = 0.0
    for step in range(1, steps + 1):
        speed = motion(t)
        shear = SHEAR_FRACTION * speed / BLOCK_HALF_HEIGHT
        prescribe_motion(speed, shear)
        solver.step(current_time=t)
        t += config.DT
        if step % export_every == 0 and step != steps:
            export_frame(solver, step, output_directory)
    prescribe_motion(0.0, 0.0)
    export_frame(solver, steps, output_directory)
    final = material_summary(solver)
    n = solver.particles.n_active()
    positions = solver.particles.x.to_numpy()[:n]
    velocity = solver.particles.v.to_numpy()[:n]
    if not np.isfinite(positions).all() or not np.isfinite(velocity).all():
        raise AssertionError(f"{mode} produced non-finite particle state")
    if not np.isclose(initial[material_name]["mass"], final[material_name]["mass"], rtol=1e-12, atol=1e-14):
        raise AssertionError(f"{mode}/{material_name} mass changed: {initial[material_name]['mass']} -> {final[material_name]['mass']}")
    if mode == "quadtree" and final[material_name]["particles"] > math.ceil(1.25 * initial[material_name]["particles"]):
        raise AssertionError(f"{mode}/{material_name} did not coarsen after leaving the refinement corridor")
    clearance = 0.1 * solver.grid.dx[0]
    if np.any(positions[:, 0] <= clearance) or np.any(positions[:, 0] >= config.AMR_DOMAIN_WIDTH - clearance):
        raise AssertionError(f"{mode} block reached a horizontal boundary")
    return {
        "mode": mode,
        "material": material_name,
        "steps": steps,
        "time": t,
        "prescribed_speed": motion(t),
        "max_particle_speed": float(np.linalg.norm(velocity, axis=1).max()),
        "initial": initial,
        "final": final,
        "output_directory": output_directory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--mode", choices=("quadtree", "gradient"))
    parser.add_argument("--material", choices=tuple(MATERIAL_IDS))
    parser.add_argument("--export-every", type=int, default=1000)
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
    output_path = os.path.join(os.path.dirname(__file__), "three_blocks_results.json")
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
