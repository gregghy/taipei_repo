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
from physics.constitutive_model import StressUsingWaterAdaptive
from solver.adaptive_engine import AdaptiveMPMSolver2D
from utils.exporter import write_boundary_vtk, write_vtk

COLUMN_MATERIAL_IDS = {"fluid": 0, "jelly": 1, "snow": 2}
MATERIAL_NAMES = {0: "fluid", 1: "jelly", 2: "snow", 4: "solid_block"}
BLOCK_MATERIAL = 4
COLUMN_BOUNDS = (0.050, 0.060, 0.010, 0.050)
OBJECT_BOUNDS = (0.010, 0.030, 0.020, 0.040)
OBJECT_SPEED = 6.0
DRIVE_STOP_X = 0.090


@ti.data_oriented
class ThreeMaterialAdaptiveMPMSolver2D(AdaptiveMPMSolver2D):
    def __init__(self, *args, drive_block=True, **kwargs):
        self.drive_block = drive_block
        super().__init__(*args, **kwargs)

    @ti.func
    def _g2p_level(self, p, level: ti.template(), t: float):
        x_p = self.particles.x[p]
        base, w0, w1, w2, _, _, _ = self._weights(level, x_p)
        v_new = ti.Vector.zero(ti.f64, 2)
        B_new = ti.Matrix.zero(ti.f64, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            I = base + ti.Vector([i, j])
            if self.grid.in_bounds(level, I):
                weight = self._weight_component(0, i, w0, w1, w2) * self._weight_component(1, j, w0, w1, w2)
                v_I = self.grid.v[level][I]
                dpos = self.grid.node_position(level, I) - x_p
                v_new += weight * v_I
                B_new += weight * v_I.outer_product(dpos)
        C_new = B_new * (4.0 * self.grid.level_inv_dx[level] * self.grid.level_inv_dx[level])
        v_speed = v_new.norm()
        if v_speed > 10.0:
            v_new = v_new * (10.0 / v_speed)
        c_norm = C_new.norm()
        c_clamp = 10.0 * self.grid.level_inv_dx[level]
        if c_norm > c_clamp:
            C_new = C_new * (c_clamp / c_norm)
        material = self.particles.material[p]
        if ti.static(self.drive_block):
            if material == BLOCK_MATERIAL and x_p[0] < ti.cast(DRIVE_STOP_X, ti.f64):
                v_new[0] = ti.max(v_new[0], ti.cast(OBJECT_SPEED, ti.f64))
        self.particles.v[p] = v_new
        self.particles.C[p] = C_new
        clearance = 0.1 * self.grid.level_dx[level]
        domain_min = self.grid.dynamic_domain_min[None]
        domain_max = self.grid.dynamic_domain_max[None]
        new_x = x_p + v_new * config.DT
        new_x[0] = ti.max(domain_min[0] + clearance, ti.min(new_x[0], domain_max[0] - clearance))
        new_x[1] = ti.max(domain_min[1] + clearance, ti.min(new_x[1], domain_max[1] - clearance))
        self.particles.x[p] = new_x
        identity = ti.Matrix.identity(ti.f64, 2)
        F_new = (identity + C_new * config.DT) @ self.particles.F[p]
        stress_new = ti.Matrix.zero(ti.f64, 2, 2)
        if material == 0:
            J = ti.max(F_new.determinant(), 0.96)
            F_new = identity * ti.sqrt(J)
            stress_new = StressUsingWaterAdaptive(F_new, C_new, self.grid.level_dx[level])
        else:
            U, sig, V = ti.svd(F_new)
            for d in ti.static(range(2)):
                sig[d, d] = ti.max(sig[d, d], 1e-6)
            hardening = 0.3
            if material == 2:
                hardening = ti.exp(10.0 * (1.0 - self.particles.Jp[p]))
            mu = ti.cast(1000.0 / (2.0 * 1.2), ti.f64) * hardening
            la = ti.cast(1000.0 * 0.2 / (1.2 * 0.6), ti.f64) * hardening
            J = 1.0
            for d in ti.static(range(2)):
                old_sig = sig[d, d]
                new_sig = old_sig
                if material == 2:
                    new_sig = ti.min(ti.max(old_sig, 1.0 - 2.5e-2), 1.0 + 4.5e-3)
                    self.particles.Jp[p] *= old_sig / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if material == 2:
                F_new = U @ sig @ V.transpose()
            stress_new = 2.0 * mu * (F_new - U @ V.transpose()) @ F_new.transpose()
            stress_new += identity * la * J * (J - 1.0)
        self.particles.F[p] = F_new
        self.particles.stress[p] = stress_new
        J_out = ti.max(F_new.determinant(), 0.1)
        self.particles.pressure[p] = ti.max(config.C_0**2 * config.RHO_0 * (1.0 / J_out - 1.0), 0.0)


def configure(mode):
    config.ACTIVE_SCENARIO = "ADAPTIVE_MPM"
    config.USE_ADAPTIVE_MPM = True
    config.DIM = 2
    config.RHO_0 = 1.0
    config.GRAVITY = [0.0, 0.0]
    config.C_0 = 32.0
    config.V_MAX_ESTIMATE = OBJECT_SPEED
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
    config.AMR_MATERIAL_COUNT = 5
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
        return ((0.036, 0.010), (0.084, 0.050))
    return ((0.0, 0.0), (config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT))


def contains(bounds, x):
    return bounds[0] <= x[0] < bounds[1] and bounds[2] <= x[1] < bounds[3]


def particle_state(solver, column_material):
    grid = solver.grid
    ppc = grid.ppc_axis
    positions = []
    levels = []
    masses = []
    volumes = []
    materials = []
    velocities = []
    if config.AMR_GRADIENT_REFINE:
        cells = [(0, grid.domain_min, grid.base_dx, grid.base_cells_x, grid.base_cells_y)]
    else:
        cells = []
        for level, origin, size in zip(grid.leaf_level, grid.leaf_origin, grid.leaf_size):
            cells.append((int(level), origin, float(size), 1, 1))
    for level, origin, dx, nx, ny in cells:
        for i in range(nx):
            for j in range(ny):
                cell_origin = origin + np.array([i * dx, j * dx])
                for pi in range(ppc):
                    for pj in range(ppc):
                        x = cell_origin + np.array([(pi + 0.5) * dx / ppc, (pj + 0.5) * dx / ppc])
                        object_particle = contains(OBJECT_BOUNDS, x)
                        if not object_particle and not contains(COLUMN_BOUNDS, x):
                            continue
                        positions.append(x)
                        levels.append(level)
                        volumes.append((dx / ppc) ** 2)
                        masses.append(config.RHO_0 * volumes[-1])
                        materials.append(BLOCK_MATERIAL if object_particle else column_material)
                        velocities.append([OBJECT_SPEED, 0.0] if object_particle else [0.0, 0.0])
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(levels, dtype=np.int32),
        np.asarray(masses, dtype=np.float64),
        np.asarray(volumes, dtype=np.float64),
        np.asarray(materials, dtype=np.int32),
        np.asarray(velocities, dtype=np.float64),
    )


def initialize_case(solver, column_material):
    x, level, mass, volume, material_ids, velocity = particle_state(solver, column_material)
    n = len(x)
    if n == 0 or n > solver.particles.capacity:
        raise RuntimeError(f"invalid benchmark particle count {n} for capacity {solver.particles.capacity}")
    def pad(values):
        shape = (solver.particles.capacity - n,) + values.shape[1:]
        return np.concatenate([values, np.zeros(shape, dtype=values.dtype)], axis=0)
    solver.particles.x.from_numpy(pad(x))
    solver.particles.v.from_numpy(pad(velocity))
    solver.particles.level.from_numpy(pad(level))
    solver.particles.mass.from_numpy(pad(mass))
    solver.particles.volume0.from_numpy(pad(volume))
    solver.particles.material.from_numpy(pad(material_ids))
    solver.particles.C.from_numpy(np.zeros((solver.particles.capacity, 2, 2), dtype=np.float64))
    solver.particles.F.from_numpy(np.tile(np.eye(2, dtype=np.float64), (solver.particles.capacity, 1, 1)))
    solver.particles.stress.from_numpy(np.zeros((solver.particles.capacity, 2, 2), dtype=np.float64))
    solver.particles.pressure.from_numpy(np.zeros(solver.particles.capacity, dtype=np.float64))
    solver.particles.Jp.from_numpy(np.ones(solver.particles.capacity, dtype=np.float64))
    solver.particles.gradient_level.from_numpy(np.zeros(solver.particles.capacity, dtype=np.int32))
    solver.particles.active_count[None] = n
    if mode_needs_adaptation(solver):
        solver._update_gradient_levels()
        solver._adapt_particles(complete=True)


def mode_needs_adaptation(solver):
    return solver.grid.max_level > 0


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
        for code, name in MATERIAL_NAMES.items()
        if np.any(material == code)
    }


def export_frame(solver, frame, output_directory):
    n = solver.particles.n_active()
    write_vtk(
        frame,
        solver.particles.x.to_numpy()[:n],
        solver.particles.pressure.to_numpy()[:n],
        solver.particles.v.to_numpy()[:n],
        output_dir=output_directory,
        material=solver.particles.material.to_numpy()[:n],
    )


def run_case(mode, column_material_name, steps, export_every, drive_block=True):
    refinement_box = configure(mode)
    solver = ThreeMaterialAdaptiveMPMSolver2D(refinement_box=refinement_box, drive_block=drive_block)
    initialize_case(solver, COLUMN_MATERIAL_IDS[column_material_name])
    output_directory = os.path.join(os.path.dirname(__file__), "output", f"{mode}_{column_material_name}")
    write_boundary_vtk(0.0, 0.0, config.AMR_DOMAIN_WIDTH, config.AMR_DOMAIN_HEIGHT, output_dir=output_directory)
    export_frame(solver, 0, output_directory)
    initial = material_summary(solver)
    t = 0.0
    for step in range(1, steps + 1):
        solver.step(current_time=t)
        t += config.DT
        if step % export_every == 0 or step == steps:
            export_frame(solver, step, output_directory)
    final = material_summary(solver)
    block_initial = initial["solid_block"]
    block_final = final.get("solid_block", {"mass": 0.0, "particles": 0, "levels": []})
    column_initial = initial[column_material_name]
    column_final = final.get(column_material_name, {"mass": 0.0, "particles": 0, "levels": []})
    if not np.isclose(block_final["mass"], block_initial["mass"], rtol=1e-12, atol=1e-14):
        raise AssertionError(f"{mode}/{column_material_name} block mass changed: {block_initial['mass']} -> {block_final['mass']}")
    if not np.isclose(column_final["mass"], column_initial["mass"], rtol=1e-12, atol=1e-14):
        raise AssertionError(f"{mode}/{column_material_name} column mass changed: {column_initial['mass']} -> {column_final['mass']}")
    positions = solver.particles.x.to_numpy()[:solver.particles.n_active()]
    materials = solver.particles.material.to_numpy()[:solver.particles.n_active()]
    if not np.isfinite(positions).all():
        raise AssertionError(f"{mode}/{column_material_name} produced non-finite particle positions")
    block_positions = positions[materials == BLOCK_MATERIAL]
    block_cleared_column = bool(np.all(block_positions[:, 0] >= COLUMN_BOUNDS[1]))
    return {
        "mode": mode,
        "column_material": column_material_name,
        "steps": steps,
        "time": t,
        "driven_block": drive_block,
        "block": {"initial": block_initial, "final": block_final},
        "column": {"initial": column_initial, "final": column_final},
        "block_cleared_column": block_cleared_column,
        "output_directory": output_directory,
        "active_particles": solver.particles.n_active(),
        "mass_total": float(solver.particles.mass.to_numpy()[:solver.particles.n_active()].sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=22000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--mode", choices=("quadtree", "gradient"))
    parser.add_argument("--column-material", choices=tuple(COLUMN_MATERIAL_IDS))
    parser.add_argument("--free-block", action="store_true")
    parser.add_argument("--export-every", type=int, default=1000)
    args = parser.parse_args()
    if args.export_every < 1:
        raise ValueError("--export-every must be positive")
    ti.init(arch=ti.cpu if args.arch == "cpu" else ti.gpu, default_fp=ti.f64)
    steps = 24 if args.smoke else args.steps
    results = []
    modes = (args.mode,) if args.mode else ("quadtree", "gradient")
    column_materials = (args.column_material,) if args.column_material else tuple(COLUMN_MATERIAL_IDS)
    for mode in modes:
        for column_material_name in column_materials:
            result = run_case(mode, column_material_name, steps, args.export_every, drive_block=not args.free_block)
            results.append(result)
            print(json.dumps(result, sort_keys=True))
    output_path = os.path.join(os.path.dirname(__file__), "three_materials_results.json")
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
