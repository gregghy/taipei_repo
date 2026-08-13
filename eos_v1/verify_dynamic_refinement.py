import math
import os
import numpy as np
import taichi as ti
import config

config.ACTIVE_SCENARIO = "IMMERSED"
config.USE_ADAPTIVE_MPM = True
config.DIM = 2
config.PADDING = 3
config.GRID_WIDTH = 0.02
config.GRID_HEIGHT = 0.04
config.N_CELL_WIDTH = 16
config.N_CELL_HEIGHT = 32
config.DX = config.GRID_WIDTH / config.N_CELL_WIDTH
config.DY = config.GRID_HEIGHT / config.N_CELL_HEIGHT
config.INV_DX = 1.0 / config.DX
config.INV_DY = 1.0 / config.DY
config.GRID_RES_X = config.N_CELL_WIDTH + 2 * config.PADDING + 1
config.GRID_RES_Y = config.N_CELL_HEIGHT + 2 * config.PADDING + 1
config.POS_MP_LEFT_BOTTOM = [config.PADDING * config.DX, config.PADDING * config.DY]
config.MP_WIDTH = config.GRID_WIDTH
config.MP_HEIGHT = 0.03
config.GRAVITY = [0.0, -9.81]
config.RHO_0 = 1000.0
config.CFL = 0.1
config.G_MAG = 9.81
config.C_0 = 10.0 * math.sqrt(2.0 * config.G_MAG * config.MP_HEIGHT)
config.V_MAX_ESTIMATE = math.sqrt(2.0 * config.G_MAG * config.MP_HEIGHT)
config.DT = 1e-5
config.INT_MOVINGRECT_XMIN = 0.007
config.INT_MOVINGRECT_XMAX = 0.013
config.INT_MOVINGRECT_YMIN = 0.012
config.INT_MOVINGRECT_YMAX = 0.016
config.PLATFORM_VELOCITY_X = 0.002
config.PLATFORM_VELOCITY_Y = -0.01
config.PLATFORM_STOP_TIME = 0.5
config.PLATFORM_DECEL_TIME = 0.2
config.AMR_MAX_LEVEL = 2
config.AMR_REFINEMENT_BUFFER_CELLS = 2
config.AMR_GHOST_BAND_CELLS = 2
config.AMR_PARTICLES_PER_CELL_AXIS = 2
config.AMR_SPLIT_PARTICLES = True
config.AMR_MERGE_PARTICLES = True
config.AMR_MERGE_MIN_PARTICLES = 4
config.AMR_PARTICLE_CAPACITY_FACTOR = 4.0
config.AMR_DYNAMIC_REFINEMENT = True
config.AMR_GRADIENT_REFINE = True
config.AMR_DYNAMIC_REGRID_INTERVAL = 1
config.AMR_PROCESS_MARGIN = 0.0015
config.AMR_DYNAMIC_PLATFORM_MARGIN_Y = 0.002

ti.init(arch=ti.gpu if os.environ.get("TAICHI_ARCH") == "gpu" else ti.cpu)

from solver.adaptive_engine import AdaptiveMPMSolver2D

solver = AdaptiveMPMSolver2D()
grid = solver.grid
initial_min = grid.region_min.to_numpy()
initial_max = grid.region_max.to_numpy()
initial_origin = grid.origin.to_numpy()

grid.clear()
grid.initialize_dynamic_penalty_mass()
dynamic_domain_mass = [field.to_numpy() for field in grid.boundary_mass]
reference_domain_mass = [field.to_numpy() for field in grid.domain_boundary_mass]
for dynamic, reference in zip(dynamic_domain_mass, reference_domain_mass):
    assert np.allclose(dynamic.sum(axis=(0, 1)), reference.sum(axis=(0, 1)), rtol=1e-12, atol=1e-12)

penalty_time = 0.4
grid.update_moving_platform_penalty_mass(penalty_time)
reference_platform_mass = [field.to_numpy() for field in grid.moving_boundary_mass]
reference_platform_momentum = [field.to_numpy() for field in grid.moving_boundary_momentum]
grid.clear()
grid.initialize_dynamic_penalty_mass()
grid.add_moving_platform_penalty_mass_gpu(penalty_time)
for dynamic, domain, reference in zip(grid.boundary_mass, dynamic_domain_mass, reference_platform_mass):
    dynamic_sum = (dynamic.to_numpy() - domain).sum(axis=(0, 1))
    reference_sum = reference.sum(axis=(0, 1))
    assert np.allclose(dynamic_sum, reference_sum, rtol=1e-12, atol=1e-12), (dynamic_sum, reference_sum)
for dynamic, reference in zip(grid.boundary_momentum, reference_platform_momentum):
    assert np.allclose(dynamic.to_numpy().sum(axis=(0, 1)), reference.sum(axis=(0, 1)), rtol=1e-12, atol=1e-12)

sample_time = 0.75
assert grid.update_dynamic_refinement(sample_time)
updated_min = grid.region_min.to_numpy()
updated_max = grid.region_max.to_numpy()
updated_origin = grid.origin.to_numpy()
level_shifts = grid.level_refinement_shift.to_numpy()
shift = grid.refinement_shift.to_numpy()
assert not grid.update_dynamic_refinement(sample_time)

assert shift[0] > 0.0 and shift[1] < 0.0
assert np.allclose(level_shifts[0], 0.0)
assert level_shifts[1, 1] > level_shifts[-1, 1]
assert np.allclose(shift, level_shifts[-1])
assert np.allclose(updated_min[0], initial_min[0])
assert np.allclose(updated_max[0], initial_max[0])
assert np.allclose(updated_origin[0], initial_origin[0])
assert np.allclose(updated_min - initial_min, level_shifts)
assert np.allclose(updated_max - initial_max, level_shifts)
assert np.allclose(updated_origin - initial_origin, level_shifts)
assert np.allclose(grid.region_min_np, updated_min)
assert np.allclose(grid.region_max_np, updated_max)
assert np.allclose(grid.origin_np, updated_origin)

for time in np.linspace(0.0, config.PLATFORM_STOP_TIME + config.PLATFORM_DECEL_TIME, 33):
    grid.update_dynamic_refinement(time)
    regions_min = grid.region_min.to_numpy()
    regions_max = grid.region_max.to_numpy()
    shifts = grid.level_refinement_shift.to_numpy()
    assert np.all(regions_min >= grid.domain_min - 1e-12)
    assert np.all(regions_max <= grid.domain_max + 1e-12)
    assert np.allclose(shifts[0], 0.0)
    for level in range(1, grid.num_levels):
        assert np.all(regions_min[level] >= regions_min[level - 1] - 1e-12)
        assert np.all(regions_max[level] <= regions_max[level - 1] + 1e-12)
        assert np.allclose(shifts[level] / grid.dx[level - 1], np.round(shifts[level] / grid.dx[level - 1]))
assert not grid.update_dynamic_refinement(sample_time)

old_probe = initial_max[-1] - np.array([0.25 * grid.dx[-1], 0.25 * grid.dx[-1]])
new_probe = updated_min[-1] + np.array([0.25 * grid.dx[-1], 0.25 * grid.dx[-1]])
assert grid.max_level == 2

probes = ti.Vector.field(2, dtype=ti.f64, shape=2)
levels = ti.field(dtype=ti.i32, shape=2)
probes[0] = old_probe
probes[1] = new_probe


@ti.kernel
def sample_levels():
    for i in range(2):
        levels[i] = grid.finest_level_at(probes[i])


sample_levels()
probe_levels = levels.to_numpy()
assert probe_levels[0] < grid.max_level
assert probe_levels[1] == grid.max_level

assert grid.update_dynamic_refinement(0.0)
solver._adapt_particles(complete=True)


def expected_particle_levels(points):
    expected = np.zeros(points.shape[0], dtype=np.int32)
    for level in range(1, grid.num_levels):
        minimum = grid.region_min_np[level]
        maximum = grid.region_max_np[level]
        inside = np.all((points >= minimum) & (points < maximum), axis=1)
        expected[inside] = level
    return expected


n_before = solver.particles.n_active()
initial_positions = solver.particles.x.to_numpy()[:n_before]
initial_levels = solver.particles.level.to_numpy()[:n_before]
mass_before = solver.particles.mass.to_numpy()[:n_before].sum()
assert np.array_equal(initial_levels, expected_particle_levels(initial_positions))
assert np.any(initial_levels == grid.max_level)
solver.step(current_time=0.2)
first_shift = grid.refinement_shift.to_numpy()
assert first_shift[0] > 0.0 and first_shift[1] < 0.0
solver.step(current_time=sample_time)
n_after = solver.particles.n_active()
positions = solver.particles.x.to_numpy()[:n_after]
particle_levels = solver.particles.level.to_numpy()[:n_after]
mass_after = solver.particles.mass.to_numpy()[:n_after].sum()
assert np.allclose(grid.refinement_shift.to_numpy(), shift)
assert n_before > 0 and n_after > 0
assert np.isfinite(positions).all()
assert np.all((particle_levels >= 0) & (particle_levels <= grid.max_level))
assert np.array_equal(particle_levels, expected_particle_levels(positions))
platform_displacement, _ = grid._platform_motion_numpy(sample_time)
platform_min = np.array([config.INT_MOVINGRECT_XMIN, config.INT_MOVINGRECT_YMIN]) + platform_displacement
platform_max = np.array([config.INT_MOVINGRECT_XMAX, config.INT_MOVINGRECT_YMAX]) + platform_displacement
inside_platform = np.all((positions > platform_min) & (positions < platform_max), axis=1)
assert not np.any(inside_platform)
assert np.isclose(mass_after, mass_before, rtol=1e-12, atol=1e-15)
assert solver.particles.split_overflow[None] == 0
solver.particles.split_overflow[None] = 1
try:
    solver._check_dynamic_split_capacity()
except RuntimeError as error:
    assert "AMR_PARTICLE_CAPACITY_FACTOR" in str(error)
else:
    raise AssertionError("dynamic split overflow did not fail fast")
solver.particles.split_overflow[None] = 0

print(f"refinement shift = {shift.tolist()}")
print(f"probe levels = {probe_levels.tolist()}")
print(f"active particles = {n_before} -> {n_after}")
print("dynamic moving refinement verification passed")
