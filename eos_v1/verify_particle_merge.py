import math
import numpy as np
import taichi as ti
import config

config.ACTIVE_SCENARIO = "ADAPTIVE_MPM"
config.DIM = 2
config.FLUID = "WATER"
config.AMR_DOMAIN_WIDTH = 0.02
config.AMR_DOMAIN_HEIGHT = 0.04
config.AMR_BASE_CELLS_X = 16
config.AMR_BASE_CELLS_Y = 32
config.AMR_BASE_DX = config.AMR_DOMAIN_WIDTH / config.AMR_BASE_CELLS_X
config.AMR_MAX_LEVEL = 2
config.AMR_PROCESS_ZONE_HEIGHT = 0.02
config.AMR_FINE_REGION_WIDTH = 0.010
config.AMR_FINE_REGION_CENTER_X = 0.015
config.AMR_FINE_REGION_YMIN = 0.0
config.AMR_GRID_PADDING = 3
config.AMR_REFINEMENT_BUFFER_CELLS = 1
config.AMR_PARTICLES_PER_CELL_AXIS = 2
config.AMR_SCATTER_TO_ANCESTORS = True
config.AMR_SPLIT_PARTICLES = True
config.AMR_MERGE_PARTICLES = True
config.AMR_MERGE_MIN_PARTICLES = 4
config.AMR_GRADIENT_REFINE = False
config.AMR_PARTICLE_CAPACITY_FACTOR = 2.0
config.AMR_INITIAL_FLUID_XMIN = 0.0
config.AMR_INITIAL_FLUID_XMAX = config.AMR_DOMAIN_WIDTH
config.AMR_INITIAL_FLUID_YMIN = 0.0
config.AMR_INITIAL_FLUID_YMAX = config.AMR_DOMAIN_HEIGHT
config.GRID_WIDTH = config.AMR_DOMAIN_WIDTH
config.GRID_HEIGHT = config.AMR_DOMAIN_HEIGHT
config.N_CELL_WIDTH = config.AMR_BASE_CELLS_X
config.N_CELL_HEIGHT = config.AMR_BASE_CELLS_Y
config.DX = config.AMR_BASE_DX
config.DY = config.AMR_BASE_DX
config.INV_DX = 1.0 / config.DX
config.INV_DY = 1.0 / config.DY
config.GRID_RES_X = config.N_CELL_WIDTH + 2 * config.PADDING + 1
config.GRID_RES_Y = config.N_CELL_HEIGHT + 2 * config.PADDING + 1
config.GRAVITY = [0.0, -9.81]
config.RHO_0 = 1000.0
config.G_MAG = 9.81
config.CFL = 0.1
config.C_0 = 10.0 * math.sqrt(2.0 * config.G_MAG * config.AMR_DOMAIN_HEIGHT)
config.V_MAX_ESTIMATE = math.sqrt(2.0 * config.G_MAG * config.AMR_DOMAIN_HEIGHT)
config.DT = 1e-5

ti.init(arch=ti.cpu)

from solver.adaptive_engine import AdaptiveMPMSolver2D

solver = AdaptiveMPMSolver2D(max_level=config.AMR_MAX_LEVEL)
ps = solver.particles
grid = solver.grid
native_mass = ps.native_mass.to_numpy()
parent_velocity = np.array([1.25, -0.5])


@ti.kernel
def seed_parent(level: ti.i32, x: ti.f64, y: ti.f64):
    ps.active_count[None] = 1
    ps.x[0] = ti.Vector([x, y])
    ps.v[0] = ti.Vector([1.25, -0.5])
    ps.C[0] = ti.Matrix([[0.2, -0.1], [0.3, 0.4]])
    ps.F[0] = ti.Matrix.identity(ti.f64, 2)
    ps.stress[0] = ti.Matrix.zero(ti.f64, 2, 2)
    ps.pressure[0] = 0.0
    ps.material[0] = 0
    ps.Jp[0] = 1.0
    ps.level[0] = level
    ps.mass[0] = ps.native_mass[level]
    ps.volume0[0] = ps.native_mass[level] / config.RHO_0
    ps.gradient_level[0] = level + 1


@ti.kernel
def shift_active_particles(dx: ti.f64):
    for p in range(ps.active_count[None]):
        ps.x[p][0] += dx


@ti.kernel
def append_parent_level_particle(level: ti.i32, x: ti.f64, y: ti.f64):
    p = ps.active_count[None]
    ps.active_count[None] = p + 1
    ps.x[p] = ti.Vector([x, y])
    ps.v[p] = ti.Vector([1.25, -0.5])
    ps.C[p] = ti.Matrix.zero(ti.f64, 2, 2)
    ps.F[p] = ti.Matrix.identity(ti.f64, 2)
    ps.stress[p] = ti.Matrix.zero(ti.f64, 2, 2)
    ps.pressure[p] = 0.0
    ps.material[p] = 0
    ps.Jp[p] = 1.0
    ps.level[p] = level
    ps.mass[p] = ps.native_mass[level]
    ps.volume0[p] = ps.native_mass[level] / config.RHO_0
    ps.gradient_level[p] = level


def geometric_targets(points):
    targets = np.zeros(len(points), dtype=np.int32)
    for level in range(1, grid.num_levels):
        inside = np.all((points >= grid.region_min_np[level]) & (points < grid.region_max_np[level]), axis=1)
        targets[inside] = level
    return targets


def prepare_children(parent_level):
    child_level = parent_level + 1
    interface_x = float(grid.region_min_np[child_level, 0])
    spacing = grid.dx[parent_level] / ps.ppc_axis
    origin_y = grid.origin_np[parent_level, 1]
    slot = math.floor((0.01 - origin_y) / spacing)
    parent_y = origin_y + (slot + 0.5) * spacing
    parent_mass = float(native_mass[parent_level])
    child_offset = 0.25 * math.sqrt(parent_mass / config.RHO_0)
    epsilon = 1e-10
    seed_parent(parent_level, interface_x + epsilon, parent_y)
    ps.split_particles()
    assert ps.n_active() == 1
    shift_active_particles(child_offset + epsilon)
    ps.split_particles()
    assert ps.n_active() == 4
    positions = ps.x.to_numpy()[:4]
    levels = ps.level.to_numpy()[:4]
    masses = ps.mass.to_numpy()[:4]
    velocities = ps.v.to_numpy()[:4]
    assert np.all(levels == child_level)
    assert np.all(geometric_targets(positions) == child_level)
    assert np.allclose(masses, parent_mass / 4.0, rtol=1e-12, atol=1e-15)
    assert np.allclose((masses[:, None] * velocities).sum(axis=0), parent_mass * parent_velocity,
                       rtol=1e-12, atol=1e-15)
    shift_active_particles(-3.0 * epsilon)
    assert np.array_equal(np.sort(geometric_targets(ps.x.to_numpy()[:4])),
                          np.array([parent_level] * 2 + [child_level] * 2))
    ps.merge_particles()
    assert ps.n_active() == 4
    assert np.all(ps.level.to_numpy()[:4] == child_level)
    assert np.allclose(ps.mass.to_numpy()[:4], parent_mass / 4.0, rtol=1e-12, atol=1e-15)
    shift_active_particles(-3.0 * child_offset)
    assert np.all(geometric_targets(ps.x.to_numpy()[:4]) == parent_level)
    return parent_mass, parent_y


def verify_transition(parent_level, verify_mixed_bin=False):
    parent_mass, parent_y = prepare_children(parent_level)
    if verify_mixed_bin:
        position = ps.x.to_numpy()[:4].mean(axis=0)
        append_parent_level_particle(parent_level, float(position[0]), parent_y)
        ps.merge_particles()
        assert ps.n_active() == 5
        levels = ps.level.to_numpy()[:5]
        assert int((levels == parent_level).sum()) == 1
        assert int((levels == parent_level + 1).sum()) == 4
        parent_mass, _ = prepare_children(parent_level)
    masses = ps.mass.to_numpy()[:4]
    velocities = ps.v.to_numpy()[:4]
    momentum = (masses[:, None] * velocities).sum(axis=0)
    ps.merge_particles()
    assert ps.n_active() == 1
    assert int(ps.level.to_numpy()[0]) == parent_level
    assert np.isclose(float(ps.mass.to_numpy()[0]), parent_mass, rtol=1e-12, atol=1e-15)
    assert np.allclose(float(ps.mass.to_numpy()[0]) * ps.v.to_numpy()[0], momentum,
                       rtol=1e-12, atol=1e-15)


def verify_round_trip():
    parent_level = 0
    parent_mass = float(native_mass[parent_level])
    spacing = grid.dx[parent_level] / ps.ppc_axis
    origin_y = grid.origin_np[parent_level, 1]
    slot = math.floor((0.01 - origin_y) / spacing)
    parent_y = origin_y + (slot + 0.5) * spacing
    parent_offset = 0.25 * math.sqrt(parent_mass / config.RHO_0)
    start_x = float(grid.region_min_np[1, 0]) - 2.0 * parent_offset
    end_x = float(grid.region_min_np[2, 0]) + 4.0 * parent_offset
    step = parent_offset / 4.0
    steps = math.ceil((end_x - start_x) / step)
    seed_parent(parent_level, start_x, parent_y)
    expected_momentum = parent_mass * parent_velocity
    counts = []
    for direction in (1.0, -1.0):
        for _ in range(steps):
            shift_active_particles(direction * step)
            ps.merge_particles()
            ps.split_particles()
            n = ps.n_active()
            levels = ps.level.to_numpy()[:n]
            masses = ps.mass.to_numpy()[:n]
            velocities = ps.v.to_numpy()[:n]
            counts.append(n)
            assert np.allclose(masses, native_mass[levels], rtol=1e-12, atol=1e-15)
            assert np.isclose(masses.sum(), parent_mass, rtol=1e-12, atol=1e-15)
            assert np.allclose((masses[:, None] * velocities).sum(axis=0), expected_momentum,
                               rtol=1e-12, atol=1e-15)
    assert max(counts) == 16
    assert ps.n_active() == 1
    assert int(ps.level.to_numpy()[0]) == 0


verify_transition(0, verify_mixed_bin=True)
verify_transition(1)
verify_round_trip()
print("level 0/1 and level 1/2 split/merge conservation OK")
