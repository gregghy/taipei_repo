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
config.AMR_MAX_LEVEL = 1
config.AMR_PROCESS_ZONE_HEIGHT = 0.02
config.AMR_FINE_REGION_WIDTH = 0.010
config.AMR_FINE_REGION_CENTER_X = 0.015
config.AMR_FINE_REGION_YMIN = 0.0
config.AMR_GRID_PADDING = 3
config.AMR_REFINEMENT_BUFFER_CELLS = 0
config.AMR_PARTICLES_PER_CELL_AXIS = 2
config.AMR_SCATTER_TO_ANCESTORS = True
config.AMR_SPLIT_PARTICLES = True
config.AMR_MERGE_PARTICLES = True
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

solver = AdaptiveMPMSolver2D(max_level=1)
ps = solver.particles
n0 = ps.n_active()
mass0 = ps.mass.to_numpy()[:n0].sum()


@ti.kernel
def seed_children(n0_arg: ti.i32):
    base = n0_arg
    ps.active_count[None] = n0_arg + 3
    c = ti.Vector([0.0040625, 0.0040625])
    off = 0.00005
    for k in ti.static(range(4)):
        idx = base + k - 1
        if ti.static(k == 0):
            idx = 0
        sx = -1.0 if k % 2 == 0 else 1.0
        sy = -1.0 if k // 2 == 0 else 1.0
        ps.x[idx] = c + ti.Vector([sx * off, sy * off])
        ps.v[idx] = ti.Vector([1.0 + 0.1 * sx, -2.0 + 0.2 * sy])
        ps.C[idx] = ti.Matrix([[0.1, 0.2], [0.3, 0.4]])
        ps.F[idx] = ti.Matrix([[1.0, 0.0], [0.0, 0.99]])
        ps.stress[idx] = ti.Matrix([[-10.0, 0.0], [0.0, -10.0]])
        ps.pressure[idx] = 10.0
        ps.level[idx] = 1
        ps.mass[idx] = ps.native_mass[1]
        ps.volume0[idx] = ps.native_mass[1] / config.RHO_0


seed_children(n0)
mass_seeded = ps.mass.to_numpy()[:ps.n_active()].sum()
ps.merge_particles()
n1 = ps.n_active()
mass1 = ps.mass.to_numpy()[:n1].sum()
levels = ps.level.to_numpy()[:n1]
print("n0", n0, "n1", n1)
print("mass0", mass0, "mass_seeded", mass_seeded, "mass1", mass1)
print("level0_count", int((levels == 0).sum()), "level1_count", int((levels == 1).sum()))
assert n1 == n0
assert abs(float(mass1 - mass0)) < 1e-6
print("merge conservation OK")
