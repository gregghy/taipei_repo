import math
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
config.AMR_MAX_LEVEL = 3
config.AMR_PROCESS_ZONE_HEIGHT = 0.005
config.AMR_FINE_REGION_WIDTH = 0.004
config.AMR_FINE_REGION_CENTER_X = 0.5 * config.AMR_DOMAIN_WIDTH
config.AMR_FINE_REGION_YMIN = 0.0
config.AMR_GRID_PADDING = 3
config.AMR_REFINEMENT_BUFFER_CELLS = 2
config.AMR_PARTICLES_PER_CELL_AXIS = 2
config.AMR_SCATTER_TO_ANCESTORS = True
config.AMR_ALLOW_LEVEL_PROMOTION_WITHOUT_SPLIT = False
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
config.DT = 1e-5

ti.init(arch=ti.gpu)

from solver.adaptive_engine import AdaptiveMPMSolver2D

solver = AdaptiveMPMSolver2D()
counts = ti.field(dtype=ti.i32, shape=solver.grid.num_levels)
solver.count_particles_by_level(counts)
print("levels", solver.grid.num_levels)
print("particles", solver.particles.n_active(), "capacity", solver.particles.capacity)
print("leaf_cells", solver.grid.leaf_count)
print("grid_shapes", solver.grid.res)
print("particle_counts", counts.to_numpy().tolist())
for step in range(3):
    solver.step(damping=1.0, current_time=step * config.DT)
solver.count_particles_by_level(counts)
print("post_step_particle_counts", counts.to_numpy().tolist())
print("post_step_active", solver.particles.n_active())
print("sample_position", solver.particles.x.to_numpy()[0].tolist())
