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
config.INT_MOVINGRECT_XMIN = 0.009
config.INT_MOVINGRECT_XMAX = 0.019
config.INT_MOVINGRECT_YMIN = 0.024
config.INT_MOVINGRECT_YMAX = 0.028
config.PLATFORM_VELOCITY_Y = -0.01
config.PLATFORM_STOP_TIME = 0.5
config.PLATFORM_DECEL_TIME = 0.2
config.AMR_MAX_LEVEL = 2
config.AMR_REFINEMENT_BUFFER_CELLS = 2
config.AMR_GHOST_BAND_CELLS = 2
config.AMR_PARTICLES_PER_CELL_AXIS = 2
config.AMR_SPLIT_PARTICLES = True
config.AMR_MERGE_PARTICLES = True
config.AMR_PARTICLE_CAPACITY_FACTOR = 4.0
config.AMR_DYNAMIC_REFINEMENT = True
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

sample_time = 0.75
grid.update_dynamic_refinement(sample_time)
updated_min = grid.region_min.to_numpy()
updated_max = grid.region_max.to_numpy()
updated_origin = grid.origin.to_numpy()
shift = grid.refinement_shift.to_numpy()

assert shift[1] < 0.0
assert np.allclose(updated_min[0], initial_min[0])
assert np.allclose(updated_max[0], initial_max[0])
assert np.allclose(updated_origin[0], initial_origin[0])
assert np.allclose(updated_min[1:] - initial_min[1:], shift)
assert np.allclose(updated_max[1:] - initial_max[1:], shift)
assert np.allclose(updated_origin[1:] - initial_origin[1:], shift)

old_center = 0.5 * (initial_min[-1] + initial_max[-1])
new_center = 0.5 * (updated_min[-1] + updated_max[-1])
assert grid.max_level == 2

probes = ti.Vector.field(2, dtype=ti.f64, shape=2)
levels = ti.field(dtype=ti.i32, shape=2)
probes[0] = old_center
probes[1] = new_center


@ti.kernel
def sample_levels():
    for i in range(2):
        levels[i] = grid.finest_level_at(probes[i])


sample_levels()
probe_levels = levels.to_numpy()
assert probe_levels[0] < grid.max_level
assert probe_levels[1] == grid.max_level

solver._adapt_particles(complete=True)
n_before = solver.particles.n_active()
solver.step(current_time=sample_time)
n_after = solver.particles.n_active()
positions = solver.particles.x.to_numpy()[:n_after]
particle_levels = solver.particles.level.to_numpy()[:n_after]
assert n_before > 0 and n_after > 0
assert np.isfinite(positions).all()
assert np.all((particle_levels >= 0) & (particle_levels <= grid.max_level))

print(f"refinement shift = {shift.tolist()}")
print(f"probe levels = {probe_levels.tolist()}")
print(f"active particles = {n_before} -> {n_after}")
print("dynamic moving refinement verification passed")
