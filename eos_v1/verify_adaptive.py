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
config.AMR_MAX_LEVEL = 3
config.AMR_PROCESS_ZONE_HEIGHT = 0.005
config.AMR_FINE_REGION_WIDTH = 0.004
config.AMR_FINE_REGION_CENTER_X = 0.5 * config.AMR_DOMAIN_WIDTH
config.AMR_FINE_REGION_YMIN = 0.0
config.AMR_GRID_PADDING = 3
config.AMR_REFINEMENT_BUFFER_CELLS = 2
config.AMR_PARTICLES_PER_CELL_AXIS = 2
config.AMR_SCATTER_TO_ANCESTORS = True
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

solver = AdaptiveMPMSolver2D()
print(f"DT = {config.DT:.3e}  active = {solver.particles.n_active()}")

t = 0.0
for step in range(6000):
    solver.step(damping=0.98, current_time=t)
    t += config.DT
for step in range(1000):
    solver.step(damping=1.0, current_time=t)
    t += config.DT

n = solver.particles.n_active()
x = solver.particles.x.to_numpy()[:n]
v = solver.particles.v.to_numpy()[:n]
p = solver.particles.pressure.to_numpy()[:n]
lvl = solver.particles.level.to_numpy()[:n]
mass = solver.particles.mass.to_numpy()[:n]

assert not np.isnan(x).any() and not np.isnan(v).any(), "NaN detected"
print(f"t = {t*1e3:.2f} ms, active = {n}")
print(f"total mass = {mass.sum():.6e} (expected {1000.0*0.02*0.04:.6e})")
speed = np.linalg.norm(v, axis=1)
print(f"max |v| = {speed.max():.4f} m/s, mean |v| = {speed.mean():.5f} m/s")

h = config.AMR_DOMAIN_HEIGHT
p_ref = 1000.0 * 9.81 * np.maximum(h - x[:, 1], 0.0)
deep = x[:, 1] < 0.5 * h
rel_err = np.abs(p[deep] - p_ref[deep]) / (1000.0 * 9.81 * h)
print(f"hydrostatic pressure rel-err (bottom half): mean {rel_err.mean():.4f}, max {rel_err.max():.4f}")

fine_min = solver.grid.region_min_np[-1]
fine_max = solver.grid.region_max_np[-1]
band = (x[:, 1] > 0.001) & (x[:, 1] < 0.004)
inside = band & (x[:, 0] > fine_min[0]) & (x[:, 0] < fine_max[0])
outside = band & ((x[:, 0] < fine_min[0] - 0.002) | (x[:, 0] > fine_max[0] + 0.002))
depth_in = (h - x[inside, 1])
depth_out = (h - x[outside, 1])
print(f"p/(rho g depth) inside fine region : {np.mean(p[inside] / (1000.0*9.81*depth_in)):.4f}")
print(f"p/(rho g depth) outside fine region: {np.mean(p[outside] / (1000.0*9.81*depth_out)):.4f}")

counts = ti.field(dtype=ti.i32, shape=solver.grid.num_levels)
solver.count_particles_by_level(counts)
print("particles per level:", counts.to_numpy().tolist())
print("split overflow:", int(solver.particles.split_overflow[None]))
