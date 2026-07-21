import numpy as np
import taichi as ti
import config

config.AMR_DOMAIN_WIDTH = 0.02
config.AMR_DOMAIN_HEIGHT = 0.04
config.AMR_BASE_CELLS_X = 4
config.AMR_BASE_CELLS_Y = 8
config.AMR_BASE_DX = config.AMR_DOMAIN_WIDTH / config.AMR_BASE_CELLS_X
config.AMR_MAX_LEVEL = 0
config.AMR_GRID_PADDING = 3
config.RHO_0 = 1000.0


ti.init(arch=ti.cpu)

from core.quadtree_grid import QuadtreeGrid2D


grid = QuadtreeGrid2D(max_level=0)
grid.clear()
grid.initialize_penalty_mass()

mass = grid.boundary_mass[0].to_numpy()
beta = config.AMR_BOUNDARY_PENALTY_NORMAL * config.RHO_0 * grid.dx[0] ** 2
expected_x_mass = 2.0 * beta * grid.domain_height / grid.dx[0]
expected_y_mass = 2.0 * beta * grid.domain_width / grid.dx[0]

assert np.isclose(mass[:, :, 0].sum(), expected_x_mass)
assert np.isclose(mass[:, :, 1].sum(), expected_y_mass)

bottom_center = (grid.padding + 2, grid.padding)
left_center = (grid.padding, grid.padding + 4)
interior = (grid.padding + 2, grid.padding + 4)

assert mass[bottom_center][1] > 0.0 and mass[bottom_center][0] == 0.0
assert mass[left_center][0] > 0.0 and mass[left_center][1] == 0.0
assert np.all(mass[interior] == 0.0)


@ti.kernel
def set_bottom_probe():
    grid.m[0][bottom_center] = 2.0
    grid.v[0][bottom_center] = ti.Vector([1.0, -1.0])


set_bottom_probe()
grid.normalize_momentum()
velocity = grid.v[0].to_numpy()[bottom_center]

assert np.isclose(velocity[0], 0.5)
assert np.isclose(velocity[1], -1.0 / (2.0 + mass[bottom_center][1]))

config.INT_MOVINGRECT_XMIN = 0.005
config.INT_MOVINGRECT_XMAX = 0.015
config.INT_MOVINGRECT_YMIN = 0.010
config.INT_MOVINGRECT_YMAX = 0.020
config.PLATFORM_VELOCITY_Y = -0.5
config.PLATFORM_STOP_TIME = 3.2
config.PLATFORM_DECEL_TIME = 0.5

grid.update_moving_platform_penalty_mass(0.0)
grid.initialize_penalty_mass()
grid.add_moving_penalty_mass()
moving_mass = grid.moving_boundary_mass[0].to_numpy()
moving_momentum = grid.moving_boundary_momentum[0].to_numpy()

assert moving_mass[:, :, 0].sum() > 0.0
assert moving_mass[:, :, 1].sum() > 0.0
assert np.isclose(
    moving_momentum[:, :, 1].sum(),
    config.PLATFORM_VELOCITY_Y * moving_mass[:, :, 1].sum(),
)

print(f"beta = {beta:.6e}")
print(f"boundary mass sums = {mass[:, :, 0].sum():.6e}, {mass[:, :, 1].sum():.6e}")
print(f"bottom probe velocity = {velocity.tolist()}")
print("penalty boundary verification passed")
