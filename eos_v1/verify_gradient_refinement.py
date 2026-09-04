import numpy as np
import taichi as ti

import config
from benchmarks.three_blocks.run_experiment import (ACCEL_TIME, GRADIENT_RELEASE_TIME, MATERIAL_IDS, configure,
                                                     initialize_case, motion, prescribed_shear)
from benchmarks.three_materials.run_experiment import ThreeMaterialAdaptiveMPMSolver2D

refinement_box = configure("gradient")
ti.init(arch=ti.cpu, default_fp=ti.f64)
solver = ThreeMaterialAdaptiveMPMSolver2D(refinement_box=refinement_box, drive_block=False)
initialize_case(solver, MATERIAL_IDS["fluid"])
particles = solver.particles
initial_count = particles.n_active()
initial_mass = particles.mass.to_numpy()[:initial_count].sum()


@ti.kernel
def prescribe_rigid_state(speed: ti.f64):
    for p in range(particles.active_count[None]):
        particles.v[p] = ti.Vector([speed, 0.0])
        particles.C[p] = ti.Matrix.zero(ti.f64, 2, 2)
        particles.F[p] = ti.Matrix.identity(ti.f64, 2)
        particles.stress[p] = ti.Matrix.zero(ti.f64, 2, 2)
        particles.pressure[p] = 0.0
        particles.Jp[p] = 1.0


@ti.kernel
def prescribe_refinement_gradient(shear: ti.f64):
    for p in range(particles.active_count[None]):
        particles.C[p] = ti.Matrix([[0.0, shear], [0.0, 0.0]])


def particle_momentum():
    n = particles.n_active()
    mass = particles.mass.to_numpy()[:n]
    velocity = particles.v.to_numpy()[:n]
    return (mass[:, None] * velocity).sum(axis=0)


speed = motion(ACCEL_TIME)
shear = prescribed_shear("gradient", ACCEL_TIME)
assert shear > 0.0
assert prescribed_shear("gradient", np.nextafter(GRADIENT_RELEASE_TIME, 0.0)) > 0.0
assert prescribed_shear("gradient", GRADIENT_RELEASE_TIME) == 0.0
assert prescribed_shear("quadtree", ACCEL_TIME) == 0.0
prescribe_rigid_state(speed)
prescribe_refinement_gradient(shear)
momentum_before = particle_momentum()
solver._update_gradient_levels()
solver._adapt_particles(complete=True)
n_refined = particles.n_active()
levels_refined = particles.level.to_numpy()[:n_refined]
mass_refined = particles.mass.to_numpy()[:n_refined]
assert n_refined == 16 * initial_count
assert np.all(levels_refined == solver.grid.max_level)
assert np.isclose(mass_refined.sum(), initial_mass, rtol=1e-12, atol=1e-15)
assert np.allclose(particle_momentum(), momentum_before, rtol=1e-12, atol=1e-15)
prescribe_rigid_state(0.0)
prescribe_refinement_gradient(0.0)
solver._update_gradient_levels()
solver._adapt_particles(complete=True)
n_coarse = particles.n_active()
levels_coarse = particles.level.to_numpy()[:n_coarse]
mass_coarse = particles.mass.to_numpy()[:n_coarse]
assert n_coarse == initial_count
assert np.all(levels_coarse == 0)
assert np.isclose(mass_coarse.sum(), initial_mass, rtol=1e-12, atol=1e-15)
print(f"gradient refinement particles = {initial_count} -> {n_refined} -> {n_coarse}")
print("gradient refinement verification passed")
