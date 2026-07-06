import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taichi as ti
import config


@ti.data_oriented
class AdaptiveParticleSystem2D:
    def __init__(self, quadtree_grid):
        self.grid = quadtree_grid
        self.fluid_ymax = float(getattr(config, 'AMR_INITIAL_FLUID_YMAX', self.grid.domain_max[1]))
        particle_data = self._build_particles()
        x_np, level_np, mass_np, volume_np = particle_data
        self.n_particles = x_np.shape[0]
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.n_particles)
        self.stress = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.n_particles)
        self.pressure = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.level = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.mass = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.volume0 = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.x.from_numpy(x_np)
        self.level.from_numpy(level_np)
        self.mass.from_numpy(mass_np)
        self.volume0.from_numpy(volume_np)
        self.init_state()

    def _build_particles(self):
        ppc_axis = int(getattr(config, 'AMR_PARTICLES_PER_CELL_AXIS', 2))
        fluid_xmin = float(getattr(config, 'AMR_INITIAL_FLUID_XMIN', self.grid.domain_min[0]))
        fluid_xmax = float(getattr(config, 'AMR_INITIAL_FLUID_XMAX', self.grid.domain_max[0]))
        fluid_ymin = float(getattr(config, 'AMR_INITIAL_FLUID_YMIN', self.grid.domain_min[1]))
        fluid_ymax = float(getattr(config, 'AMR_INITIAL_FLUID_YMAX', self.grid.domain_max[1]))
        positions = []
        levels = []
        masses = []
        volumes = []
        for cell_level, origin, dx in zip(self.grid.leaf_level, self.grid.leaf_origin, self.grid.leaf_size):
            center = origin + 0.5 * dx
            if center[0] < fluid_xmin or center[0] >= fluid_xmax or center[1] < fluid_ymin or center[1] >= fluid_ymax:
                continue
            spacing = dx / ppc_axis
            volume = dx * dx / (ppc_axis * ppc_axis)
            mass = volume * float(config.RHO_0)
            for i in range(ppc_axis):
                for j in range(ppc_axis):
                    positions.append(origin + np.array([(i + 0.5) * spacing, (j + 0.5) * spacing], dtype=np.float32))
                    levels.append(cell_level)
                    masses.append(mass)
                    volumes.append(volume)
        return (
            np.array(positions, dtype=np.float32),
            np.array(levels, dtype=np.int32),
            np.array(masses, dtype=np.float32),
            np.array(volumes, dtype=np.float32),
        )

    @ti.kernel
    def init_state(self):
        fluid_ymax = ti.cast(self.fluid_ymax, ti.f32)
        bulk_modulus = config.C_0**2 * config.RHO_0
        for p in range(self.n_particles):
            self.v[p] = ti.Vector.zero(ti.f32, 2)
            self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            local_depth = ti.max(fluid_ymax - self.x[p][1], 0.0)
            p_hydro = config.RHO_0 * 9.81 * local_depth
            J_init = 1.0 / (p_hydro / bulk_modulus + 1.0)
            self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, J_init]])
            self.stress[p] = ti.Matrix([[-p_hydro, 0.0], [0.0, -p_hydro]])
            self.pressure[p] = p_hydro
