import sys
import os
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taichi as ti
import config


@ti.data_oriented
class AdaptiveParticleSystem2D:
    def __init__(self, quadtree_grid):
        self.grid = quadtree_grid
        self.fluid_ymax = float(getattr(config, 'AMR_INITIAL_FLUID_YMAX', self.grid.domain_max[1]))
        self.ppc_axis = int(getattr(config, 'AMR_PARTICLES_PER_CELL_AXIS', 2))
        self.split_enabled = bool(getattr(config, 'AMR_SPLIT_PARTICLES', True))
        capacity_factor = float(getattr(config, 'AMR_PARTICLE_CAPACITY_FACTOR', 2.0))
        particle_data = self._build_particles()
        x_np, level_np, mass_np, volume_np = particle_data
        self.n_initial = x_np.shape[0]
        self.capacity = int(math.ceil(self.n_initial * max(capacity_factor, 1.0))) if self.split_enabled else self.n_initial
        self.n_particles = self.capacity
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=self.capacity)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=self.capacity)
        self.C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.capacity)
        self.F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.capacity)
        self.stress = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.capacity)
        self.pressure = ti.field(dtype=ti.f32, shape=self.capacity)
        self.level = ti.field(dtype=ti.i32, shape=self.capacity)
        self.mass = ti.field(dtype=ti.f32, shape=self.capacity)
        self.volume0 = ti.field(dtype=ti.f32, shape=self.capacity)
        self.active_count = ti.field(dtype=ti.i32, shape=())
        self.split_overflow = ti.field(dtype=ti.i32, shape=())
        # Native (fully refined) particle mass of each level, used to decide
        # whether a promoted particle should be split or just re-assigned.
        self.native_mass = ti.field(dtype=ti.f32, shape=self.grid.num_levels)
        self.native_mass.from_numpy(np.array(
            [config.RHO_0 * (self.grid.dx[l] / self.ppc_axis) ** 2 for l in range(self.grid.num_levels)],
            dtype=np.float32))
        self.x.from_numpy(self._pad(x_np))
        self.level.from_numpy(self._pad(level_np))
        self.mass.from_numpy(self._pad(mass_np))
        self.volume0.from_numpy(self._pad(volume_np))
        self.active_count[None] = self.n_initial
        self.split_overflow[None] = 0
        self.init_state()

    def _pad(self, arr):
        if arr.shape[0] == self.capacity:
            return arr
        pad_shape = (self.capacity - arr.shape[0],) + arr.shape[1:]
        return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)

    def _build_particles(self):
        ppc_axis = self.ppc_axis
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
        for p in range(self.n_initial):
            self.v[p] = ti.Vector.zero(ti.f32, 2)
            self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            local_depth = ti.max(fluid_ymax - self.x[p][1], 0.0)
            p_hydro = config.RHO_0 * 9.81 * local_depth
            J_init = 1.0 / (p_hydro / bulk_modulus + 1.0)
            self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, J_init]])
            self.stress[p] = ti.Matrix([[-p_hydro, 0.0], [0.0, -p_hydro]])
            self.pressure[p] = p_hydro

    @ti.kernel
    def split_particles(self):
        # Promote particles that moved into a finer region. A particle that is
        # still coarser than its target level is split into 4 children (one
        # level at a time, mass/volume conserved); particles that are already
        # at (or below) the native mass of the target level are re-assigned
        # without splitting so repeated interface crossings cannot refine
        # particles without bound.
        n_before = self.active_count[None]
        for p in range(n_before):
            lvl = self.level[p]
            target = self.grid.finest_level_at(self.x[p])
            if target > lvl:
                new_level = lvl + 1
                if self.mass[p] > 1.5 * self.native_mass[new_level]:
                    base = ti.atomic_add(self.active_count[None], 3)
                    if base + 3 <= self.capacity:
                        x0 = self.x[p]
                        v0 = self.v[p]
                        C0 = self.C[p]
                        F0 = self.F[p]
                        S0 = self.stress[p]
                        pr0 = self.pressure[p]
                        m_child = 0.25 * self.mass[p]
                        vol_child = 0.25 * self.volume0[p]
                        off = 0.25 * ti.sqrt(self.volume0[p])
                        for k in ti.static(range(4)):
                            sx = -1.0 if k % 2 == 0 else 1.0
                            sy = -1.0 if k // 2 == 0 else 1.0
                            d = ti.Vector([sx * off, sy * off])
                            idx = base + k - 1
                            if ti.static(k == 0):
                                idx = p
                            self.x[idx] = x0 + d
                            self.v[idx] = v0 + C0 @ d
                            self.C[idx] = C0
                            self.F[idx] = F0
                            self.stress[idx] = S0
                            self.pressure[idx] = pr0
                            self.level[idx] = new_level
                            self.mass[idx] = m_child
                            self.volume0[idx] = vol_child
                    else:
                        ti.atomic_sub(self.active_count[None], 3)
                        ti.atomic_add(self.split_overflow[None], 1)
                else:
                    self.level[p] = new_level

    def n_active(self):
        return int(self.active_count[None])
