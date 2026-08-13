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
        self.merge_enabled = self.split_enabled and bool(getattr(config, 'AMR_MERGE_PARTICLES', True))
        self.merge_min_particles = max(4, int(getattr(config, 'AMR_MERGE_MIN_PARTICLES', 4)))
        capacity_factor = float(getattr(config, 'AMR_PARTICLE_CAPACITY_FACTOR', 2.0))
        particle_data = self._build_particles()
        x_np, level_np, mass_np, volume_np = particle_data
        self.n_initial = x_np.shape[0]
        self.capacity = int(math.ceil(self.n_initial * max(capacity_factor, 1.0))) if self.split_enabled else self.n_initial
        self.n_particles = self.capacity
        self.x = ti.Vector.field(2, dtype=ti.f64, shape=self.capacity)
        self.v = ti.Vector.field(2, dtype=ti.f64, shape=self.capacity)
        self.C = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.capacity)
        self.F = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.capacity)
        self.stress = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.capacity)
        self.pressure = ti.field(dtype=ti.f64, shape=self.capacity)
        self.level = ti.field(dtype=ti.i32, shape=self.capacity)
        self.mass = ti.field(dtype=ti.f64, shape=self.capacity)
        self.volume0 = ti.field(dtype=ti.f64, shape=self.capacity)
        self.gradient_level = ti.field(dtype=ti.i32, shape=self.capacity)
        self.active_count = ti.field(dtype=ti.i32, shape=())
        self.split_overflow = ti.field(dtype=ti.i32, shape=())
        # Native (fully refined) particle mass of each level, used to decide
        # whether a promoted particle should be split or just re-assigned.
        self.native_mass = ti.field(dtype=ti.f64, shape=self.grid.num_levels)
        self.native_mass.from_numpy(np.array(
            [config.RHO_0 * (self.grid.dx[l] / self.ppc_axis) ** 2 for l in range(self.grid.num_levels)],
            dtype=np.float64))
        slot_counts = [self.grid.res_x[l] * self.ppc_axis * self.grid.res_y[l] * self.ppc_axis
                       for l in range(max(self.grid.num_levels - 1, 1))]
        self.merge_slot_capacity = max(self.capacity, max(slot_counts) if slot_counts else self.capacity)
        self.merge_count = ti.field(dtype=ti.i32, shape=self.merge_slot_capacity)
        self.merge_keep = ti.field(dtype=ti.i32, shape=self.merge_slot_capacity)
        self.merge_mass = ti.field(dtype=ti.f64, shape=self.merge_slot_capacity)
        self.merge_volume = ti.field(dtype=ti.f64, shape=self.merge_slot_capacity)
        self.merge_pressure = ti.field(dtype=ti.f64, shape=self.merge_slot_capacity)
        self.merge_x = ti.Vector.field(2, dtype=ti.f64, shape=self.merge_slot_capacity)
        self.merge_v = ti.Vector.field(2, dtype=ti.f64, shape=self.merge_slot_capacity)
        self.merge_C = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.merge_slot_capacity)
        self.merge_F = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.merge_slot_capacity)
        self.merge_stress = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.merge_slot_capacity)
        self.compact_count = ti.field(dtype=ti.i32, shape=())
        self.x_tmp = ti.Vector.field(2, dtype=ti.f64, shape=self.capacity)
        self.v_tmp = ti.Vector.field(2, dtype=ti.f64, shape=self.capacity)
        self.C_tmp = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.capacity)
        self.F_tmp = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.capacity)
        self.stress_tmp = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.capacity)
        self.pressure_tmp = ti.field(dtype=ti.f64, shape=self.capacity)
        self.level_tmp = ti.field(dtype=ti.i32, shape=self.capacity)
        self.mass_tmp = ti.field(dtype=ti.f64, shape=self.capacity)
        self.volume0_tmp = ti.field(dtype=ti.f64, shape=self.capacity)
        self.gradient_level_tmp = ti.field(dtype=ti.i32, shape=self.capacity)
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
        # When AMR_INITIAL_PARTICLE_LEVEL is set, all particles start at that
        # level regardless of the quadtree leaf structure.  This keeps the
        # initial particle count low and lets the gradient-based split criterion
        # refine particles on the fly where the physics demands it.
        init_level = int(getattr(config, 'AMR_INITIAL_PARTICLE_LEVEL', -1))
        positions = []
        levels = []
        masses = []
        volumes = []
        if init_level >= 0:
            # Start all particles at the specified base level
            dx = self.grid.dx[init_level]
            origin = self.grid.region_min_np[init_level]
            region_max = self.grid.region_max_np[init_level]
            spacing = dx / ppc_axis
            volume = dx * dx / (ppc_axis * ppc_axis)
            mass = volume * float(config.RHO_0)
            nx = int(round((region_max[0] - origin[0]) / dx))
            ny = int(round((region_max[1] - origin[1]) / dx))
            for i in range(nx):
                for j in range(ny):
                    cell_origin = origin + np.array([i * dx, j * dx], dtype=np.float64)
                    center = cell_origin + 0.5 * dx
                    if center[0] < fluid_xmin or center[0] >= fluid_xmax or center[1] < fluid_ymin or center[1] >= fluid_ymax:
                        continue
                    for pi in range(ppc_axis):
                        for pj in range(ppc_axis):
                            positions.append(cell_origin + np.array([(pi + 0.5) * spacing, (pj + 0.5) * spacing], dtype=np.float64))
                            levels.append(init_level)
                            masses.append(mass)
                            volumes.append(volume)
        else:
            for cell_level, origin, dx in zip(self.grid.leaf_level, self.grid.leaf_origin, self.grid.leaf_size):
                center = origin + 0.5 * dx
                if center[0] < fluid_xmin or center[0] >= fluid_xmax or center[1] < fluid_ymin or center[1] >= fluid_ymax:
                    continue
                spacing = dx / ppc_axis
                volume = dx * dx / (ppc_axis * ppc_axis)
                mass = volume * float(config.RHO_0)
                for i in range(ppc_axis):
                    for j in range(ppc_axis):
                        positions.append(origin + np.array([(i + 0.5) * spacing, (j + 0.5) * spacing], dtype=np.float64))
                        levels.append(cell_level)
                        masses.append(mass)
                        volumes.append(volume)
        return (
            np.array(positions, dtype=np.float64),
            np.array(levels, dtype=np.int32),
            np.array(masses, dtype=np.float64),
            np.array(volumes, dtype=np.float64),
        )

    @ti.kernel
    def init_state(self):
        fluid_ymax = ti.cast(self.fluid_ymax, ti.f64)
        bulk_modulus = config.C_0**2 * config.RHO_0
        for p in range(self.n_initial):
            self.v[p] = ti.Vector.zero(ti.f64, 2)
            self.C[p] = ti.Matrix.zero(ti.f64, 2, 2)
            local_depth = ti.max(fluid_ymax - self.x[p][1], 0.0)
            p_hydro = config.RHO_0 * 9.81 * local_depth
            J_init = 1.0 / (p_hydro / bulk_modulus + 1.0)
            self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, J_init]])
            self.stress[p] = ti.Matrix([[-p_hydro, 0.0], [0.0, -p_hydro]])
            self.pressure[p] = p_hydro

    @ti.func
    def _target_level(self, p):
        geometric_target = self.grid.finest_level_at(self.x[p])
        target = geometric_target
        if ti.static(not self.grid.dynamic_refinement and getattr(config, 'AMR_GRADIENT_REFINE', True)):
            target = ti.min(self.gradient_level[p], geometric_target)
        return target

    @ti.kernel
    def split_particles(self):
        # Promote particles to a finer level.  When AMR_GRADIENT_REFINE is True,
        # the target level comes from the gradient criterion only (no geometric
        # split).  When False, the original geometric behavior is used.
        n_before = self.active_count[None]
        for p in range(n_before):
            lvl = self.level[p]
            target = self._target_level(p)
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

    @ti.func
    def _merge_slot(self, level: ti.template(), x):
        parent = ti.static(level - 1)
        slot_nx = ti.static(self.grid.res_x[parent] * self.ppc_axis)
        slot_ny = ti.static(self.grid.res_y[parent] * self.ppc_axis)
        spacing = self.grid.level_dx[parent] / ti.cast(self.ppc_axis, ti.f64)
        fx = (x - self.grid.origin[parent]) / spacing
        si = ti.cast(ti.floor(fx[0]), ti.i32)
        sj = ti.cast(ti.floor(fx[1]), ti.i32)
        si = ti.max(0, ti.min(si, slot_nx - 1))
        sj = ti.max(0, ti.min(sj, slot_ny - 1))
        return si * slot_ny + sj

    @ti.kernel
    def _clear_merge_bins(self):
        for i in self.merge_count:
            self.merge_count[i] = 0
            self.merge_keep[i] = self.capacity
            self.merge_mass[i] = 0.0
            self.merge_volume[i] = 0.0
            self.merge_pressure[i] = 0.0
            self.merge_x[i] = ti.Vector.zero(ti.f64, 2)
            self.merge_v[i] = ti.Vector.zero(ti.f64, 2)
            self.merge_C[i] = ti.Matrix.zero(ti.f64, 2, 2)
            self.merge_F[i] = ti.Matrix.zero(ti.f64, 2, 2)
            self.merge_stress[i] = ti.Matrix.zero(ti.f64, 2, 2)

    @ti.kernel
    def _accumulate_merge_bins(self, level: ti.template()):
        for p in range(self.active_count[None]):
            should_merge = False
            if self.level[p] == level:
                should_merge = self._target_level(p) < level
            if should_merge:
                slot = self._merge_slot(level, self.x[p])
                m = self.mass[p]
                vol = self.volume0[p]
                ti.atomic_add(self.merge_count[slot], 1)
                ti.atomic_min(self.merge_keep[slot], p)
                ti.atomic_add(self.merge_mass[slot], m)
                ti.atomic_add(self.merge_volume[slot], vol)
                ti.atomic_add(self.merge_pressure[slot], vol * self.pressure[p])
                for a in ti.static(range(2)):
                    ti.atomic_add(self.merge_x[slot][a], m * self.x[p][a])
                    ti.atomic_add(self.merge_v[slot][a], m * self.v[p][a])
                    for b in ti.static(range(2)):
                        ti.atomic_add(self.merge_C[slot][a, b], m * self.C[p][a, b])
                        ti.atomic_add(self.merge_F[slot][a, b], vol * self.F[p][a, b])
                        ti.atomic_add(self.merge_stress[slot][a, b], vol * self.stress[p][a, b])

    @ti.kernel
    def _finalize_merge_bins(self, level: ti.template()):
        parent = ti.static(level - 1)
        for p in range(self.active_count[None]):
            if self.level[p] == level:
                target = self._target_level(p)
                if target < level:
                    slot = self._merge_slot(level, self.x[p])
                    m = self.merge_mass[slot]
                    vol = self.merge_volume[slot]
                    can_merge = self.merge_count[slot] >= ti.static(self.merge_min_particles)
                    if m < 0.5 * self.native_mass[parent] or m > 1.5 * self.native_mass[parent]:
                        can_merge = False
                    if can_merge and vol > 0.0:
                        if p == self.merge_keep[slot]:
                            inv_m = 1.0 / m
                            inv_vol = 1.0 / vol
                            self.x[p] = self.merge_x[slot] * inv_m
                            self.v[p] = self.merge_v[slot] * inv_m
                            self.C[p] = self.merge_C[slot] * inv_m
                            self.F[p] = self.merge_F[slot] * inv_vol
                            self.stress[p] = self.merge_stress[slot] * inv_vol
                            self.pressure[p] = self.merge_pressure[slot] * inv_vol
                            self.level[p] = parent
                            self.mass[p] = m
                            self.volume0[p] = vol
                        else:
                            self.level[p] = -1
                            self.mass[p] = 0.0
                            self.volume0[p] = 0.0
                    else:
                        self.level[p] = target

    @ti.kernel
    def _reset_compaction(self):
        self.compact_count[None] = 0

    @ti.kernel
    def _scatter_compaction(self):
        for p in range(self.active_count[None]):
            if self.level[p] >= 0 and self.mass[p] > 0.0:
                q = ti.atomic_add(self.compact_count[None], 1)
                self.x_tmp[q] = self.x[p]
                self.v_tmp[q] = self.v[p]
                self.C_tmp[q] = self.C[p]
                self.F_tmp[q] = self.F[p]
                self.stress_tmp[q] = self.stress[p]
                self.pressure_tmp[q] = self.pressure[p]
                self.level_tmp[q] = self.level[p]
                self.mass_tmp[q] = self.mass[p]
                self.volume0_tmp[q] = self.volume0[p]
                self.gradient_level_tmp[q] = self.gradient_level[p]

    @ti.kernel
    def _apply_compaction(self):
        n = self.compact_count[None]
        for p in range(n):
            self.x[p] = self.x_tmp[p]
            self.v[p] = self.v_tmp[p]
            self.C[p] = self.C_tmp[p]
            self.F[p] = self.F_tmp[p]
            self.stress[p] = self.stress_tmp[p]
            self.pressure[p] = self.pressure_tmp[p]
            self.level[p] = self.level_tmp[p]
            self.mass[p] = self.mass_tmp[p]
            self.volume0[p] = self.volume0_tmp[p]
            self.gradient_level[p] = self.gradient_level_tmp[p]
        self.active_count[None] = n

    def merge_particles(self):
        if not self.merge_enabled or self.grid.num_levels <= 1:
            return
        for level in range(self.grid.num_levels - 1, 0, -1):
            self._clear_merge_bins()
            self._accumulate_merge_bins(level)
            self._finalize_merge_bins(level)
        self._reset_compaction()
        self._scatter_compaction()
        self._apply_compaction()

    def n_active(self):
        return int(self.active_count[None])
