import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import taichi as ti
import config
from core.quadtree_grid import QuadtreeGrid2D
from core.adaptive_particles import AdaptiveParticleSystem2D
from physics.constitutive_model import StressUsingWater
import physics.boundary as bnd


@ti.data_oriented
class AdaptiveMPMSolver2D:
    def __init__(self, refinement_box=None, max_level=None):
        self._configure_from_standard_scenario()
        self.grid = QuadtreeGrid2D(refinement_box=refinement_box, max_level=max_level)
        self._apply_fine_level_timestep()
        self.particles = AdaptiveParticleSystem2D(self.grid)
        self.scatter_to_ancestors = bool(getattr(config, 'AMR_SCATTER_TO_ANCESTORS', True))
        self.allow_promotion_without_split = (bool(getattr(config, 'AMR_ALLOW_LEVEL_PROMOTION_WITHOUT_SPLIT', False))
                                              and not self.particles.split_enabled)
        self._split_overflow_warned = False
        self._step_count = 0

    def _configure_from_standard_scenario(self):
        if config.ACTIVE_SCENARIO in ["DAM_BREAK", "IMMERSED"]:
            config.AMR_DOMAIN_MIN_X = config.PADDING * config.DX
            config.AMR_DOMAIN_MIN_Y = config.PADDING * config.DY
            config.AMR_DOMAIN_WIDTH = config.GRID_WIDTH
            config.AMR_DOMAIN_HEIGHT = config.GRID_HEIGHT
            config.AMR_BASE_DX = config.DX
            config.AMR_BASE_CELLS_X = config.N_CELL_WIDTH
            config.AMR_BASE_CELLS_Y = config.N_CELL_HEIGHT
            config.AMR_FINE_REGION_CENTER_X = config.AMR_DOMAIN_MIN_X + 0.5 * config.GRID_WIDTH
            config.AMR_FINE_REGION_YMIN = config.AMR_DOMAIN_MIN_Y
            config.AMR_INITIAL_FLUID_XMIN = config.POS_MP_LEFT_BOTTOM[0]
            config.AMR_INITIAL_FLUID_XMAX = config.POS_MP_LEFT_BOTTOM[0] + config.MP_WIDTH
            config.AMR_INITIAL_FLUID_YMIN = config.POS_MP_LEFT_BOTTOM[1]
            config.AMR_INITIAL_FLUID_YMAX = config.POS_MP_LEFT_BOTTOM[1] + config.MP_HEIGHT
            if config.ACTIVE_SCENARIO == "IMMERSED":
                part_width = config.INT_MOVINGRECT_XMAX - config.INT_MOVINGRECT_XMIN
                config.AMR_FINE_REGION_CENTER_X = 0.5 * (config.INT_MOVINGRECT_XMIN + config.INT_MOVINGRECT_XMAX)
                config.AMR_FINE_REGION_WIDTH = part_width + 2.0 * getattr(config, 'AMR_PROCESS_MARGIN', 0.05)
                config.AMR_FINE_REGION_YMIN = config.AMR_DOMAIN_MIN_Y
                config.AMR_PROCESS_ZONE_HEIGHT = max(config.AMR_PROCESS_ZONE_HEIGHT, config.INT_MOVINGRECT_YMAX - config.AMR_DOMAIN_MIN_Y)

    def _apply_fine_level_timestep(self):
        if bool(getattr(config, 'AMR_USE_FINE_LEVEL_DT', True)):
            fine_dx = self.grid.dx[-1]
            max_wave_speed = config.C_0 + getattr(config, 'V_MAX_ESTIMATE', 0.0)
            dt_fine = config.CFL * fine_dx / max_wave_speed
            config.DT = min(config.DT, dt_fine)
            if hasattr(config, 'FRAME_DT'):
                config.SUBSTEPS = int(math.ceil(config.FRAME_DT / config.DT))
                config.DT = config.FRAME_DT / config.SUBSTEPS
            else:
                config.SUBSTEPS = 1

    @ti.func
    def _weights(self, level: ti.template(), x):
        inv_dx = self.grid.level_inv_dx[level]
        fx = (x - self.grid.origin[level]) * inv_dx
        base = ti.cast(fx - 0.5, ti.i32)
        d = fx - ti.cast(base, ti.f64)
        w0 = 0.5 * (1.5 - d)**2
        w1 = 0.75 - (d - 1.0)**2
        w2 = 0.5 * (d - 0.5)**2
        dw0 = (d - 1.5) * inv_dx
        dw1 = -2.0 * (d - 1.0) * inv_dx
        dw2 = (d - 0.5) * inv_dx
        return base, w0, w1, w2, dw0, dw1, dw2

    @ti.func
    def _weight_component(self, axis: ti.template(), idx: ti.template(), w0, w1, w2):
        value = w0[axis]
        if ti.static(idx == 1):
            value = w1[axis]
        if ti.static(idx == 2):
            value = w2[axis]
        return value

    @ti.func
    def _grad_component(self, axis: ti.template(), idx: ti.template(), dw0, dw1, dw2):
        value = dw0[axis]
        if ti.static(idx == 1):
            value = dw1[axis]
        if ti.static(idx == 2):
            value = dw2[axis]
        return value

    @ti.func
    def _p2g_level(self, p, level: ti.template()):
        x_p = self.particles.x[p]
        v_p = self.particles.v[p]
        C_p = self.particles.C[p]
        p_mass = self.particles.mass[p]
        base, w0, w1, w2, _, _, _ = self._weights(level, x_p)
        for i, j in ti.static(ti.ndrange(3, 3)):
            I = base + ti.Vector([i, j])
            if self.grid.in_bounds(level, I):
                weight = self._weight_component(0, i, w0, w1, w2) * self._weight_component(1, j, w0, w1, w2)
                x_I = self.grid.node_position(level, I)
                dpos = x_I - x_p
                affine = C_p @ dpos
                self.grid.m[level][I] += weight * p_mass
                self.grid.v[level][I] += weight * p_mass * (v_p + affine)

    @ti.kernel
    def p2g_APIC(self):
        for p in range(self.particles.active_count[None]):
            particle_level = self.particles.level[p]
            for level in ti.static(range(self.grid.num_levels)):
                if ti.static(self.scatter_to_ancestors):
                    if level <= particle_level:
                        self._p2g_level(p, level)
                else:
                    if level == particle_level:
                        self._p2g_level(p, level)

    @ti.func
    def _compute_forces_level(self, p, level: ti.template()):
        x_p = self.particles.x[p]
        stress_p = self.particles.stress[p]
        J_p = self.particles.F[p].determinant()
        current_vol = self.particles.volume0[p] * J_p
        base, w0, w1, w2, dw0, dw1, dw2 = self._weights(level, x_p)
        for i, j in ti.static(ti.ndrange(3, 3)):
            I = base + ti.Vector([i, j])
            if self.grid.in_bounds(level, I):
                wx = self._weight_component(0, i, w0, w1, w2)
                wy = self._weight_component(1, j, w0, w1, w2)
                dwx = self._grad_component(0, i, dw0, dw1, dw2)
                dwy = self._grad_component(1, j, dw0, dw1, dw2)
                sf_weight = wx * wy
                sf_grad = ti.Vector([dwx * wy, wx * dwy])
                f_int = (stress_p @ sf_grad) * current_vol
                f_ext = sf_weight * self.particles.mass[p] * ti.Vector(config.GRAVITY)
                self.grid.f[level][I] += f_ext - f_int

    @ti.kernel
    def compute_forces(self):
        for p in range(self.particles.active_count[None]):
            particle_level = self.particles.level[p]
            for level in ti.static(range(self.grid.num_levels)):
                if ti.static(self.scatter_to_ancestors):
                    if level <= particle_level:
                        self._compute_forces_level(p, level)
                else:
                    if level == particle_level:
                        self._compute_forces_level(p, level)

    @ti.kernel
    def compute_moving_ebc_forces(self, t: float):
        for level in ti.static(range(self.grid.num_levels)):
            for I in ti.grouped(self.grid.m[level]):
                m_I = self.grid.m[level][I]
                if m_I > self.grid.node_mass_cutoff[level]:
                    x_I = self.grid.node_position(level, I)
                    v_platform_x = 0.0
                    v_platform_y = 0.0
                    displacement_y = 0.0
                    if t < config.PLATFORM_STOP_TIME:
                        v_platform_y = config.PLATFORM_VELOCITY_Y
                        displacement_y = v_platform_y * t
                    elif t < config.PLATFORM_STOP_TIME + config.PLATFORM_DECEL_TIME:
                        time_in_decel = t - config.PLATFORM_STOP_TIME
                        progress = time_in_decel / config.PLATFORM_DECEL_TIME
                        v_platform_y = config.PLATFORM_VELOCITY_Y * (1.0 - progress)
                        dist_before_stop = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                        dist_during_decel = config.PLATFORM_VELOCITY_Y * time_in_decel - 0.5 * (config.PLATFORM_VELOCITY_Y / config.PLATFORM_DECEL_TIME) * (time_in_decel**2)
                        displacement_y = dist_before_stop + dist_during_decel
                    else:
                        dist_before_stop = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                        total_decel_dist = 0.5 * config.PLATFORM_VELOCITY_Y * config.PLATFORM_DECEL_TIME
                        displacement_y = dist_before_stop + total_decel_dist
                    r, normal = bnd.Get_Rect_SDF(
                        x_I,
                        config.INT_MOVINGRECT_XMIN,
                        config.INT_MOVINGRECT_XMAX,
                        config.INT_MOVINGRECT_YMIN + displacement_y,
                        config.INT_MOVINGRECT_YMAX + displacement_y,
                    )
                    f_bc = bnd.Compute_EBC_Force(
                        m_I,
                        self.grid.v[level][I],
                        self.grid.f[level][I],
                        r,
                        normal,
                        3,
                        ti.Vector([v_platform_x, v_platform_y]),
                    )
                    self.grid.f[level][I] += f_bc

    @ti.kernel
    def grid_update(self, damping: float):
        for level in ti.static(range(self.grid.num_levels)):
            for I in ti.grouped(self.grid.m[level]):
                m_I = self.grid.m[level][I]
                if m_I > self.grid.node_mass_cutoff[level]:
                    self.grid.v_old[level][I] = self.grid.v[level][I]
                    self.grid.v[level][I] += (self.grid.f[level][I] / m_I) * config.DT
                    self.grid.v[level][I] *= damping

    @ti.kernel
    def apply_boundaries(self):
        # Free-slip walls: zero only the wall-normal velocity component,
        # identical logic to the standard (non-AMR) solver.
        for level in ti.static(range(self.grid.num_levels)):
            for I in ti.grouped(self.grid.v[level]):
                x_I = self.grid.node_position(level, I)
                if x_I[0] <= self.grid.domain_min[0] and self.grid.v[level][I][0] < 0.0:
                    self.grid.v[level][I][0] = 0.0
                if x_I[0] >= self.grid.domain_max[0] and self.grid.v[level][I][0] > 0.0:
                    self.grid.v[level][I][0] = 0.0
                if x_I[1] <= self.grid.domain_min[1] and self.grid.v[level][I][1] < 0.0:
                    self.grid.v[level][I][1] = 0.0
                if x_I[1] >= self.grid.domain_max[1] and self.grid.v[level][I][1] > 0.0:
                    self.grid.v[level][I][1] = 0.0

    @ti.func
    def _g2p_level(self, p, level: ti.template(), t: float):
        x_p = self.particles.x[p]
        base, w0, w1, w2, _, _, _ = self._weights(level, x_p)
        v_new = ti.Vector.zero(ti.f64, 2)
        B_new = ti.Matrix.zero(ti.f64, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            I = base + ti.Vector([i, j])
            if self.grid.in_bounds(level, I):
                weight = self._weight_component(0, i, w0, w1, w2) * self._weight_component(1, j, w0, w1, w2)
                v_I = self.grid.v[level][I]
                dpos = self.grid.node_position(level, I) - x_p
                v_new += weight * v_I
                B_new += weight * v_I.outer_product(dpos)
        C_new = B_new * (4.0 * self.grid.level_inv_dx[level] * self.grid.level_inv_dx[level])
        self.particles.v[p] = v_new
        self.particles.C[p] = C_new
        new_x = x_p + v_new * config.DT
        eps = 0.25 * self.grid.level_dx[level]
        new_x[0] = ti.max(self.grid.domain_min[0] + eps, ti.min(new_x[0], self.grid.domain_max[0] - eps))
        new_x[1] = ti.max(self.grid.domain_min[1] + eps, ti.min(new_x[1], self.grid.domain_max[1] - eps))
        if ti.static(config.ACTIVE_SCENARIO == "IMMERSED"):
            displacement_y = 0.0
            if t < config.PLATFORM_STOP_TIME:
                displacement_y = config.PLATFORM_VELOCITY_Y * t
            elif t < config.PLATFORM_STOP_TIME + config.PLATFORM_DECEL_TIME:
                time_in_decel = t - config.PLATFORM_STOP_TIME
                dist_before = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                dist_during = config.PLATFORM_VELOCITY_Y * time_in_decel - 0.5 * (config.PLATFORM_VELOCITY_Y / config.PLATFORM_DECEL_TIME) * (time_in_decel**2)
                displacement_y = dist_before + dist_during
            else:
                dist_before = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                dist_during = 0.5 * config.PLATFORM_VELOCITY_Y * config.PLATFORM_DECEL_TIME
                displacement_y = dist_before + dist_during
            box_ymin = config.INT_MOVINGRECT_YMIN + displacement_y
            box_ymax = config.INT_MOVINGRECT_YMAX + displacement_y
            r, n = bnd.Get_Rect_SDF(new_x, config.INT_MOVINGRECT_XMIN, config.INT_MOVINGRECT_XMAX, box_ymin, box_ymax)
            if r < 0.0:
                new_x = new_x + (ti.abs(r) + 1e-5) * n
        self.particles.x[p] = new_x
        identity = ti.Matrix.identity(ti.f64, 2)
        F_new = (identity + C_new * config.DT) @ self.particles.F[p]
        self.particles.F[p] = F_new
        stress_new = StressUsingWater(F_new, C_new)
        self.particles.stress[p] = stress_new
        J = ti.max(F_new.determinant(), 0.1)
        K = config.C_0**2 * config.RHO_0
        self.particles.pressure[p] = ti.max(K * (1.0 / J - 1.0), 0.0)
        new_level = self.grid.finest_level_at(new_x)
        if ti.static(self.allow_promotion_without_split):
            self.particles.level[p] = new_level
        else:
            if ti.static(not self.particles.merge_enabled):
                if new_level < self.particles.level[p]:
                    self.particles.level[p] = new_level

    @ti.kernel
    def g2p_APIC(self, t: float):
        for p in range(self.particles.active_count[None]):
            particle_level = self.particles.level[p]
            for level in ti.static(range(self.grid.num_levels)):
                if particle_level == level:
                    self._g2p_level(p, level, t)

    @ti.kernel
    def count_particles_by_level(self, counts: ti.template()):
        for level in ti.static(range(self.grid.num_levels)):
            counts[level] = 0
        for p in range(self.particles.active_count[None]):
            ti.atomic_add(counts[self.particles.level[p]], 1)

    def step(self, damping=1.0, current_time=0.0):
        self.grid.clear()
        self.p2g_APIC()
        self.grid.normalize_momentum()
        self.compute_forces()
        if config.ACTIVE_SCENARIO == "IMMERSED":
            self.compute_moving_ebc_forces(current_time)
        self.grid_update(damping)
        self.apply_boundaries()
        self.grid.fill_fine_boundary_velocities()
        self.apply_boundaries()
        self.g2p_APIC(current_time)
        if self.particles.split_enabled:
            self.particles.merge_particles()
            self.particles.split_particles()
            self._step_count += 1
            if self._step_count % 500 == 0 and not self._split_overflow_warned:
                if self.particles.split_overflow[None] > 0:
                    print("WARNING: particle split capacity exhausted; increase AMR_PARTICLE_CAPACITY_FACTOR")
                    self._split_overflow_warned = True
