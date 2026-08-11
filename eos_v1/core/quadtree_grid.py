import sys
import os
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taichi as ti
import config


@ti.data_oriented
class QuadtreeGrid2D:
    def __init__(self, refinement_box=None, max_level=None):
        self.max_level = int(max_level if max_level is not None else getattr(config, 'AMR_MAX_LEVEL', 3))
        self.num_levels = self.max_level + 1
        self.padding = int(getattr(config, 'AMR_GRID_PADDING', config.PADDING))
        self.ghost_band_cells = int(getattr(config, 'AMR_GHOST_BAND_CELLS', 2))
        self.ppc_axis = int(getattr(config, 'AMR_PARTICLES_PER_CELL_AXIS', 2))
        self.domain_width = float(getattr(config, 'AMR_DOMAIN_WIDTH', config.GRID_WIDTH))
        self.domain_height = float(getattr(config, 'AMR_DOMAIN_HEIGHT', config.GRID_HEIGHT))
        self.domain_min = np.array([
            float(getattr(config, 'AMR_DOMAIN_MIN_X', 0.0)),
            float(getattr(config, 'AMR_DOMAIN_MIN_Y', 0.0)),
        ], dtype=np.float64)
        self.domain_max = self.domain_min + np.array([self.domain_width, self.domain_height], dtype=np.float64)
        self.base_cells_x = int(getattr(config, 'AMR_BASE_CELLS_X', config.N_CELL_WIDTH))
        self.base_cells_y = int(getattr(config, 'AMR_BASE_CELLS_Y', config.N_CELL_HEIGHT))
        self.base_dx = float(getattr(config, 'AMR_BASE_DX', self.domain_width / self.base_cells_x))
        self.refine_buffer_cells = int(getattr(config, 'AMR_REFINEMENT_BUFFER_CELLS', 4))
        self.dynamic_refinement = bool(getattr(config, 'AMR_DYNAMIC_REFINEMENT', False))
        for name, extent in (('width', self.domain_width), ('height', self.domain_height)):
            n_cells = extent / self.base_dx
            if abs(n_cells - round(n_cells)) > 1e-6:
                raise ValueError(f"Domain {name} {extent} must be an integer multiple of AMR_BASE_DX {self.base_dx}")

        if refinement_box is None:
            if self.dynamic_refinement and config.ACTIVE_SCENARIO in ("IMMERSED",):
                refinement_box = self._moving_platform_refinement_box()
            else:
                refinement_box = self._default_refinement_box()
        fine_min, fine_max = refinement_box
        self.fine_region_min = np.clip(np.asarray(fine_min, dtype=np.float64), self.domain_min, self.domain_max)
        self.fine_region_max = np.clip(np.asarray(fine_max, dtype=np.float64), self.domain_min, self.domain_max)
        if np.any(self.fine_region_max <= self.fine_region_min):
            raise ValueError(f"Refinement box {refinement_box} is empty after clipping to the domain")

        self.dx = []
        self.inv_dx = []
        self.region_min_np = np.zeros((self.num_levels, 2), dtype=np.float64)
        self.region_max_np = np.zeros((self.num_levels, 2), dtype=np.float64)
        self.origin_np = np.zeros((self.num_levels, 2), dtype=np.float64)
        self.res = []
        self.res_x = []
        self.res_y = []

        self._build_level_geometry()
        self.reference_region_min_np = self.region_min_np.copy()
        self.reference_region_max_np = self.region_max_np.copy()
        self.reference_origin_np = self.origin_np.copy()
        self.level_refinement_shift_np = np.zeros((self.num_levels, 2), dtype=np.float64)

        self.node_mass_cutoff = [1e-6 * float(config.RHO_0) * (self.dx[l] / self.ppc_axis) ** 2
                                 for l in range(self.num_levels)]
        self.face_interior = []
        for level in range(self.num_levels):
            tol = 0.5 * self.dx[level]
            self.face_interior.append((
                bool(self.region_min_np[level][0] > self.domain_min[0] + tol),
                bool(self.region_max_np[level][0] < self.domain_max[0] - tol),
                bool(self.region_min_np[level][1] > self.domain_min[1] + tol),
                bool(self.region_max_np[level][1] < self.domain_max[1] - tol),
            ))

        self.region_min = ti.Vector.field(2, dtype=ti.f64, shape=self.num_levels)
        self.region_max = ti.Vector.field(2, dtype=ti.f64, shape=self.num_levels)
        self.origin = ti.Vector.field(2, dtype=ti.f64, shape=self.num_levels)
        self.reference_region_min = ti.Vector.field(2, dtype=ti.f64, shape=self.num_levels)
        self.reference_region_max = ti.Vector.field(2, dtype=ti.f64, shape=self.num_levels)
        self.reference_origin = ti.Vector.field(2, dtype=ti.f64, shape=self.num_levels)
        self.refinement_shift = ti.Vector.field(2, dtype=ti.f64, shape=())
        self.level_refinement_shift = ti.Vector.field(2, dtype=ti.f64, shape=self.num_levels)
        self.platform_bounds = ti.Vector.field(4, dtype=ti.f64, shape=())
        self.platform_motion_parameters = ti.Vector.field(4, dtype=ti.f64, shape=())
        self.dynamic_domain_min = ti.Vector.field(2, dtype=ti.f64, shape=())
        self.dynamic_domain_max = ti.Vector.field(2, dtype=ti.f64, shape=())
        self.level_dx = ti.field(dtype=ti.f64, shape=self.num_levels)
        self.level_inv_dx = ti.field(dtype=ti.f64, shape=self.num_levels)
        # Dynamic refinement criterion
        self.refinement_criterion = str(getattr(config, 'AMR_REFINEMENT_CRITERION', 'platform'))
        self._initial_criterion_center = None  # set on first update_dynamic_refinement call
        self.region_min.from_numpy(self.region_min_np.astype(np.float64))
        self.region_max.from_numpy(self.region_max_np.astype(np.float64))
        self.origin.from_numpy(self.origin_np.astype(np.float64))
        self.reference_region_min.from_numpy(self.reference_region_min_np.astype(np.float64))
        self.reference_region_max.from_numpy(self.reference_region_max_np.astype(np.float64))
        self.reference_origin.from_numpy(self.origin_np.astype(np.float64))
        self.refinement_shift[None] = [0.0, 0.0]
        self.level_refinement_shift.from_numpy(self.level_refinement_shift_np)
        self.platform_bounds[None] = [
            config.INT_MOVINGRECT_XMIN,
            config.INT_MOVINGRECT_XMAX,
            config.INT_MOVINGRECT_YMIN,
            config.INT_MOVINGRECT_YMAX,
        ]
        self.platform_motion_parameters[None] = [
            config.PLATFORM_VELOCITY_X,
            config.PLATFORM_VELOCITY_Y,
            config.PLATFORM_STOP_TIME,
            config.PLATFORM_DECEL_TIME,
        ]
        self.dynamic_domain_min[None] = self.domain_min
        self.dynamic_domain_max[None] = self.domain_max
        self.level_dx.from_numpy(np.array(self.dx, dtype=np.float64))
        self.level_inv_dx.from_numpy(np.array(self.inv_dx, dtype=np.float64))

        self.m = []
        self.v = []
        self.f = []
        self.v_old = []
        self.boundary_mass = []
        self.boundary_momentum = []
        self.domain_boundary_mass = []
        self.moving_boundary_mass = []
        self.moving_boundary_momentum = []
        for level in range(self.num_levels):
            shape = self.res[level]
            self.m.append(ti.field(dtype=ti.f64, shape=shape))
            self.v.append(ti.Vector.field(2, dtype=ti.f64, shape=shape))
            self.f.append(ti.Vector.field(2, dtype=ti.f64, shape=shape))
            self.v_old.append(ti.Vector.field(2, dtype=ti.f64, shape=shape))
            self.boundary_mass.append(ti.Vector.field(2, dtype=ti.f64, shape=shape))
            self.boundary_momentum.append(ti.Vector.field(2, dtype=ti.f64, shape=shape))
            self.domain_boundary_mass.append(ti.Vector.field(2, dtype=ti.f64, shape=shape))
            self.moving_boundary_mass.append(ti.Vector.field(2, dtype=ti.f64, shape=shape))
            self.moving_boundary_momentum.append(ti.Vector.field(2, dtype=ti.f64, shape=shape))

        for field, values in zip(self.domain_boundary_mass, self._build_domain_boundary_mass()):
            field.from_numpy(values)

        self.leaf_level, self.leaf_origin, self.leaf_size = self._build_leaf_cells()
        self.leaf_count = len(self.leaf_level)
        self._validate_leaf_tiling()

        # Platform penalty cache: recompute the B-spline mass stencil only
        # when the platform has moved by a meaningful fraction of the finest
        # cell.  Between updates, a GPU kernel reuses the cached mass and
        # scales it by the current platform velocity for the momentum.
        self._platform_penalty_position_cache = None
        self._platform_penalty_threshold = 0.25 * self.dx[-1]

    def _add_penalty_quadrature(self, mass, momentum, level, x_q, switch, wall_velocity, weight):
        dx = self.dx[level]
        fx = (x_q - self.origin_np[level]) / dx
        base = np.floor(fx - 0.5).astype(np.int32)
        d = fx - base
        weights = (
            0.5 * (1.5 - d) ** 2,
            0.75 - (d - 1.0) ** 2,
            0.5 * (d - 0.5) ** 2,
        )
        beta = config.AMR_BOUNDARY_PENALTY_NORMAL * config.RHO_0 * dx ** 2
        for i in range(3):
            for j in range(3):
                I = base + np.array([i, j], dtype=np.int32)
                if 0 <= I[0] < self.res_x[level] and 0 <= I[1] < self.res_y[level]:
                    contribution = beta * weight * weights[i][0] * weights[j][1]
                    mass[I[0], I[1]] += contribution * switch
                    momentum[I[0], I[1]] += contribution * switch * wall_velocity

    def _add_penalty_segment(self, mass, momentum, level, start, end, switch, wall_velocity):
        dx = self.dx[level]
        segment = end - start
        length = float(np.linalg.norm(segment))
        count = max(1, int(math.ceil(length / dx)))
        weight = 0.5 * length / (count * dx)
        # Vectorize over all quadrature points (count cells × 2 quad points).
        xis = np.array([0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)])
        cells = np.arange(count, dtype=np.float64)
        # points shape: (count, 2, 2) → (count*2, 2) [cell, quad_point, xy]
        points = start[None, None, :] + (cells[:, None, None] + xis[None, :, None]) * segment[None, None, :] / count
        points = points.reshape(-1, 2)
        fx = (points - self.origin_np[level][None, :]) / dx
        base = np.floor(fx - 0.5).astype(np.int32)
        d = fx - base
        w = np.stack([
            0.5 * (1.5 - d) ** 2,
            0.75 - (d - 1.0) ** 2,
            0.5 * (d - 0.5) ** 2,
        ], axis=-1)  # (N, 2, 3)
        beta = config.AMR_BOUNDARY_PENALTY_NORMAL * config.RHO_0 * dx ** 2
        contrib = beta * weight
        switch = np.asarray(switch, dtype=np.float64)
        wall_velocity = np.asarray(wall_velocity, dtype=np.float64)
        for i in range(3):
            for j in range(3):
                I = base + np.array([i, j], dtype=np.int32)  # (N, 2)
                valid = (I[:, 0] >= 0) & (I[:, 0] < self.res_x[level]) & (I[:, 1] >= 0) & (I[:, 1] < self.res_y[level])
                if not np.any(valid):
                    continue
                Iv = I[valid]
                wx = w[valid, 0, i]
                wy = w[valid, 1, j]
                c = contrib * wx * wy  # (Nv,)
                np.add.at(mass, (Iv[:, 0], Iv[:, 1]), c[:, None] * switch)
                np.add.at(momentum, (Iv[:, 0], Iv[:, 1]), c[:, None] * switch * wall_velocity)

    def _build_domain_boundary_mass(self):
        masses = [np.zeros((*shape, 2), dtype=np.float64) for shape in self.res]
        zero_momentum = [np.zeros_like(mass) for mass in masses]
        zero_velocity = np.zeros(2, dtype=np.float64)
        for level in range(self.num_levels):
            minimum = self.region_min_np[level]
            maximum = self.region_max_np[level]
            left, right, bottom, top = self.face_interior[level]
            if not bottom:
                self._add_penalty_segment(
                    masses[level], zero_momentum[level], level,
                    np.array([minimum[0], self.domain_min[1]]),
                    np.array([maximum[0], self.domain_min[1]]),
                    np.array([0.0, 1.0]), zero_velocity,
                )
            if not top:
                self._add_penalty_segment(
                    masses[level], zero_momentum[level], level,
                    np.array([minimum[0], self.domain_max[1]]),
                    np.array([maximum[0], self.domain_max[1]]),
                    np.array([0.0, 1.0]), zero_velocity,
                )
            if not left:
                self._add_penalty_segment(
                    masses[level], zero_momentum[level], level,
                    np.array([self.domain_min[0], minimum[1]]),
                    np.array([self.domain_min[0], maximum[1]]),
                    np.array([1.0, 0.0]), zero_velocity,
                )
            if not right:
                self._add_penalty_segment(
                    masses[level], zero_momentum[level], level,
                    np.array([self.domain_max[0], minimum[1]]),
                    np.array([self.domain_max[0], maximum[1]]),
                    np.array([1.0, 0.0]), zero_velocity,
                )
        return masses

    def update_moving_platform_penalty_mass(self, t):
        masses = [np.zeros((*shape, 2), dtype=np.float64) for shape in self.res]
        momenta = [np.zeros_like(mass) for mass in masses]
        displacement, velocity = self._platform_motion_numpy(t)
        x_min = config.INT_MOVINGRECT_XMIN + displacement[0]
        x_max = config.INT_MOVINGRECT_XMAX + displacement[0]
        y_min = config.INT_MOVINGRECT_YMIN + displacement[1]
        y_max = config.INT_MOVINGRECT_YMAX + displacement[1]
        for level in range(self.num_levels):
            self._add_penalty_segment(
                masses[level], momenta[level], level,
                np.array([x_min, y_min]), np.array([x_max, y_min]),
                np.array([0.0, 1.0]), velocity,
            )
            self._add_penalty_segment(
                masses[level], momenta[level], level,
                np.array([x_min, y_max]), np.array([x_max, y_max]),
                np.array([0.0, 1.0]), velocity,
            )
            self._add_penalty_segment(
                masses[level], momenta[level], level,
                np.array([x_min, y_min]), np.array([x_min, y_max]),
                np.array([1.0, 0.0]), velocity,
            )
            self._add_penalty_segment(
                masses[level], momenta[level], level,
                np.array([x_max, y_min]), np.array([x_max, y_max]),
                np.array([1.0, 0.0]), velocity,
            )
        for field, values in zip(self.moving_boundary_mass, masses):
            field.from_numpy(values)
        for field, values in zip(self.moving_boundary_momentum, momenta):
            field.from_numpy(values)

    def _moving_platform_refinement_box(self):
        margin_x = float(getattr(config, 'AMR_PROCESS_MARGIN', 0.0))
        margin_y = float(getattr(config, 'AMR_DYNAMIC_PLATFORM_MARGIN_Y', margin_x))
        return (
            (config.INT_MOVINGRECT_XMIN - margin_x, config.INT_MOVINGRECT_YMIN - margin_y),
            (config.INT_MOVINGRECT_XMAX + margin_x, config.INT_MOVINGRECT_YMAX + margin_y),
        )

    def _platform_motion_numpy(self, t):
        velocity = np.array([config.PLATFORM_VELOCITY_X, config.PLATFORM_VELOCITY_Y], dtype=np.float64)
        stop_time = float(config.PLATFORM_STOP_TIME)
        decel_time = float(config.PLATFORM_DECEL_TIME)
        if t < stop_time:
            return velocity * t, velocity
        if t < stop_time + decel_time:
            time_in_decel = t - stop_time
            return (
                velocity * (stop_time + time_in_decel - 0.5 * time_in_decel ** 2 / decel_time),
                velocity * (1.0 - time_in_decel / decel_time),
            )
        return velocity * (stop_time + 0.5 * decel_time), np.zeros(2, dtype=np.float64)

    def _snap_dynamic_shift(self, displacement, spacing):
        return np.copysign(np.floor(np.abs(displacement) / spacing + 0.5), displacement) * spacing

    def _shifts_for_displacement(self, displacement):
        """Compute per-level shifts for a given displacement vector, respecting
        parent-child nesting and domain bounds."""
        shifts = np.zeros((self.num_levels, 2), dtype=np.float64)
        for level in range(1, self.num_levels):
            spacing = self.dx[level - 1]
            desired = self._snap_dynamic_shift(displacement, spacing)
            parent_min = self.reference_region_min_np[level - 1] + shifts[level - 1]
            parent_max = self.reference_region_max_np[level - 1] + shifts[level - 1]
            lower = np.maximum(self.domain_min - self.reference_region_min_np[level], parent_min - self.reference_region_min_np[level])
            upper = np.minimum(self.domain_max - self.reference_region_max_np[level], parent_max - self.reference_region_max_np[level])
            lower = np.ceil(lower / spacing - 1e-9) * spacing
            upper = np.floor(upper / spacing + 1e-9) * spacing
            if np.any(lower > upper):
                raise ValueError(f"Dynamic level {level} cannot remain nested inside its parent")
            shifts[level] = np.minimum(np.maximum(desired, lower), upper)
        return shifts

    def _level_dynamic_shifts(self, t):
        """Legacy interface: compute shifts from platform motion at time t."""
        displacement, _ = self._platform_motion_numpy(t)
        return self._shifts_for_displacement(displacement)

    def _compute_criterion_center(self, particles):
        """Compute the mass-weighted centroid of particles that exceed the
        configured refinement criterion threshold.

        Returns None if no particles exceed the threshold (e.g. at rest).
        """
        n = particles.n_active()
        if n == 0:
            return None
        x = particles.x.to_numpy()[:n]
        mass = particles.mass.to_numpy()[:n]
        criterion = self.refinement_criterion

        if criterion == "platform":
            # Platform criterion is handled by the caller via _platform_motion_numpy
            return None

        # Build a weight mask for the selected criterion
        weights = np.zeros(n, dtype=np.float64)
        if criterion in ("velocity", "combined"):
            v = particles.v.to_numpy()[:n]
            speed = np.linalg.norm(v, axis=1)
            threshold = float(getattr(config, 'AMR_REFINEMENT_VELOCITY_FRACTION', 0.05)) * float(getattr(config, 'V_MAX_ESTIMATE', 1.0))
            mask = speed > threshold
            weights[mask] += mass[mask] * speed[mask]
        if criterion in ("pressure", "combined"):
            p = particles.pressure.to_numpy()[:n]
            threshold = float(getattr(config, 'AMR_REFINEMENT_PRESSURE_FRACTION', 0.01)) * float(config.RHO_0 * config.C_0 ** 2)
            mask = p > threshold
            weights[mask] += mass[mask] * p[mask]
        if criterion in ("deformation", "combined"):
            F = particles.F.to_numpy()[:n]
            J = np.linalg.det(F)
            deformation = np.abs(J - 1.0)
            threshold = float(getattr(config, 'AMR_REFINEMENT_DEFORMATION_THRESHOLD', 0.01))
            mask = deformation > threshold
            weights[mask] += mass[mask] * deformation[mask]

        total_weight = weights.sum()
        if total_weight < 1e-15:
            return None
        center = (weights[:, None] * x).sum(axis=0) / total_weight
        return center

    @ti.func
    def _platform_motion(self, t: ti.f64):
        parameters = self.platform_motion_parameters[None]
        velocity = ti.Vector([parameters[0], parameters[1]])
        stop_time = parameters[2]
        decel_time = parameters[3]
        displacement = ti.Vector.zero(ti.f64, 2)
        velocity_out = ti.Vector.zero(ti.f64, 2)
        if t < stop_time:
            velocity_out = velocity
            displacement = velocity * t
        elif t < stop_time + decel_time:
            time_in_decel = t - stop_time
            velocity_out = velocity * (1.0 - time_in_decel / decel_time)
            displacement = velocity * (stop_time + time_in_decel - 0.5 * time_in_decel ** 2 / decel_time)
        else:
            displacement = velocity * (stop_time + 0.5 * decel_time)
        return displacement, velocity_out

    def update_dynamic_refinement(self, t, particles=None):
        if not self.dynamic_refinement:
            return False
        # Compute the displacement vector based on the configured criterion
        if self.refinement_criterion == "platform" or particles is None:
            displacement, _ = self._platform_motion_numpy(t)
        else:
            center = self._compute_criterion_center(particles)
            if center is None:
                # No particles exceed the threshold yet — fall back to platform
                displacement, _ = self._platform_motion_numpy(t)
            else:
                if self._initial_criterion_center is None:
                    self._initial_criterion_center = center.copy()
                displacement = center - self._initial_criterion_center
        shifts = self._shifts_for_displacement(displacement)
        if np.allclose(shifts, self.level_refinement_shift_np, rtol=0.0, atol=1e-12):
            return False
        self.level_refinement_shift_np = shifts
        self.region_min_np = self.reference_region_min_np + shifts
        self.region_max_np = self.reference_region_max_np + shifts
        self.origin_np = self.reference_origin_np + shifts
        self.region_min.from_numpy(self.region_min_np)
        self.region_max.from_numpy(self.region_max_np)
        self.origin.from_numpy(self.origin_np)
        self.level_refinement_shift.from_numpy(shifts)
        self.refinement_shift[None] = shifts[-1]
        self._rebuild_dynamic_domain_boundary_mass()
        self._platform_penalty_position_cache = None
        return True

    @ti.func
    def _add_dynamic_penalty_quadrature(self, level: ti.template(), x_q, switch, wall_velocity, weight):
        half = ti.cast(0.5, ti.f64)
        one = ti.cast(1.0, ti.f64)
        dx = self.level_dx[level]
        fx = (x_q - self.origin[level]) / dx
        base = ti.cast(ti.floor(fx - half), ti.i32)
        d = fx - ti.cast(base, ti.f64)
        w0 = half * (one + half - d)**2
        w1 = ti.cast(0.75, ti.f64) - (d - one)**2
        w2 = half * (d - half)**2
        beta = ti.cast(config.AMR_BOUNDARY_PENALTY_NORMAL, ti.f64) * ti.cast(config.RHO_0, ti.f64) * dx**2
        for i, j in ti.static(ti.ndrange(3, 3)):
            I = base + ti.Vector([i, j])
            if self.in_bounds(level, I):
                wx = w0[0]
                wy = w0[1]
                if ti.static(i == 1):
                    wx = w1[0]
                if ti.static(i == 2):
                    wx = w2[0]
                if ti.static(j == 1):
                    wy = w1[1]
                if ti.static(j == 2):
                    wy = w2[1]
                contribution = beta * weight * wx * wy
                for axis in ti.static(range(2)):
                    self.boundary_mass[level][I][axis] += contribution * switch[axis]
                    self.boundary_momentum[level][I][axis] += contribution * switch[axis] * wall_velocity[axis]

    @ti.func
    def _add_dynamic_penalty_segment(self, level: ti.template(), start, end, switch, wall_velocity):
        dx = self.level_dx[level]
        segment = end - start
        length = segment.norm()
        count = ti.max(1, ti.cast(ti.ceil(length / dx), ti.i32))
        weight = 0.5 * length / (ti.cast(count, ti.f64) * dx)
        ti.loop_config(serialize=True)
        for cell in range(count):
            for quadrature_point in ti.static(range(2)):
                xi = ti.cast(0.21132486540518713, ti.f64)
                if ti.static(quadrature_point == 1):
                    xi = ti.cast(0.7886751345948129, ti.f64)
                point = start + (ti.cast(cell, ti.f64) + xi) * segment / ti.cast(count, ti.f64)
                self._add_dynamic_penalty_quadrature(level, point, switch, wall_velocity, weight)

    def _rebuild_dynamic_domain_boundary_mass(self):
        """Recompute domain boundary penalty mass on CPU after a dynamic grid shift.

        The precomputed ``domain_boundary_mass`` from ``__init__`` reflects the
        initial refinement geometry.  When the refinement window moves, the
        faces that coincide with the domain boundary change, so the cached
        mass is stale.  This method rebuilds it from the current
        ``region_min_np`` / ``region_max_np`` / ``origin_np`` and uploads the
        result, allowing the fast GPU ``initialize_penalty_mass`` kernel to be
        reused every step instead of the serial dynamic kernel.
        """
        masses = [np.zeros((*shape, 2), dtype=np.float64) for shape in self.res]
        zero_momentum = [np.zeros_like(mass) for mass in masses]
        zero_velocity = np.zeros(2, dtype=np.float64)
        for level in range(self.num_levels):
            minimum = self.region_min_np[level]
            maximum = self.region_max_np[level]
            tol = 0.5 * self.dx[level]
            if minimum[1] <= self.domain_min[1] + tol:
                self._add_penalty_segment(
                    masses[level], zero_momentum[level], level,
                    np.array([minimum[0], self.domain_min[1]]),
                    np.array([maximum[0], self.domain_min[1]]),
                    np.array([0.0, 1.0]), zero_velocity,
                )
            if maximum[1] >= self.domain_max[1] - tol:
                self._add_penalty_segment(
                    masses[level], zero_momentum[level], level,
                    np.array([minimum[0], self.domain_max[1]]),
                    np.array([maximum[0], self.domain_max[1]]),
                    np.array([0.0, 1.0]), zero_velocity,
                )
            if minimum[0] <= self.domain_min[0] + tol:
                self._add_penalty_segment(
                    masses[level], zero_momentum[level], level,
                    np.array([self.domain_min[0], minimum[1]]),
                    np.array([self.domain_min[0], maximum[1]]),
                    np.array([1.0, 0.0]), zero_velocity,
                )
            if maximum[0] >= self.domain_max[0] - tol:
                self._add_penalty_segment(
                    masses[level], zero_momentum[level], level,
                    np.array([self.domain_max[0], minimum[1]]),
                    np.array([self.domain_max[0], maximum[1]]),
                    np.array([1.0, 0.0]), zero_velocity,
                )
        for field, values in zip(self.domain_boundary_mass, masses):
            field.from_numpy(values)

    def initialize_dynamic_penalty_mass(self):
        self.initialize_penalty_mass()

    def add_cached_moving_platform_penalty(self, t):
        """Add the moving platform penalty mass/momentum, caching the B-spline
        mass stencil across steps where the platform has barely moved.

        The platform displacement per timestep is typically orders of
        magnitude smaller than the finest cell, so the mass distribution
        changes negligibly.  We recompute it only when the platform has
        moved by more than ``_platform_penalty_threshold`` since the last
        update; between updates we reuse the cached ``moving_boundary_mass``
        and scale it by the current platform velocity on the GPU.
        """
        displacement, velocity = self._platform_motion_numpy(t)
        if (self._platform_penalty_position_cache is None
                or np.linalg.norm(displacement - self._platform_penalty_position_cache)
                > self._platform_penalty_threshold):
            self.update_moving_platform_penalty_mass(t)
            self._platform_penalty_position_cache = displacement.copy()
        self.add_moving_penalty_mass_with_velocity(float(velocity[0]), float(velocity[1]))

    def add_moving_platform_penalty_mass_gpu(self, t):
        displacement, velocity = self._platform_motion_numpy(t)
        self.update_moving_platform_penalty_mass(t)
        self._platform_penalty_position_cache = displacement.copy()
        self.add_moving_penalty_mass_with_velocity(float(velocity[0]), float(velocity[1]))

    def _default_refinement_box(self):
        width = float(getattr(config, 'AMR_FINE_REGION_WIDTH', min(0.002, self.domain_width)))
        height = float(getattr(config, 'AMR_PROCESS_ZONE_HEIGHT', min(0.005, self.domain_height)))
        cx = float(getattr(config, 'AMR_FINE_REGION_CENTER_X', self.domain_min[0] + 0.5 * self.domain_width))
        xmin = max(self.domain_min[0], cx - 0.5 * width)
        xmax = min(self.domain_max[0], cx + 0.5 * width)
        ymin = float(getattr(config, 'AMR_FINE_REGION_YMIN', self.domain_min[1]))
        ymax = min(self.domain_max[1], ymin + height)
        return (xmin, ymin), (xmax, ymax)

    def _build_level_geometry(self):
        for level in range(self.num_levels):
            self.dx.append(self.base_dx / (2 ** level))
            self.inv_dx.append(1.0 / self.dx[level])
        for level in range(self.num_levels):
            dx = self.dx[level]
            if level == 0:
                region_min = self.domain_min.copy()
                region_max = self.domain_max.copy()
            else:
                # Snap the refinement region outward to the parent-level cell
                # lattice so that every level tiles exactly onto its parent.
                parent_dx = self.dx[level - 1]
                grow = (self.max_level - level) * self.refine_buffer_cells * parent_dx
                desired_min = self.fine_region_min - grow
                desired_max = self.fine_region_max + grow
                lo = np.floor((desired_min - self.domain_min) / parent_dx + 1e-6)
                hi = np.ceil((desired_max - self.domain_min) / parent_dx - 1e-6)
                region_min = self.domain_min + lo * parent_dx
                region_max = self.domain_min + hi * parent_dx
                region_min = np.maximum(region_min, self.domain_min)
                region_max = np.minimum(region_max, self.domain_max)
                if np.any(region_max - region_min < 0.5 * parent_dx):
                    raise ValueError(f"Refinement region at level {level} is degenerate: {region_min} {region_max}")
                if level >= 2:
                    prev_min = self.region_min_np[level - 1]
                    prev_max = self.region_max_np[level - 1]
                    eps = 1e-9 * parent_dx
                    if np.any(region_min < prev_min - eps) or np.any(region_max > prev_max + eps):
                        raise ValueError(f"Level {level} region {region_min}..{region_max} is not nested "
                                         f"inside level {level - 1} region {prev_min}..{prev_max}")
            self.region_min_np[level] = region_min
            self.region_max_np[level] = region_max
            self.origin_np[level] = region_min - self.padding * dx
            nx = int(round((region_max[0] - region_min[0]) / dx)) + 2 * self.padding + 1
            ny = int(round((region_max[1] - region_min[1]) / dx)) + 2 * self.padding + 1
            self.res.append((nx, ny))
            self.res_x.append(nx)
            self.res_y.append(ny)

    def _build_leaf_cells(self):
        levels = []
        origins = []
        sizes = []
        for level in range(self.num_levels):
            dx = self.dx[level]
            region_min = self.region_min_np[level]
            region_max = self.region_max_np[level]
            nx = int(round((region_max[0] - region_min[0]) / dx))
            ny = int(round((region_max[1] - region_min[1]) / dx))
            ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
            ox = region_min[0] + ii * dx
            oy = region_min[1] + jj * dx
            keep = np.ones((nx, ny), dtype=bool)
            if level < self.max_level:
                cx = ox + 0.5 * dx
                cy = oy + 0.5 * dx
                nmn = self.region_min_np[level + 1]
                nmx = self.region_max_np[level + 1]
                covered = (cx >= nmn[0]) & (cx < nmx[0]) & (cy >= nmn[1]) & (cy < nmx[1])
                keep = ~covered
            levels.append(np.full(int(keep.sum()), level, dtype=np.int32))
            origins.append(np.stack([ox[keep], oy[keep]], axis=1))
            sizes.append(np.full(int(keep.sum()), dx, dtype=np.float64))
        return (
            np.concatenate(levels),
            np.concatenate(origins).astype(np.float64),
            np.concatenate(sizes).astype(np.float64),
        )

    def _validate_leaf_tiling(self):
        leaf_area = 0.0
        for level in range(self.num_levels):
            count = int(np.sum(self.leaf_level == level))
            leaf_area += count * self.dx[level] ** 2
        domain_area = self.domain_width * self.domain_height
        if abs(leaf_area - domain_area) > 1e-6 * domain_area:
            raise RuntimeError(f"Leaf cells do not tile the domain: leaf area {leaf_area} vs domain {domain_area}")

    @ti.func
    def finest_level_at(self, x):
        level_out = 0
        for level in ti.static(range(1, self.num_levels)):
            mn = self.region_min[level]
            mx = self.region_max[level]
            if x[0] >= mn[0] and x[0] < mx[0] and x[1] >= mn[1] and x[1] < mx[1]:
                level_out = level
        return level_out

    @ti.func
    def node_position(self, level: ti.template(), I):
        return self.origin[level] + ti.cast(I, ti.f64) * self.level_dx[level]

    @ti.func
    def in_bounds(self, level: ti.template(), I):
        return I[0] >= 0 and I[0] < ti.static(self.res_x[level]) and I[1] >= 0 and I[1] < ti.static(self.res_y[level])

    @ti.kernel
    def clear(self):
        for level in ti.static(range(self.num_levels)):
            for I in ti.grouped(self.m[level]):
                self.m[level][I] = 0.0
                self.v[level][I] = ti.Vector.zero(ti.f64, 2)
                self.f[level][I] = ti.Vector.zero(ti.f64, 2)
                self.v_old[level][I] = ti.Vector.zero(ti.f64, 2)
                self.boundary_mass[level][I] = ti.Vector.zero(ti.f64, 2)
                self.boundary_momentum[level][I] = ti.Vector.zero(ti.f64, 2)

    @ti.kernel
    def initialize_penalty_mass(self):
        for level in ti.static(range(self.num_levels)):
            for I in ti.grouped(self.m[level]):
                self.boundary_mass[level][I] = self.domain_boundary_mass[level][I]
                self.boundary_momentum[level][I] = ti.Vector.zero(ti.f64, 2)

    @ti.kernel
    def add_moving_penalty_mass(self):
        for level in ti.static(range(self.num_levels)):
            for I in ti.grouped(self.m[level]):
                self.boundary_mass[level][I] += self.moving_boundary_mass[level][I]
                self.boundary_momentum[level][I] += self.moving_boundary_momentum[level][I]

    @ti.kernel
    def add_moving_penalty_mass_with_velocity(self, vx: ti.f64, vy: ti.f64):
        velocity = ti.Vector([vx, vy])
        for level in ti.static(range(self.num_levels)):
            for I in ti.grouped(self.m[level]):
                mass = self.moving_boundary_mass[level][I]
                self.boundary_mass[level][I] += mass
                self.boundary_momentum[level][I] += mass * velocity

    @ti.kernel
    def normalize_momentum(self):
        for level in ti.static(range(self.num_levels)):
            for I in ti.grouped(self.m[level]):
                if self.m[level][I] > self.node_mass_cutoff[level]:
                    for axis in ti.static(range(2)):
                        effective_mass = self.m[level][I] + self.boundary_mass[level][I][axis]
                        self.v[level][I][axis] = (
                            self.v[level][I][axis] + self.boundary_momentum[level][I][axis]
                        ) / effective_mass
                else:
                    self.v[level][I] = ti.Vector.zero(ti.f64, 2)

    @ti.func
    def sample_velocity(self, level: ti.template(), x):
        inv_dx = self.level_inv_dx[level]
        fx = (x - self.origin[level]) * inv_dx
        base = ti.cast(fx - 0.5, ti.i32)
        d = fx - ti.cast(base, ti.f64)
        w0 = 0.5 * (1.5 - d)**2
        w1 = 0.75 - (d - 1.0)**2
        w2 = 0.5 * (d - 0.5)**2
        v_out = ti.Vector.zero(ti.f64, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            I = base + offset
            if self.in_bounds(level, I):
                wx = w0[0]
                wy = w0[1]
                if ti.static(i == 1): wx = w1[0]
                if ti.static(i == 2): wx = w2[0]
                if ti.static(j == 1): wy = w1[1]
                if ti.static(j == 2): wy = w2[1]
                v_out += wx * wy * self.v[level][I]
        return v_out

    @ti.kernel
    def fill_fine_boundary_velocities(self):
        # Overwrite the velocities of "ghost" nodes on each fine level with an
        # interpolation from the parent level. Ghost nodes are the nodes whose
        # B-spline support reaches outside the level's refinement region (their
        # mass/forces are incomplete because coarse particles outside the region
        # only scatter to coarser levels), plus all (near-)massless nodes.
        # Faces that coincide with the domain boundary are excluded: there is
        # no matter beyond them, so the fine solution there is already complete.
        for level in ti.static(range(1, self.num_levels)):
            for I in ti.grouped(self.m[level]):
                x = self.node_position(level, I)
                band = self.ghost_band_cells * self.level_dx[level]
                ghost = self.m[level][I] <= self.node_mass_cutoff[level]
                tolerance = 0.5 * self.level_dx[level]
                if self.region_min[level][0] > self.domain_min[0] + tolerance:
                    if x[0] < self.region_min[level][0] + band:
                        ghost = True
                if self.region_max[level][0] < self.domain_max[0] - tolerance:
                    if x[0] > self.region_max[level][0] - band:
                        ghost = True
                if self.region_min[level][1] > self.domain_min[1] + tolerance:
                    if x[1] < self.region_min[level][1] + band:
                        ghost = True
                if self.region_max[level][1] < self.domain_max[1] - tolerance:
                    if x[1] > self.region_max[level][1] - band:
                        ghost = True
                if ghost:
                    self.v[level][I] = self.sample_velocity(level - 1, x)
