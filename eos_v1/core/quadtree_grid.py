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
        self.dynamic_refinement = bool(getattr(config, 'AMR_DYNAMIC_REFINEMENT', False)) and config.ACTIVE_SCENARIO == "IMMERSED"
        for name, extent in (('width', self.domain_width), ('height', self.domain_height)):
            n_cells = extent / self.base_dx
            if abs(n_cells - round(n_cells)) > 1e-6:
                raise ValueError(f"Domain {name} {extent} must be an integer multiple of AMR_BASE_DX {self.base_dx}")

        if refinement_box is None:
            if self.dynamic_refinement and config.ACTIVE_SCENARIO == "IMMERSED":
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
        self.dynamic_shift_min_np, self.dynamic_shift_max_np = self._dynamic_shift_bounds()

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
        self.dynamic_shift_min = ti.Vector.field(2, dtype=ti.f64, shape=())
        self.dynamic_shift_max = ti.Vector.field(2, dtype=ti.f64, shape=())
        self.level_dx = ti.field(dtype=ti.f64, shape=self.num_levels)
        self.level_inv_dx = ti.field(dtype=ti.f64, shape=self.num_levels)
        self.region_min.from_numpy(self.region_min_np.astype(np.float64))
        self.region_max.from_numpy(self.region_max_np.astype(np.float64))
        self.origin.from_numpy(self.origin_np.astype(np.float64))
        self.reference_region_min.from_numpy(self.reference_region_min_np.astype(np.float64))
        self.reference_region_max.from_numpy(self.reference_region_max_np.astype(np.float64))
        self.reference_origin.from_numpy(self.origin_np.astype(np.float64))
        self.refinement_shift[None] = [0.0, 0.0]
        self.dynamic_shift_min[None] = self.dynamic_shift_min_np
        self.dynamic_shift_max[None] = self.dynamic_shift_max_np
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
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / self.dx[level])))
        for cell in range(count):
            for xi in (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)):
                point = start + (cell + xi) * (end - start) / count
                weight = 0.5 * length / (count * self.dx[level])
                self._add_penalty_quadrature(mass, momentum, level, point, switch, wall_velocity, weight)

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
        if t < config.PLATFORM_STOP_TIME:
            velocity_y = config.PLATFORM_VELOCITY_Y
            displacement_y = velocity_y * t
        elif t < config.PLATFORM_STOP_TIME + config.PLATFORM_DECEL_TIME:
            time_in_decel = t - config.PLATFORM_STOP_TIME
            velocity_y = config.PLATFORM_VELOCITY_Y * (1.0 - time_in_decel / config.PLATFORM_DECEL_TIME)
            displacement_y = (
                config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                + config.PLATFORM_VELOCITY_Y * time_in_decel
                - 0.5 * config.PLATFORM_VELOCITY_Y * time_in_decel ** 2 / config.PLATFORM_DECEL_TIME
            )
        else:
            velocity_y = 0.0
            displacement_y = config.PLATFORM_VELOCITY_Y * (
                config.PLATFORM_STOP_TIME + 0.5 * config.PLATFORM_DECEL_TIME
            )
        x_min = config.INT_MOVINGRECT_XMIN
        x_max = config.INT_MOVINGRECT_XMAX
        y_min = config.INT_MOVINGRECT_YMIN + displacement_y
        y_max = config.INT_MOVINGRECT_YMAX + displacement_y
        velocity = np.array([0.0, velocity_y], dtype=np.float64)
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

    def _dynamic_shift_bounds(self):
        if not self.dynamic_refinement or self.num_levels <= 1:
            return np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
        lower_y = -np.inf
        upper_y = np.inf
        for level in range(1, self.num_levels):
            inset = self.ghost_band_cells * self.dx[level]
            lower_y = max(lower_y, self.domain_min[1] + inset - self.region_min_np[level][1])
            upper_y = min(upper_y, self.domain_max[1] - inset - self.region_max_np[level][1])
        if lower_y > upper_y:
            raise ValueError("Dynamic refinement window cannot fit inside the domain")
        return (
            np.array([0.0, lower_y], dtype=np.float64),
            np.array([0.0, upper_y], dtype=np.float64),
        )

    @ti.func
    def _platform_motion(self, t: ti.f64):
        velocity = ti.cast(config.PLATFORM_VELOCITY_Y, ti.f64)
        stop_time = ti.cast(config.PLATFORM_STOP_TIME, ti.f64)
        decel_time = ti.cast(config.PLATFORM_DECEL_TIME, ti.f64)
        displacement_y = ti.cast(0.0, ti.f64)
        velocity_y = ti.cast(0.0, ti.f64)
        if t < stop_time:
            velocity_y = velocity
            displacement_y = velocity_y * t
        elif t < stop_time + decel_time:
            time_in_decel = t - stop_time
            velocity_y = velocity * (1.0 - time_in_decel / decel_time)
            displacement_y = velocity * stop_time + velocity * time_in_decel - 0.5 * velocity * time_in_decel ** 2 / decel_time
        else:
            displacement_y = velocity * (stop_time + 0.5 * decel_time)
        return displacement_y, velocity_y

    @ti.kernel
    def _update_dynamic_refinement(self, t: ti.f64):
        displacement_y, _ = self._platform_motion(t)
        raw_steps = displacement_y / ti.cast(self.base_dx, ti.f64)
        snapped_steps = ti.floor(raw_steps + 0.5)
        if raw_steps < 0.0:
            snapped_steps = -ti.floor(-raw_steps + 0.5)
        shift = ti.Vector([0.0, snapped_steps * ti.cast(self.base_dx, ti.f64)])
        shift[1] = ti.max(self.dynamic_shift_min[None][1], ti.min(self.dynamic_shift_max[None][1], shift[1]))
        self.refinement_shift[None] = shift
        for level in ti.static(range(self.num_levels)):
            if ti.static(level == 0):
                self.region_min[level] = self.reference_region_min[level]
                self.region_max[level] = self.reference_region_max[level]
                self.origin[level] = self.reference_origin[level]
            else:
                self.region_min[level] = self.reference_region_min[level] + shift
                self.region_max[level] = self.reference_region_max[level] + shift
                self.origin[level] = self.reference_origin[level] + shift

    def update_dynamic_refinement(self, t):
        if self.dynamic_refinement:
            self._update_dynamic_refinement(t)

    @ti.func
    def _line_weight(self, distance, dx):
        q = ti.abs(distance) / dx
        weight = ti.cast(0.0, ti.f64)
        if q < 0.5:
            weight = 0.75 - q**2
        elif q < 1.5:
            weight = 0.5 * (1.5 - q)**2
        return weight

    @ti.func
    def _add_horizontal_penalty(self, level: ti.template(), I, x_min, x_max, y, velocity):
        dx = self.level_dx[level]
        x = self.node_position(level, I)
        if x[0] >= x_min and x[0] <= x_max:
            contribution = config.AMR_BOUNDARY_PENALTY_NORMAL * config.RHO_0 * dx**2 * self._line_weight(x[1] - y, dx)
            self.boundary_mass[level][I][1] += contribution
            self.boundary_momentum[level][I][1] += contribution * velocity[1]

    @ti.func
    def _add_vertical_penalty(self, level: ti.template(), I, x, y_min, y_max, velocity):
        dx = self.level_dx[level]
        position = self.node_position(level, I)
        if position[1] >= y_min and position[1] <= y_max:
            contribution = config.AMR_BOUNDARY_PENALTY_NORMAL * config.RHO_0 * dx**2 * self._line_weight(position[0] - x, dx)
            self.boundary_mass[level][I][0] += contribution
            self.boundary_momentum[level][I][0] += contribution * velocity[0]

    @ti.func
    def _apply_rectangular_penalty(self, level: ti.template(), I, x_min, x_max, y_min, y_max, velocity):
        self._add_horizontal_penalty(level, I, x_min, x_max, y_min, velocity)
        self._add_horizontal_penalty(level, I, x_min, x_max, y_max, velocity)
        self._add_vertical_penalty(level, I, x_min, y_min, y_max, velocity)
        self._add_vertical_penalty(level, I, x_max, y_min, y_max, velocity)

    @ti.kernel
    def initialize_dynamic_penalty_mass(self):
        zero_velocity = ti.Vector([0.0, 0.0])
        for level in ti.static(range(self.num_levels)):
            minimum = self.region_min[level]
            maximum = self.region_max[level]
            tolerance = 0.5 * self.level_dx[level]
            for I in ti.grouped(self.m[level]):
                if minimum[1] <= self.domain_min[1] + tolerance:
                    self._apply_rectangular_penalty(
                        level, I, minimum[0], maximum[0], self.domain_min[1], self.domain_min[1], zero_velocity,
                    )
                if maximum[1] >= self.domain_max[1] - tolerance:
                    self._apply_rectangular_penalty(
                        level, I, minimum[0], maximum[0], self.domain_max[1], self.domain_max[1], zero_velocity,
                    )
                if minimum[0] <= self.domain_min[0] + tolerance:
                    self._apply_rectangular_penalty(
                        level, I, self.domain_min[0], self.domain_min[0], minimum[1], maximum[1], zero_velocity,
                    )
                if maximum[0] >= self.domain_max[0] - tolerance:
                    self._apply_rectangular_penalty(
                        level, I, self.domain_max[0], self.domain_max[0], minimum[1], maximum[1], zero_velocity,
                    )

    @ti.kernel
    def add_moving_platform_penalty_mass_gpu(self, t: ti.f64):
        displacement_y, velocity_y = self._platform_motion(t)
        x_min = ti.cast(config.INT_MOVINGRECT_XMIN, ti.f64)
        x_max = ti.cast(config.INT_MOVINGRECT_XMAX, ti.f64)
        y_min = ti.cast(config.INT_MOVINGRECT_YMIN, ti.f64) + displacement_y
        y_max = ti.cast(config.INT_MOVINGRECT_YMAX, ti.f64) + displacement_y
        velocity = ti.Vector([0.0, velocity_y])
        for level in ti.static(range(self.num_levels)):
            for I in ti.grouped(self.m[level]):
                self._apply_rectangular_penalty(level, I, x_min, x_max, y_min, y_max, velocity)

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
