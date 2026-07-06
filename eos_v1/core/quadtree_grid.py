import math
import sys
import os
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
        self.domain_width = float(getattr(config, 'AMR_DOMAIN_WIDTH', config.GRID_WIDTH))
        self.domain_height = float(getattr(config, 'AMR_DOMAIN_HEIGHT', config.GRID_HEIGHT))
        self.domain_min = np.array([
            float(getattr(config, 'AMR_DOMAIN_MIN_X', 0.0)),
            float(getattr(config, 'AMR_DOMAIN_MIN_Y', 0.0)),
        ], dtype=np.float32)
        self.domain_max = self.domain_min + np.array([self.domain_width, self.domain_height], dtype=np.float32)
        self.base_cells_x = int(getattr(config, 'AMR_BASE_CELLS_X', config.N_CELL_WIDTH))
        self.base_cells_y = int(getattr(config, 'AMR_BASE_CELLS_Y', config.N_CELL_HEIGHT))
        self.base_dx = float(getattr(config, 'AMR_BASE_DX', self.domain_width / self.base_cells_x))
        self.refine_buffer_cells = int(getattr(config, 'AMR_REFINEMENT_BUFFER_CELLS', 4))

        if refinement_box is None:
            refinement_box = self._default_refinement_box()
        fine_min, fine_max = refinement_box
        self.fine_region_min = np.array(fine_min, dtype=np.float32)
        self.fine_region_max = np.array(fine_max, dtype=np.float32)

        self.dx = []
        self.inv_dx = []
        self.region_min_np = np.zeros((self.num_levels, 2), dtype=np.float32)
        self.region_max_np = np.zeros((self.num_levels, 2), dtype=np.float32)
        self.origin_np = np.zeros((self.num_levels, 2), dtype=np.float32)
        self.res = []
        self.res_x = []
        self.res_y = []

        self._build_level_geometry()

        self.region_min = ti.Vector.field(2, dtype=ti.f32, shape=self.num_levels)
        self.region_max = ti.Vector.field(2, dtype=ti.f32, shape=self.num_levels)
        self.origin = ti.Vector.field(2, dtype=ti.f32, shape=self.num_levels)
        self.level_dx = ti.field(dtype=ti.f32, shape=self.num_levels)
        self.level_inv_dx = ti.field(dtype=ti.f32, shape=self.num_levels)
        self.region_min.from_numpy(self.region_min_np)
        self.region_max.from_numpy(self.region_max_np)
        self.origin.from_numpy(self.origin_np)
        self.level_dx.from_numpy(np.array(self.dx, dtype=np.float32))
        self.level_inv_dx.from_numpy(np.array(self.inv_dx, dtype=np.float32))

        self.m = []
        self.v = []
        self.f = []
        self.v_old = []
        for level in range(self.num_levels):
            shape = self.res[level]
            self.m.append(ti.field(dtype=ti.f32, shape=shape))
            self.v.append(ti.Vector.field(2, dtype=ti.f32, shape=shape))
            self.f.append(ti.Vector.field(2, dtype=ti.f32, shape=shape))
            self.v_old.append(ti.Vector.field(2, dtype=ti.f32, shape=shape))

        self.leaf_level, self.leaf_origin, self.leaf_size = self._build_leaf_cells()
        self.leaf_count = len(self.leaf_level)

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
            dx = self.base_dx / (2 ** level)
            self.dx.append(dx)
            self.inv_dx.append(1.0 / dx)
            if level == 0:
                region_min = self.domain_min.copy()
                region_max = self.domain_max.copy()
            else:
                grow = (self.max_level - level) * self.refine_buffer_cells * self.dx[level - 1]
                region_min = self.fine_region_min - grow
                region_max = self.fine_region_max + grow
                region_min = np.maximum(region_min, self.domain_min)
                region_max = np.minimum(region_max, self.domain_max)
            self.region_min_np[level] = region_min
            self.region_max_np[level] = region_max
            origin = region_min - self.padding * dx
            self.origin_np[level] = origin
            nx = int(math.ceil((region_max[0] - region_min[0]) / dx)) + 2 * self.padding + 1
            ny = int(math.ceil((region_max[1] - region_min[1]) / dx)) + 2 * self.padding + 1
            self.res.append((nx, ny))
            self.res_x.append(nx)
            self.res_y.append(ny)

    def _inside_region(self, level, point):
        return np.all(point >= self.region_min_np[level]) and np.all(point < self.region_max_np[level])

    def _build_leaf_cells(self):
        levels = []
        origins = []
        sizes = []
        for level in range(self.num_levels):
            dx = self.dx[level]
            region_min = self.region_min_np[level]
            region_max = self.region_max_np[level]
            nx = int(math.ceil((region_max[0] - region_min[0]) / dx))
            ny = int(math.ceil((region_max[1] - region_min[1]) / dx))
            for i in range(nx):
                for j in range(ny):
                    origin = region_min + np.array([i * dx, j * dx], dtype=np.float32)
                    center = origin + 0.5 * dx
                    if center[0] >= self.domain_max[0] or center[1] >= self.domain_max[1]:
                        continue
                    covered_by_finer = False
                    if level < self.max_level:
                        covered_by_finer = self._inside_region(level + 1, center)
                    if not covered_by_finer:
                        levels.append(level)
                        origins.append(origin)
                        sizes.append(dx)
        return np.array(levels, dtype=np.int32), np.array(origins, dtype=np.float32), np.array(sizes, dtype=np.float32)

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
        return self.origin[level] + ti.cast(I, ti.f32) * self.level_dx[level]

    @ti.func
    def in_bounds(self, level: ti.template(), I):
        return I[0] >= 0 and I[0] < ti.static(self.res_x[level]) and I[1] >= 0 and I[1] < ti.static(self.res_y[level])

    @ti.kernel
    def clear(self):
        for level in ti.static(range(self.num_levels)):
            for I in ti.grouped(self.m[level]):
                self.m[level][I] = 0.0
                self.v[level][I] = ti.Vector.zero(ti.f32, 2)
                self.f[level][I] = ti.Vector.zero(ti.f32, 2)
                self.v_old[level][I] = ti.Vector.zero(ti.f32, 2)

    @ti.kernel
    def normalize_momentum(self):
        for level in ti.static(range(self.num_levels)):
            for I in ti.grouped(self.m[level]):
                if self.m[level][I] > 0.0:
                    self.v[level][I] /= self.m[level][I]

    @ti.func
    def sample_velocity(self, level: ti.template(), x):
        inv_dx = self.level_inv_dx[level]
        fx = (x - self.origin[level]) * inv_dx
        base = ti.cast(fx - 0.5, ti.i32)
        d = fx - ti.cast(base, ti.f32)
        w0 = 0.5 * (1.5 - d)**2
        w1 = 0.75 - (d - 1.0)**2
        w2 = 0.5 * (d - 0.5)**2
        v_out = ti.Vector.zero(ti.f32, 2)
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
        for level in ti.static(range(1, self.num_levels)):
            for I in ti.grouped(self.m[level]):
                if self.m[level][I] <= 0.0:
                    x = self.node_position(level, I)
                    self.v[level][I] = self.sample_velocity(level - 1, x)
