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
        for name, extent in (('width', self.domain_width), ('height', self.domain_height)):
            n_cells = extent / self.base_dx
            if abs(n_cells - round(n_cells)) > 1e-6:
                raise ValueError(f"Domain {name} {extent} must be an integer multiple of AMR_BASE_DX {self.base_dx}")

        if refinement_box is None:
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

        self.region_min = ti.Vector.field(2, dtype=ti.f32, shape=self.num_levels)
        self.region_max = ti.Vector.field(2, dtype=ti.f32, shape=self.num_levels)
        self.origin = ti.Vector.field(2, dtype=ti.f32, shape=self.num_levels)
        self.level_dx = ti.field(dtype=ti.f32, shape=self.num_levels)
        self.level_inv_dx = ti.field(dtype=ti.f32, shape=self.num_levels)
        self.region_min.from_numpy(self.region_min_np.astype(np.float32))
        self.region_max.from_numpy(self.region_max_np.astype(np.float32))
        self.origin.from_numpy(self.origin_np.astype(np.float32))
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
        self._validate_leaf_tiling()

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
            np.concatenate(origins).astype(np.float32),
            np.concatenate(sizes).astype(np.float32),
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
                if self.m[level][I] > self.node_mass_cutoff[level]:
                    self.v[level][I] /= self.m[level][I]
                else:
                    self.v[level][I] = ti.Vector.zero(ti.f32, 2)

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
                if ti.static(self.face_interior[level][0]):
                    if x[0] < self.region_min[level][0] + band:
                        ghost = True
                if ti.static(self.face_interior[level][1]):
                    if x[0] > self.region_max[level][0] - band:
                        ghost = True
                if ti.static(self.face_interior[level][2]):
                    if x[1] < self.region_min[level][1] + band:
                        ghost = True
                if ti.static(self.face_interior[level][3]):
                    if x[1] > self.region_max[level][1] - band:
                        ghost = True
                if ghost:
                    self.v[level][I] = self.sample_velocity(level - 1, x)
