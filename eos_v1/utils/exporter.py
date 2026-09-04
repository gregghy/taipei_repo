# utils/export.py
import physics.boundary as bnd
import os
import numpy as np
import config

def write_vtk(frame_number, pos, pressure, velocity, output_dir="output", material=None,
              point_scalars=None, point_vectors=None): # use polyvertex
    """
    Exports particle positions, pressure, and velocity to a VTK file.
    Uses POLYDATA / VERTICES format with chunked lines to stay within
    ParaView's ASCII reader buffer limit.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"mpm_fluid_{frame_number:06d}.vtk")
    num_particles = len(pos)

    # Sanitize NaN/inf to 0 so ParaView can load the file without errors.
    pos = np.where(np.isfinite(pos), pos, 0.0).astype(np.float64)
    pressure = np.where(np.isfinite(pressure), pressure, 0.0).astype(np.float64)
    velocity = np.where(np.isfinite(velocity), velocity, 0.0).astype(np.float64)
    if material is not None:
        material = np.asarray(material, dtype=np.int32)
        if len(material) != num_particles:
            raise ValueError("material must have one value per particle")

    # Reserved names already written as base attributes — skip duplicates.
    _reserved_scalars = {"Pressure", "Material"}
    _reserved_vectors = {"Velocity"}
    point_scalars = {} if point_scalars is None else {
        str(name): np.asarray(values, dtype=np.float64)
        for name, values in point_scalars.items() if str(name) not in _reserved_scalars
    }
    point_vectors = {} if point_vectors is None else {
        str(name): np.asarray(values, dtype=np.float64)
        for name, values in point_vectors.items() if str(name) not in _reserved_vectors
    }
    for name, values in point_scalars.items():
        if values.shape != (num_particles,):
            raise ValueError(f"point scalar {name} must have shape ({num_particles},)")
        point_scalars[name] = np.where(np.isfinite(values), values, 0.0)
    for name, values in point_vectors.items():
        if values.ndim != 2 or values.shape[0] != num_particles or values.shape[1] not in (2, 3):
            raise ValueError(f"point vector {name} must have shape ({num_particles}, 2) or ({num_particles}, 3)")
        point_vectors[name] = np.where(np.isfinite(values), values, 0.0)

    is_3d = config.DIM == 3

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM Simulation Data\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # --- POINTS ---
        f.write(f"POINTS {num_particles} float\n")
        if is_3d:
            pts = pos
        else:
            pts = np.column_stack([pos[:, 0], pos[:, 1], np.zeros(num_particles)])
        np.savetxt(f, pts, fmt="%.17g")

        # --- VERTICES (chunked to avoid reader buffer overflow) ---
        f.write(f"\nVERTICES 1 {num_particles + 1}\n")
        f.write(f"{num_particles}\n")
        ids = np.arange(num_particles)
        chunk = 12  # 12 ints per line keeps lines well under 4096 chars
        for start in range(0, num_particles, chunk):
            row = ids[start:start + chunk]
            f.write(" ".join(str(v) for v in row) + "\n")

        # --- POINT_DATA ---
        f.write(f"\nPOINT_DATA {num_particles}\n")

        # Pressure
        f.write("SCALARS Pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, pressure, fmt="%.17g")

        # Velocity
        f.write("VECTORS Velocity float\n")
        if is_3d:
            np.savetxt(f, velocity, fmt="%.17g %.17g %.17g")
        else:
            np.savetxt(f, velocity, fmt="%.17g %.17g 0.0")

        # Material
        if material is not None:
            f.write("SCALARS Material int 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, material.reshape(-1, 1), fmt="%d")

        # Extra point scalars
        for name, values in point_scalars.items():
            f.write(f"SCALARS {name} double 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, values, fmt="%.17g")

        # Extra point vectors
        for name, values in point_vectors.items():
            f.write(f"VECTORS {name} double\n")
            if values.shape[1] == 3:
                np.savetxt(f, values, fmt="%.17g %.17g %.17g")
            else:
                np.savetxt(f, values, fmt="%.17g %.17g 0.0")

    print(f"Exported frame {frame_number} to {filename}")

def write_boundary_vtk(min_x, min_y, max_x, max_y, output_dir="output"):
    """
    Exports a single, static box representing the true physical boundary walls.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.join(output_dir, "domain_boundary.vtk")
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM Domain Boundary\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Use the passed coordinates for the corners
        f.write("POINTS 4 float\n")
        f.write(f"{min_x} {min_y} 0.0\n")         # Bottom-Left
        f.write(f"{max_x} {min_y} 0.0\n")         # Bottom-Right
        f.write(f"{max_x} {max_y} 0.0\n")         # Top-Right
        f.write(f"{min_x} {max_y} 0.0\n")         # Top-Left
        
        f.write("\nLINES 1 6\n")
        f.write("5 0 1 2 3 0\n")
        
    print(f"Exported static domain boundary to {filename}")

def write_quadtree_grid_vtk(grid, output_dir="output"):
    """Exports the quadtree refinement regions as one wireframe VTK file.

    Each AMR level is drawn as a rectangle outlining its current refinement
    region.  For dynamic refinement, call this after each grid update to
    capture the moving patch.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, "quadtree_grid.vtk")
    regions = []
    for level in range(grid.num_levels):
        mn = grid.region_min_np[level]
        mx = grid.region_max_np[level]
        regions.append((level, float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])))
    point_count = 4 * len(regions)
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM Quadtree Refinement Regions\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {point_count} float\n")
        for _, xmin, ymin, xmax, ymax in regions:
            f.write(f"{xmin} {ymin} 0.0\n")
            f.write(f"{xmax} {ymin} 0.0\n")
            f.write(f"{xmax} {ymax} 0.0\n")
            f.write(f"{xmin} {ymax} 0.0\n")
        f.write(f"\nLINES {len(regions)} {5 * len(regions)}\n")
        for i in range(len(regions)):
            base = 4 * i
            f.write(f"5 {base} {base + 1} {base + 2} {base + 3} {base}\n")
        f.write(f"\nCELL_DATA {len(regions)}\n")
        f.write("SCALARS Level int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for level, *_ in regions:
            f.write(f"{level}\n")
    print(f"Exported quadtree grid to {filename}")

def write_mpm_grid_vtk(grid, output_dir="output"):
    """Exports the composite leaf-cell MPM background grid as wireframe."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, "mpm_background_grid.vtk")
    cell_count = int(grid.leaf_count)
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM Composite Background Grid\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {4 * cell_count} double\n")
        for origin, size in zip(grid.leaf_origin, grid.leaf_size):
            x0, y0 = float(origin[0]), float(origin[1])
            x1, y1 = x0 + float(size), y0 + float(size)
            f.write(f"{x0:.17g} {y0:.17g} 0.0\n")
            f.write(f"{x1:.17g} {y0:.17g} 0.0\n")
            f.write(f"{x1:.17g} {y1:.17g} 0.0\n")
            f.write(f"{x0:.17g} {y1:.17g} 0.0\n")
        f.write(f"\nLINES {cell_count} {6 * cell_count}\n")
        for i in range(cell_count):
            base = 4 * i
            f.write(f"5 {base} {base + 1} {base + 2} {base + 3} {base}\n")
        f.write(f"\nCELL_DATA {cell_count}\n")
        f.write("SCALARS RefinementLevel int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for level in grid.leaf_level:
            f.write(f"{int(level)}\n")
        f.write("SCALARS CellSize double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for size in grid.leaf_size:
            f.write(f"{float(size):.17g}\n")
    print(f"Exported MPM background grid to {filename}")

def _current_leaf_cells(grid):
    levels = []
    origins = []
    sizes = []
    for level in range(grid.num_levels):
        dx = float(grid.dx[level])
        region_min = grid.region_min_np[level]
        region_max = grid.region_max_np[level]
        nx = int(round((region_max[0] - region_min[0]) / dx))
        ny = int(round((region_max[1] - region_min[1]) / dx))
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        ox = region_min[0] + ii * dx
        oy = region_min[1] + jj * dx
        keep = np.ones((nx, ny), dtype=bool)
        if level < grid.max_level:
            cx = ox + 0.5 * dx
            cy = oy + 0.5 * dx
            next_min = grid.region_min_np[level + 1]
            next_max = grid.region_max_np[level + 1]
            keep = ~((cx >= next_min[0]) & (cx < next_max[0]) & (cy >= next_min[1]) & (cy < next_max[1]))
        levels.append(np.full(int(keep.sum()), level, dtype=np.int32))
        origins.append(np.stack([ox[keep], oy[keep]], axis=1))
        sizes.append(np.full(int(keep.sum()), dx, dtype=np.float64))
    levels = np.concatenate(levels)
    origins = np.concatenate(origins)
    sizes = np.concatenate(sizes)
    area = float(np.sum(sizes**2))
    domain_area = float(grid.domain_width * grid.domain_height)
    if not np.isclose(area, domain_area, rtol=1e-12, atol=1e-14):
        raise RuntimeError(f"dynamic leaf cells do not tile the domain: {area} vs {domain_area}")
    return levels, origins, sizes

def write_dynamic_mpm_grid_vtk(grid, frame, output_dir="output"):
    """Exports the current dynamic composite leaf grid as a VTK frame."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    levels, origins, sizes = _current_leaf_cells(grid)
    cell_count = len(levels)
    filename = os.path.join(output_dir, f"mpm_background_grid_{frame:06d}.vtk")
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM Dynamic Composite Background Grid\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {4 * cell_count} double\n")
        for origin, size in zip(origins, sizes):
            x0, y0 = float(origin[0]), float(origin[1])
            x1, y1 = x0 + float(size), y0 + float(size)
            f.write(f"{x0:.17g} {y0:.17g} 0.0\n")
            f.write(f"{x1:.17g} {y0:.17g} 0.0\n")
            f.write(f"{x1:.17g} {y1:.17g} 0.0\n")
            f.write(f"{x0:.17g} {y1:.17g} 0.0\n")
        f.write(f"\nLINES {cell_count} {6 * cell_count}\n")
        for i in range(cell_count):
            base = 4 * i
            f.write(f"5 {base} {base + 1} {base + 2} {base + 3} {base}\n")
        f.write(f"\nCELL_DATA {cell_count}\n")
        f.write("SCALARS RefinementLevel int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for level in levels:
            f.write(f"{int(level)}\n")
        f.write("SCALARS CellSize double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for size in sizes:
            f.write(f"{float(size):.17g}\n")
    print(f"Exported dynamic MPM grid frame {frame} to {filename}")

def write_dynamic_quadtree_grid_vtk(grid, frame, output_dir="output"):
    """Exports the current dynamic refinement-region outlines as a VTK frame."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"quadtree_grid_{frame:06d}.vtk")
    regions = [(level, grid.region_min_np[level], grid.region_max_np[level]) for level in range(grid.num_levels)]
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM Dynamic Refinement Regions\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {4 * len(regions)} double\n")
        for _, minimum, maximum in regions:
            f.write(f"{minimum[0]:.17g} {minimum[1]:.17g} 0.0\n")
            f.write(f"{maximum[0]:.17g} {minimum[1]:.17g} 0.0\n")
            f.write(f"{maximum[0]:.17g} {maximum[1]:.17g} 0.0\n")
            f.write(f"{minimum[0]:.17g} {maximum[1]:.17g} 0.0\n")
        f.write(f"\nLINES {len(regions)} {6 * len(regions)}\n")
        for i in range(len(regions)):
            base = 4 * i
            f.write(f"5 {base} {base + 1} {base + 2} {base + 3} {base}\n")
        f.write(f"\nCELL_DATA {len(regions)}\n")
        f.write("SCALARS Level int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for level, _, _ in regions:
            f.write(f"{level}\n")
    print(f"Exported dynamic quadtree frame {frame} to {filename}")

def write_mpm_grid_level_vtk(grid, level, output_dir="output"):
    """Exports a single MPM grid level as a wireframe VTK file.

    This is useful in gradient mode where every level covers the entire
    domain: the composite leaf grid only shows the finest level, so the
    coarser levels are invisible. Per-level files let you overlay the
    grid that matches each particle's ParticleLevel in ParaView.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if level < 0 or level >= grid.num_levels:
        raise ValueError(f"level {level} out of range [0, {grid.num_levels})")
    dx = float(grid.dx[level])
    region_min = grid.region_min_np[level]
    region_max = grid.region_max_np[level]
    nx = int(round((region_max[0] - region_min[0]) / dx))
    ny = int(round((region_max[1] - region_min[1]) / dx))
    cell_count = nx * ny
    filename = os.path.join(output_dir, f"mpm_grid_level_{level}.vtk")
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"MPM Background Grid Level {level}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {4 * cell_count} double\n")
        for j in range(ny):
            for i in range(nx):
                x0 = region_min[0] + i * dx
                y0 = region_min[1] + j * dx
                x1 = x0 + dx
                y1 = y0 + dx
                f.write(f"{x0:.17g} {y0:.17g} 0.0\n")
                f.write(f"{x1:.17g} {y0:.17g} 0.0\n")
                f.write(f"{x1:.17g} {y1:.17g} 0.0\n")
                f.write(f"{x0:.17g} {y1:.17g} 0.0\n")
        f.write(f"\nLINES {cell_count} {6 * cell_count}\n")
        for k in range(cell_count):
            base = 4 * k
            f.write(f"5 {base} {base + 1} {base + 2} {base + 3} {base}\n")
        f.write(f"\nCELL_DATA {cell_count}\n")
        f.write("SCALARS RefinementLevel int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for _ in range(cell_count):
            f.write(f"{level}\n")
        f.write("SCALARS CellSize double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for _ in range(cell_count):
            f.write(f"{dx:.17g}\n")
    print(f"Exported MPM grid level {level} to {filename}")

def write_mpm_grid_levels_vtk(grid, output_dir="output"):
    """Exports every MPM grid level as a separate wireframe VTK file."""
    for level in range(grid.num_levels):
        write_mpm_grid_level_vtk(grid, level, output_dir=output_dir)

def write_boundary_vtk_3d(min_x, min_y, min_z, max_x, max_y, max_z, output_dir="output"):
    """Exports a 3D wireframe box representing the domain bucket."""
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, "domain_boundary.vtk")
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\nMPM 3D Domain Boundary\nASCII\nDATASET POLYDATA\n")
        f.write("POINTS 8 float\n")
        f.write(f"{min_x} {min_y} {min_z}\n") # 0
        f.write(f"{max_x} {min_y} {min_z}\n") # 1
        f.write(f"{max_x} {max_y} {min_z}\n") # 2
        f.write(f"{min_x} {max_y} {min_z}\n") # 3
        f.write(f"{min_x} {min_y} {max_z}\n") # 4
        f.write(f"{max_x} {min_y} {max_z}\n") # 5
        f.write(f"{max_x} {max_y} {max_z}\n") # 6
        f.write(f"{min_x} {max_y} {max_z}\n") # 7
        
        # 12 Lines mapping the wireframe edges
        f.write("\nLINES 12 36\n")
        f.write("2 0 1\n2 1 2\n2 2 3\n2 3 0\n") # Bottom face edges
        f.write("2 4 5\n2 5 6\n2 6 7\n2 7 4\n") # Top face edges
        f.write("2 0 4\n2 1 5\n2 2 6\n2 3 7\n") # Vertical pillars

def write_interior_square_vtk(min_x, min_y, max_x, max_y, output_dir="output"):
    """
    Exports the interior square obstacle as a VTK file so it is visible in ParaView.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.join(output_dir, "interior_square.vtk")
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM Interior Obstacle\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        f.write("POINTS 4 float\n")
        f.write(f"{min_x} {min_y} 0.0\n")         # Bottom-Left
        f.write(f"{max_x} {min_y} 0.0\n")         # Bottom-Right
        f.write(f"{max_x} {max_y} 0.0\n")         # Top-Right
        f.write(f"{min_x} {max_y} 0.0\n")         # Top-Left
        
        # Connect the 4 points to draw the square
        f.write("\nPOLYGONS 1 5\n")
        f.write("4 0 1 2 3\n")
        
    print(f"Exported interior obstacle boundary to {filename}")

def WriteInteriorMoving(frame_number, min_x, min_y, max_x, max_y, output_dir="output"):
    """
    Exports the interior square obstacle as a time-series VTK file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Append the frame number so ParaView reads it as a sequence
    filename = os.path.join(output_dir, f"interior_square_{frame_number:04d}.vtk")
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM Interior Obstacle\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        f.write("POINTS 4 float\n")
        f.write(f"{min_x} {min_y} 0.0\n")         # Bottom-Left
        f.write(f"{max_x} {min_y} 0.0\n")         # Bottom-Right
        f.write(f"{max_x} {max_y} 0.0\n")         # Top-Right
        f.write(f"{min_x} {max_y} 0.0\n")         # Top-Left
        
        # Connect the 4 points to draw the square
        f.write("\nPOLYGONS 1 5\n")
        f.write("4 0 1 2 3\n")

def WriteInteriorMoving_3d(frame_number, min_x, min_y, min_z, max_x, max_y, max_z, output_dir="output"):
    """Exports the 3D solid box representing the moving immersed platform."""
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"interior_box_{frame_number:04d}.vtk")
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\nMPM Interior Box 3D\nASCII\nDATASET POLYDATA\n")
        f.write("POINTS 8 float\n")
        f.write(f"{min_x} {min_y} {min_z}\n") # 0
        f.write(f"{max_x} {min_y} {min_z}\n") # 1
        f.write(f"{max_x} {max_y} {min_z}\n") # 2
        f.write(f"{min_x} {max_y} {min_z}\n") # 3
        f.write(f"{min_x} {min_y} {max_z}\n") # 4
        f.write(f"{max_x} {min_y} {max_z}\n") # 5
        f.write(f"{max_x} {max_y} {max_z}\n") # 6
        f.write(f"{min_x} {max_y} {max_z}\n") # 7
        
        # 6 faces connecting the 8 corners
        f.write("\nPOLYGONS 6 30\n")
        f.write("4 0 1 2 3\n") # Bottom
        f.write("4 4 5 6 7\n") # Top
        f.write("4 0 1 5 4\n") # Front
        f.write("4 3 2 6 7\n") # Back
        f.write("4 0 3 7 4\n") # Left
        f.write("4 1 2 6 5\n") # Right

def write_ntu_wireframe_vtk(output_dir="output"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, "ntu_boundary_wireframe.vtk")
    
    OX = 0.375
    OY = 0.8
    tip_offset = 0.02 # Match this to your boundary.py value

    # Construct the points with the split tips
    # Lower Wall (Indices 0-17)
    # P4 split into: 4 (left), 5 (right)
    lower_wall = [
        (0.10, OY+0.40), (OX+0.00, OY+0.40), (OX+0.00, OY+0.00), (OX+0.13, OY+0.00),
        (OX+0.13 - tip_offset, OY+0.27), (OX+0.13 + tip_offset, OY+0.27), # p4_left, p4_right
        (OX+0.33, OY+0.00), (OX+0.46, OY+0.00), (OX+0.46, OY+0.35),
        (OX+0.61, OY+0.35), (OX+0.61, OY+0.00), (OX+0.76, OY+0.00), (OX+0.76, OY+0.35),
        (OX+0.91, OY+0.35), (OX+0.91, OY+0.00), (OX+1.25, OY+0.00), (OX+1.25, OY+0.40),
        (OX+1.45, OY+0.40)
    ]
    
    # Upper Wall (Indices 18-27)
    # T2 split into: 20 (left), 21 (right)
    upper_wall = [
        (0.30, OY+0.55), (OX+0.13, OY+0.55), 
        (OX+0.33 - tip_offset, OY+0.30), (OX+0.33 + tip_offset, OY+0.30), # t2_left, t2_right
        (OX+0.33, OY+0.50), (OX+1.03, OY+0.50), (OX+1.03, OY+0.10), 
        (OX+1.13, OY+0.10), (OX+1.13, OY+0.55), (OX+1.45, OY+0.55)
    ]
    
    points = lower_wall + upper_wall

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\nNTU Microfluidic Pipe\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for pt in points:
            f.write(f"{pt[0]:.5f} {pt[1]:.5f} 0.0\n")
            
        # VTK Lines format: "N p0 p1 ... pN-1"
        # Line 1: 18 points (Indices 0 to 17)
        # Line 2: 10 points (Indices 18 to 27)
        # Total integers in LINES section: (1 + 18) + (1 + 10) = 30
        f.write("\nLINES 2 30\n")
        
        # Lower Wall line string
        lower_idx = " ".join([str(i) for i in range(18)])
        f.write(f"18 {lower_idx}\n")
        
        # Upper Wall line string
        upper_idx = " ".join([str(i) for i in range(18, 28)])
        f.write(f"10 {upper_idx}\n")
        
def write_normals_vtk(output_dir, grid_normals_np):
    """Writes the grid normals to a VTK file for vector visualization."""
    filepath = os.path.join(output_dir, "normals.vtk")
    res_x, res_y = grid_normals_np.shape[0], grid_normals_np.shape[1]
    
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("SDF Normal Visualization\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {res_x} {res_y} 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {config.DX} {config.DY} 1\n")
        f.write(f"POINT_DATA {res_x * res_y}\n")
        f.write("VECTORS Normals float\n")
        
        # Flatten the array in column-major order to match Taichi's memory layout
        for j in range(res_y):
            for i in range(res_x):
                nx, ny = grid_normals_np[i, j]
                f.write(f"{nx} {ny} 0.0\n")