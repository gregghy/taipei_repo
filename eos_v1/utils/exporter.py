# utils/export.py
import physics.boundary as bnd
import os
import numpy as np
import config

# def write_vtk(frame_number, pos, pressure, velocity, output_dir="output"):
#     """
#     Exports particle positions, pressure, and velocity to a VTK file.
#     Automatically scales between 2D and 3D based on config.DIM.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     filename = os.path.join(output_dir, f"mpm_fluid_{frame_number:04d}.vtk")
#     num_particles = len(pos)
    
#     with open(filename, 'w') as f:
#         f.write("# vtk DataFile Version 3.0\n")
#         f.write("MPM Simulation Data\n")
#         f.write("ASCII\n")
#         f.write("DATASET UNSTRUCTURED_GRID\n")
        
#         # 1. POSITIONS
#         f.write(f"POINTS {num_particles} float\n")
#         if config.DIM == 3:
#             for i in range(num_particles):
#                 f.write(f"{pos[i, 0]} {pos[i, 1]} {pos[i, 2]}\n")
#         else:
#             for i in range(num_particles):
#                 f.write(f"{pos[i, 0]} {pos[i, 1]} 0.0\n")
            
#         f.write(f"\nCELLS {num_particles} {num_particles * 2}\n")
#         for i in range(num_particles): f.write(f"1 {i}\n")
            
#         f.write(f"\nCELL_TYPES {num_particles}\n")
#         for i in range(num_particles): f.write("1\n")
            
#         # 2. PRESSURE (SCALAR)
#         f.write(f"\nPOINT_DATA {num_particles}\n")
#         f.write("SCALARS Pressure float 1\n")
#         f.write("LOOKUP_TABLE default\n")
#         for i in range(num_particles):
#             f.write(f"{pressure[i]}\n")
        
#         # 3. VELOCITY (VECTOR)
#         f.write("VECTORS Velocity float\n")
#         if config.DIM == 3:
#             for i in range(num_particles):
#                 f.write(f"{velocity[i, 0]} {velocity[i, 1]} {velocity[i, 2]}\n")
#         else:
#             for i in range(num_particles):
#                 f.write(f"{velocity[i, 0]} {velocity[i, 1]} 0.0\n")

#     print(f"Exported frame {frame_number} to {filename}")

def write_vtk(frame_number, pos, pressure, velocity, output_dir="output"): # use polyvertex
    """
    Exports particle positions, pressure, and velocity to a VTK file.
    Uses highly optimized POLYDATA / VERTICES for massive 3D point clouds.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.join(output_dir, f"mpm_fluid_{frame_number:04d}.vtk")
    num_particles = len(pos)
    
    # Sanitize NaN/inf to 0 so ParaView can load the file without errors.
    pos = np.where(np.isfinite(pos), pos, 0.0)
    pressure = np.where(np.isfinite(pressure), pressure, 0.0)
    velocity = np.where(np.isfinite(velocity), velocity, 0.0)
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MPM Simulation Data\n")
        f.write("ASCII\n")
        # Change 1: Use POLYDATA instead of UNSTRUCTURED_GRID
        f.write("DATASET POLYDATA\n")
        
        # 1. POSITIONS (Stays the same)
        f.write(f"POINTS {num_particles} float\n")
        if config.DIM == 3:
            for i in range(num_particles):
                f.write(f"{pos[i, 0]} {pos[i, 1]} {pos[i, 2]}\n")
        else:
            for i in range(num_particles):
                f.write(f"{pos[i, 0]} {pos[i, 1]} 0.0\n")
            
        # Change 2: The Polyvertex / Vertices block
        # Format: VERTICES <number_of_cells> <total_number_of_integers_to_read>
        # We have 1 cell, containing `num_particles` vertices. 
        # So it needs to read (1 for the count + num_particles for the IDs) integers.
        f.write(f"\nVERTICES 1 {num_particles + 1}\n")
        
        # Write the number of particles, followed by all their IDs (0 to N-1)
        f.write(f"{num_particles} ")
        # Using a generator to write them space-separated efficiently
        id_string = " ".join(str(i) for i in range(num_particles))
        f.write(id_string + "\n")
            
        # 2. PRESSURE (SCALAR) (Stays the same)
        f.write(f"\nPOINT_DATA {num_particles}\n")
        f.write("SCALARS Pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(num_particles):
            f.write(f"{pressure[i]}\n")
        
        # 3. VELOCITY (VECTOR) (Stays the same)
        f.write("VECTORS Velocity float\n")
        if config.DIM == 3:
            for i in range(num_particles):
                f.write(f"{velocity[i, 0]} {velocity[i, 1]} {velocity[i, 2]}\n")
        else:
            for i in range(num_particles):
                f.write(f"{velocity[i, 0]} {velocity[i, 1]} 0.0\n")

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