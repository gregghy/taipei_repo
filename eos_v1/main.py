# main.py
import os
import taichi as ti
import numpy as np
import config
import utils.visualization as ut
import physics.boundary as bnd

# Import the VTK exporter
from utils.exporter import write_vtk
from utils.exporter import write_boundary_vtk
from utils.exporter import write_boundary_vtk_3d
from utils.exporter import write_interior_square_vtk
from utils.exporter import WriteInteriorMoving
from utils.exporter import WriteInteriorMoving_3d
from utils.exporter import write_ntu_wireframe_vtk
from utils.exporter import write_normals_vtk

def get_particle_arrays(solver):
    pos = solver.particles.x.to_numpy()
    pressure = solver.particles.pressure.to_numpy()
    velocity = solver.particles.v.to_numpy()
    if callable(getattr(solver.particles, 'n_active', None)):
        n_active = solver.particles.n_active()
        pos, pressure, velocity = pos[:n_active], pressure[:n_active], velocity[:n_active]
    return pos, pressure, velocity

def main():
    # 1. Initialize Taichi on the GPU
    ti.init(arch=ti.gpu)
    
    bnd.init_boundary_fields()

    dim_case = config.DIM
    
    # 2. Instantiate the correct solver based on config BEFORE the loop
    print(f"Initializing Taichi solver for {config.ACTIVE_SCENARIO} in {config.DIM}D...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if config.ACTIVE_SCENARIO == "DAM_BREAK":
        if getattr(config, 'USE_ADAPTIVE_MPM', False) and config.DIM == 2:
            from solver.adaptive_engine import AdaptiveMPMSolver2D
            solver = AdaptiveMPMSolver2D()
            output_directory = os.path.join(base_dir, 'hydrostatic_2D_adaptive')
        else:
            from solver.standard_engine import StandardSolver
            solver = StandardSolver()
            output_directory = os.path.join(base_dir, 'hydrostatic_2D')
        
    elif config.ACTIVE_SCENARIO == "POISEUILLE":
        from solver.Poiseuille_engine import PoiseuilleSolver
        solver = PoiseuilleSolver()
        output_directory = os.path.join(base_dir, 'vtk_poiseuille')
        
    elif config.ACTIVE_SCENARIO == "INFLOW":
        from solver.Inflow_engine import Inflow_Solver
        solver = Inflow_Solver()
        output_directory = os.path.join(base_dir, 'inflow_result')
        
    elif config.ACTIVE_SCENARIO == "IMMERSED":
        if getattr(config, 'USE_ADAPTIVE_MPM', False) and config.DIM == 2:
            from solver.adaptive_engine import AdaptiveMPMSolver2D
            solver = AdaptiveMPMSolver2D()
            output_directory = os.path.join(base_dir, f'vppcase_immersed_{dim_case}D_adaptive')
        else:
            from solver.standard_engine import StandardSolver
            solver = StandardSolver()
            output_directory = os.path.join(base_dir, f'vppcase_immersed_{dim_case}D')
    elif config.ACTIVE_SCENARIO == "ADAPTIVE_MPM":
        from solver.adaptive_engine import AdaptiveMPMSolver2D
        solver = AdaptiveMPMSolver2D()
        output_directory = os.path.join(base_dir, 'adaptive_mpm')
    else:
        raise ValueError("Unknown scenario!")

    os.makedirs(output_directory, exist_ok=True)
    
    TOTAL_FRAMES = 300 # default 300
    
    print(f"Simulation will run for {TOTAL_FRAMES} frames.")
    print(f"Substeps per frame: {config.SUBSTEPS}")
    print(f"Time Step (DT): {config.DT:.7f}")
    print("=========================================")
    
    # =========================================================================
    # DRAW INITIAL CONDITION BC
    # =========================================================================
    # DRAW THE OUTER BOUNDARY
    min_x = config.PADDING * config.DX
    min_y = config.PADDING * config.DY
    max_x = min_x + config.GRID_WIDTH
    max_y = min_y + config.GRID_HEIGHT
    
    if config.DIM == 3:
        min_z = config.PADDING * config.DZ
        max_z = min_z + config.GRID_DEPTH
        write_boundary_vtk_3d(min_x, min_y, min_z, max_x, max_y, max_z, output_dir=output_directory)
    else:
        write_boundary_vtk(min_x, min_y, max_x, max_y, output_dir=output_directory)
    
    print("Exporting initial state (t=0)...")
    pos_initial, pressure_initial, velocity_initial = get_particle_arrays(solver)
    write_vtk(0, pos_initial, pressure_initial, velocity_initial, output_dir=output_directory)
    
    if config.ACTIVE_SCENARIO == "DAM_BREAK" and config.IS_DAMBREAK_WITH_OBSTACLE:
        write_interior_square_vtk(
            config.INT_SQUARE_XMIN, config.INT_SQUARE_YMIN_DRAW, 
            config.INT_SQUARE_XMAX, config.INT_SQUARE_YMAX, 
            output_dir=output_directory
        )
    elif config.ACTIVE_SCENARIO == "IMMERSED":
        if config.DIM == 3:
            WriteInteriorMoving_3d(
                0,
                config.INT_MOVINGRECT_XMIN, config.INT_MOVINGRECT_YMIN, config.INT_MOVINGRECT_ZMIN,
                config.INT_MOVINGRECT_XMAX, config.INT_MOVINGRECT_YMAX, config.INT_MOVINGRECT_ZMAX,
                output_dir=output_directory
            )
        else:
            WriteInteriorMoving(
                0,
                config.INT_MOVINGRECT_XMIN, config.INT_MOVINGRECT_YMIN,
                config.INT_MOVINGRECT_XMAX, config.INT_MOVINGRECT_YMAX,
                output_dir=output_directory
            )
    elif config.ACTIVE_SCENARIO == "INFLOW":
        grid_mask = ti.field(dtype=ti.f64, shape=(config.GRID_RES_X, config.GRID_RES_Y))
        write_ntu_wireframe_vtk(output_dir=output_directory)
        mask_np = grid_mask.to_numpy()
        geom_file = os.path.join(output_directory, 'geometry_check.vtk')
        with open(geom_file, 'w') as f:
            f.write("# vtk DataFile Version 3.0\nNTU Maze Geometry Check\nASCII\nDATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {config.GRID_RES_X} {config.GRID_RES_Y} 1\nORIGIN 0 0 0\n")
            f.write(f"SPACING {config.DX} {config.DY} 1\nPOINT_DATA {config.GRID_RES_X * config.GRID_RES_Y}\n")
            f.write("SCALARS SolidWall float 1\nLOOKUP_TABLE default\n")
            for val in mask_np.flatten(order='F'): f.write(f"{val}\n")
        bnd.compute_all_normals()
        normals_np = bnd.grid_normals.to_numpy()
        write_normals_vtk(output_directory, normals_np)
    
    # =========================================================
    # DYNAMIC RELAXATION FOR DAM-BREAK OR IMMERSED
    # =========================================================
    if config.ACTIVE_SCENARIO in ["DAM_BREAK", "IMMERSED"]:
        RELAXATION_FRAMES = 30     
        DAMPING_FACTOR = 0.98      
        
        print(f"Starting Dynamic Relaxation Phase ({RELAXATION_FRAMES} frames)...")
        relaxation_time = 0.0
        for frame in range(1, RELAXATION_FRAMES + 1):
            for _ in range(config.SUBSTEPS):
                solver.step(damping=DAMPING_FACTOR, current_time=relaxation_time)
                relaxation_time += config.DT
        print("Dynamic Relaxation Complete. Fluid is settled.\n=========================================")
        
        pos_initial, pressure_initial, velocity_initial = get_particle_arrays(solver)
        write_vtk(0, pos_initial, pressure_initial, velocity_initial, output_dir=output_directory)  
    
    # =========================================================
    # 3. THE MAIN SIMULATION LOOP
    # =========================================================
    current_time = 0.0
    for frame in range(1, TOTAL_FRAMES + 1):
        
        # ACTUALLY STEP THE PHYSICS FORWARD!
        for _ in range(config.SUBSTEPS):
            solver.step(damping=1.0, current_time=current_time)
            current_time += config.DT
            
        # Extract data to CPU for export
        pos, pressure, velocity = get_particle_arrays(solver)
        
        # Export to VTK
        write_vtk(frame, pos, pressure, velocity, output_dir=output_directory)
        
        if config.ACTIVE_SCENARIO == "IMMERSED":
            # calculate displacement for drawing
            if current_time < config.PLATFORM_STOP_TIME:
                displacement_y = config.PLATFORM_VELOCITY_Y * current_time
            elif current_time < config.PLATFORM_STOP_TIME + config.PLATFORM_DECEL_TIME:
                time_in_decel = current_time - config.PLATFORM_STOP_TIME
                dist_before_stop = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                dist_during_decel = config.PLATFORM_VELOCITY_Y * time_in_decel - 0.5 * (config.PLATFORM_VELOCITY_Y / config.PLATFORM_DECEL_TIME) * (time_in_decel**2)
                displacement_y = dist_before_stop + dist_during_decel
            else:
                dist_before_stop = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                total_decel_dist = 0.5 * config.PLATFORM_VELOCITY_Y * config.PLATFORM_DECEL_TIME
                displacement_y = dist_before_stop + total_decel_dist
            
            current_y_min = config.INT_MOVINGRECT_YMIN + displacement_y
            current_y_max = config.INT_MOVINGRECT_YMAX + displacement_y

            if config.DIM == 3:
                WriteInteriorMoving_3d(
                    frame,
                    config.INT_MOVINGRECT_XMIN, current_y_min, config.INT_MOVINGRECT_ZMIN,
                    config.INT_MOVINGRECT_XMAX, current_y_max, config.INT_MOVINGRECT_ZMAX,
                    output_dir=output_directory
                )
            else:
                WriteInteriorMoving(
                    frame,
                    config.INT_MOVINGRECT_XMIN, current_y_min,
                    config.INT_MOVINGRECT_XMAX, current_y_max,
                    output_dir=output_directory
                )

        print(f"Frame {frame}/{TOTAL_FRAMES} exported.")
        
    print("=========================================")

    if config.ACTIVE_SCENARIO == "POISEUILLE":
        ut.plot_velocity_profile(pos, velocity)

if __name__ == "__main__":
    main()
