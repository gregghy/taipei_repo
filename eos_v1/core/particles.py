# core/particles.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taichi as ti
import config

@ti.data_oriented
class ParticleSystem:
    def __init__(self):
        self.n_particles = config.TOTAL_NUM_MP
        
        # Kinematics
        self.x = ti.Vector.field(config.DIM, dtype=ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(config.DIM, dtype=ti.f32, shape=self.n_particles)
        
        # TRACKER: How many particles are currently in the simulation
        self.active_count = ti.field(dtype=ti.i32, shape=())
        
        # Matrices
        self.C = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f32, shape=self.n_particles)
        self.stress = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f32, shape=self.n_particles)
        self.pressure = ti.field(dtype=ti.f32, shape=self.n_particles)
        
    def init_particles(self):
        """Router for dimensional initialization (Python Scope)"""
        if config.DIM == 3:
            # MOVED HERE: Safely fetch 3D variables in pure Python
            spacing_x = getattr(config, 'DX', 0.0) / getattr(config, 'P_PER_CELL_AXIS', 1)
            spacing_y = getattr(config, 'DY', 0.0) / getattr(config, 'P_PER_CELL_AXIS', 1)
            spacing_z = getattr(config, 'DZ', 0.0) / getattr(config, 'P_PER_CELL_AXIS', 1)
            
            num_w = getattr(config, 'NUM_MP_WIDTH', 1)
            num_h = getattr(config, 'NUM_MP_HEIGHT', 1)
            num_d = getattr(config, 'NUM_MP_DEPTH', 1)
            
            # Pass them as arguments to the GPU kernel below
            self.init_particles_3d(spacing_x, spacing_y, spacing_z, num_w, num_h, num_d)
        else:
            self.init_particles_2d()

    @ti.kernel
    def init_particles_2d(self):
        spacing_x = config.DX / config.P_PER_CELL_AXIS
        spacing_y = config.DY / config.P_PER_CELL_AXIS
        
        for i, j in ti.ndrange(config.NUM_MP_WIDTH, config.NUM_MP_HEIGHT):
            p = i * config.NUM_MP_HEIGHT + j
            
            x_pos = config.POS_MP_LEFT_BOTTOM[0] + (i + 0.5) * spacing_x
            y_pos = config.POS_MP_LEFT_BOTTOM[1] + (j + 0.5) * spacing_y
            
            self.x[p][0] = x_pos
            self.x[p][1] = y_pos
            # Use ti.Vector.zero dynamically based on config.DIM
            self.v[p] = ti.Vector.zero(ti.f32, config.DIM)
            self.C[p] = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            
            # --- HYDROSTATIC INITIALIZATION ---
            local_y = y_pos - config.POS_MP_LEFT_BOTTOM[1]
            p_hydro = config.RHO_0 * 9.81 * (config.MP_HEIGHT - local_y)
            p_hydro = ti.max(0.0, p_hydro)
            
            J_init = 1.0 / (p_hydro / (config.C_0**2 * config.RHO_0) + 1.0)
            self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, J_init]])
            
            self.stress[p] = ti.Matrix([
                [-p_hydro, 0.0], 
                [0.0, -p_hydro]
            ])
            self.pressure[p] = p_hydro

    @ti.kernel
    def init_particles_3d(self, spacing_x: float, spacing_y: float, spacing_z: float, num_w: int, num_h: int, num_d: int):
        """GPU Kernel (No Python built-ins allowed here, only arguments)"""
        for i, j, k in ti.ndrange(num_w, num_h, num_d):
            # Calculate 3D flattened index
            p = i * (num_h * num_d) + j * num_d + k
            
            x_pos = config.POS_MP_LEFT_BOTTOM[0] + (i + 0.5) * spacing_x
            y_pos = config.POS_MP_LEFT_BOTTOM[1] + (j + 0.5) * spacing_y
            z_pos = config.POS_MP_LEFT_BOTTOM[2] + (k + 0.5) * spacing_z
            
            self.x[p][0] = x_pos
            self.x[p][1] = y_pos
            self.x[p][2] = z_pos
            self.v[p] = ti.Vector.zero(ti.f32, config.DIM)
            self.C[p] = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            
            # --- HYDROSTATIC INITIALIZATION ---
            local_y = y_pos - config.POS_MP_LEFT_BOTTOM[1]
            p_hydro = config.RHO_0 * 9.81 * (config.MP_HEIGHT - local_y)
            p_hydro = ti.max(0.0, p_hydro)
            
            J_init = 1.0 / (p_hydro / (config.C_0**2 * config.RHO_0) + 1.0)
            
            self.F[p] = ti.Matrix([
                [1.0, 0.0, 0.0], 
                [0.0, J_init, 0.0],
                [0.0, 0.0, 1.0]
            ])
            
            self.stress[p] = ti.Matrix([
                [-p_hydro, 0.0, 0.0], 
                [0.0, -p_hydro, 0.0],
                [0.0, 0.0, -p_hydro]
            ])
            self.pressure[p] = p_hydro

    @ti.kernel
    def init_particles_inflow(self):
        """Initializes the empty memory pool for the continuous inflow scenario."""
        self.active_count[None] = 0
        
        for p in range(self.n_particles):
            if ti.static(config.DIM == 3):
                self.x[p] = ti.Vector([-1000.0, -1000.0, -1000.0])
            else:
                self.x[p] = ti.Vector([-1000.0, -1000.0])
                
            self.v[p] = ti.Vector.zero(ti.f32, config.DIM)
            self.C[p] = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            self.F[p] = ti.Matrix.identity(ti.f32, config.DIM)
            self.stress[p] = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            self.pressure[p] = 0.0
