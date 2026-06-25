import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import taichi as ti
import config
from core.particles import ParticleSystem
from core.grid import Grid
import core.functions as cf
from physics.constitutive_model import StressUsingBingham
from physics.constitutive_model import StressUsingWater
from physics.constitutive_model import StressUsingWater_WithArtificialViscosity
import physics.boundary as bnd

@ti.data_oriented
class Inflow_Solver:
    def __init__(self):
        self.particles = ParticleSystem()
        self.grid = Grid()
        self.particles.init_particles_inflow()
        
        # Tracks fractional particle distances between substeps
        self.unspawned_distance = 0.0
    
    def emit_fluid(self, current_time: float):
        """Calculates how many columns of water to inject this substep."""
        if current_time > config.INFLOW_DURATION:
            return
            
        # Distance the fluid stream should travel in one substep (v * dt)
        travel_dist = config.INFLOW_VELOCITY[0] * config.DT
        self.unspawned_distance += travel_dist
        
        spacing_x = config.DX / config.P_PER_CELL_AXIS
        spacing_y = config.DY / config.P_PER_CELL_AXIS
        
        # Check if enough distance has accumulated to spawn a new vertical column
        columns_to_spawn = int(self.unspawned_distance / spacing_x)
        
        if columns_to_spawn > 0:
            self.unspawned_distance -= (columns_to_spawn * spacing_x)
            self.spawn_particle_columns(columns_to_spawn, spacing_x, spacing_y)

    @ti.kernel
    def spawn_particle_columns(self, num_columns: int, spacing_x: float, spacing_y: float):
        """Injects new particles into the active pool at the inlet boundary."""
        num_rows = config.NUM_MP_HEIGHT
        particles_to_spawn = num_columns * num_rows
        
        # Safely reserve space in the active pool
        start_idx = ti.atomic_add(self.particles.active_count[None], particles_to_spawn)
        
        for i in range(particles_to_spawn):
            p = start_idx + i
            
            # Prevent out-of-bounds memory crash if we hit MAX_PARTICLE_INFLOW
            if p < config.MAX_PARTICLE_INFLOW:
                col = i // num_rows
                row = i % num_rows
                
                x_pos = config.POS_MP_LEFT_BOTTOM[0] + (col + 0.5) * spacing_x
                y_pos = config.POS_MP_LEFT_BOTTOM[1] + (row + 0.5) * spacing_y
                
                self.particles.x[p] = ti.Vector([x_pos, y_pos])
                self.particles.v[p] = ti.Vector(config.INFLOW_VELOCITY)
                self.particles.F[p] = ti.Matrix.identity(ti.f64, config.DIM)
                self.particles.stress[p] = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                self.particles.pressure[p] = 0.0
                
    @ti.kernel
    def p2g_APIC(self):
        # CRITICAL FIX: Only loop over spawned particles!
        for p in range(self.particles.active_count[None]):
            x_p = self.particles.x[p]
            v_p = self.particles.v[p]
            C_p = self.particles.C[p] 
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                x_I = ti.cast(I, ti.f64) * config.DX
                dpos = x_I - x_p
                affine_momentum = C_p @ dpos
                
                self.grid.m[I] += sf_weight * config.P_MASS
                self.grid.v[I] += sf_weight * config.P_MASS * (v_p + affine_momentum)

        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 0.0:
                self.grid.v[I] /= self.grid.m[I]
    
    @ti.kernel
    def compute_forces(self):
        # CRITICAL FIX: Only loop over spawned particles!
        for p in range(self.particles.active_count[None]):
            x_p = self.particles.x[p]
            stress_p = self.particles.stress[p]
            
            J_p = self.particles.F[p].determinant()
            current_vol = config.P_VOL * J_p
            
            base, _, w_0, w_1, w_2, dw_0, dw_1, dw_2 = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            dw = [dw_0, dw_1, dw_2]
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                
                sf_weight = w[i][0] * w[j][1]
                sf_grad_weight = ti.Vector([dw[i][0] * w[j][1], w[i][0] * dw[j][1]])
                
                f_int = (stress_p @ sf_grad_weight) * current_vol 
                f_ext = sf_weight * config.P_MASS * ti.Vector(config.GRAVITY) 
                
                self.grid.f[I] += (f_ext - f_int)
    
    @ti.kernel
    def ComputeForces_NTU_Maze(self):
        """
        Evaluates the Enhanced Boundary Conditions for the NTU line-segment geometry
        and updates the grid force BEFORE integrating the momentum.
        """
        for I in ti.grouped(self.grid.m):
            m_I = self.grid.m[I]
            
            # Only process active grid nodes
            if m_I > 1e-8:
                x_I = ti.Vector([ti.cast(I[0], ti.f64) * config.DX, 
                                 ti.cast(I[1], ti.f64) * config.DY])
                
                # Query the custom NTU line-segment SDF
                r, n = bnd.Get_NTU_Sketch_SDF(x_I)
                
                # Set boundary properties
                bc_type = 3  # 0 = Slip (best for internal pipe flow so it doesn't snag)
                v_target = ti.Vector([0.0, 0.0]) # The maze walls are stationary
                
                # Compute the boundary force
                f_bc = bnd.Compute_EBC_Force(
                    m_I, 
                    self.grid.v[I], 
                    self.grid.f[I], 
                    r, 
                    n, 
                    bc_type, 
                    v_target
                )
                
                # Inject the boundary force into the grid's net force
                self.grid.f[I] += f_bc
    
    @ti.kernel
    def enforce_inflow_valve(self):
        """Enforces a one-way check-valve zone at the channel inlet"""
        inlet_xmin = config.POS_MP_LEFT_BOTTOM[0] - 0.05
        inlet_xmax = config.POS_MP_LEFT_BOTTOM[0] + 0.15 
        
        # 1. Define a Y-minimum to isolate the inlet pipe
        inlet_ymin = config.POS_MP_LEFT_BOTTOM[1] - 0.05
        
        for p in range(self.particles.active_count[None]):
            x_p = self.particles.x[p][0]
            y_p = self.particles.x[p][1] # Get Y coordinate
            
            # 2. Check BOTH X and Y coordinates
            if x_p >= inlet_xmin and x_p <= inlet_xmax and y_p >= inlet_ymin:
                if self.particles.v[p][0] < config.INFLOW_VELOCITY[0]:
                    self.particles.v[p][0] = config.INFLOW_VELOCITY[0]
    
    @ti.kernel
    def grid_update(self, damping: float): #ignore
        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 1e-7:
                self.grid.v_old[I] = self.grid.v[I]
                acceleration = self.grid.f[I] / self.grid.m[I]
                self.grid.v[I] += acceleration * config.DT
                
                # Apply Dynamic Relaxation damping
                self.grid.v[I] *= damping

                max_speed = 15.0 
                self.grid.v[I][0] = ti.max(-max_speed, ti.min(max_speed, self.grid.v[I][0]))
                self.grid.v[I][1] = ti.max(-max_speed, ti.min(max_speed, self.grid.v[I][1]))
                
            else: # SAFETY NET: for splashed particle
                self.grid.m[I] = 0.0 
                self.grid.v[I] = ti.Vector.zero(ti.f64, config.DIM)
                self.grid.f[I] = ti.Vector.zero(ti.f64, config.DIM)
    
    @ti.kernel
    def apply_boundaries(self):
        for i, j in self.grid.v:
            if i <= config.PADDING and self.grid.v[i, j][0] < 0.0:
                self.grid.v[i, j][0] = 0.0  
            if i >= config.GRID_RES_X - config.PADDING - 1 and self.grid.v[i, j][0] > 0.0:
                self.grid.v[i, j][0] = 0.0  
            if j <= config.PADDING and self.grid.v[i, j][1] < 0.0:
                self.grid.v[i, j][1] = 0.0  
            if j >= config.GRID_RES_Y - config.PADDING - 1 and self.grid.v[i, j][1] > 0.0:
                self.grid.v[i, j][1] = 0.0
    
    @ti.kernel
    def g2p_APIC_NTU(self):
        """Standard APIC gather with NTU static boundary kinematic velocity projection"""
        for p in range(self.particles.active_count[None]):
            x_p = self.particles.x[p]
            
            # Skip completely unspawned particles
            if x_p[0] > -500.0:
                
                # =========================================================
                # 1. STANDARD APIC GATHER
                # =========================================================
                base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
                w = [w_0, w_1, w_2]
                
                v_new = ti.Vector.zero(ti.f64, config.DIM)
                B_new = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                
                for i, j in ti.static(ti.ndrange(3, 3)):
                    offset = ti.Vector([i, j])
                    I = ti.cast(base + offset, ti.i32)
                    sf_weight = w[i][0] * w[j][1]
                    
                    v_I = self.grid.v[I]      
                    dpos = (ti.cast(I, ti.f64) * config.DX) - x_p
                    
                    v_new += v_I * sf_weight
                    B_new += sf_weight * v_I.outer_product(dpos)
                
                C_new = B_new * (4.0 * config.INV_DX ** 2)
                
                # =========================================================
                # 2. PARTICLE-BASED ENHANCED BOUNDARY CONDITION (Velocity Projection)
                # =========================================================
                # Query distance and normal based on CURRENT particle position
                r, n = bnd.Get_NTU_Sketch_SDF(x_p)
                
                # Check if particle is inside the regularization zone
                if r < config.DECAY_ZONE:
                    v_target = ti.Vector([0.0, 0.0]) # The NTU maze is stationary
                    v_rel = v_new - v_target
                    
                    # Check for penetration (Fluid moving INTO the boundary)
                    if v_rel.dot(n) < 0.0:
                        # Calculate Regularized Delta
                        delta_c = 1.0 if r <= 0.0 else 1.0 - ti.math.pow(r / config.DECAY_ZONE, 3.0)
                        
                        # Construct Projection Matrix (Lambda_c)
                        I_mat = ti.Matrix.identity(ti.f64, config.DIM)
                        Lambda_c = I_mat - (n.outer_product(n) * delta_c)
                        
                        # Apply constraint to Velocity! 
                        # This mathematically zeroes out the velocity pushing into the wall.
                        v_new = (Lambda_c @ v_new) + (delta_c * v_target.dot(n) * n)

                # Finalize the correctly bounded velocity and affine momentum
                self.particles.v[p] = v_new
                self.particles.C[p] = C_new
                
                # =========================================================
                # 3. PROVISIONAL POSITION UPDATE
                # =========================================================
                new_x = x_p + v_new * config.DT
                
                # =========================================================
                # 4. KINEMATIC POSITION CLAMP (The "Safety Bumper")
                # =========================================================
                # Query SDF again at the newly predicted position
                r_new, n_new = bnd.Get_NTU_Sketch_SDF(new_x)
                
                if r_new < 0.0:
                    epsilon = 1e-5
                    new_x = new_x + (ti.abs(r_new) + epsilon) * n_new

                # =========================================================
                # 5. STRICT OUTER DOMAIN CLAMP
                # =========================================================
                pad_x = config.PADDING * config.DX
                pad_y = config.PADDING * config.DY
                max_x = (config.GRID_RES_X - config.PADDING - 1) * config.DX
                max_y = (config.GRID_RES_Y - config.PADDING - 1) * config.DY
                
                new_x[0] = ti.max(pad_x + 0.001, ti.min(new_x[0], max_x - 0.001))
                new_x[1] = ti.max(pad_y + 0.001, ti.min(new_x[1], max_y - 0.001))
                
                self.particles.x[p] = new_x
    
    @ti.kernel
    def update_kinematics(self):
        """Step 1: Standard incremental update of the deformation gradient F"""
        for p in range(self.particles.active_count[None]):
            F_p = self.particles.F[p]
            grad_v = self.particles.C[p] 
            identity = ti.Matrix.identity(ti.f64, config.DIM)
            self.particles.F[p] = (identity + grad_v * config.DT) @ F_p
    
    @ti.kernel
    def nodal_Fbar_P2G(self):
        """Step 1: Scatter particle volumes to grid nodes using B-splines"""
        for p in range(self.particles.active_count[None]):
            x_p = self.particles.x[p]
            J_p = ti.max(self.particles.F[p].determinant(), 0.1) # Safety clamp
            
            v_init = config.P_VOL
            v_curr = v_init * J_p
            
            # Get B-spline weights
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                # Scatter volume to nodes
                self.grid.vol_init[I] += sf_weight * v_init
                self.grid.vol_curr[I] += sf_weight * v_curr
    
    @ti.kernel
    def nodal_Fbar_G2P_and_stress(self):
        """Step 2: Gather smoothed J back to particles, scale F, and compute stress"""
        for p in range(self.particles.active_count[None]):
            x_p = self.particles.x[p]
            J_p = ti.max(self.particles.F[p].determinant(), 0.1)
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            J_bar_p = 0.0
            weight_sum = 0.0
            
            # Gather smoothed volume ratio from nodes
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                V_init_node = self.grid.vol_init[I]
                V_curr_node = self.grid.vol_curr[I]
                
                if V_init_node > 1e-8:
                    J_node = V_curr_node / V_init_node
                    J_bar_p += sf_weight * J_node
                    weight_sum += sf_weight
            
            # Normalize to handle boundary truncation (free surfaces)
            if weight_sum > 1e-5:
                J_bar_p /= weight_sum
            else:
                J_bar_p = J_p # Free surface fallback (Standard MPM)
                
            # Equation 48: Scale F
            # dim_factor = 1.0 / config.DIM
            # scale = ti.math.pow(J_bar_p / J_p, dim_factor)
            # F_new = self.particles.F[p] * scale
            # self.particles.F[p] = F_new
            
            # Omit Deviatoric part - Incompressible Fluid
            dim_factor = 1.0 / config.DIM
            scale = ti.math.pow(J_bar_p / J_p, dim_factor)

            scale = ti.max(0.95, ti.min(1.05, scale))
            
            J_new = J_p * (scale ** config.DIM)
            vol_scale = ti.math.pow(J_new, 1.0 / config.DIM)
            F_new = ti.Matrix([
                [vol_scale, 0.0], 
                [0.0, vol_scale]
            ])
            self.particles.F[p] = F_new
            
            
            # Compute stress using the newly stabilized F
            grad_v = self.particles.C[p]
            stress_new = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            
            if ti.static(config.FLUID == "BINGHAM_PLASTIC"):
                stress_new = StressUsingBingham(F_new, grad_v)
            elif ti.static(config.FLUID == "WATER"):
                stress_new = StressUsingWater(F_new, grad_v)
                
            self.particles.stress[p] = stress_new
            
            # Export smoothed pressure to ParaView
            K = config.C_0**2 * config.RHO_0
            self.particles.pressure[p] = ti.max(K * (1.0 / J_bar_p - 1.0), 0.0)
    
    
    def step(self, damping=1.0, current_time=0.0):
        
        self.emit_fluid(current_time)
        self.grid.clear()
        self.p2g_APIC()
        
        self.compute_forces()
        self.ComputeForces_NTU_Maze()
        
        self.grid_update(damping)
        self.apply_boundaries() # free slip for outer boundary
        
        self.g2p_APIC_NTU()
        
        # 2. ENFORCE THE INFLOW VALVE HERE!
        self.enforce_inflow_valve()
        
        # Nodal-averaged F-bar method | 04052026
        self.update_kinematics()
        self.nodal_Fbar_P2G()
        self.nodal_Fbar_G2P_and_stress()