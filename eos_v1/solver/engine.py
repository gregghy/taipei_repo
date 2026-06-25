import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import taichi as ti
import config
from core.particles import ParticleSystem
from core.grid import Grid
import core.functions as cf
from physics.constitutive_model import StressUsingWater
from physics.constitutive_model import StressUsingBingham
from physics.boundary import apply_grid_boundary_conditions

@ti.data_oriented
class MPMSolver:
    def __init__(self):
        self.particles = ParticleSystem()
        self.grid = Grid()
        self.particles.init_particles()

    @ti.kernel
    def p2g_kinematics(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            v_p = self.particles.v[p]
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                self.grid.m[I] += sf_weight * config.P_MASS
                self.grid.v[I] += sf_weight * config.P_MASS * v_p

        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 0.0:
                self.grid.v[I] /= self.grid.m[I]

    @ti.kernel
    def p2g_APIC(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            v_p = self.particles.v[p]
            C_p = self.particles.C[p] # APIC affine matrix
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                # Distance vector from particle to node (x_I - x_p)
                x_I = ti.cast(I, ti.f64) * config.DX
                dpos = x_I - x_p
                
                # APIC Affine momentum transfer
                affine_momentum = C_p @ dpos
                
                self.grid.m[I] += sf_weight * config.P_MASS
                self.grid.v[I] += sf_weight * config.P_MASS * (v_p + affine_momentum)

        # Normalize momentum to get velocity
        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 0.0:
                self.grid.v[I] /= self.grid.m[I]

    @ti.kernel
    def p2g_APIC_noNormalized(self):
        for p in range(self.particles.n_particles):
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
                # NOTE: grid.v is acting as a MOMENTUM accumulator here
                self.grid.v[I] += sf_weight * config.P_MASS * (v_p + affine_momentum)

    @ti.kernel
    def g2p(self):
        FLIP_RATIO = 0.98 # Reduced slightly from 0.99 for better stability

        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            v_p = self.particles.v[p]
            
            base, _, w_0, w_1, w_2, dw_0, dw_1, dw_2 = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            dw = [dw_0, dw_1, dw_2]
            
            # FIXED: Made sure all matrices are ti.f64 instead of ti.f32
            v_pic = ti.Vector.zero(ti.f64, config.DIM)
            v_flip_change = ti.Vector.zero(ti.f64, config.DIM)
            grad_v = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                
                sf_weight = w[i][0] * w[j][1]
                sf_grad_weight = ti.Vector([dw[i][0] * w[j][1], w[i][0] * dw[j][1]])
                
                v_I_new = self.grid.v[I]      
                v_I_old = self.grid.v_old[I]  
                
                v_pic += v_I_new * sf_weight
                v_flip_change += (v_I_new - v_I_old) * sf_weight
                grad_v += v_I_new.outer_product(sf_grad_weight)
            
            v_flip = v_p + v_flip_change
            v_p_new = FLIP_RATIO * v_flip + (1.0 - FLIP_RATIO) * v_pic
            
            self.particles.v[p] = v_p_new
            self.particles.x[p] += v_p_new * config.DT
            self.particles.C[p] = grad_v
    
    @ti.kernel
    def g2p_APIC(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            v_new = ti.Vector.zero(ti.f64, config.DIM)
            
            # Explicitly declare B and D matrices
            B_new = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            D_new = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                v_I_new = self.grid.v[I]      
                
                x_I = ti.cast(I, ti.f64) * config.DX
                dpos = x_I - x_p
                
                # 1. Update Particle Velocity
                v_new += v_I_new * sf_weight
                
                # 2. Compute the B matrix (Momentum matrix)
                B_new += sf_weight * v_I_new.outer_product(dpos)
                
                # 3. Compute the D matrix (Inertia matrix)
                D_new += sf_weight * dpos.outer_product(dpos)
            
            # 4. Invert the D matrix
            # Note: For standard inner particles, this will always be invertible.
            D_inv = D_new.inverse()
            
            # 5. Calculate the true Affine matrix (C = B * D^-1)
            C_new = B_new @ D_inv
            
            # Update particle states
            self.particles.v[p] = v_new
            self.particles.x[p] += v_new * config.DT
            
            # Store the Affine matrix
            self.particles.C[p] = C_new

    @ti.kernel
    def g2p_APIC_Poiseuille(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            v_new = ti.Vector.zero(ti.f64, config.DIM)
            
            # Explicitly declare B and D matrices
            B_new = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            D_new = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                v_I_new = self.grid.v[I]      
                
                x_I = ti.cast(I, ti.f64) * config.DX
                dpos = x_I - x_p
                
                # 1. Update Particle Velocity
                v_new += v_I_new * sf_weight
                
                # 2. Compute the B matrix (Momentum matrix)
                B_new += sf_weight * v_I_new.outer_product(dpos)
                
                # 3. Compute the D matrix (Inertia matrix)
                D_new += sf_weight * dpos.outer_product(dpos)
            
            # 4. Invert the D matrix
            # Note: For standard inner particles, this will always be invertible.
            D_inv = D_new.inverse()
            
            # 5. Calculate the true Affine matrix (C = B * D^-1)
            C_new = B_new @ D_inv
            
            # Update particle states
            self.particles.v[p] = v_new
            self.particles.x[p] += v_new * config.DT
            
            # Store the Affine matrix
            self.particles.C[p] = C_new

            # --- PERIODIC PARTICLE WRAP ---
            # Shift position relative to the start of the physical domain
            x_physical = self.particles.x[p][0] - (config.PADDING * config.DX)
            
            # Wrap it using modulo
            x_wrapped = x_physical % config.GRID_WIDTH
            
            # Shift it back to absolute world coordinates
            self.particles.x[p][0] = x_wrapped + (config.PADDING * config.DX)

    @ti.kernel
    def BC_map_mass_momentum(self):
        """Maps mass and momentum (stored in grid.v) across periodic boundaries."""
        # 1. Add ghost node data into the internal domain
        for i, j in self.grid.m:
            if i < config.PADDING: # LEFT GHOST
                internal_i = i + config.N_CELL_WIDTH
                self.grid.m[internal_i, j] += self.grid.m[i,j]
                self.grid.v[internal_i, j] += self.grid.v[i,j] 
            elif i >= config.PADDING + config.N_CELL_WIDTH: # RIGHT GHOST
                internal_i = i - config.N_CELL_WIDTH
                self.grid.m[internal_i, j] += self.grid.m[i,j]
                self.grid.v[internal_i, j] += self.grid.v[i,j]
                
        # 2. Copy the combined internal data back to ghost nodes
        for i, j in self.grid.m:
            if i < config.PADDING:
                internal_i = i + config.N_CELL_WIDTH
                self.grid.m[i, j] = self.grid.m[internal_i, j]
                self.grid.v[i, j] = self.grid.v[internal_i, j]
            elif i >= config.PADDING + config.N_CELL_WIDTH:
                internal_i = i - config.N_CELL_WIDTH
                self.grid.m[i, j] = self.grid.m[internal_i, j]
                self.grid.v[i, j] = self.grid.v[internal_i, j]

    @ti.kernel
    def normalize_momentum(self):
        """Safely divides momentum by mass to get velocity after boundaries are mapped."""
        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 0.0:
                self.grid.v[I] /= self.grid.m[I]

    @ti.kernel
    def BC_map_forces(self):
        """Ensures forces scattered to ghost nodes are applied to internal nodes."""
        for i, j in self.grid.f:
            if i < config.PADDING:
                internal_i = i + config.N_CELL_WIDTH
                self.grid.f[internal_i, j] += self.grid.f[i, j]
            elif i >= config.PADDING + config.N_CELL_WIDTH:
                internal_i = i - config.N_CELL_WIDTH
                self.grid.f[internal_i, j] += self.grid.f[i, j]
                
        for i, j in self.grid.f:
            if i < config.PADDING:
                internal_i = i + config.N_CELL_WIDTH
                self.grid.f[i, j] = self.grid.f[internal_i, j]
            elif i >= config.PADDING + config.N_CELL_WIDTH:
                internal_i = i - config.N_CELL_WIDTH
                self.grid.f[i, j] = self.grid.f[internal_i, j]

    @ti.kernel
    def BC_map_velocities(self):
        """Copies updated internal velocities to ghost nodes for g2p interpolation."""
        # We only need to copy here, not add, because the internal grid was already updated
        for i, j in self.grid.v:
            if i < config.PADDING:
                internal_i = i + config.N_CELL_WIDTH
                self.grid.v[i, j] = self.grid.v[internal_i, j]
            elif i >= config.PADDING + config.N_CELL_WIDTH:
                internal_i = i - config.N_CELL_WIDTH
                self.grid.v[i, j] = self.grid.v[internal_i, j]

    @ti.kernel
    def compute_forces(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            stress_p = self.particles.stress[p]
            
            # CRITICAL FIX: Calculate current volume of particle for Cauchy stress
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
                
                # Apply force using the CURRENT volume
                f_int = (stress_p @ sf_grad_weight) * current_vol 
                f_ext = sf_weight * config.P_MASS * ti.Vector(config.GRAVITY) 
                
                self.grid.f[I] += (f_ext - f_int)
    
    @ti.kernel
    def grid_update(self):
        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 0.0:
                self.grid.v_old[I] = self.grid.v[I]
                acceleration = self.grid.f[I] / self.grid.m[I]
                self.grid.v[I] += acceleration * config.DT
    
    @ti.kernel
    def apply_boundaries(self):
        for i, j in self.grid.v:
            # Left and Right
            if i <= config.PADDING and self.grid.v[i, j][0] < 0.0:
                self.grid.v[i, j][0] = 0.0  
            if i >= config.GRID_RES_X - config.PADDING - 1 and self.grid.v[i, j][0] > 0.0:
                self.grid.v[i, j][0] = 0.0  

            # Bottom and Top
            if j <= config.PADDING and self.grid.v[i, j][1] < 0.0:
                self.grid.v[i, j][1] = 0.0  
            if j >= config.GRID_RES_Y - config.PADDING - 1 and self.grid.v[i, j][1] > 0.0:
                self.grid.v[i, j][1] = 0.0

    @ti.kernel
    def apply_BC_Poiseuille(self): # The boundary of x axis is handled by periodic BC
        for i, j in self.grid.v:
            # NO-SLIP bottom and top walls
            if j <= config.PADDING:
                self.grid.v[i,j] = ti.Vector([0.0, 0.0])
            elif j >= config.GRID_RES_Y - config.PADDING - 1:
                self.grid.v[i,j] = ti.Vector([0.0, 0.0])
            else:
                continue

    @ti.kernel
    def BC_for_PoiseuilleFlow(self):
        """
        The particle goes through the ghost node, and transfer the momentum on to the opposite side.
        """
        # 1. Add ghost node data (outside the physical bounds) into the internal
        for i, j in self.grid.m:
            if i < config.PADDING: # LEFT-GHOST-NODE
                internal_i = i + config.N_CELL_WIDTH
                self.grid.m[internal_i, j] += self.grid.m[i,j]
                self.grid.v[internal_i, j] += self.grid.v[i,j]
            elif i >= config.PADDING + config.N_CELL_WIDTH:
                internal_i = i - config.N_CELL_WIDTH
                self.grid.m[internal_i, j] += self.grid.m[i,j]
                self.grid.v[internal_i, j] += self.grid.v[i,j]
            else:
                continue
        
        for i, j in self.grid.m:
            if i < config.PADDING:
                internal_i = i + config.N_CELL_WIDTH
                self.grid.m[i, j] = self.grid.m[internal_i, j]
                self.grid.v[i, j] = self.grid.v[internal_i, j]
                
            elif i >= config.PADDING + config.N_CELL_WIDTH:
                internal_i = i - config.N_CELL_WIDTH
                self.grid.m[i, j] = self.grid.m[internal_i, j]
                self.grid.v[i, j] = self.grid.v[internal_i, j]
    
    @ti.kernel
    def mp_update(self):
        for p in range(self.particles.n_particles):
            F_p = self.particles.F[p]
            grad_v = self.particles.C[p] 
            
            identity = ti.Matrix.identity(ti.f64, config.DIM) # Fixed to ti.f64
            F_new = (identity + grad_v * config.DT) @ F_p
            self.particles.F[p] = F_new
            
            # If strain projection active, comment the p_new and below
            # stress_new, P_new = compute_fluid_stress(F_new, grad_v)
            stress_new, _ = StressUsingBingham(F_new, grad_v)
            self.particles.stress[p] = stress_new
            # self.particles.pressure[p] = P_new
    
    @ti.kernel
    def smooth_strain_rate(self):
        # 1. Clear a temporary grid field for smoothing (we can reuse grid.m)
        for I in ti.grouped(self.grid.m):
            self.grid.m[I] = 0.0
            self.grid.v[I] = ti.Vector.zero(ti.f64, config.DIM) # Using v[0] to store scalar div

        # 2. Project particle divergence to grid
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            # div(v) is the trace of the velocity gradient C
            div_v = self.particles.C[p].trace()
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                # We use grid.v[I][0] as a scalar accumulator for smoothed div(v)
                self.grid.v[I][0] += sf_weight * div_v
                self.grid.m[I] += sf_weight

        # 3. Normalize on grid and map back to particles
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            smoothed_div_p = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                if self.grid.m[I] > 0:
                    node_div = self.grid.v[I][0] / self.grid.m[I]
                    smoothed_div_p += sf_weight * node_div
            
            # 4. Update pressure using the RATE form EOS
            # P_new = P_old - K * div(v) * dt
            # (Bulk modulus K = C_0^2 * rho_0)
            bulk_modulus = config.C_0**2 * config.RHO_0
            self.particles.pressure[p] -= bulk_modulus * smoothed_div_p * config.DT
            
            # Update stress tensor with the new smoothed pressure
            p_new = self.particles.pressure[p]
            if p_new < 0.0: p_new = 0.0 # Water tension limit
            
            identity = ti.Matrix.identity(ti.f64, config.DIM)
            self.particles.stress[p] = -p_new * identity
            
    def step(self):
        self.grid.clear()
        self.p2g_APIC()
        self.BC_for_PoiseuilleFlow()
        self.compute_forces()
        self.grid_update()

        # self.apply_boundaries()
        self.apply_BC_Poiseuille()
        self.BC_for_PoiseuilleFlow()

        # self.g2p_APIC()
        self.g2p_APIC_Poiseuille()
        self.smooth_strain_rate()
        self.mp_update()