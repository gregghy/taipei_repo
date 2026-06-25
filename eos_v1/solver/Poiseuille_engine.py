# solver/engine_poiseuille.py
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

@ti.data_oriented
class PoiseuilleSolver:
    def __init__(self):
        self.particles = ParticleSystem()
        self.grid = Grid()
        self.particles.init_particles()

    @ti.kernel
    def p2g_APIC(self):
        """Scatters mass and momentum. NO division by mass here."""
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
                self.grid.v[I] += sf_weight * config.P_MASS * (v_p + affine_momentum)

    @ti.kernel
    def BC_map_mass_momentum(self):
        """Resolves periodic boundaries for mass and momentum."""
        for i, j in self.grid.m:
            if i < config.PADDING: 
                internal_i = i + config.N_CELL_WIDTH
                self.grid.m[internal_i, j] += self.grid.m[i,j]
                self.grid.v[internal_i, j] += self.grid.v[i,j] 
            elif i >= config.PADDING + config.N_CELL_WIDTH: 
                internal_i = i - config.N_CELL_WIDTH
                self.grid.m[internal_i, j] += self.grid.m[i,j]
                self.grid.v[internal_i, j] += self.grid.v[i,j]
                
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
        """Converts momentum to velocity after mapping."""
        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 0.0:
                self.grid.v[I] /= self.grid.m[I]

    @ti.kernel
    def compute_forces(self):
        for p in range(self.particles.n_particles):
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
    def BC_map_forces(self):
        """Resolves periodic boundaries for forces."""
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
    def grid_update(self):
        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 0.0:
                self.grid.v_old[I] = self.grid.v[I]
                acceleration = self.grid.f[I] / self.grid.m[I]
                self.grid.v[I] += acceleration * config.DT

    @ti.kernel
    def apply_boundaries(self):
        """NO-SLIP bottom and top walls."""
        for i, j in self.grid.v:
            if j <= config.PADDING:
                self.grid.v[i,j] = ti.Vector([0.0, 0.0])
            elif j >= config.GRID_RES_Y - config.PADDING - 1:
                self.grid.v[i,j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def BC_map_velocities(self):
        """Copies final integrated velocities to ghost nodes for clean G2P."""
        for i, j in self.grid.v:
            if i < config.PADDING:
                internal_i = i + config.N_CELL_WIDTH
                self.grid.v[i, j] = self.grid.v[internal_i, j]
            elif i >= config.PADDING + config.N_CELL_WIDTH:
                internal_i = i - config.N_CELL_WIDTH
                self.grid.v[i, j] = self.grid.v[internal_i, j]

    @ti.kernel
    def g2p_APIC(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            v_new = ti.Vector.zero(ti.f64, config.DIM)
            B_new = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            D_new = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                v_I_new = self.grid.v[I]      
                x_I = ti.cast(I, ti.f64) * config.DX
                dpos = x_I - x_p
                
                v_new += v_I_new * sf_weight
                B_new += sf_weight * v_I_new.outer_product(dpos)
                D_new += sf_weight * dpos.outer_product(dpos)
            
            D_inv = D_new.inverse()
            C_new = B_new @ D_inv
            
            self.particles.v[p] = v_new
            self.particles.x[p] += v_new * config.DT
            self.particles.C[p] = C_new

            # --- PERIODIC PARTICLE WRAP ---
            x_physical = self.particles.x[p][0] - (config.PADDING * config.DX)
            x_wrapped = x_physical % config.GRID_WIDTH
            self.particles.x[p][0] = x_wrapped + (config.PADDING * config.DX)

    @ti.kernel
    def update_kinematics(self):
        """Step 1: Standard incremental update of the deformation gradient F"""
        for p in range(self.particles.n_particles):
            F_p = self.particles.F[p]
            grad_v = self.particles.C[p] 
            identity = ti.Matrix.identity(ti.f64, config.DIM)
            self.particles.F[p] = (identity + grad_v * config.DT) @ F_p

    @ti.kernel
    def nodal_Fbar_P2G(self):
        """Step 1: Scatter particle volumes to grid nodes using B-splines"""
        for p in range(self.particles.n_particles):
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
    
    # For Nodal F-bar method | 04052026
    @ti.kernel
    def nodal_Fbar_G2P_and_stress(self):
        """Step 2: Gather smoothed J back to particles, scale F, and compute stress"""
        for p in range(self.particles.n_particles):
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

    def step(self):
        self.grid.clear()
        self.p2g_APIC()

        self.BC_map_mass_momentum()
        self.normalize_momentum()
        self.compute_forces()
        self.BC_map_forces()

        self.grid_update()

        self.apply_boundaries()
        self.BC_map_velocities()
        
        self.g2p_APIC()

        self.update_kinematics()
        self.nodal_Fbar_P2G()
        self.nodal_Fbar_G2P_and_stress()

        