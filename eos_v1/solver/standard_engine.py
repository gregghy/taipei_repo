# solver/engine_standard.py
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
class StandardSolver:
    def __init__(self):
        self.particles = ParticleSystem()
        self.grid = Grid()
        self.particles.init_particles()

    @ti.kernel
    def p2g_APIC(self):
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
                
                x_I = ti.cast(I, ti.f32) * config.DX
                dpos = x_I - x_p
                affine_momentum = C_p @ dpos
                
                self.grid.m[I] += sf_weight * config.P_MASS
                self.grid.v[I] += sf_weight * config.P_MASS * (v_p + affine_momentum)

        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 0.0:
                self.grid.v[I] /= self.grid.m[I]
    
    # For Nodal F-bar method | 04052026
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

    @ti.kernel
    def p2g_APIC_3d(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            v_p = self.particles.v[p]
            C_p = self.particles.C[p] 
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                I = ti.cast(base + offset, ti.i32)
                
                # 3D Scalar Weight
                sf_weight = w[i][0] * w[j][1] * w[k][2]
                
                x_I = ti.cast(I, ti.f32) * config.DX
                dpos = x_I - x_p
                affine_momentum = C_p @ dpos
                
                self.grid.m[I] += sf_weight * config.P_MASS
                self.grid.v[I] += sf_weight * config.P_MASS * (v_p + affine_momentum)

        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 0.0:
                self.grid.v[I] /= self.grid.m[I]

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
            stress_new = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            
            if ti.static(config.FLUID == "BINGHAM_PLASTIC"):
                stress_new = StressUsingBingham(F_new, grad_v)
            elif ti.static(config.FLUID == "WATER"):
                stress_new = StressUsingWater(F_new, grad_v)
                
            self.particles.stress[p] = stress_new
            
            # Export smoothed pressure to ParaView
            K = config.C_0**2 * config.RHO_0
            self.particles.pressure[p] = ti.max(K * (1.0 / J_bar_p - 1.0), 0.0)
    
    @ti.kernel
    def nodal_Fbar_G2P_and_stress_3d(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            J_p = ti.max(self.particles.F[p].determinant(), 0.1)
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            J_bar_p = 0.0
            weight_sum = 0.0
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1] * w[k][2]
                
                V_init_node = self.grid.vol_init[I]
                V_curr_node = self.grid.vol_curr[I]
                
                eps_vol = 1e-5 * config.P_VOL

                if V_init_node > eps_vol:
                    J_bar_p += sf_weight * (V_curr_node / V_init_node)
                    weight_sum += sf_weight
            
            if weight_sum > 1e-5:
                J_bar_p /= weight_sum
            else:
                J_bar_p = J_p
                
            dim_factor = 1.0 / config.DIM
            scale = ti.math.pow(J_bar_p / J_p, dim_factor)
            scale = ti.max(0.95, ti.min(1.05, scale)) 
            
            J_new = J_p * (scale ** config.DIM)
            vol_scale = ti.math.pow(J_new, dim_factor)
            
            # Explicit 3D scaling matrix
            F_new = ti.Matrix([
                [vol_scale, 0.0, 0.0], 
                [0.0, vol_scale, 0.0],
                [0.0, 0.0, vol_scale]
            ])
            self.particles.F[p] = F_new
            
            grad_v = self.particles.C[p]
            stress_new = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            
            if ti.static(config.FLUID == "BINGHAM_PLASTIC"):
                stress_new = StressUsingBingham(F_new, grad_v)
            elif ti.static(config.FLUID == "WATER"):
                stress_new = StressUsingWater(F_new, grad_v)
                
            self.particles.stress[p] = stress_new
            K = config.C_0**2 * config.RHO_0
            self.particles.pressure[p] = ti.max(K * (1.0 / J_bar_p - 1.0), 0.0)

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
    def compute_forces_3d(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            stress_p = self.particles.stress[p]
            
            J_p = self.particles.F[p].determinant()
            current_vol = config.P_VOL * J_p
            
            base, _, w_0, w_1, w_2, dw_0, dw_1, dw_2 = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            dw = [dw_0, dw_1, dw_2]
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                I = ti.cast(base + offset, ti.i32)
                
                sf_weight = w[i][0] * w[j][1] * w[k][2]
                
                # 3D Gradient Vector
                sf_grad_weight = ti.Vector([
                    dw[i][0] * w[j][1] * w[k][2],
                     w[i][0] * dw[j][1] * w[k][2],
                     w[i][0] * w[j][1] * dw[k][2]
                ])
                
                f_int = (stress_p @ sf_grad_weight) * current_vol 
                f_ext = sf_weight * config.P_MASS * ti.Vector(config.GRAVITY) 
                
                self.grid.f[I] += (f_ext - f_int)

    @ti.kernel
    def ComputeForces_EBC(self):
        """
        Evaluates the Enhanced Boundary Conditions on the background grid 
        and updates the grid force BEFORE integrating the momentum.
        """
        for I in ti.grouped(self.grid.m):
            m_I = self.grid.m[I]
            
            # Only process active grid nodes
            if m_I > 1e-8:
                # 1. Get the physical coordinate of the grid node
                # Note: ti.cast(I) gets the [i, j] index, multiply by [DX, DY]
                x_I = ti.Vector([ti.cast(I[0], ti.f32) * config.DX, 
                                 ti.cast(I[1], ti.f32) * config.DY])
                
                # 2. Define the obstacle (using the interior square from config)
                box_xmin = config.INT_SQUARE_XMIN
                box_xmax = config.INT_SQUARE_XMAX
                box_ymin = config.INT_SQUARE_YMIN
                box_ymax = config.INT_SQUARE_YMAX
                
                # 3. Get Distance (r) and Normal (n) using the SDF
                r, n = bnd.Get_Rect_SDF(x_I, box_xmin, box_xmax, box_ymin, box_ymax)
                
                # 4. Set the Boundary Type and Target Velocity
                # bc_type: 0 = Slip, 1 = No-Slip, 2 = Velocity
                bc_type = 0
                v_target = ti.Vector([0.0, 0.0]) # The obstacle is stationary
                
                # 5. Compute the required boundary force
                f_bc = bnd.Compute_EBC_Force(
                    m_I, 
                    self.grid.v[I], 
                    self.grid.f[I], 
                    r, 
                    n, 
                    bc_type, 
                    v_target
                )
                
                # 6. Inject the boundary force into the grid's net force
                self.grid.f[I] += f_bc
    
    @ti.kernel
    def ComputeForces_MovingEBC(self, t: float):
        """
        Evaluates the Enhanced Boundary Conditions on the background grid 
        with a dynamically moving platform.
        """
        for I in ti.grouped(self.grid.m):
            m_I = self.grid.m[I]
            
            if m_I > 1e-8:
                x_I = ti.Vector([ti.cast(I[0], ti.f32) * config.DX, 
                                 ti.cast(I[1], ti.f32) * config.DY])
                
                # 1. Calculate Platform Displacement inside the kernel
                displacement_y = 0.0
                if t < config.PLATFORM_STOP_TIME:
                    displacement_y = config.PLATFORM_VELOCITY_Y * t
                else:
                    displacement_y = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                
                # 2. Dynamically define the bounding box
                box_xmin = config.INT_MOVINGRECT_XMIN
                box_xmax = config.INT_MOVINGRECT_XMAX
                box_ymin = config.INT_MOVINGRECT_YMIN + displacement_y
                box_ymax = config.INT_MOVINGRECT_YMAX + displacement_y
                
                # 3. Get Distance (r) and Normal (n)
                r, n = bnd.Get_Rect_SDF(x_I, box_xmin, box_xmax, box_ymin, box_ymax)
                
                # 4. Set kinematics for the boundary collision
                v_platform_x = config.PLATFORM_VELOCITY_X if t < config.PLATFORM_STOP_TIME else 0.0
                v_platform_y = config.PLATFORM_VELOCITY_Y if t < config.PLATFORM_STOP_TIME else 0.0
                
                bc_type = 3 # generalized Slip wall (when moving use velocity, when stop it is slip)
                v_target = ti.Vector([v_platform_x, v_platform_y])
                
                # 5. Compute EBC force
                f_bc = bnd.Compute_EBC_Force(
                    m_I, self.grid.v[I], self.grid.f[I], r, n, bc_type, v_target
                )
                
                self.grid.f[I] += f_bc
    
    @ti.kernel
    def ComputeForces_MovingEBC_smoothdec(self, t: float):
        """
        Evaluates the Enhanced Boundary Conditions on the background grid 
        with a dynamically moving platform featuring smooth deceleration.
        """
        for I in ti.grouped(self.grid.m):
            m_I = self.grid.m[I]
            
            if m_I > 1e-8:
                x_I = ti.Vector([ti.cast(I[0], ti.f32) * config.DX, 
                                 ti.cast(I[1], ti.f32) * config.DY])
                
                # =========================================================
                # KINEMATICS: Smooth Braking Ramp
                # =========================================================
                DECEL_TIME = 0.5 # The platform takes 0.5 seconds to brake to a halt
                
                v_platform_x = 0.0
                v_platform_y = 0.0
                displacement_y = 0.0
                
                if t < config.PLATFORM_STOP_TIME:
                    # 1. Constant velocity phase
                    v_platform_y = config.PLATFORM_VELOCITY_Y
                    displacement_y = v_platform_y * t
                    
                elif t < config.PLATFORM_STOP_TIME + DECEL_TIME:
                    # 2. Deceleration (Braking) phase
                    time_in_decel = t - config.PLATFORM_STOP_TIME
                    progress = time_in_decel / DECEL_TIME
                    
                    # Linearly reduce velocity to 0
                    v_platform_y = config.PLATFORM_VELOCITY_Y * (1.0 - progress)
                    
                    # Calculate distance traveled during braking
                    dist_before_stop = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                    dist_during_decel = config.PLATFORM_VELOCITY_Y * time_in_decel - 0.5 * (config.PLATFORM_VELOCITY_Y / DECEL_TIME) * (time_in_decel**2)
                    displacement_y = dist_before_stop + dist_during_decel
                    
                else:
                    # 3. Fully stopped phase
                    v_platform_y = 0.0
                    
                    # Total distance remains locked
                    dist_before_stop = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                    total_decel_dist = 0.5 * config.PLATFORM_VELOCITY_Y * DECEL_TIME
                    displacement_y = dist_before_stop + total_decel_dist
                
                # =========================================================
                # BOUNDARY COLLISION
                # =========================================================
                # Dynamically define the bounding box based on smooth displacement
                box_xmin = config.INT_MOVINGRECT_XMIN
                box_xmax = config.INT_MOVINGRECT_XMAX
                box_ymin = config.INT_MOVINGRECT_YMIN + displacement_y
                box_ymax = config.INT_MOVINGRECT_YMAX + displacement_y
                
                # Get Distance (r) and Normal (n)
                r, n = bnd.Get_Rect_SDF(x_I, box_xmin, box_xmax, box_ymin, box_ymax)
                
                # Set kinematics for the boundary collision
                bc_type = 3 # Generalized Slip wall
                v_target = ti.Vector([v_platform_x, v_platform_y])
                
                # Compute EBC force
                f_bc = bnd.Compute_EBC_Force(
                    m_I, self.grid.v[I], self.grid.f[I], r, n, bc_type, v_target
                )
                
                self.grid.f[I] += f_bc

    @ti.kernel
    def ComputeForces_EBC_Scenarios_3d(self, t: float):
        for I in ti.grouped(self.grid.m):
            m_I = self.grid.m[I]
            eps_mass = 1e-5 * config.P_MASS
            if m_I > eps_mass:
                x_I = ti.Vector([ti.cast(I[0], ti.f32) * config.DX, 
                                 ti.cast(I[1], ti.f32) * config.DY,
                                 ti.cast(I[2], ti.f32) * config.DZ])
                # Initialize 3D force vector
                f_bc = ti.Vector([0.0, 0.0, 0.0])
                
                if ti.static(config.ACTIVE_SCENARIO == "IMMERSED"):
                    displacement_y = 0.0
                    DECEL_TIME = 0.5
                    v_p_y = 0.0
                    
                    if t < config.PLATFORM_STOP_TIME:
                        v_p_y = config.PLATFORM_VELOCITY_Y
                        displacement_y = config.PLATFORM_VELOCITY_Y * t
                    elif t < config.PLATFORM_STOP_TIME + DECEL_TIME:
                        time_in_decel = t - config.PLATFORM_STOP_TIME
                        v_p_y = config.PLATFORM_VELOCITY_Y * (1.0 - (time_in_decel / DECEL_TIME))
                        displacement_y = (config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME) + (config.PLATFORM_VELOCITY_Y * time_in_decel - 0.5 * (config.PLATFORM_VELOCITY_Y / DECEL_TIME) * (time_in_decel**2))
                    else:
                        displacement_y = (config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME) + (0.5 * config.PLATFORM_VELOCITY_Y * DECEL_TIME)
                        
                    # Query 3D Box SDF
                    r, n = bnd.Get_Box_SDF(
                        x_I, 
                        config.INT_MOVINGRECT_XMIN, config.INT_MOVINGRECT_XMAX, 
                        config.INT_MOVINGRECT_YMIN + displacement_y, config.INT_MOVINGRECT_YMAX + displacement_y,
                        config.INT_MOVINGRECT_ZMIN, config.INT_MOVINGRECT_ZMAX
                    )
                    f_bc = bnd.Compute_EBC_Force(m_I, self.grid.v[I], self.grid.f[I], r, n, 3, ti.Vector([0.0, v_p_y, 0.0]))
                    
                self.grid.f[I] += f_bc

    @ti.kernel
    def grid_update(self, damping: float): #ignore
        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 1e-7:
                self.grid.v_old[I] = self.grid.v[I]
                acceleration = self.grid.f[I] / self.grid.m[I]
                self.grid.v[I] += acceleration * config.DT
                
                # Apply Dynamic Relaxation damping
                self.grid.v[I] *= damping
                
                # --- SAFETYNET ---
                # This protects both velocity AND the C matrix!
                max_speed = 15.0
                self.grid.v[I][0] = ti.max(-max_speed, ti.min(max_speed, self.grid.v[I][0]))
                self.grid.v[I][1] = ti.max(-max_speed, ti.min(max_speed, self.grid.v[I][1]))
            else: # SAFETY NET: for splashed particle
                self.grid.m[I] = 0.0 
                self.grid.v[I] = ti.Vector.zero(ti.f32, config.DIM)
                self.grid.f[I] = ti.Vector.zero(ti.f32, config.DIM)
    
    @ti.kernel
    def grid_update_3d(self, damping: float):
        for I in ti.grouped(self.grid.m):
            eps_mass = 1e-5 * config.P_MASS
            if self.grid.m[I] > eps_mass:
                self.grid.v_old[I] = self.grid.v[I]
                self.grid.v[I] += (self.grid.f[I] / self.grid.m[I]) * config.DT
                self.grid.v[I] *= damping
                
                max_speed = 15.0
                self.grid.v[I][0] = ti.max(-max_speed, ti.min(max_speed, self.grid.v[I][0]))
                self.grid.v[I][1] = ti.max(-max_speed, ti.min(max_speed, self.grid.v[I][1]))
                self.grid.v[I][2] = ti.max(-max_speed, ti.min(max_speed, self.grid.v[I][2]))
            else:
                self.grid.m[I] = 0.0 
                self.grid.v[I] = ti.Vector.zero(ti.f32, config.DIM)
                self.grid.f[I] = ti.Vector.zero(ti.f32, config.DIM)

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
    def apply_boundaries_3d(self):
        for i, j, k in self.grid.v:
            # X limits
            if i <= config.PADDING and self.grid.v[i, j, k][0] < 0.0: self.grid.v[i, j, k][0] = 0.0  
            if i >= config.GRID_RES_X - config.PADDING - 1 and self.grid.v[i, j, k][0] > 0.0: self.grid.v[i, j, k][0] = 0.0  
            
            # Y limits
            if j <= config.PADDING and self.grid.v[i, j, k][1] < 0.0: self.grid.v[i, j, k][1] = 0.0  
            if j >= config.GRID_RES_Y - config.PADDING - 1 and self.grid.v[i, j, k][1] > 0.0: self.grid.v[i, j, k][1] = 0.0

            # Z limits
            if k <= config.PADDING and self.grid.v[i, j, k][2] < 0.0: self.grid.v[i, j, k][2] = 0.0  
            if k >= config.GRID_RES_Z - config.PADDING - 1 and self.grid.v[i, j, k][2] > 0.0: self.grid.v[i, j, k][2] = 0.0

    @ti.kernel
    def g2p_APIC(self):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            v_new = ti.Vector.zero(ti.f32, config.DIM)
            B_new = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            # D_new = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                v_I_new = self.grid.v[I]      
                x_I = ti.cast(I, ti.f32) * config.DX
                dpos = x_I - x_p
                
                v_new += v_I_new * sf_weight
                B_new += sf_weight * v_I_new.outer_product(dpos)
                # D_new += sf_weight * dpos.outer_product(dpos)
            
            # D_inv = D_new.inverse()
            # C_new = B_new @ D_inv
            D_inv_scalar = 4.0 * (config.INV_DX ** 2)
            C_new = B_new * D_inv_scalar
            
            # --- SAFETY NET: need to adjust according to the maximum speed limit ---
            self.particles.v[p] = v_new
            self.particles.C[p] = C_new
            
            # Update position
            new_x = self.particles.x[p] + v_new * config.DT
            
            # --- STRICT DOMAIN CLAMP ---
            # Don't let particles enter the padding zone!
            fluid_min_x = config.PADDING * config.DX
            fluid_max_x = (config.GRID_RES_X - config.PADDING - 1) * config.DX
            fluid_min_y = config.PADDING * config.DY
            fluid_max_y = (config.GRID_RES_Y - config.PADDING - 1) * config.DY
            
            new_x[0] = ti.max(fluid_min_x + 0.001, ti.min(new_x[0], fluid_max_x - 0.001))
            new_x[1] = ti.max(fluid_min_y + 0.001, ti.min(new_x[1], fluid_max_y - 0.001))
            
            self.particles.x[p] = new_x

    @ti.kernel
    def g2p_APIC_EBCp(self, t: float):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            
            # =========================================================
            # 1. STANDARD APIC GATHER (GRID TO PARTICLE)
            # =========================================================
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            v_new = ti.Vector.zero(ti.f32, config.DIM)
            B_new = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                v_I = self.grid.v[I]      
                dpos = (ti.cast(I, ti.f32) * config.DX) - x_p
                
                v_new += v_I * sf_weight
                B_new += sf_weight * v_I.outer_product(dpos)
            
            # Base Affine Matrix (Unconstrained)
            C_new = B_new * (4.0 * config.INV_DX ** 2)
            
            # =========================================================
            # 2. PLATFORM KINEMATICS & BOUNDING BOX
            # =========================================================
            v_platform_y = 0.0
            displacement_y = 0.0
            
            if t < config.PLATFORM_STOP_TIME:
                v_platform_y = config.PLATFORM_VELOCITY_Y
                displacement_y = v_platform_y * t
            elif t < config.PLATFORM_STOP_TIME + config.PLATFORM_DECEL_TIME:
                time_in_decel = t - config.PLATFORM_STOP_TIME
                progress = time_in_decel / config.PLATFORM_DECEL_TIME
                v_platform_y = config.PLATFORM_VELOCITY_Y * (1.0 - progress)
                dist_before = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                dist_during = config.PLATFORM_VELOCITY_Y * time_in_decel - 0.5 * (config.PLATFORM_VELOCITY_Y / config.PLATFORM_DECEL_TIME) * (time_in_decel**2)
                displacement_y = dist_before + dist_during
            else:
                dist_before = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                dist_during = 0.5 * config.PLATFORM_VELOCITY_Y * config.PLATFORM_DECEL_TIME
                displacement_y = dist_before + dist_during
                
            box_xmin = config.INT_MOVINGRECT_XMIN
            box_xmax = config.INT_MOVINGRECT_XMAX
            box_ymin = config.INT_MOVINGRECT_YMIN + displacement_y
            box_ymax = config.INT_MOVINGRECT_YMAX + displacement_y
            
            # =========================================================
            # 3. NITSCHE'S KINEMATIC PROJECTION
            # =========================================================
            r, n = bnd.Get_Rect_SDF(x_p, box_xmin, box_xmax, box_ymin, box_ymax)
            
            # Check if particle is inside the regularization zone
            if r < config.DECAY_ZONE:
                v_target = ti.Vector([0.0, v_platform_y])
                v_rel = v_new - v_target
                
                # Check for penetration (Fluid moving INTO the boundary)
                if v_rel.dot(n) < 0.0:
                    # Calculate Regularized Delta
                    delta_c = 1.0 if r <= 0.0 else 1.0 - ti.math.pow(r / config.DECAY_ZONE, 3.0)
                    
                    # Construct Projection Matrix (Lambda_c)
                    I_mat = ti.Matrix.identity(ti.f32, config.DIM)
                    Lambda_c = I_mat - (n.outer_product(n) * delta_c)
                    
                    # Apply constraints to BOTH Velocity and Affine Matrix
                    v_new = (Lambda_c @ v_new) + (delta_c * v_target.dot(n) * n)

            # =========================================================
            # 4. FINALIZE PARTICLE UPDATE & DOMAIN CLAMPING
            # =========================================================
            self.particles.v[p] = v_new
            self.particles.C[p] = C_new
            
            new_x = self.particles.x[p] + v_new * config.DT
            
            # Strict Outer Domain Clamp
            pad_x = config.PADDING * config.DX
            pad_y = config.PADDING * config.DY
            max_x = (config.GRID_RES_X - config.PADDING - 1) * config.DX
            max_y = (config.GRID_RES_Y - config.PADDING - 1) * config.DY
            
            new_x[0] = ti.max(pad_x + 0.001, ti.min(new_x[0], max_x - 0.001))
            new_x[1] = ti.max(pad_y + 0.001, ti.min(new_x[1], max_y - 0.001))
            
            self.particles.x[p] = new_x

    @ti.kernel
    def g2p_APIC_EBCpos(self, t: float):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            
            # =========================================================
            # 1. STANDARD APIC GATHER (No velocity/affine tampering!)
            # =========================================================
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            v_new = ti.Vector.zero(ti.f32, config.DIM)
            B_new = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1]
                
                v_I = self.grid.v[I]      
                dpos = (ti.cast(I, ti.f32) * config.DX) - x_p
                
                v_new += v_I * sf_weight
                B_new += sf_weight * v_I.outer_product(dpos)
            
            C_new = B_new * (4.0 * config.INV_DX ** 2)
            
            # Commit the UNMODIFIED physics back to the particle
            self.particles.v[p] = v_new
            self.particles.C[p] = C_new
            
            # =========================================================
            # 2. PROVISIONAL POSITION UPDATE
            # =========================================================
            new_x = x_p + v_new * config.DT
            
            # =========================================================
            # 3. PLATFORM KINEMATICS & BOUNDING BOX
            # =========================================================
            displacement_y = 0.0
            if t < config.PLATFORM_STOP_TIME:
                displacement_y = config.PLATFORM_VELOCITY_Y * t
            elif t < config.PLATFORM_STOP_TIME + config.PLATFORM_DECEL_TIME:
                time_in_decel = t - config.PLATFORM_STOP_TIME
                dist_before = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                dist_during = config.PLATFORM_VELOCITY_Y * time_in_decel - 0.5 * (config.PLATFORM_VELOCITY_Y / config.PLATFORM_DECEL_TIME) * (time_in_decel**2)
                displacement_y = dist_before + dist_during
            else:
                dist_before = config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                dist_during = 0.5 * config.PLATFORM_VELOCITY_Y * config.PLATFORM_DECEL_TIME
                displacement_y = dist_before + dist_during
                
            box_xmin = config.INT_MOVINGRECT_XMIN
            box_xmax = config.INT_MOVINGRECT_XMAX
            box_ymin = config.INT_MOVINGRECT_YMIN + displacement_y
            box_ymax = config.INT_MOVINGRECT_YMAX + displacement_y
            
            # =========================================================
            # 4. KINEMATIC POSITION CLAMP (The "Bumper")
            # =========================================================
            # Check the SDF against the NEW predicted position
            r, n = bnd.Get_Rect_SDF(new_x, box_xmin, box_xmax, box_ymin, box_ymax)
            
            # If r < 0, the particle has penetrated the boundary due to numerical leakage
            if r < 0.0:
                # Push it exactly back to the surface along the normal, 
                # plus a tiny epsilon to prevent it from getting mathematically trapped on the line.
                epsilon = 1e-5
                new_x = new_x + (ti.abs(r) + epsilon) * n

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
    def g2p_APIC_BndProjection_3d(self, t: float):
        for p in range(self.particles.n_particles):
            x_p = self.particles.x[p]
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            v_new = ti.Vector.zero(ti.f32, config.DIM)
            B_new = ti.Matrix.zero(ti.f32, config.DIM, config.DIM)
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1] * w[k][2]
                
                v_I = self.grid.v[I]      
                dpos = (ti.cast(I, ti.f32) * config.DX) - x_p
                v_new += v_I * sf_weight
                B_new += sf_weight * v_I.outer_product(dpos)
            
            C_new = B_new * (4.0 * config.INV_DX ** 2)
            
            # --- Kinematic 3D SDF Particle Corrections ---
            r, n = 1e6, ti.Vector([0.0, 0.0, 0.0])
            if ti.static(config.ACTIVE_SCENARIO == "IMMERSED"):
                displacement_y = config.PLATFORM_VELOCITY_Y * t if t < config.PLATFORM_STOP_TIME else config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                r, n = bnd.Get_Box_SDF(
                    x_p, 
                    config.INT_MOVINGRECT_XMIN, config.INT_MOVINGRECT_XMAX, 
                    config.INT_MOVINGRECT_YMIN + displacement_y, config.INT_MOVINGRECT_YMAX + displacement_y,
                    config.INT_MOVINGRECT_ZMIN, config.INT_MOVINGRECT_ZMAX
                )

            if r < config.DECAY_ZONE:
                v_target = ti.Vector([0.0, 0.0, 0.0])
                if ti.static(config.ACTIVE_SCENARIO == "IMMERSED") and t < config.PLATFORM_STOP_TIME:
                    v_target[1] = config.PLATFORM_VELOCITY_Y
                if (v_new - v_target).dot(n) < 0.0:
                    # delta_c = 1.0
                    delta_c = 1.0 if r <= 0.0 else 1.0 - ti.math.pow(r / config.DECAY_ZONE, 3.0)
                    v_new = ( (ti.Matrix.identity(ti.f32, config.DIM) - n.outer_product(n)*delta_c) @ v_new ) + (delta_c * v_target.dot(n) * n)

            self.particles.v[p] = v_new
            self.particles.C[p] = C_new
            
            # Predictive positioning and bounding bumpers
            new_x = x_p + v_new * config.DT
            r_new, n_new = 1e6, ti.Vector([0.0, 0.0, 0.0])
            if ti.static(config.ACTIVE_SCENARIO == "IMMERSED"):
                displacement_y = config.PLATFORM_VELOCITY_Y * t if t < config.PLATFORM_STOP_TIME else config.PLATFORM_VELOCITY_Y * config.PLATFORM_STOP_TIME
                r_new, n_new = bnd.Get_Box_SDF(
                    new_x, 
                    config.INT_MOVINGRECT_XMIN, config.INT_MOVINGRECT_XMAX, 
                    config.INT_MOVINGRECT_YMIN + displacement_y, config.INT_MOVINGRECT_YMAX + displacement_y,
                    config.INT_MOVINGRECT_ZMIN, config.INT_MOVINGRECT_ZMAX
                )
                
            if r_new < 0.0:
                new_x += (ti.abs(r_new) + 1e-5) * n_new

            # 3D Global Clamping Boundary
            eps = 0.01 * config.DX

            pad_x, pad_y, pad_z = config.PADDING * config.DX, config.PADDING * config.DY, config.PADDING * config.DZ
            max_x = (config.GRID_RES_X - config.PADDING - 1) * config.DX
            max_y = (config.GRID_RES_Y - config.PADDING - 1) * config.DY
            max_z = (config.GRID_RES_Z - config.PADDING - 1) * config.DZ
            
            new_x[0] = ti.max(pad_x + eps, ti.min(new_x[0], max_x - eps))
            new_x[1] = ti.max(pad_y + eps, ti.min(new_x[1], max_y - eps))
            new_x[2] = ti.max(pad_z + eps, ti.min(new_x[2], max_z - eps))
            
            self.particles.x[p] = new_x
    
    @ti.kernel
    def check_particle_health(self) -> int:
        healthy_count = 0
        
        # Define the exact bounds from your clamp
        fluid_min_x = config.PADDING * config.DX + 0.001
        fluid_max_x = (config.GRID_RES_X - config.PADDING - 1) * config.DX - 0.001
        fluid_min_y = config.PADDING * config.DY + 0.001
        fluid_max_y = (config.GRID_RES_Y - config.PADDING - 1) * config.DY - 0.001
        
        for p in range(self.particles.n_particles):
            x = self.particles.x[p]
            v = self.particles.v[p]
            
            # 1. Check for NaNs or Infinity
            is_nan = ti.math.isnan(x[0]) or ti.math.isnan(x[1]) or ti.math.isnan(v[0]) or ti.math.isnan(v[1])
            is_inf = ti.math.isinf(x[0]) or ti.math.isinf(x[1]) or ti.math.isinf(v[0]) or ti.math.isinf(v[1])
            
            if not is_nan and not is_inf:
                healthy_count += 1                   
                    
        return healthy_count
    
    @ti.kernel
    def update_kinematics(self):
        """Step 1: Standard incremental update of the deformation gradient F"""
        for p in range(self.particles.n_particles):
            F_p = self.particles.F[p]
            grad_v = self.particles.C[p] 
            identity = ti.Matrix.identity(ti.f32, config.DIM)
            self.particles.F[p] = (identity + grad_v * config.DT) @ F_p

    @ti.kernel
    def update_kinematics_and_nodal_Fbar_P2G_3d(self):
        for p in range(self.particles.n_particles):
            F_p = self.particles.F[p]
            grad_v = self.particles.C[p] 
            identity = ti.Matrix.identity(ti.f32, config.DIM)
            F_new = (identity + grad_v * config.DT) @ F_p
            self.particles.F[p] = F_new

            x_p = self.particles.x[p]
            J_p = ti.max(F_new.determinant(), 0.1)
            
            v_init = config.P_VOL
            v_curr = v_init * J_p
            
            base, _, w_0, w_1, w_2, _, _, _ = cf.GetSF_QuadBspline(x_p, config.INV_DX)
            w = [w_0, w_1, w_2]
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                I = ti.cast(base + offset, ti.i32)
                sf_weight = w[i][0] * w[j][1] * w[k][2]
                
                self.grid.vol_init[I] += sf_weight * v_init
                self.grid.vol_curr[I] += sf_weight * v_curr

    def step(self, damping=1.0, current_time=0.0):
        if config.DIM == 3:
            self.step_3d(damping, current_time)
        else:
            self.step_2d(damping, current_time)

    def step_2d(self, damping=1.0, current_time=0.0):
        self.grid.clear()
        self.p2g_APIC()
        self.compute_forces()
        if config.ACTIVE_SCENARIO == "DAM_BREAK":
            if config.IS_DAMBREAK_WITH_OBSTACLE:
                self.ComputeForces_EBC()
        elif config.ACTIVE_SCENARIO == "IMMERSED":
            self.ComputeForces_MovingEBC_smoothdec(current_time) # for moving BC smooth dec
            # self.ComputeForces_MovingEBC(current_time) # for moving BC

        self.grid_update(damping)
        self.apply_boundaries() # free slip for outer boundary
    
        if config.ACTIVE_SCENARIO == "IMMERSED":
            self.g2p_APIC_EBCpos(current_time)
            # self.g2p_APIC_EBCp(current_time) # developed for particle EBC: 07052026
        else:
            self.g2p_APIC()
        
        # Nodal-averaged F-bar method | 04052026
        self.update_kinematics()
        self.nodal_Fbar_P2G()
        self.nodal_Fbar_G2P_and_stress()

    def step_3d(self, damping=1.0, current_time=0.0):
        self.grid.clear()
        self.p2g_APIC_3d()
        
        self.compute_forces_3d()
        self.ComputeForces_EBC_Scenarios_3d(current_time)
        
        self.grid_update_3d(damping)
        self.apply_boundaries_3d()
        
        self.g2p_APIC_BndProjection_3d(current_time)
        
        self.update_kinematics_and_nodal_Fbar_P2G_3d()
        self.nodal_Fbar_G2P_and_stress_3d()
