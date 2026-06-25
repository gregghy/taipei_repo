import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taichi as ti
import config

VISCOSITY = 1e-3
ALPHA_L = 0.5  # Linear coefficient: damps high-frequency acoustic ringing
ALPHA_Q = 1.0  # Quadratic coefficient: damps sharp shockwaves (like boundary impacts)
        

@ti.func
def StressUsingWater(F, C): 
    # Use J_bar to update the pressure
    J = ti.max(F.determinant(), 0.96) # clamped means its max compressed ratio
    K = config.C_0**2 * config.RHO_0
    p_physics = K * (1.0 / J - 1.0)
    
    if p_physics < 0.0: p_physics = 0.0
    
    # 2. Viscous Shear Stress
    strain_rate = 0.5 * (C + C.transpose())
    trace_D = strain_rate[0, 0] + strain_rate[1, 1]
    identity = ti.Matrix.identity(ti.f32, config.DIM)
    viscous_stress = 2.0 * VISCOSITY * (strain_rate - 0.5 * trace_D * identity)
    
    # 3. Return Cauchy Stress using the Stable Physics Pressure
    stress = -p_physics * identity + viscous_stress
    # stress = -p_particle * identity + viscous_stress
    return stress

@ti.func
def StressUsingWater_WithArtificialViscosity(F, C): 
    # 1. Standard EoS Physics Pressure
    J = ti.max(F.determinant(), 0.95) # clamped to 0.95
    K = config.C_0**2 * config.RHO_0
    p_physics = K * (1.0 / J - 1.0)
    
    if p_physics < 0.0: 
        p_physics = 0.0
    
    # 2. Kinematics
    strain_rate = 0.5 * (C + C.transpose())
    # The trace of the strain rate is the divergence of velocity (rate of volume change)
    div_v = strain_rate[0, 0] + strain_rate[1, 1] 
    identity = ti.Matrix.identity(ti.f32, config.DIM)
    
    # ========================================================
    # 3. ARTIFICIAL VISCOSITY (The "Smart" Shock Absorber)
    # ========================================================
    q_art = 0.0
    if div_v < 0.0:  # ONLY apply when the fluid is compressing!
        # Linear term (targets baseline acoustic ringing)
        q_linear = -ALPHA_L * config.RHO_0 * config.C_0 * config.DX * div_v
        
        # Quadratic term (targets sharp, sudden impacts from the boundary)
        q_quad = ALPHA_Q * config.RHO_0 * (config.DX * div_v)**2
        
        q_art = q_linear + q_quad
        
    # Add the artificial damping pressure to the physical pressure
    p_eff = p_physics + q_art
    # ========================================================
    
    # 4. Standard Physical Viscous Shear Stress
    viscous_stress = 2.0 * VISCOSITY * (strain_rate - 0.5 * div_v * identity)
    
    # 5. Return Cauchy Stress using the Effective Pressure
    stress = -p_eff * identity + viscous_stress
    return stress

@ti.func
def StressUsingBingham(F, C): 
    # 1. Physics Pressure from F (Stable!)
    J = ti.max(F.determinant(), 0.95)
    K = config.C_0**2 * config.RHO_0
    p_physics = K * (1.0 / J - 1.0)
    
    if p_physics < 0.0: p_physics = 0.0

    identity = ti.Matrix.identity(ti.f32, config.DIM)
    
    # 2. Kinematics & Rheology
    D = 0.5 * (C + C.transpose())
    trace_D = D[0, 0] + D[1, 1]
    D_prime = D - 0.5 * trace_D * identity
    
    D_prime_sq_sum = D_prime[0,0]**2 + D_prime[0,1]**2 + D_prime[1,0]**2 + D_prime[1,1]**2
    gamma_dot = ti.math.sqrt(2.0 * D_prime_sq_sum)
    gamma_dot = ti.max(gamma_dot, 1e-8) 
    
    exponent = ti.math.exp(-config.M_PARAM * gamma_dot)
    mu_eff = config.MU_P + (config.TAU_Y / gamma_dot) * (1.0 - exponent)
    viscous_stress = 2.0 * mu_eff * D_prime
    
    # 3. Return Cauchy Stress using the Stable Physics Pressure
    stress = -p_physics * identity + viscous_stress
    return stress