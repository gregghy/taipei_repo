import taichi as ti

@ti.func
def GetBaseGrid(x_p, inv_dx):
    fx = x_p * inv_dx
    base = ti.cast(fx - 0.5, ti.i32)
    return base

@ti.func
def GetSF_QuadBspline(x_p, inv_dx):
    # 1. Calculate the base node index (bottom-left of the 3x3 support domain)
    fx = x_p * inv_dx
    base = ti.cast(fx - 0.5, ti.i32)
    
    # 2. Calculate the fractional distance from the node centers
    d = fx - ti.cast(base, ti.f32)
    
    # 3. Quadratic B-spline Weights (Vectorized for X and Y simultaneously)
    w_0 = 0.5 * (1.5 - d)**2
    w_1 = 0.75 - (d - 1.0)**2
    w_2 = 0.5 * (d - 0.5)**2
    
    # 4. Gradients of the Weights 
    # The chain rule dictates we must multiply by inv_dx because W(x) = W(x_p / dx)
    dw_0 = (d - 1.5) * inv_dx
    dw_1 = -2.0 * (d - 1.0) * inv_dx
    dw_2 = (d - 0.5) * inv_dx
    
    return base, fx, w_0, w_1, w_2, dw_0, dw_1, dw_2