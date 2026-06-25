import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taichi as ti
import config

# 1. Initialize as None
grid_normals = None

def init_boundary_fields():
    """Call this function AFTER ti.init() has been run."""
    global grid_normals
    grid_normals = ti.Vector.field(
        2, 
        dtype=ti.f32, 
        shape=(config.GRID_RES_X, config.GRID_RES_Y)
    )

@ti.kernel
def apply_grid_boundary_conditions(grid_v: ti.template()): # type: ignore
    for i, j in grid_v:
        # Left and Right
        if i < config.PADDING and grid_v[i, j][0] < 0.0:
            grid_v[i, j][0] = 0.0  
        if i >= config.GRID_RES_X - config.PADDING and grid_v[i, j][0] > 0.0:
            grid_v[i, j][0] = 0.0  

        # Bottom and Top
        if j < config.PADDING and grid_v[i, j][1] < 0.0:
            grid_v[i, j][1] = 0.0  
        if j >= config.GRID_RES_Y - config.PADDING and grid_v[i, j][1] > 0.0:
            grid_v[i, j][1] = 0.0

@ti.func
def Compute_EBC_Force(m_I, v_I, f_net_I, r, n, bc_type, v_target):
    """
    Computes the Enhanced Boundary Condition (EBC) force for a grid node.
    bc_type: 0 = Slip, 1 = No-Slip, 2 = Velocity
    n: Normal vector pointing FROM the solid INTO the fluid.
    r: Signed distance from the node to the solid surface (Negative = inside solid).
    """
    f_bc = ti.Vector.zero(ti.f32, config.DIM)
    
    if m_I > 1e-8:
        # 1. Calculate Decay Zone Thickness (h) based on grid resolution and normal
        # h = ti.math.sqrt((config.DX * n[0])**2 + (config.DY * n[1])**2)
        h = config.DECAY_ZONE # match with PEBC m=3
        
        # 2. Calculate Spatial Influence Factor (gamma_SF)
        gamma_SF = 0.0
        if r <= 0.0:
            # Constrained Zone (Inside boundary)
            gamma_SF = 1.0
        elif r < h:
            # Decay Zone
            gamma_SF = (1.0 - r / h)**3 # match with PEBC m=3
            
        # 3. Compute Required Force based on BC Type
        if gamma_SF > 0.0:
            
            if bc_type == 1:
                # ---------------------------------------------
                # NO-SLIP BOUNDARY (Fixed)
                # ---------------------------------------------
                # delta(mv) = -m * v_n
                f_bc_raw = (-m_I * v_I) / config.DT - f_net_I
                f_bc = gamma_SF * f_bc_raw
                
            elif bc_type == 0:
                # ---------------------------------------------
                # SLIP-WALL BOUNDARY
                # ---------------------------------------------
                # Check predicted normal momentum
                mv_predicted = f_net_I * config.DT + m_I * v_I
                p_normal = mv_predicted.dot(n)
                
                # If p_normal < 0, it means it's moving against the normal (into the wall)
                if p_normal < 0.0:
                    # Cancel out the normal internal/external forces and normal momentum
                    f_bc_scalar = (-f_net_I - (m_I * v_I) / config.DT).dot(n)
                    f_bc_raw = f_bc_scalar * n
                    f_bc = gamma_SF * f_bc_raw
                    
            elif bc_type == 2:
                # ---------------------------------------------
                # VELOCITY BOUNDARY (Constrained Normal)
                # ---------------------------------------------
                target_p = m_I * v_target
                current_p = m_I * v_I
                
                # delta(mv)_normal = (target_p - current_p \dot n) * n
                delta_p_normal = (target_p - current_p.dot(n) * n)
                
                # ( -f_net \dot N ) + delta_p / dt
                f_bc_scalar = (-f_net_I).dot(n)
                f_bc_raw = f_bc_scalar * n + delta_p_normal / config.DT
                f_bc = gamma_SF * f_bc_raw
            
            elif bc_type == 3:
                # ---------------------------------------------
                # GENERALIZED SLIP-WALL BOUNDARY
                # ---------------------------------------------
                # 1. Predict momentum without boundary forces
                mv_predicted = f_net_I * config.DT + m_I * v_I
                
                # 2. Target momentum of the rigid body
                p_wall = m_I * v_target
                
                # 3. Calculate relative normal momentum
                # Negative means the fluid is moving INTO the wall faster than the wall is moving away
                p_rel_normal = (mv_predicted - p_wall).dot(n)
                
                if p_rel_normal < 0.0:
                    # Cancel out ONLY the penetrating relative normal momentum
                    # This forces the fluid's normal velocity to match the wall's normal velocity
                    f_bc_scalar = (-f_net_I - (m_I * v_I) / config.DT).dot(n) + p_wall.dot(n) / config.DT
                    f_bc_raw = f_bc_scalar * n
                    f_bc = gamma_SF * f_bc_raw
                
    return f_bc

@ti.func
def Get_Rect_SDF(x_I, box_xmin, box_xmax, box_ymin, box_ymax):
    """
    Returns the signed distance 'r' and outward normal 'n' for a rectangle.
    Normal points OUT of the rectangle (INTO the fluid).
    r < 0 means the point is inside the rectangle.
    """
    # Vector from center to point
    cx = (box_xmin + box_xmax) * 0.5
    cy = (box_ymin + box_ymax) * 0.5
    
    # Half extents
    hx = (box_xmax - box_xmin) * 0.5
    hy = (box_ymax - box_ymin) * 0.5
    
    dx = ti.abs(x_I[0] - cx) - hx
    dy = ti.abs(x_I[1] - cy) - hy
    
    # Signed distance
    dist_outside = ti.math.sqrt(ti.max(dx, 0.0)**2 + ti.max(dy, 0.0)**2)
    dist_inside = ti.min(ti.max(dx, dy), 0.0)
    r = dist_outside + dist_inside
    
    # Normal calculation
    n = ti.Vector([0.0, 0.0])
    
    if r > 0.0:
        # Outside: normal points from nearest edge to the point
        n_x = ti.max(dx, 0.0) * (1.0 if x_I[0] > cx else -1.0)
        n_y = ti.max(dy, 0.0) * (1.0 if x_I[1] > cy else -1.0)
        length = ti.math.sqrt(n_x**2 + n_y**2)
        if length > 1e-8:
            n = ti.Vector([n_x / length, n_y / length])
    else:
        # Inside: normal points toward the closest edge
        if dx > dy:
            n = ti.Vector([1.0 if x_I[0] > cx else -1.0, 0.0])
        else:
            n = ti.Vector([0.0, 1.0 if x_I[1] > cy else -1.0])
            
    return r, n

@ti.func
def Get_Circle_SDF(x_I, cx, cy, radius):
    """
    SDF for a circular obstacle (a curve).
    Returns distance r and outward normal n for a given grid node x_I.
    """
    dx = x_I[0] - cx
    dy = x_I[1] - cy
    dist_to_center = ti.math.sqrt(dx**2 + dy**2)
    
    # 1. Signed Distance (r < 0 means the grid node is INSIDE the circle)
    r = dist_to_center - radius
    
    # 2. Outward Normal (pointing away from the circle into the fluid)
    n = ti.Vector([0.0, 0.0])
    
    # Prevent division by zero if a grid node is exactly at the circle's center
    if dist_to_center > 1e-8:
        n = ti.Vector([dx / dist_to_center, dy / dist_to_center])
    else:
        n = ti.Vector([1.0, 0.0]) # Arbitrary fallback
        
    return r, n

@ti.func
def Get_Box_SDF(x_I, box_xmin, box_xmax, box_ymin, box_ymax, box_zmin, box_zmax):
    """3D Signed Distance Field for a Box (Immersed Platform)"""
    cx = (box_xmin + box_xmax) * 0.5
    cy = (box_ymin + box_ymax) * 0.5
    cz = (box_zmin + box_zmax) * 0.5
    
    hx = (box_xmax - box_xmin) * 0.5
    hy = (box_ymax - box_ymin) * 0.5
    hz = (box_zmax - box_zmin) * 0.5
    
    dx = ti.abs(x_I[0] - cx) - hx
    dy = ti.abs(x_I[1] - cy) - hy
    dz = ti.abs(x_I[2] - cz) - hz
    
    dist_outside = ti.math.sqrt(ti.max(dx, 0.0)**2 + ti.max(dy, 0.0)**2 + ti.max(dz, 0.0)**2)
    dist_inside = ti.min(ti.max(dx, ti.max(dy, dz)), 0.0)
    r = dist_outside + dist_inside
    
    n = ti.Vector([0.0, 0.0, 0.0])
    if r > 0.0:
        n_x = ti.max(dx, 0.0) * (1.0 if x_I[0] > cx else -1.0)
        n_y = ti.max(dy, 0.0) * (1.0 if x_I[1] > cy else -1.0)
        n_z = ti.max(dz, 0.0) * (1.0 if x_I[2] > cz else -1.0)
        length = ti.math.sqrt(n_x**2 + n_y**2 + n_z**2)
        if length > 1e-8:
            n = ti.Vector([n_x / length, n_y / length, n_z / length])
    else:
        if dx > dy and dx > dz:
            n = ti.Vector([1.0 if x_I[0] > cx else -1.0, 0.0, 0.0])
        elif dy > dz:
            n = ti.Vector([0.0, 1.0 if x_I[1] > cy else -1.0, 0.0])
        else:
            n = ti.Vector([0.0, 0.0, 1.0 if x_I[2] > cz else -1.0])
            
    return r, n

@ti.func
def sdLineSegment_Abs(p, a, b, normal_dir):
    """Calculates absolute distance and geometric vectors without premature sign flipping."""
    pa = p - a
    ba = b - a
    ba_len = ti.max(ba.norm(), 1e-10)
    
    # 1. Calculate the fixed geometric normal of the wall
    n_geom = ti.Vector([-ba[1] * normal_dir, ba[0] * normal_dir]) / ba_len
    
    # 2. Project point onto the line segment
    h = ti.math.clamp(pa.dot(ba) / (ba_len * ba_len), 0.0, 1.0)
    closest_point = a + ba * h
    
    # 3. Vector from the closest point to the particle
    v = p - closest_point
    dist = v.norm()
    
    return dist, v, n_geom

@ti.func
def check_segment(p, a, b, current_min_d, current_v, current_n_geom, normal_dir):
    """Tracks the absolute closest geometric distance to any wall."""
    d, v, n_geom = sdLineSegment_Abs(p, a, b, normal_dir)
    
    if d < current_min_d:
        current_min_d = d
        current_v = v
        current_n_geom = n_geom
        
    return current_min_d, current_v, current_n_geom

@ti.func
def Get_NTU_Sketch_SDF(x_I):
    min_d = 1e6
    best_v = ti.Vector([0.0, 0.0])
    best_n = ti.Vector([0.0, 0.0])
    
    OX = 0.375 
    OY = 0.8
    tip_offset = 0.02

    # =========================================================
    # 1. THE LOWER WALL (Fluid is on the Left -> normal_dir = 1.0)
    # =========================================================
    p0  = ti.Vector([0.10, OY + 0.40]) 
    p1  = ti.Vector([OX + 0.00, OY + 0.40]) 
    p2  = ti.Vector([OX + 0.00, OY + 0.00]) 
    p3  = ti.Vector([OX + 0.13, OY + 0.00]) 
    # p4  = ti.Vector([OX + 0.13, OY + 0.27])
    p4_left  = ti.Vector([OX + 0.13 - tip_offset, OY + 0.27]) 
    p4_right = ti.Vector([OX + 0.13 + tip_offset, OY + 0.27])
    p5  = ti.Vector([OX + 0.33, OY + 0.00]) 
    p6  = ti.Vector([OX + 0.46, OY + 0.00]) 
    p7  = ti.Vector([OX + 0.46, OY + 0.35]) 
    p8  = ti.Vector([OX + 0.61, OY + 0.35]) 
    p9  = ti.Vector([OX + 0.61, OY + 0.00]) 
    p10 = ti.Vector([OX + 0.76, OY + 0.00]) 
    p11 = ti.Vector([OX + 0.76, OY + 0.35]) 
    p12 = ti.Vector([OX + 0.91, OY + 0.35]) 
    p13 = ti.Vector([OX + 0.91, OY + 0.00]) 
    p14 = ti.Vector([OX + 1.25, OY + 0.00]) 
    p15 = ti.Vector([OX + 1.25, OY + 0.40]) 
    p16 = ti.Vector([OX + 1.45, OY + 0.40]) 

    # FIXED: Unpacking 3 variables
    min_d, best_v, best_n = check_segment(x_I, p0, p1, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p1, p2, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p2, p3, min_d, best_v, best_n, 1.0)
    
    # min_d, best_v, best_n = check_segment(x_I, p3, p4, min_d, best_v, best_n, 1.0)
    # min_d, best_v, best_n = check_segment(x_I, p4, p5, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p3, p4_left, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p4_left, p4_right, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p4_right, p5, min_d, best_v, best_n, 1.0)
    
    min_d, best_v, best_n = check_segment(x_I, p5, p6, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p6, p7, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p7, p8, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p8, p9, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p9, p10, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p10, p11, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p11, p12, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p12, p13, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p13, p14, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p14, p15, min_d, best_v, best_n, 1.0)
    min_d, best_v, best_n = check_segment(x_I, p15, p16, min_d, best_v, best_n, 1.0)

    # =========================================================
    # 2. THE UPPER WALL (Fluid is on the Right -> normal_dir = -1.0)
    # =========================================================
    t0 = ti.Vector([0.30, OY + 0.55]) 
    t1 = ti.Vector([OX + 0.13, OY + 0.55]) 
    # t2 = ti.Vector([OX + 0.33, OY + 0.30])
    t2_left  = ti.Vector([OX + 0.33 - tip_offset, OY + 0.30]) 
    t2_right = ti.Vector([OX + 0.33 + tip_offset, OY + 0.30])
    t3 = ti.Vector([OX + 0.33, OY + 0.50]) 
    t4 = ti.Vector([OX + 1.03, OY + 0.50]) 
    t5 = ti.Vector([OX + 1.03, OY + 0.10]) 
    t6 = ti.Vector([OX + 1.13, OY + 0.10]) 
    t7 = ti.Vector([OX + 1.13, OY + 0.55]) 
    t8 = ti.Vector([OX + 1.45, OY + 0.55]) 

    min_d, best_v, best_n = check_segment(x_I, t0, t1, min_d, best_v, best_n, -1.0)
    # min_d, best_v, best_n = check_segment(x_I, t1, t2, min_d, best_v, best_n, -1.0)
    # min_d, best_v, best_n = check_segment(x_I, t2, t3, min_d, best_v, best_n, -1.0)
    min_d, best_v, best_n = check_segment(x_I, t1, t2_left, min_d, best_v, best_n, -1.0)
    min_d, best_v, best_n = check_segment(x_I, t2_left, t2_right, min_d, best_v, best_n, -1.0)
    min_d, best_v, best_n = check_segment(x_I, t2_right, t3, min_d, best_v, best_n, -1.0)
    
    min_d, best_v, best_n = check_segment(x_I, t3, t4, min_d, best_v, best_n, -1.0)
    min_d, best_v, best_n = check_segment(x_I, t4, t5, min_d, best_v, best_n, -1.0)
    min_d, best_v, best_n = check_segment(x_I, t5, t6, min_d, best_v, best_n, -1.0)
    min_d, best_v, best_n = check_segment(x_I, t6, t7, min_d, best_v, best_n, -1.0)
    min_d, best_v, best_n = check_segment(x_I, t7, t8, min_d, best_v, best_n, -1.0)

    return min_d, best_n

@ti.kernel
def compute_all_normals():
    for I in ti.grouped(grid_normals):
        # Convert grid index to world space
        x_I = ti.Vector([ti.cast(I[0], ti.f32) * config.DX, 
                         ti.cast(I[1], ti.f32) * config.DY])
        
        # Calculate normal using your existing function
        r, n = Get_NTU_Sketch_SDF(x_I)
        
        # Store it (only keep it if we are near the wall to save space)
        if ti.abs(r) < 2.0 * config.DX:
            grid_normals[I] = n
        else:
            grid_normals[I] = ti.Vector([0.0, 0.0])