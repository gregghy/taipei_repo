# config.py
import math

# =======================================================================
# 1. SCENARIO & DIMENSION SELECTOR
# =======================================================================
ACTIVE_SCENARIO = "DAM_BREAK"
FLUID = "WATER" # Options: WATER | BINGHAM_PLASTIC
IS_DAMBREAK_WITH_OBSTACLE = False    

# THE MASTER TOGGLE: 2 for 2D, 3 for 3D
DIM = 2

USE_ADAPTIVE_MPM = False
AMR_REFERENCE_CHAMBER_HEIGHT = 0.1
AMR_REFERENCE_FINE_DX = 10e-6
AMR_REFERENCE_PROCESS_ZONE_HEIGHT = 5e-3
AMR_DOMAIN_MIN_X = 0.0
AMR_DOMAIN_MIN_Y = 0.0
AMR_DOMAIN_WIDTH = 0.02
AMR_DOMAIN_HEIGHT = AMR_REFERENCE_CHAMBER_HEIGHT
AMR_BASE_CELLS_X = 32
AMR_BASE_CELLS_Y = 160
AMR_BASE_DX = AMR_DOMAIN_WIDTH / AMR_BASE_CELLS_X
AMR_MAX_LEVEL = 6
AMR_FINE_DX = AMR_BASE_DX / (2 ** AMR_MAX_LEVEL)
AMR_USE_FINE_LEVEL_DT = True
AMR_PROCESS_ZONE_HEIGHT = AMR_REFERENCE_PROCESS_ZONE_HEIGHT
AMR_PROCESS_MARGIN = 0.05
AMR_FINE_REGION_WIDTH = 0.002
AMR_FINE_REGION_CENTER_X = 0.5 * AMR_DOMAIN_WIDTH
AMR_FINE_REGION_YMIN = 0.0
AMR_GRID_PADDING = 3
AMR_REFINEMENT_BUFFER_CELLS = 4
AMR_GHOST_BAND_CELLS = 2
AMR_PARTICLES_PER_CELL_AXIS = 2
AMR_SCATTER_TO_ANCESTORS = True
AMR_SPLIT_PARTICLES = True
AMR_PARTICLE_CAPACITY_FACTOR = 2.0
AMR_ALLOW_LEVEL_PROMOTION_WITHOUT_SPLIT = False
AMR_BOUNDARY_PENALTY_NORMAL = 1e4

# --- Gradient-based adaptive refinement -------------------------------------
# Particles with high velocity gradient |C|*dx or high |J-1| are split to
# finer levels.  Each level halves the threshold, so the finest level captures
# the sharpest gradients.  Set AMR_GRADIENT_REFINE to False to disable.
AMR_GRADIENT_REFINE = True
AMR_GRADIENT_REFINE_THRESHOLD = 0.1    # |C|*dx threshold for level 0
AMR_GRADIENT_PRESSURE_THRESHOLD = 0.05  # |J-1| threshold for level 0
# Cap the maximum level that gradient-based splitting can trigger.  This keeps
# the particle count manageable: gradient-driven particles refine up to this
# level, and finer levels are only used by the geometric refinement box.
AMR_GRADIENT_MAX_LEVEL = 2
# When >= 0, all particles start at this level regardless of quadtree leaf
# structure.  Set to 0 for gradient-driven refinement (start coarse, split
# on demand).  Set to -1 for the default (fill all leaf cells).
AMR_INITIAL_PARTICLE_LEVEL = -1
AMR_INITIAL_FLUID_XMIN = 0.0
AMR_INITIAL_FLUID_XMAX = AMR_DOMAIN_WIDTH
AMR_INITIAL_FLUID_YMIN = 0.0
AMR_INITIAL_FLUID_YMAX = AMR_DOMAIN_HEIGHT

# --- Dynamic refinement criterion -------------------------------------------
# When AMR_DYNAMIC_REFINEMENT is True, the finest patch follows a criterion
# instead of being pinned.  Options:
#   "platform"    – follow the immersed platform position (default)
#   "velocity"    – follow the mass-weighted centroid of fast-moving particles
#   "pressure"    – follow the mass-weighted centroid of high-pressure particles
#   "deformation" – follow the mass-weighted centroid of highly deformed particles
#   "combined"    – weighted union of velocity + pressure + deformation
AMR_REFINEMENT_CRITERION = "platform"
AMR_REFINEMENT_MARGIN = 0.02       # half-size of the finest box around the criterion center
AMR_REFINEMENT_VELOCITY_FRACTION = 0.05   # threshold = fraction * V_MAX_ESTIMATE
AMR_REFINEMENT_PRESSURE_FRACTION = 0.01   # threshold = fraction * RHO_0 * C_0^2
AMR_REFINEMENT_DEFORMATION_THRESHOLD = 0.01  # |J - 1| above this triggers refinement
AMR_DYNAMIC_REGRID_INTERVAL = 16   # re-evaluate criterion every N steps

# =======================================================================
# 2. GLOBAL GRID DISCRETIZATION
# =======================================================================
PADDING = 3                         

if DIM == 2:
    GRID_WIDTH = 1.0        
    GRID_HEIGHT = 1.0       
    N_CELL_WIDTH = 100                  
    N_CELL_HEIGHT = 100                 
    
    GRID_RES_X = N_CELL_WIDTH + 2 * PADDING + 1   
    GRID_RES_Y = N_CELL_HEIGHT + 2 * PADDING + 1
    
    DX = GRID_WIDTH / N_CELL_WIDTH      
    DY = GRID_HEIGHT / N_CELL_HEIGHT    
    INV_DX = 1.0 / DX                   
    INV_DY = 1.0 / DY
    
    P_PER_CELL_AXIS = 2

elif DIM == 3:
    # Safe 3D Baseline for RTX 3060 (12GB VRAM)
    GRID_WIDTH = 2.0        
    GRID_HEIGHT = 0.75 
    GRID_DEPTH = 2.0
    
    N_CELL_WIDTH = 64                  
    N_CELL_HEIGHT = 64                 
    N_CELL_DEPTH = 64
    
    GRID_RES_X = N_CELL_WIDTH + 2 * PADDING + 1   
    GRID_RES_Y = N_CELL_HEIGHT + 2 * PADDING + 1
    GRID_RES_Z = N_CELL_DEPTH + 2 * PADDING + 1
    
    DX = GRID_WIDTH / N_CELL_WIDTH      
    DY = GRID_HEIGHT / N_CELL_HEIGHT  
    DZ = GRID_DEPTH / N_CELL_DEPTH
    
    INV_DX = 1.0 / DX                   
    INV_DY = 1.0 / DY
    INV_DZ = 1.0 / DZ
    
    # 2x2x2 = 8 particles per cell (Standard for 3D)
    P_PER_CELL_AXIS = 2 

# ==========================================
# EBC (Enhanced Boundary Condition) Parameters
# ==========================================
DECAY_ZONE = 0.50 * DX       
MU_FRIC = 0.3               

# ===========================================
# INTERIOR BOUNDARY CONDITION (IMMERSED PLATFORM)
# ===========================================
# 1. Define the size of your platform
PLATFORM_WIDTH = 1.5

# 2. Find the exact mathematical center of the shifted fluid pool
FLUID_CENTER_X = (PADDING * DX) + (GRID_WIDTH / 2.0)

# 3. Dynamically center the platform boundaries
INT_MOVINGRECT_XMIN = FLUID_CENTER_X - (PLATFORM_WIDTH / 2.0)
INT_MOVINGRECT_XMAX = FLUID_CENTER_X + (PLATFORM_WIDTH / 2.0)

# --- 3D ONLY PLATFORM DEPTH ---
if DIM == 3:
    PLATFORM_DEPTH = 1.5
    FLUID_CENTER_Z = (PADDING * DZ) + (GRID_DEPTH / 2.0)
    INT_MOVINGRECT_ZMIN = FLUID_CENTER_Z - (PLATFORM_DEPTH / 2.0)
    INT_MOVINGRECT_ZMAX = FLUID_CENTER_Z + (PLATFORM_DEPTH / 2.0)
    PLATFORM_VELOCITY_Z = 0.0

# 4. Fluid Height and Platform Z-Axis Placement
MP_HEIGHT = 0.15

# Dynamically calculate the exact fluid surface height
FLUID_BOTTOM_Y = PADDING * DY
FLUID_TOP_Y = FLUID_BOTTOM_Y + MP_HEIGHT

# Place the platform exactly resting on the fluid surface
INT_MOVINGRECT_YMIN = FLUID_TOP_Y + 0.05
INT_MOVINGRECT_YMAX = FLUID_TOP_Y + 0.15 # Makes the plate 0.10 units thick

PLATFORM_VELOCITY_X = 0.0
PLATFORM_VELOCITY_Y = -0.05

PLATFORM_STOP_TIME = 3.2
PLATFORM_DECEL_TIME = 0.5

# =======================================================================
# 3. SCENARIO-SPECIFIC CONFIGURATIONS
# =======================================================================
if ACTIVE_SCENARIO in ["DAM_BREAK", "IMMERSED"]:                 
    
    if DIM == 2:
        POS_MP_LEFT_BOTTOM = [PADDING * DX, PADDING * DY]
        GRAVITY = [0.0, -9.81]
    else:
        POS_MP_LEFT_BOTTOM = [PADDING * DX, PADDING * DY, PADDING * DZ]
        GRAVITY = [0.0, -9.81, 0.0]
        MP_DEPTH = GRID_DEPTH

    if ACTIVE_SCENARIO == "DAM_BREAK":
        MP_WIDTH = GRID_WIDTH
        if DIM == 3: MP_DEPTH = GRID_DEPTH * 0.2
    else:
        MP_WIDTH = GRID_WIDTH
elif ACTIVE_SCENARIO == "ADAPTIVE_MPM":
    DIM = 2
    GRID_WIDTH = AMR_DOMAIN_WIDTH
    GRID_HEIGHT = AMR_DOMAIN_HEIGHT
    N_CELL_WIDTH = AMR_BASE_CELLS_X
    N_CELL_HEIGHT = AMR_BASE_CELLS_Y
    GRID_RES_X = N_CELL_WIDTH + 2 * PADDING + 1
    GRID_RES_Y = N_CELL_HEIGHT + 2 * PADDING + 1
    DX = AMR_BASE_DX
    DY = AMR_BASE_DX
    INV_DX = 1.0 / DX
    INV_DY = 1.0 / DY
    DECAY_ZONE = 0.50 * DX
    POS_MP_LEFT_BOTTOM = [0.0, 0.0]
    GRAVITY = [0.0, -9.81]
    MP_WIDTH = AMR_DOMAIN_WIDTH
    MP_HEIGHT = AMR_DOMAIN_HEIGHT

# =======================================================================
# 4. DERIVED PARTICLE GENERATION
# =======================================================================
NUM_MP_WIDTH = int((MP_WIDTH / DX) * P_PER_CELL_AXIS)   
NUM_MP_HEIGHT = int((MP_HEIGHT / DY) * P_PER_CELL_AXIS) 

if DIM == 2:
    TOTAL_NUM_MP = NUM_MP_WIDTH * NUM_MP_HEIGHT 
    P_VOL = (MP_WIDTH * MP_HEIGHT) / TOTAL_NUM_MP
elif DIM == 3:
    NUM_MP_DEPTH = int((MP_DEPTH / DZ) * P_PER_CELL_AXIS)
    TOTAL_NUM_MP = NUM_MP_WIDTH * NUM_MP_HEIGHT * NUM_MP_DEPTH
    P_VOL = (MP_WIDTH * MP_HEIGHT * MP_DEPTH) / TOTAL_NUM_MP

# =======================================================================
# 5. MATERIAL & FLUID PROPERTIES
# =======================================================================
RHO_0 = 1000.0
P_MASS = P_VOL * RHO_0

M_PARAM = 100 
TAU_Y = 5
MU_P = 20

# =======================================================================
# 6. TIME STEPPING (CFL & VISCOUS LIMITS)
# =======================================================================
CFL = 0.1                       
G_MAG = 9.81
H = MP_HEIGHT

C_0 = 10.0 * math.sqrt(2.0 * G_MAG * H) 
V_MAX_ESTIMATE = math.sqrt(2.0 * G_MAG * H)
MAX_WAVE_SPEED = C_0 + V_MAX_ESTIMATE
DT_ACOUSTIC = CFL * (DX / MAX_WAVE_SPEED)

if FLUID == "BINGHAM_PLASTIC":
    NU_MAX = (MU_P + M_PARAM * TAU_Y) / RHO_0
    DT_VISCOUS = 0.5 * (DX**2 / NU_MAX)
    DT_RAW = min(DT_ACOUSTIC, DT_VISCOUS)
elif FLUID == "WATER":
    DT_RAW = DT_ACOUSTIC

FPS = 60
FRAME_DT = 1.0 / FPS
SUBSTEPS = int(math.ceil(FRAME_DT / DT_RAW))
DT = FRAME_DT / SUBSTEPS
