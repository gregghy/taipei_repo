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
