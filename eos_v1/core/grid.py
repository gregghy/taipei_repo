# core/grid.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taichi as ti
import config

@ti.data_oriented
class Grid:
    def __init__(self):
        # 1. Dynamically set the grid resolution tuple based on dimension
        if config.DIM == 3:
            self.res = (config.GRID_RES_X, config.GRID_RES_Y, config.GRID_RES_Z)
        else:
            self.res = (config.GRID_RES_X, config.GRID_RES_Y)
        
        # ==========================================
        # Taichi Field Allocations (Strictly f32)
        # ==========================================
        self.v = ti.Vector.field(config.DIM, dtype=ti.f32, shape=self.res)
        self.m = ti.field(dtype=ti.f32, shape=self.res)
        self.f = ti.Vector.field(config.DIM, dtype=ti.f32, shape=self.res)
        self.v_old = ti.Vector.field(config.DIM, dtype=ti.f32, shape=self.res)
        
        # FOR F-BAR Method
        self.cell_V_current = ti.field(dtype=ti.f32, shape=self.res)
        self.cell_V_initial = ti.field(dtype=ti.f32, shape=self.res)
        
        # For Nodal F-bar method
        self.vol_init = ti.field(dtype=ti.f32, shape=self.res)
        self.vol_curr = ti.field(dtype=ti.f32, shape=self.res)

    @ti.kernel
    def clear(self):
        """
        Wipes the grid clean at the start of every time step.
        ti.grouped automatically handles 2D or 3D looping!
        """
        for I in ti.grouped(self.m):
            self.m[I] = 0.0
            self.v[I] = ti.Vector.zero(ti.f32, config.DIM)
            self.f[I] = ti.Vector.zero(ti.f32, config.DIM)
            self.v_old[I] = ti.Vector.zero(ti.f32, config.DIM)
            
            self.cell_V_current[I] = 0.0
            self.cell_V_initial[I] = 0.0
            self.vol_init[I] = 0.0
            self.vol_curr[I] = 0.0