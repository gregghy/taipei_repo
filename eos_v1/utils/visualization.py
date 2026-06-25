import sys
import os

# Add the parent directory (mpm_project root) to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import config

def visualize_initial_state():
    # 1. Calculate Particle Spacing
    # Distance between each particle
    spacing_x = config.DX / config.P_PER_CELL_AXIS
    spacing_y = config.DY / config.P_PER_CELL_AXIS

    # 2. Generate Particle Coordinates
    px = []
    py = []
    for i in range(config.NUM_MP_WIDTH):
        for j in range(config.NUM_MP_HEIGHT):
            # We add 0.5 * spacing so the particle sits in the center of its sub-cell,
            # rather than directly on the grid lines.
            x = config.POS_MP_LEFT_BOTTOM[0] + (i + 0.5) * spacing_x
            y = config.POS_MP_LEFT_BOTTOM[1] + (j + 0.5) * spacing_y
            px.append(x)
            py.append(y)

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    
    # Calculate the bounds of the computational grid (Starts at exactly 0.0 in Taichi)
    x_min_pad = 0.0
    x_max_pad = config.GRID_RES_X * config.DX
    y_min_pad = 0.0
    y_max_pad = config.GRID_RES_Y * config.DY
    
    # Calculate the bounds of the true Physical Domain (shifted by PADDING)
    phys_x_start = config.PADDING * config.DX
    phys_y_start = config.PADDING * config.DY

    # 4. Draw the Computational Grid
    for i in range(config.GRID_RES_X + 1):
        x = x_min_pad + i * config.DX
        ax.axvline(x, color='lightgray', linestyle='-', linewidth=0.3)
    for j in range(config.GRID_RES_Y + 1):
        y = y_min_pad + j * config.DY
        ax.axhline(y, color='lightgray', linestyle='-', linewidth=0.3)

    # 5. Draw the True Physical Boundary
    physical_domain = patches.Rectangle(
        (phys_x_start, phys_y_start), config.GRID_WIDTH, config.GRID_HEIGHT, 
        linewidth=2, edgecolor='black', facecolor='none', label='Physical Boundary'
    )
    ax.add_patch(physical_domain)

    # 6. Draw the Computational Grid Boundary
    padded_domain = patches.Rectangle(
        (x_min_pad, y_min_pad), x_max_pad, y_max_pad,
        linewidth=2, edgecolor='red', linestyle='--', facecolor='none', label='Computational Grid'
    )
    ax.add_patch(padded_domain)

    # 7. Plot Material Points
    # We use a very small marker size (s=0.5) because there are 10,000 of them
    ax.scatter(px, py, s=0.5, color='blue', label='Material Points')

    # Formatting
    ax.set_aspect('equal', 'box')
    ax.set_xlim(x_min_pad - config.DX*5, x_max_pad + config.DX*5)
    ax.set_ylim(y_min_pad - config.DY*5, y_max_pad + config.DY*5)
    ax.set_title("MPM Initial Discretization", fontsize=14)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    # Use a legend outside the plot so it doesn't cover the grid
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.tight_layout()
    plt.show()

def plot_velocity_profile(pos, velocity, title="Steady State Velocity Profile"):
    """
    Plots the X-velocity of particles against their Y-position.
    """
    y_coords = pos[:, 1]
    x_vel = velocity[:, 0]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # Scatter plot of the velocity profile
    ax.scatter(x_vel, y_coords, s=2.0, color='blue', alpha=0.5, label='MPM Particles')
    
    # Formatting
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Velocity in X-direction (u)")
    ax.set_ylabel("Y Position (Height across channel)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axvline(0, color='black', linewidth=1) # Zero velocity line
    
    plt.tight_layout()
    plt.show()