import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nu = 0.3  # Poisson's ratio
theta = np.linspace(-np.pi, np.pi, 360)  # Angle from -180 to 180 degrees in radians

# Define normalized radius formulas
# R = (pi * sigma_YS^2 / K_II^2) * r_y
R_stress = 0.25 * (7 - np.cos(theta) - 4.5 * np.sin(theta)**2)
R_strain = 0.25 * ((1 - 2*nu)**2 * (1 - np.cos(theta)) + 6 - 4.5 * np.sin(theta)**2)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(np.degrees(theta), R_stress, label='Plane Stress', linewidth=2)
plt.plot(np.degrees(theta), R_strain, '--', label='Plane Strain', linewidth=2)

# Formatting the plot
plt.xlabel('Angle $\\theta$ (Degrees)')
plt.ylabel('Normalized Radius $(\\pi\\sigma_{YS}^2 / K_{II}^2) r_y$')
plt.title('Mode II Crack Plastic Zone Size vs Angle')
plt.legend(loc='best')
plt.grid(True)
plt.xlim([-180, 180])

# Display the plot
plt.tight_layout()
plt.show()


# Parameters
nu = 0.3
theta = np.linspace(-np.pi, np.pi, 360)

# Equations
R_stress = 0.25 * (7 - np.cos(theta) - 4.5 * np.sin(theta)**2)
R_strain = 0.25 * ((1 - 2*nu)**2 * (1 - np.cos(theta)) + 6 - 4.5 * np.sin(theta)**2)

# Create a POLAR plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
ax.plot(theta, R_stress, label='Plane Stress', linewidth=2)
ax.plot(theta, R_strain, '--', label='Plane Strain', linewidth=2)

# Formatting
ax.set_title("Mode II Plastic Zone (Physical Shape)\nCrack tip is at center, crack faces left (180°)")
ax.legend(loc='best')
plt.show()