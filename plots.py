import matplotlib
matplotlib.use('TkAgg') # Or 'Agg', 'Qt5Agg', etc.
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
# Function to set up a consistent plot appearance
def setup_plot(ax, title, xlim=(-6, 6), ylim=(-6, 6)):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color='black', lw=0.75)
    ax.axvline(0, color='black', lw=0.75)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")

# Create the figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Illustrations of Constraint Geometries", fontsize=16)

# --- Plot 1: Two L2 Norm Constraints (Concentric Circles) ---
ax1 = axs[0]
circle_large_radius = 4.0
circle_small_radius = 3.0

# Large L2 norm constraint
circle_large = patches.Circle((0, 0), circle_large_radius, edgecolor='blue', facecolor='none', lw=2, label=f'L2 Norm Constraint (radius={circle_large_radius})')
ax1.add_patch(circle_large)

# Small L2 norm constraint
circle_small = patches.Circle((0, 0), circle_small_radius, edgecolor='red', facecolor='none', lw=2, label=f'L2 Norm Constraint (radius={circle_small_radius})')
ax1.add_patch(circle_small)

setup_plot(ax1, "Fig 1: Two l2 constraints (small/large)")
ax1.legend(fontsize=8)

# --- Plot 2: L1 Norm Inscribed in L2 Norm Constraint ---
ax2 = axs[1]
l2_radius_fig2 = 4.0
l1_radius_fig2 = l2_radius_fig2 # For a square inscribed touching the circle at axes

# L2 norm constraint (circle)
circle_fig2 = patches.Circle((0, 0), l2_radius_fig2, edgecolor='blue', facecolor='none', lw=2, label=f'L2 Norm Constraint (radius={l2_radius_fig2})')
ax2.add_patch(circle_fig2)

# L1 norm constraint (diamond/square)
# Vertices for an L1 ball with "radius" r are (r,0), (0,r), (-r,0), (0,-r)
l1_vertices = [
    [l1_radius_fig2, 0],
    [0, l1_radius_fig2],
    [-l1_radius_fig2, 0],
    [0, -l1_radius_fig2]
]
l1_diamond = patches.Polygon(l1_vertices, closed=True, edgecolor='green', facecolor='none', lw=2, label=f'L1 Norm Constraint (vertices at ±{l1_radius_fig2})')
ax2.add_patch(l1_diamond)

setup_plot(ax2, "Fig 2: Transition from L2 to L1 Norm Constraint")
ax2.legend(fontsize=8)


# --- Plot 3: Union of L2 Constraints within a larger L1 Boundary (Non-convex example) ---
ax3 = axs[2]

# Define a larger L1 boundary (diamond shape)
l1_outer_radius = 5.0
outer_l1_vertices = [
    [l1_outer_radius, 0],
    [0, l1_outer_radius],
    [-l1_outer_radius, 0],
    [0, -l1_outer_radius]
]
outer_l1_boundary = patches.Polygon(outer_l1_vertices, closed=True, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.2, lw=1, label=f'Overall L1 Boundary (vertices at ±{l1_outer_radius})')
ax3.add_patch(outer_l1_boundary)


# First L2 constraint (circle 1)
center1 = (-1.5, -1)
radius1 = 1.8
circle1_fig3 = patches.Circle(center1, radius1, edgecolor='blue', facecolor='lightblue', lw=2, alpha=0.7, label=f'L2 Constraint 1 (center=({center1[0]},{center1[1]}), r={radius1})')
ax3.add_patch(circle1_fig3)

# Second L2 constraint (circle 2)
center2 = (1.5, 2)
radius2 = 1.5
circle2_fig3 = patches.Circle(center2, radius2, edgecolor='blue', facecolor='lightblue', lw=2, alpha=0.7, label=f'L2 Constraint 2 (center=({center2[0]},{center2[1]}), r={radius2})')
ax3.add_patch(circle2_fig3)



setup_plot(ax3, "Fig 3: Transition from L1 Norm Constraint to Union of Two L2 Norm Constraints")
ax3.legend(fontsize=8, loc='lower right')


plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.show()