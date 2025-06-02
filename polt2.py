import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Import Line2D for custom legend handles


# --- 定义单个 L2-norm 圆的约束函数 ---
def cfunc_l2_norm(x, y, center_x, center_y, radius):
    return np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius


# --- 定义 Log-Sum-Exp 平滑函数 ---
def smooth_union_sdf_lse(x, y, cfunc1_vals, cfunc2_vals, tau):
    A = -cfunc1_vals / tau
    B = -cfunc2_vals / tau
    max_val = np.maximum(A, B)
    # Add a small epsilon to prevent log(0) if both terms are extremely small after exp(A-max_val)
    log_sum_exp_terms = max_val + np.log(np.exp(A - max_val) + np.exp(B - max_val) + 1e-30)
    return -tau * log_sum_exp_terms


# --- 定义绘图函数 ---
def plot_constraints_and_smoothed_union(
        c1_params, c2_params, tau_values,
        x_range=(-3, 3), y_range=(-3, 3), n_points=200
):
    x_coords = np.linspace(x_range[0], x_range[1], n_points)
    y_coords = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x_coords, y_coords)

    Z1 = cfunc_l2_norm(X, Y, c1_params['center_x'], c1_params['center_y'], c1_params['radius'])
    Z2 = cfunc_l2_norm(X, Y, c2_params['center_x'], c2_params['center_y'], c2_params['radius'])

    num_taus = len(tau_values)
    fig, axes = plt.subplots(1, num_taus, figsize=(5 * num_taus, 5.5),
                             squeeze=False)  # Increased height slightly for suptitle
    fig.suptitle(f"Union of Two L2-Norm Circles and LSE Smoothing\n"
                 f"Circle 1: C=({c1_params['center_x']},{c1_params['center_y']}), R={c1_params['radius']}\n"
                 f"Circle 2: C=({c2_params['center_x']},{c2_params['center_y']}), R={c2_params['radius']}",
                 fontsize=10)

    for i, tau in enumerate(tau_values):
        ax = axes[0, i]
        Z_smooth_union = smooth_union_sdf_lse(X, Y, Z1, Z2, tau)

        # Plot原始圆的边界 (cfunc = 0)
        # These contour objects themselves don't directly take a 'label' for the legend
        ax.contour(X, Y, Z1, levels=[0], colors='blue', linestyles='dashed', linewidths=1.5)
        ax.contour(X, Y, Z2, levels=[0], colors='green', linestyles='dashed', linewidths=1.5)

        # Plot log-sum-exp 平滑后的约束边界 (phi_approx = 0)
        ax.contour(X, Y, Z_smooth_union, levels=[0], colors='red', linewidths=2)

        ax.set_title(f'LSE Smoothing with $\\tau = {tau}$')
        ax.set_xlabel('x')
        if i == 0:  # Set y-label only for the first subplot
            ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.7)

        # Create custom legend handles
        legend_handles = [
            Line2D([0], [0], label='Circle 1 Boundary', color='blue', linestyle='dashed', linewidth=1.5),
            Line2D([0], [0], label='Circle 2 Boundary', color='green', linestyle='dashed', linewidth=1.5),
            Line2D([0], [0], label=f'LSE Smoothed ($\\tau={tau}$)', color='red', linewidth=2)
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize='small')

    plt.tight_layout(rect=[0, 0.05, 1, 0.9])  # Adjust layout to make space for suptitle
    plt.show()


# --- 示例参数 ---
circle1_params_overlap = {'center_x': -0.8, 'center_y': 0, 'radius': 1.0}
circle2_params_overlap = {'center_x': 0.8, 'center_y': 0, 'radius': 1.0}

circle1_params_disjoint = {'center_x': -1.5, 'center_y': 0, 'radius': 0.7}
circle2_params_disjoint = {'center_x': 1.5, 'center_y': 0, 'radius': 0.7}

circle1_params_contain = {'center_x': 0, 'center_y': 0, 'radius': 1.5}
circle2_params_contain = {'center_x': 0.5, 'center_y': 0, 'radius': 0.5}

tau_values_to_test = [0.5, 0.1, 0.05]

print("Plotting overlapping circles...")
plot_constraints_and_smoothed_union(circle1_params_overlap, circle2_params_overlap, tau_values_to_test)

print("Plotting disjoint circles...")
plot_constraints_and_smoothed_union(circle1_params_disjoint, circle2_params_disjoint, tau_values_to_test,
                                    x_range=(-3, 3), y_range=(-2, 2))

print("Plotting one circle containing another...")
plot_constraints_and_smoothed_union(circle1_params_contain, circle2_params_contain, tau_values_to_test)