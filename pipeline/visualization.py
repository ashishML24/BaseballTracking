import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# 1. 3D Trajectory with Velocity Vector Overlay
# --------------------------------------------------

def plot_trajectory_with_velocity_vectors(
    positions_3d,
    velocities,
    stride=1,
    scale=0.05,
    path = None
):
    """
    positions_3d : (N, 3) array in meters
    velocities   : (N-1, 3) array in m/s
    stride       : plot every k-th vector
    scale        : arrow length scaling
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = positions_3d.T
    vx, vy, vz = velocities.T

    ax.plot(x, y, z, 'o-', label="Ball trajectory", linewidth=2)

    ax.quiver(
        x[:-1:stride],
        y[:-1:stride],
        z[:-1:stride],
        vx[::stride],
        vy[::stride],
        vz[::stride],
        length=scale,
        normalize=True,
        color='r',
        label="Velocity vectors"
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory with Velocity Vector Overlay")

    ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --------------------------------------------------
# 2. Speed Magnitude vs Time (with Optional CI)
# --------------------------------------------------

def plot_speed_vs_time(
    timestamps,
    velocities,
    speed_ci=None,
    path = None
):
    """
    timestamps : (N-1,) array
    velocities : (N-1, 3) array
    speed_ci   : (N-1, 2) array [lower, upper] or None
    """

    speed = np.linalg.norm(velocities, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, speed, 'b-o', label="Speed magnitude")

    if speed_ci is not None:
        plt.fill_between(
            timestamps,
            speed_ci[:, 0],
            speed_ci[:, 1],
            alpha=0.3,
            label="Confidence interval"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed Magnitude vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --------------------------------------------------
# 3. Component-wise Velocity Plot
# --------------------------------------------------

def plot_velocity_components(
    timestamps,
    velocities,
    path = None
):
    """
    velocities : (N-1, 3) array
    """

    labels = ['Vx', 'Vy', 'Vz']
    colors = ['r', 'g', 'b']

    plt.figure(figsize=(8, 5))

    for i in range(3):
        plt.plot(
            timestamps,
            velocities[:, i],
            color=colors[i],
            marker='o',
            label=f"{labels[i]} component"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity Components vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --------------------------------------------------
# 4. Raw vs Kalman-Filtered Velocity Comparison
# --------------------------------------------------

def plot_raw_vs_filtered_velocity(
    timestamps,
    velocities_raw,
    velocities_kalman,
    path = None
):
    """
    Compare magnitude of raw vs filtered velocity
    """

    speed_raw = np.linalg.norm(velocities_raw, axis=1)
    speed_filt = np.linalg.norm(velocities_kalman, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, speed_raw, 'k--o', label="Raw velocity")
    plt.plot(timestamps, speed_filt, 'r-o', label="Kalman-filtered velocity")

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Raw vs Filtered Speed Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --------------------------------------------------
# 5. Pixel-space vs World-space Velocity Comparison
# --------------------------------------------------

def plot_pixel_vs_world_velocity(
    timestamps,
    pixel_velocity,
    world_velocity,
    path = None
):
    """
    pixel_velocity : (N-1,) scalar pixel speed
    world_velocity : (N-1,) scalar m/s
    """

    plt.figure(figsize=(8, 4))

    plt.plot(timestamps, pixel_velocity, 'b-o', label="Pixel-space speed")
    plt.plot(timestamps, world_velocity, 'r-o', label="World-space speed")

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (pixels/frame or m/s)")
    plt.title("Pixel-space vs World-space Velocity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --------------------------------------------------
# 6. Ballistic Fit Residual Plot (Optional)
# --------------------------------------------------

def plot_ballistic_residuals(
    timestamps,
    measured_positions,
    ballistic_positions,
    path = None
):
    """
    Both inputs: (N, 3)
    """

    residuals = np.linalg.norm(
        measured_positions - ballistic_positions,
        axis=1
    )

    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, residuals, 'm-o')

    plt.xlabel("Time (s)")
    plt.ylabel("Position Residual (m)")
    plt.title("Residuals Between Estimated and Ballistic Trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# --------------------------------------------------
# 7. 3D Trajectory with Error Bars
# --------------------------------------------------

def plot_trajectory_with_error(P, sigma, path):
    """
    P     : (N, 3) 3D positions in meters
    sigma : (N,) or (N,3) positional uncertainty
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        P[:, 0], P[:, 1], P[:, 2],
        color='b', linewidth=2, label="Estimated trajectory"
    )

    # Error bars (applied along Y by default for clarity)
    ax.errorbar(
        P[:, 0], P[:, 1], P[:, 2],
        yerr=sigma,
        fmt='o', color='gray',
        alpha=0.4, label="Positional uncertainty"
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory with Uncertainty")

    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# --------------------------------------------------
# 8. Speed with Confidence Interval
# --------------------------------------------------

def plot_velocity_ci(speed, mean, ci, title, path):
    """
    speed : (N,) speed values in m/s
    mean  : scalar mean speed
    ci    : scalar confidence interval half-width
    """

    x = np.arange(len(speed))

    plt.figure(figsize=(8, 4))
    plt.plot(x, speed, 'b-o', label="Instantaneous speed")

    plt.axhline(
        mean,
        color='r',
        linestyle='--',
        linewidth=2,
        label=f"Mean speed = {mean:.2f} m/s"
    )

    plt.fill_between(
        x,
        mean - ci,
        mean + ci,
        color='r',
        alpha=0.2,
        label="95% confidence interval"
    )

    plt.xlabel("Frame index")
    plt.ylabel("Speed (m/s)")
    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# --------------------------------------------------
# 9. Trajectory Comparison Across Pipeline Stages
# --------------------------------------------------

def plot_comparison(P_cam, P_world, P_kf, P_ball, path):
    """
    Each input: (N, 3) trajectory in meters
    """

    fig = plt.figure(figsize=(14, 10))

    trajectories = [
        (P_cam,   "Camera Coordinates", "--"),
        (P_world, "World Coordinates",  "-"),
        (P_kf,    "Kalman Filtered",    "-"),
        (P_ball,  "Ballistic Fit",      "-")
    ]

    for i, (P, title, style) in enumerate(trajectories, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')

        ax.plot(
            P[:, 0], P[:, 1], P[:, 2],
            style, linewidth=2
        )

        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

    fig.suptitle("Trajectory Evolution Across the Pipeline", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, dpi=300)
    plt.close()
