"""
Author: Ashish Saxena
===========================================================================
Pipeline: Baseball Tracking, 3D Reconstruction, and Speed Estimation
===========================================================================

PURPOSE
-------
This script implements an end-to-end pipeline for tracking a baseball in a
high-speed monocular image sequence, reconstructing its 3D trajectory, and
estimating its speed vector using known camera parameters and physical
constraints.

The pipeline is designed to be modular, interpretable, and physically
grounded. All estimation steps (tracking, reconstruction, velocity
computation) are completed prior to visualization, ensuring that plots do
not influence numerical results.

---------------------------------------------------------------------------
INPUTS
---------------------------------------------------------------------------
1. Image Sequence
   - Directory containing sequential image frames (e.g., PNG/JPG)
   - Frames are assumed to be captured at a fixed frame rate (FPS)

2. Detection Output (from tracking stage)
   - Dictionary of detections with the structure:
       detections = {
           frame_id: (u, v, r)
       }
     where:
       u, v : Ball center coordinates in pixel space
       r    : Ball radius in pixels (estimated accurately from the
              ball–bat separation frame)

3. Camera and Scene Parameters
   - Frame rate (FPS)
   - Camera intrinsics (focal length, pixel size, principal point)
   - Camera pose (tilt with respect to world frame)
   - Known physical radius of the baseball

---------------------------------------------------------------------------
ASSUMPTIONS
---------------------------------------------------------------------------
- The physical radius of the baseball is constant and known.
- The pixel-space radius estimated from the separation frame provides a
  reliable metric scale for depth estimation.
- Subsequent frames reuse this fixed physical scale
- Per-frame radius is estimated using radial edge detection and geometric circle fitting.
- However, per-frame radius  was not used to avoid depth jitter.
- Small center localization errors introduce bounded angular noise but do
  not affect metric scale.
- Motion is smooth over the short temporal window of the sequence.

---------------------------------------------------------------------------
OUTPUTS
---------------------------------------------------------------------------
Numerical Outputs:
- positions_3d        : (N, 3) array of reconstructed 3D positions (meters)
- velocities_raw      : (N-1, 3) array of raw 3D velocity vectors (m/s)
- world_speed         : (N-1,) speed magnitude per frame (m/s)
- mean_speed          : Scalar mean speed (m/s)
- speed_conf_interval : Scalar confidence interval for speed estimate

Visualization Outputs (saved to disk):
- 3D trajectory with positional uncertainty
- Speed vs. time with confidence interval
- Trajectory comparison across pipeline stages
  (raw, filtered, ballistic)

---------------------------------------------------------------------------
PIPELINE STAGES
---------------------------------------------------------------------------
1. Load and sort image frames
2. Track ball center in pixel space (forward and backward to the separation frame)
3. Reconstruct 3D positions using ray projection and fixed-radius depth
4. Compute velocities via finite differences
5. Optional filtering and ballistic validation
6. Visualization of final results

---------------------------------------------------------------------------
NOTES
---------------------------------------------------------------------------
- Visualization functions are strictly read-only and operate on finalized
  results.
- This script is intended for offline analysis and reporting, not
  real-time deployment.

===========================================================================
"""


import glob, cv2, os, numpy as np
from pipeline.config import *
from pipeline.p10_detection import BaseballDetectorSeparated
from pipeline.p10_1_backtrack import backtrack_ball
from pipeline.p10_2_searchupdate import update_search_radius
from pipeline.p11_rays import pixel_to_ray
from pipeline.p12_depth import estimate_depth
from pipeline.p13_pose import camera_to_world
from pipeline.p13_1_backproject import backproject
from pipeline.p14_velocity import compute_velocity
from pipeline.p20_kalman import kalman_filter
from pipeline.p30_ballistic import fit_ballistic_with_ci
from pipeline.visualization import *
from natsort import natsorted

print("loaded all the modules")

os.makedirs("outputs", exist_ok=True)

# ============================================================
# User inputs
# ============================================================

paths = natsorted(glob.glob("data/*.bmp"))  # change per directory and image type
Ball_refinement = False                     # make it False if radial edge detection and geometric circle fitting is not needed
using_fixed_depth = True                    # make it False if per-frame depth estimation is needed. True is recommended.

# ============================================================
# PER FRAME BASEBALL DETECTION PIPELINE
# ============================================================


detector = BaseballDetectorSeparated()

# --- Warm-up background model ---
for i in range(min(4, len(paths))):
    img = cv2.imread(paths[i])
    _ = detector.bg.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# --- Determine separation frame ---

detections = {}        # frame_idx -> (u, v, r)
first_detect_frame = None
ball_template = None

for idx, p in enumerate(paths):
    img = cv2.imread(p)

    try:
        u, v, r = detector.detect(
            img,
            save_debug=True,
            frame_id=idx
        )
        
        detections[idx] = (u, v, r)

        print(f"[OK] Frame {idx}: ball detected")
        break

    except RuntimeError as e:
        print(f"[INFO] Frame {idx}: {e}")
        continue

# --- Get ball template from separation frame ---

first_detect_frame = min(detections)
print("first_detect_frame", first_detect_frame)

u, v, r = detections[first_detect_frame]
img_temp = cv2.imread(paths[first_detect_frame])
gray_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    
# Extract template from this detected frame for backtracking
pad = int(0.8 * r)
x1 = int(max(0, u - pad))
y1 = int(max(0, v - pad))
x2 = int(min(img_temp.shape[1], u + pad))
y2 = int(min(img_temp.shape[0], v + pad))

ball_template_bg = gray_temp[y1:y2, x1:x2]

# --- Forward tracking using adaptive search window and ball template ---

# Forward pass
c_frame = 0
ball_template = ball_template_bg.copy()
u_bt, v_bt = u, v
search_radius = 50
for idx in range(first_detect_frame + 1, len(paths)):  
      
    img = cv2.imread(paths[idx])

    # Use last known position as reference
    prev_center = detections[idx - 1][:2]

    u_bt, v_bt, radius, confidence = backtrack_ball(
            prev_image=img,
            center= prev_center,
            radius=detections[idx - 1][2],
            template=ball_template,
            u_temp = u_bt,
            v_temp = v_bt,
            search_radius= search_radius,
            save_debug = True,
            frame_id = idx,
            refine_segmentation = Ball_refinement
        )
    new_center = (u_bt, v_bt)
    search_radius = update_search_radius(
        prev_center,
        new_center,
        search_radius
    )

    detections[idx] = (u_bt, v_bt, radius)
    print(f"[BT] Frame {idx}: backtracked; Confidence {confidence}; search radius {search_radius}")

    #Get new adaptive template
    gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract template from this detected frame for backtracking
    pad = int(0.8 * r)
    x1 = int(max(0, u_bt - pad))
    y1 = int(max(0, v_bt - pad))
    x2 = int(min(img.shape[1], u_bt + pad))
    y2 = int(min(img.shape[0], v_bt + pad))
    new_template = gray_temp[y1:y2, x1:x2]
    
    if confidence > 0.85:
        c_frame += 1
    
    eta = 0.05
    if confidence > 0.85 and c_frame == 2:
        ht = min(ball_template.shape[0], new_template.shape[0])
        wt = min(ball_template.shape[1], new_template.shape[1])

        ball_template = ball_template[:ht, :wt]
        new_template  = new_template[:ht, :wt]

        ball_template = (1-eta) * ball_template + eta * new_template
        ball_template = np.clip(ball_template, 0, 255).astype(np.uint8)
        print("using updated template")

    elif confidence < 0.7:
        print("ball not detected, a fallback should be included here")
        continue

    else:
        print(f"[WARN] Template confidence is moderate at {idx}, keeping previous template")
    
    if c_frame == 2:
        c_frame = 0

# --- Backward tracking using adaptive search window and ball template ---
c_frame = 0
ball_template = ball_template_bg.copy()
u_bt, v_bt = u, v
search_radius = 50
for idx in range(first_detect_frame - 1, -1, -1):          
    img = cv2.imread(paths[idx])

    # Use last known position as reference
    next_center = detections[idx + 1][:2]

    u_bt, v_bt, radius, confidence = backtrack_ball(
            prev_image=img,
            center=next_center,
            radius=detections[idx + 1][2],
            template=ball_template,
            u_temp = u_bt,
            v_temp = v_bt,
            search_radius=search_radius,
            save_debug = True,
            frame_id = idx,
            refine_segmentation = Ball_refinement
        )
    
    new_center = (u_bt, v_bt)
    search_radius = update_search_radius(
        next_center,
        new_center,
        search_radius
    )

    detections[idx] = (u_bt, v_bt, radius)
    print(f"[BT] Frame {idx}: backtracked; Confidence {confidence}; search radius {search_radius}")

    #Get new adaptive template
    gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract template from this detected frame for backtracking
    r = detections[idx + 1][2]
    pad = int(0.8 * r)
    x1 = int(max(0, u_bt - pad))
    y1 = int(max(0, v_bt - pad))
    x2 = int(min(img.shape[1], u_bt + pad))
    y2 = int(min(img.shape[0], v_bt + pad))
    new_template = gray_temp[y1:y2, x1:x2]

    if confidence > 0.85:
        c_frame += 1
    
    eta = 0.05
    if confidence > 0.85 and c_frame == 2:
        ht = min(ball_template.shape[0], new_template.shape[0])
        wt = min(ball_template.shape[1], new_template.shape[1])

        ball_template = ball_template[:ht, :wt]
        new_template  = new_template[:ht, :wt]

        ball_template = (1-eta) * ball_template + eta * new_template
        ball_template = np.clip(ball_template, 0, 255).astype(np.uint8)
        #ball_template = new_template
        print("using updated last frame template")
    elif confidence < 0.7:
        print("ball not detected, a fallback should be included here")
        continue
    else:
        print(f"[WARN] Template confidence is moderate at {idx}, keeping previous template")
    
    if c_frame == 2:
        c_frame = 0

# ============================================================
# 3D RECONSTRUCTION
# ============================================================

# --- Sort detections by frame id ---
frame_ids = sorted(detections.keys())

pixel_centers = np.array(
    [(detections[fid][0], detections[fid][1]) for fid in frame_ids],
    dtype=np.float32
)

radii_px = np.array(
    [detections[fid][2] for fid in frame_ids],
    dtype=np.float32
)

# --- Reconstruct 3D world positions ---
world_positions, P_cam = [], []

for idx, (u, v) in enumerate(pixel_centers):
    ray = pixel_to_ray(u, v)                          # unit ray
    if using_fixed_depth:
        fixed_radius_px = radii_px[first_detect_frame]   # separation frame depth, fixed scale
    else:
        fixed_radius_px = radii_px[idx]                  # per frame depth estimation     
    depth = estimate_depth(fixed_radius_px)           
    Pc = backproject(u, v, depth)
    P_cam.append(Pc)
    X_cam = ray * depth
    X_world = camera_to_world(X_cam)
    world_positions.append(X_world)

positions_3d = np.array(world_positions)
P_cam = np.array(P_cam)

positions_3d_kalman = kalman_filter(positions_3d)

# --- Time base ---
timestamps = np.arange(len(positions_3d_kalman)) * DT

# ---- velocity ----
velocities_raw, _ = compute_velocity(positions_3d)
velocities_pkf, _ = compute_velocity(positions_3d_kalman)

# ============================================================
# SPEED ESTIMATION & VISUALIZATION
# ============================================================

# --- World-space speed ---
world_speed = np.linalg.norm(velocities_raw, axis=1)
world_speed_kamlan = np.linalg.norm(velocities_pkf, axis=1)


# --- Pixel-space speed ---
pixel_speed = np.linalg.norm(
    pixel_centers[1:] - pixel_centers[:-1],
    axis=1
)

# --- Mean speed & confidence interval ---
mean_speed = np.mean(world_speed)
mean_speed_kamlan = np.mean(world_speed_kamlan)

# 95% CI assuming approximately Gaussian noise
speed_ci = 1.96 * np.std(world_speed) / np.sqrt(len(world_speed))
speed_ci_kamlan = 1.96 * np.std(world_speed_kamlan) / np.sqrt(len(world_speed_kamlan))

print(f"Raw Speed = {mean_speed} ± {speed_ci} m/s (95% CI)")
print(f"Kamlan Speed = {mean_speed_kamlan} ± {speed_ci_kamlan} m/s (95% CI)")

# --- Optional Kalman filtering ---
velocities_kalman = None
try:
    velocities_kalman = kalman_filter(velocities_raw)
except Exception:
    pass


# ============================================================
# 2D PLOTTING
# ============================================================

print("Generating final speed and trajectory plots...")

# --- Speed with confidence interval ---
plot_velocity_ci(
    speed=world_speed,
    mean=mean_speed,
    ci=speed_ci,
    title = "Speed estimate without filtering",
    path="outputs/detections/speed_ci.png"
)

plot_velocity_ci(
    speed= world_speed_kamlan,
    mean= mean_speed_kamlan,
    ci= speed_ci_kamlan,
    title = "Speed estimate with Kamlan filtering",
    path="outputs/detections/speed_kamlan_ci.png"
)

# --- Raw vs Kalman-filtered speed comparison ---
if velocities_kalman is not None:
    kalman_speed = np.linalg.norm(
        np.asarray(velocities_kalman), axis=1
    )

    plot_raw_vs_filtered_velocity(
        timestamps=timestamps[1:],
        velocities_raw=velocities_raw,
        velocities_kalman=velocities_kalman,
        path = "outputs/detections/plot_raw_vs_filtered_velocity.png"
    )

# --- Pixel vs world speed comparison ---
plot_pixel_vs_world_velocity(
    timestamps=timestamps[1:],
    pixel_velocity=pixel_speed,
    world_velocity=world_speed,
    path = "outputs/detections/plot_pixel_vs_world_velocity.png"
)

print("Speed estimation and visualization completed.")

# ============================================================
# 3D TRACKING VISUALIZATION
# ============================================================

# --- Optional: Ballistic trajectory ---
positions_ballistic = None
try:
    p0, v0, v0_ci = fit_ballistic_with_ci(positions_3d_kalman)
    positions_ballistic = p0 + v0*timestamps[:,None] + 0.5*G*timestamps[:,None]**2

except Exception:
    pass

# --- Positional uncertainty ---
# Simple conservative estimate from frame-to-frame motion
pos_sigma = np.linalg.norm(
    positions_3d[1:] - positions_3d[:-1],
    axis=1
)
pos_sigma = np.concatenate([[pos_sigma[0]], pos_sigma])

# ============================================================
# 3D PLOTTING
# ============================================================

print("Generating 3D tracking plots...")

# 3D trajectory with velocity vectors
plot_trajectory_with_velocity_vectors(
    positions_3d=positions_3d,
    velocities=velocities_raw,
    stride=1,
    scale=0.05,
    path = "outputs/detections/trajectory_with_velocity_vector.png"
)

# --- 3D trajectory with uncertainty ---
plot_trajectory_with_error(
    P=positions_3d,
    sigma=pos_sigma,
    path="outputs/detections/3dtrajectory_with_error.png"
)

# --- Trajectory comparison across pipeline ---
if positions_3d_kalman is not None and positions_ballistic is not None:
    plot_comparison(
        P_cam=P_cam,          # raw reconstruction
        P_world=positions_3d,        # same reference (or transformed)
        P_kf= positions_3d_kalman,
        P_ball=positions_ballistic,
        path="outputs/detections/trajectory_comparison.png"
    )

print("3D tracking visualization completed.")
