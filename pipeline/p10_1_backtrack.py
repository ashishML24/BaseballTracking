import cv2
import numpy as np
import os

def segment_ball_seeded(gray, seed, max_radius, seed_radius=1):
    """
    Seeded region growing using multiple nearby seed points
    for robustness against center localization noise.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image
    seed : tuple(float, float)
        Template-matched center (u, v)
    max_radius : int
        Maximum allowed growth radius (pixels)
    seed_radius : int, optional
        Radius (in pixels) around seed to generate additional seeds

    Returns
    -------
    mask : np.ndarray or None
        Binary segmentation mask (uint8), or None if failed
    """

    h, w = gray.shape
    mask = np.zeros_like(gray, dtype=np.uint8)

    u0, v0 = int(seed[0]), int(seed[1])
    if not (0 <= u0 < w and 0 <= v0 < h):
        return None

    # --------------------------------------------------
    # Generate multiple seed points around center
    # --------------------------------------------------
    seed_points = []
    for du in range(-seed_radius, seed_radius + 1):
        for dv in range(-seed_radius, seed_radius + 1):
            uu, vv = u0 + du, v0 + dv
            if 0 <= uu < w and 0 <= vv < h:
                seed_points.append((uu, vv))

    # Reference intensity: mean over seed points
    seed_vals = [gray[v, u] for (u, v) in seed_points]
    seed_val = int(np.mean(seed_vals))

    thresh = 5  # intensity tolerance (tunable)

    # Initialize region growing stack
    stack = []
    for (u, v) in seed_points:
        mask[v, u] = 1
        stack.append((u, v))

    # --------------------------------------------------
    # Region growing
    # --------------------------------------------------
    while stack:
        u, v = stack.pop()

        for du, dv in [(-1,0),(1,0),(0,-1),(0,1)]:
            uu, vv = u + du, v + dv
            if 0 <= uu < w and 0 <= vv < h:
                if mask[vv, uu] == 0:
                    if abs(int(gray[vv, uu]) - seed_val) < thresh:
                        if (uu - u0)**2 + (vv - v0)**2 <= max_radius**2:
                            mask[vv, uu] = 1
                            stack.append((uu, vv))
    #overlay_seg = gray.copy()
    #overlay_seg[mask > 0] = 255
    #cv2.imwrite(
    #   f"outputs/detections/region_growing_seg_debug.png",
    #   overlay_seg)
    #input('check')

    return mask

def score_segmentation(mask, seed, expected_radius_px, circ_min=0.65, sigma_r=3.0):
    """
    Returns (confidence, refined_center, refined_radius)
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return 0.0, None, None

    # Area & perimeter
    area = len(xs)
    perimeter = cv2.arcLength(
        cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32)),
        True
    )

    circularity = 4 * np.pi * area / max(perimeter**2, 1e-6)
    S_circ = min(1.0, circularity / circ_min)

    # Radius estimate
    r_seg = np.sqrt(area / np.pi)
    dr = abs(r_seg - expected_radius_px)
    S_rad = np.exp(-(dr**2) / (2 * sigma_r**2))

    # Seed containment
    u0, v0 = int(seed[0]), int(seed[1])
    S_seed = 1.0 if mask[v0, u0] > 0 else 0.0

    confidence = S_seed * (0.6 * S_circ + 0.4 * S_rad)

    # Refined center
    cx = xs.mean()
    cy = ys.mean()

    return confidence, (cx, cy), r_seg


def radial_edge_circle_fit(
    gray,
    center,
    r_min,
    r_max,
    n_rays=90,
    grad_thresh=20
):
    """
    Radial edge detection followed by circle fitting.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image
    center : tuple(float, float)
        Approximate ball center (u, v)
    r_min, r_max : float
        Expected radius bounds (pixels)
    n_rays : int
        Number of radial directions
    grad_thresh : float
        Minimum gradient magnitude for edge detection

    Returns
    -------
    edge_points : np.ndarray or None
        Detected boundary points (Nx2)
    """

    h, w = gray.shape
    u0, v0 = center
    u0, v0 = int(u0), int(v0)

    # Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    edge_points = []

    angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)

    for theta in angles:
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        best_pt = None
        best_grad = 0

        for r in range(int(r_min), int(r_max)):
            u = int(u0 + r * cos_t)
            v = int(v0 + r * sin_t)

            if not (0 <= u < w and 0 <= v < h):
                break

            g = grad_mag[v, u]
            if g > grad_thresh and g > best_grad:
                best_grad = g
                best_pt = (u, v)

        if best_pt is not None:
            edge_points.append(best_pt)

    if len(edge_points) < 10:
        return None
    
    #overlay_seg = gray.copy()
    #for (u, v) in edge_points:
    #    cv2.circle(overlay_seg, (u, v), 1, (0,0,255), -1)
    #cv2.imwrite(
    #   f"outputs/detections/region_growing_seg_debug.png",
    #   overlay_seg)
    #input('check')

    return np.array(edge_points, dtype=np.float32)

def score_circle_fit(
    edge_points,
    center_tm,
    expected_radius,
    radius_sigma=3.0,
    max_center_shift=5.0
):
    """
    Fit circle to edge points and compute confidence.

    Returns
    -------
    confidence : float
    refined_center : tuple(float, float)
    refined_radius : float
    """

    if edge_points is None or len(edge_points) < 10:
        return 0.0, None, None

    # RANSAC circle fit using OpenCV
    (cx, cy), r = cv2.minEnclosingCircle(edge_pointsistani := edge_points.astype(np.float32))

    # Inlier ratio
    dists = np.sqrt((edge_points[:,0] - cx)**2 + (edge_points[:,1] - cy)**2)
    inliers = np.abs(dists - r) < radius_sigma
    inlier_ratio = np.sum(inliers) / len(edge_points)

    S_inlier = np.clip(inlier_ratio / 0.6, 0, 1)

    # Radius consistency
    dr = abs(r - expected_radius)
    S_rad = np.exp(-(dr**2) / (2 * radius_sigma**2))

    # Center shift penalty
    shift = np.linalg.norm(np.array([cx, cy]) - np.array(center_tm))
    S_center = np.clip(1 - shift / max_center_shift, 0, 1)

    # Final confidence
    #confidence = S_inlier * (0.5 * S_rad + 0.5 * S_center)
    confidence = S_inlier * (0.5 * S_rad)

    return confidence, (cx, cy), r


def save_overlay_debug(
    image,
    u,
    v,
    out_dir,
    frame_id,
    r=15,
    roi=None,
    template_size=None,
    u_temp = None,
    v_temp = None):
    """
    roi           : (x1, y1, x2, y2)
    template_size : (h, w)
    """
    os.makedirs(out_dir, exist_ok=True)
    overlay = image.copy()

    # --- Draw ball ---
    cv2.circle(overlay, (int(u), int(v)), int(r), (0, 255, 0), 2)
    cv2.circle(overlay, (int(u), int(v)), 3, (0, 0, 255), -1)

    # --- Draw ROI ---
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # --- Draw template bounding box ---
    if template_size is not None:
        th, tw = template_size
        tx1 = int(u_temp - tw / 2)
        ty1 = int(v_temp - th / 2)
        tx2 = int(u_temp + tw / 2)
        ty2 = int(v_temp + th / 2)

        #cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)

    cv2.imwrite(
        f"{out_dir}/frame_{frame_id:03d}_overlay_backtracked_debug.png",
        overlay
    )

def preprocess(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(img)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    return gray

def backtrack_ball(
    prev_image,
    center,
    radius,
    template,
    u_temp,
    v_temp,
    search_radius=30, 
    save_debug = False,
    out_dir="outputs/detections",
    frame_id = None,
    refine_segmentation = False
):
    """
    Backward tracking using:
    - Foreground gating
    - Template matching fallback
    """

    gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #gray = clahe.apply(gray)
    #gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    
    u, v = center

    x1 = int(max(0, u - search_radius))
    y1 = int(max(0, v - search_radius))
    x2 = int(min(prev_image.shape[1], u + search_radius))
    y2 = int(min(prev_image.shape[0], v + search_radius))

    roi = gray[y1:y2, x1:x2]
    roi = preprocess(roi)
    template = preprocess(template)
    confidence = 0

    # --- template matching ---
    print("backtracking using template matching")
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    cx = x1 + max_loc[0] + template.shape[1] / 2
    cy = y1 + max_loc[1] + template.shape[0] / 2   

    # confidence calculation
    # --- Correlation stats ---
    mean_val = np.mean(res)
    std_val = np.std(res)
    peak_sharpness = (max_val - mean_val) / (std_val + 1e-6)

    confidence = 0.5 * max_val + 0.5 * np.tanh(peak_sharpness)

    # --- Motion consistency ---
    dx = cx - u
    dy = cy - v
    displacement = np.sqrt(dx*dx + dy*dy)
    motion_penalty = np.exp(-displacement / (2.0 * radius))
    #confidence *= motion_penalty

    # --- Peak ambiguity penalty ---
    res_flat = res.flatten()
    second_peak = np.partition(res_flat, -2)[-2]
    peak_ratio = (max_val - second_peak) / (abs(second_peak) + 1e-6)

    #if peak_ratio < 0.1:
     #   confidence *= 0.3

    confidence = np.clip(confidence, 0.0, 1.0)
    
    # --------------------------------------------------
    # Segmentation-based refinement (AFTER template match)
    # --------------------------------------------------
    if refine_segmentation:
        seg_conf_thresh = 0.8
        center_tm = (max_loc[0] + template.shape[1] / 2, max_loc[1] + template.shape[0] / 2)
        radius_prev = radius
        conf_seg = 0.0

        '''
        mask = segment_ball_seeded(
        roi,
        seed=center_tm,
        max_radius=int(radius_prev)
        )

        if mask is not None:
            conf_seg, center_seg, radius_seg = score_segmentation(
            mask,
            seed=center_tm,
            expected_radius_px=radius_prev
            )
        else:
            conf_seg = 0.0
    
        print("segmentation refinement confidence score", conf_seg)
        '''
        
        edge_pts = radial_edge_circle_fit(
        roi,
        center=center_tm,
        r_min=radius_prev * 0.8,
        r_max=radius_prev * 1.2
        )
        conf_seg, center_seg, radius_seg = score_circle_fit(
        edge_pts,
        center_tm=center_tm,
        expected_radius=radius_prev
        )
        print("segmentation refinement confidence score", conf_seg)

        # Decision logic
        if conf_seg > 0.3:
            print("Accept segmentation refinement with confidence score", conf_seg)
            center = center_seg
            cx = x1 + center[0]
            cy = y1 + center[1]
            radius = radius_seg
            used_segmentation = True
        else:
            print("Fallback to template matching")


    if save_debug:

        save_overlay_debug(
        prev_image,
        cx,
        cy,
        out_dir=out_dir,
        frame_id=frame_id,
        r=radius,
        roi=(x1, y1, x2, y2),
        template_size=template.shape,
        u_temp = u_temp,
        v_temp = v_temp
        )

    
    return cx, cy, radius, confidence
