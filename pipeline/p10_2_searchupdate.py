import numpy as np


def update_search_radius(
    prev_center,
    curr_center,
    prev_radius,
    base_radius=25,
    min_radius=20,
    max_radius=120,
    alpha=1.5,
    smooth=0.5
):
    """
    Adaptively updates the search window radius for template matching based on
    observed inter-frame motion of the tracked object.

    This function dynamically expands or contracts the search region according
    to the displacement of the ball center between consecutive frames, while
    enforcing lower and upper bounds for numerical stability. A smoothing term
    is used to prevent abrupt radius changes caused by noisy detections.

    ------------------------------------------------------------------------
    INPUTS
    ------------------------------------------------------------------------
    prev_center : tuple(float, float)
        Ball center coordinates (u_{t-1}, v_{t-1}) in pixel space from the
        previous frame.

    curr_center : tuple(float, float) or None
        Ball center coordinates (u_t, v_t) in pixel space from the current
        frame. If None, it indicates a detection failure or low-confidence
        frame, triggering a recovery mode.

    prev_radius : float
        Search radius (in pixels) used in the previous frame.

    base_radius : float, optional (default=25)
        Minimum nominal search radius corresponding to negligible motion.
        This acts as the baseline radius when displacement is near zero.

    min_radius : float, optional (default=20)
        Lower bound on the search radius to avoid excessively small regions
        that could miss the target due to noise.

    max_radius : float, optional (default=120)
        Upper bound on the search radius to prevent excessive growth and
        reduce false matches.

    alpha : float, optional (default=1.5)
        Scaling factor that controls how aggressively the search radius
        expands in response to inter-frame displacement.

    smooth : float in [0, 1], optional (default=0.5)
        Temporal smoothing factor for displacement estimation.
        - smooth → 1.0 emphasizes historical radius changes
        - smooth → 0.0 emphasizes instantaneous displacement

    ------------------------------------------------------------------------
    OUTPUT
    ------------------------------------------------------------------------
    new_radius : int
        Updated search radius (in pixels) to be used for the next frame,
        clipped to the range [min_radius, max_radius].

    ------------------------------------------------------------------------
    ALGORITHM DESCRIPTION
    ------------------------------------------------------------------------
    1. If curr_center is None, the tracker enters a recovery mode where the
       search radius is expanded multiplicatively (up to max_radius) to
       increase the chance of re-detecting the ball.

    2. Otherwise, the inter-frame displacement d is computed as:
           d = sqrt((u_t - u_{t-1})^2 + (v_t - v_{t-1})^2)

    3. The displacement is smoothed using a weighted combination of:
       - the previous radius deviation from the base radius
       - the current instantaneous displacement

    4. The new radius is computed as:
           new_radius = base_radius + alpha * d_smooth

    5. The result is clipped to [min_radius, max_radius] and returned.

    ------------------------------------------------------------------------
    DESIGN RATIONALE
    ------------------------------------------------------------------------
    - Ensures tight search regions during slow motion for robustness
    - Expands search area adaptively during fast motion
    - Avoids sudden radius jumps caused by noisy detections
    - Provides graceful recovery in frames with missing detections

    ------------------------------------------------------------------------
    NOTES
    ------------------------------------------------------------------------
    - All quantities are in pixel units.
    - This function is independent of the tracking method and can be reused
      for other single-object tracking tasks.
    """

    if curr_center is None:
        # Recovery mode: expand search region conservatively
        return min(max_radius, prev_radius * 1.5)

    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    d = np.sqrt(dx * dx + dy * dy)

    # Smoothed displacement to avoid abrupt radius changes
    d_smooth = smooth * (prev_radius - base_radius) / max(alpha, 1e-6) \
               + (1 - smooth) * d

    new_radius = base_radius + alpha * d_smooth

    return int(np.clip(new_radius, min_radius, max_radius))