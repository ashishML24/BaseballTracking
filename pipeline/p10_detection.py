import cv2
import numpy as np
import os


class BaseballDetectorSeparated:
    """
    Detection with explicit bat–ball separation logic and shape tolerance.
    """

    def __init__(
        self,
        expected_radius_px=15.5,
        radius_tol=0.3,        # ±30% size tolerance
        min_circularity=0.5,   # relaxed
        max_aspect_ratio=1.2
    ):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=10,
            varThreshold=16,
            detectShadows=False
        )

        self.expected_radius = expected_radius_px
        self.radius_tol = radius_tol
        self.min_circularity = min_circularity
        self.max_aspect_ratio = max_aspect_ratio

        self.prev_center = None
        self.confirmed = False
        self.confirm_count = 0

    def detect(
        self,
        image,
        save_debug=False,
        out_dir="outputs/detections",
        frame_id=None
    ):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

        # ----------------------------
        # Phase A — Foreground
        # ----------------------------
        fg = self.bg.apply(gray)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.dilate(fg, kernel, iterations=1)
        cv2.imwrite(
                f"{out_dir}/frame_{frame_id:03d}_foreground.png",
                fg
            )

        # ----------------------------
        # Phase B — Candidates
        # ----------------------------
        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

            r_px = np.sqrt(area / np.pi)

            # -------- SOFT SHAPE SCORING --------
            size_score = 1 - abs(r_px - self.expected_radius) / self.expected_radius
            circ_score = circularity
            aspect_score = max(0, 1 - abs(aspect_ratio - 1))

            score = (
                0.5 * size_score +
                0.3 * circ_score +
                0.2 * aspect_score
            )

            if score > 0.8 and frame_id <= 5:  # permissive threshold
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                candidates.append((score, cx, cy, r_px))
            elif score > 0.5 and frame_id > 5:  # permissive threshold
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                candidates.append((score, cx, cy, r_px))

        if not candidates:
            raise RuntimeError("Ball not isolated yet")

        candidates.sort(reverse=True)
        score, u, v, r = candidates[0]

        # ----------------------------
        # Phase C — Separation confirmation
        # ----------------------------
        if self.prev_center is not None:
            dist = np.linalg.norm(np.array([u, v]) - np.array(self.prev_center))
            if dist > 2.0:
                self.confirm_count += 1
            else:
                self.confirm_count = 0
        else:
            self.confirm_count += 1

        self.prev_center = (u, v)

        if self.confirm_count < 2:
            raise RuntimeError("Separation not confirmed")

        self.confirmed = True

        # ----------------------------
        # Debug
        # ----------------------------
        if save_debug and frame_id is not None:
            os.makedirs(out_dir, exist_ok=True)
            overlay = image.copy()
            cv2.circle(overlay, (int(u), int(v)), int(r), (0, 255, 0), 2)
            cv2.circle(overlay, (int(u), int(v)), 3, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                f"score={score:.2f}, r={r:.1f}",
                (int(u+5), int(v)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255,255,255),
                1
            )
            cv2.imwrite(
                f"{out_dir}/frame_{frame_id:03d}_overlay.png",
                overlay
            )

        return float(u), float(v), float(r)
