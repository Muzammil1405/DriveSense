"""
DriveSense — Head Pose Estimator
Estimates yaw (left/right), pitch (up/down), roll (tilt)
using a Perspective-n-Point (PnP) solve on 6 facial landmarks.
"""

import numpy as np
import cv2
from utils.config import (
    YAW_THRESHOLD, PITCH_THRESHOLD, HEAD_CONSEC_FRAMES
)

# ── 6 stable 3-D reference points (from a generic face model) ────────────────
# Nose tip, Chin, Left eye corner, Right eye corner, Left mouth, Right mouth
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),      # Nose tip          — landmark 1
    (0.0,   -63.6,  -12.5),     # Chin              — landmark 152
    (-43.3,  32.7,  -26.0),     # Left eye corner   — landmark 263
    (43.3,   32.7,  -26.0),     # Right eye corner  — landmark 33
    (-28.9, -28.9,  -24.1),     # Left mouth corner — landmark 287
    (28.9,  -28.9,  -24.1),     # Right mouth corner— landmark 57
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
MP_INDICES = [1, 152, 263, 33, 287, 57]


def _get_camera_matrix(frame_w, frame_h):
    focal = frame_w
    cx, cy = frame_w / 2, frame_h / 2
    return np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1 ]
    ], dtype=np.float64)


DIST_COEFFS = np.zeros((4, 1))  # assuming no lens distortion


class HeadPoseEstimator:
    """
    Stateful estimator for head pose (yaw, pitch, roll).
    Raises distraction alert when face turns away for too many frames.
    """

    def __init__(self):
        self.yaw           = 0.0
        self.pitch         = 0.0
        self.roll          = 0.0
        self.consec_frames = 0
        self.is_distracted = False

    def update(self, landmarks, frame_w, frame_h):
        """
        Call once per frame.
        Returns True if distraction is detected.
        """
        # Extract 2-D image points
        image_points_2d = []
        for idx in MP_INDICES:
            lm = landmarks[idx]
            image_points_2d.append((lm.x * frame_w, lm.y * frame_h))
        image_points_2d = np.array(image_points_2d, dtype=np.float64)

        camera_matrix = _get_camera_matrix(frame_w, frame_h)

        success, rot_vec, trans_vec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return self.is_distracted

        # Convert rotation vector → rotation matrix → Euler angles
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        proj_mat   = np.hstack((rot_mat, trans_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)

        self.pitch = float(euler[0])
        self.yaw   = float(euler[1])
        self.roll  = float(euler[2])

        # Normalise pitch to [-90, 90]
        if self.pitch > 90:
            self.pitch -= 180
        elif self.pitch < -90:
            self.pitch += 180

        off_road = (
            abs(self.yaw)   > YAW_THRESHOLD or
            abs(self.pitch) > PITCH_THRESHOLD
        )

        if off_road:
            self.consec_frames += 1
        else:
            self.consec_frames = 0

        self.is_distracted = self.consec_frames >= HEAD_CONSEC_FRAMES
        return self.is_distracted

    def direction_label(self):
        """Human-readable gaze direction."""
        if abs(self.yaw) <= YAW_THRESHOLD and abs(self.pitch) <= PITCH_THRESHOLD:
            return "Forward"
        parts = []
        if self.pitch < -PITCH_THRESHOLD:
            parts.append("Up")
        elif self.pitch > PITCH_THRESHOLD:
            parts.append("Down")
        if self.yaw < -YAW_THRESHOLD:
            parts.append("Left")
        elif self.yaw > YAW_THRESHOLD:
            parts.append("Right")
        return " + ".join(parts) if parts else "Forward"
