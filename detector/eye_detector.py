"""
DriveSense — Eye Detector
Calculates Eye Aspect Ratio (EAR), blink rate, and PERCLOS
using MediaPipe Face Mesh landmark indices.
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import deque
from utils.config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES,
    PERCLOS_WINDOW, PERCLOS_THRESHOLD
)

# ── MediaPipe landmark indices for left & right eye ──────────────────────────
# Each eye uses 6 points: 2 horizontal corners + 4 vertical points
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def compute_ear(landmarks, eye_indices, frame_w, frame_h):
    """
    Compute Eye Aspect Ratio for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    A high EAR (~0.3+) means the eye is open.
    A low EAR (~0.2 or below) means the eye is closed.
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * frame_w, lm.y * frame_h))

    # Vertical distances
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    # Horizontal distance
    C = dist.euclidean(pts[0], pts[3])

    ear = (A + B) / (2.0 * C)
    return round(ear, 3)


class EyeMonitor:
    """
    Stateful monitor that tracks:
    - Current EAR (average of both eyes)
    - Consecutive closed-eye frames
    - Blink count and blink rate (blinks per minute)
    - PERCLOS score (% of time eyes closed in rolling window)
    - Drowsiness state
    """

    def __init__(self):
        self.ear              = 0.0
        self.consec_frames    = 0      # frames below EAR threshold
        self.total_blinks     = 0
        self.blink_in_prog    = False
        self.frame_count      = 0
        self.start_time       = None

        # Rolling window for PERCLOS
        self._ear_history     = deque(maxlen=PERCLOS_WINDOW)

        # State
        self.is_drowsy        = False
        self.drowsy_frames    = 0      # consecutive drowsy frames

        import time
        self.start_time       = time.time()

    def update(self, landmarks, frame_w, frame_h):
        """
        Call once per frame with MediaPipe face landmarks.
        Returns True if drowsiness is detected.
        """
        import time

        left_ear  = compute_ear(landmarks, LEFT_EYE,  frame_w, frame_h)
        right_ear = compute_ear(landmarks, RIGHT_EYE, frame_w, frame_h)
        self.ear  = round((left_ear + right_ear) / 2.0, 3)

        # Rolling PERCLOS history
        eye_closed = self.ear < EAR_THRESHOLD
        self._ear_history.append(1 if eye_closed else 0)

        # Blink detection (quick close → open)
        if eye_closed:
            self.consec_frames += 1
            self.blink_in_prog  = True
        else:
            if self.blink_in_prog and self.consec_frames < EAR_CONSEC_FRAMES:
                # Short closure = a blink (not drowsiness)
                self.total_blinks += 1
            self.blink_in_prog = False
            self.consec_frames = 0

        # Drowsiness: eyes closed for too many consecutive frames
        if self.consec_frames >= EAR_CONSEC_FRAMES:
            self.drowsy_frames += 1
            self.is_drowsy = True
        else:
            self.drowsy_frames = 0
            self.is_drowsy     = False

        # Also trigger drowsy on high PERCLOS
        if self.perclos >= PERCLOS_THRESHOLD:
            self.is_drowsy = True

        self.frame_count += 1
        return self.is_drowsy

    @property
    def perclos(self):
        """Percentage of frames in rolling window where eyes were closed."""
        if not self._ear_history:
            return 0.0
        return round(sum(self._ear_history) / len(self._ear_history), 3)

    @property
    def blink_rate(self):
        """Blinks per minute since monitoring started."""
        import time
        elapsed_min = (time.time() - self.start_time) / 60.0
        if elapsed_min < 0.01:
            return 0.0
        return round(self.total_blinks / elapsed_min, 1)

    def reset(self):
        import time
        self.__init__()
