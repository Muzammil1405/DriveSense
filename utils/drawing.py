"""
DriveSense — Drawing / HUD utilities
All on-screen overlays: stats panel, EAR bar, PERCLOS, head angles.
"""

import cv2
import numpy as np
from utils.config import SHOW_EAR_VALUE, SHOW_HEAD_ANGLES, SHOW_LANDMARKS


# ── Colour palette (BGR) ────────────────────────────────────────────────────
COL_OK      = (100, 210, 80)
COL_WARN    = (0,   200, 230)
COL_DANGER  = (40,  40,  220)
COL_WHITE   = (255, 255, 255)
COL_DARK    = (20,  20,  20)
COL_PANEL   = (30,  30,  30)
COL_GRAY    = (140, 140, 140)


def _status_color(is_bad):
    return COL_DANGER if is_bad else COL_OK


def draw_stats_panel(frame, eye_mon, head_pose, alert_mgr, fps):
    """
    Draw the semi-transparent stats panel in the bottom-left corner.
    """
    h, w = frame.shape[:2]

    panel_w  = 310
    panel_h  = 210
    margin   = 14
    px, py   = margin, h - panel_h - margin

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), COL_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (80, 80, 80), 1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fs    = 0.52
    lh    = 28   # line height
    tx    = px + 12
    ty    = py + 24

    def put(label, value, color=COL_WHITE, bold=False):
        nonlocal ty
        cv2.putText(frame, label, (tx, ty), font, fs, COL_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (tx + 130, ty), font, fs, color, 2 if bold else 1, cv2.LINE_AA)
        ty += lh

    # ── Rows ────────────────────────────────────────────────────────────────
    put("FPS",          f"{fps:.0f}")
    put("EAR",          f"{eye_mon.ear:.3f}",
        _status_color(eye_mon.ear < 0.25))

    put("PERCLOS",      f"{eye_mon.perclos * 100:.1f}%",
        _status_color(eye_mon.perclos >= 0.35))

    put("Blinks/min",   f"{eye_mon.blink_rate:.1f}")

    if SHOW_HEAD_ANGLES:
        put("Yaw",      f"{head_pose.yaw:+.1f}°",
            _status_color(abs(head_pose.yaw) > 25))
        put("Pitch",    f"{head_pose.pitch:+.1f}°",
            _status_color(abs(head_pose.pitch) > 20))
        put("Gaze",     head_pose.direction_label())

    # ── Status badge ────────────────────────────────────────────────────────
    badge_label = alert_mgr.label()
    badge_col   = alert_mgr.color_bgr()
    bx = px + panel_w - 90
    by = py + 10
    cv2.rectangle(frame, (bx, by), (bx + 80, by + 26), badge_col, -1)
    cv2.putText(frame, badge_label, (bx + 6, by + 19),
                font, 0.55, COL_WHITE, 1, cv2.LINE_AA)


def draw_ear_bar(frame, ear_value, threshold=0.25):
    """
    Draw a vertical EAR meter on the right edge of the frame.
    """
    h, w = frame.shape[:2]
    bar_h  = 200
    bar_w  = 20
    bx     = w - bar_w - 20
    by     = (h - bar_h) // 2

    # Background track
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)

    # Fill level (EAR clamped 0 → 0.5 range)
    ratio  = min(max(ear_value / 0.5, 0.0), 1.0)
    fill_h = int(bar_h * ratio)
    color  = COL_OK if ear_value >= threshold else COL_DANGER
    cv2.rectangle(frame,
                  (bx, by + bar_h - fill_h),
                  (bx + bar_w, by + bar_h),
                  color, -1)

    # Threshold line
    th_y = by + bar_h - int(bar_h * (threshold / 0.5))
    cv2.line(frame, (bx - 4, th_y), (bx + bar_w + 4, th_y), COL_WARN, 2)

    # Border
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (120, 120, 120), 1)

    # Label
    cv2.putText(frame, "EAR", (bx, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{ear_value:.2f}", (bx - 6, by + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_face_landmarks(frame, face_landmarks, frame_w, frame_h, color=(0, 255, 180)):
    """
    Draw small dots for every MediaPipe face mesh landmark.
    Shown only when SHOW_LANDMARKS = True.
    """
    if not SHOW_LANDMARKS:
        return
    for lm in face_landmarks.landmark:
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        cv2.circle(frame, (x, y), 1, color, -1)


def draw_eye_contours(frame, landmarks, frame_w, frame_h):
    """Draw green contour lines around both eyes."""
    from detector.eye_detector import LEFT_EYE, RIGHT_EYE

    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        pts = []
        for idx in eye_indices:
            lm = landmarks[idx]
            pts.append((int(lm.x * frame_w), int(lm.y * frame_h)))
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 230, 100), thickness=1)


def draw_logo(frame):
    """Draw the DriveSense brand text in the top-left corner."""
    cv2.putText(frame, "DriveSense", (14, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Driver Monitoring System", (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
