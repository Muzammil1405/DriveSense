"""
DriveSense — Main Entry Point
Real-time driver drowsiness and distraction detection.

Usage:
    python main.py              # default webcam (index 0)
    python main.py --cam 1      # second camera
    python main.py --video path/to/video.mp4
    python main.py --no-sound
"""

import argparse
import time
import sys
import cv2
import mediapipe as mp

from detector.eye_detector import EyeMonitor
from detector.head_pose    import HeadPoseEstimator
from detector.alert        import AlertManager
from utils.drawing         import (
    draw_stats_panel, draw_ear_bar,
    draw_face_landmarks, draw_eye_contours, draw_logo
)
from utils.config import (
    FRAME_WIDTH, FRAME_HEIGHT,
    MAX_NUM_FACES, REFINE_LANDMARKS,
    MIN_DETECTION_CONF, MIN_TRACKING_CONF
)


# ── CLI arguments ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DriveSense — Driver Monitoring")
    p.add_argument("--cam",      type=int,   default=0,    help="Webcam index (default 0)")
    p.add_argument("--video",    type=str,   default=None, help="Path to video file")
    p.add_argument("--no-sound", action="store_true",      help="Disable audio alerts")
    p.add_argument("--no-land",  action="store_true",      help="Hide face landmarks")
    return p.parse_args()


# ── MediaPipe setup ───────────────────────────────────────────────────────────
def build_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces       = MAX_NUM_FACES,
        refine_landmarks    = REFINE_LANDMARKS,
        min_detection_confidence = MIN_DETECTION_CONF,
        min_tracking_confidence  = MIN_TRACKING_CONF
    )


# ── Main loop ────────────────────────────────────────────────────────────────
def run(args):
    # Source: file or webcam
    src = args.video if args.video else args.cam
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {src}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    eye_mon    = EyeMonitor()
    head_pose  = HeadPoseEstimator()
    alert_mgr  = AlertManager()

    face_mesh  = build_face_mesh()

    prev_time  = time.time()
    fps        = 0.0

    print("\n[DriveSense] Starting — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                print("[INFO] Video ended.")
            else:
                print("[ERROR] Frame grab failed.")
            break

        frame_h, frame_w = frame.shape[:2]

        # ── FPS calculation ──────────────────────────────────────────────
        now      = time.time()
        fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── MediaPipe inference ─────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        drowsy     = False
        distracted = False

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            lms  = face.landmark

            # Eye analysis
            drowsy = eye_mon.update(lms, frame_w, frame_h)

            # Head pose analysis
            distracted = head_pose.update(lms, frame_w, frame_h)

            # Drawing
            draw_face_landmarks(frame, face, frame_w, frame_h)
            draw_eye_contours(frame, lms, frame_w, frame_h)

        else:
            # No face detected — treat as distracted after a moment
            head_pose.consec_frames += 1
            if head_pose.consec_frames >= 30:
                distracted = True

        # ── Alert logic ─────────────────────────────────────────────────
        if args.no_sound:
            alert_mgr._sound_loaded = False
        alert_mgr.update(drowsy, distracted)

        # ── Draw HUD ────────────────────────────────────────────────────
        alert_mgr.draw_overlay(frame)
        draw_logo(frame)
        draw_stats_panel(frame, eye_mon, head_pose, alert_mgr, fps)
        draw_ear_bar(frame, eye_mon.ear)

        # ── Show ────────────────────────────────────────────────────────
        cv2.imshow("DriveSense — Driver Monitoring", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # Q or Esc
            break
        if key == ord('r'):
            eye_mon.reset()
            print("[INFO] Stats reset.")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[DriveSense] Stopped.")


if __name__ == "__main__":
    args = parse_args()
    run(args)
