# ─────────────────────────────────────────────
#  DriveSense — Configuration / Thresholds
# ─────────────────────────────────────────────

# ── Eye Aspect Ratio (EAR) ──────────────────
EAR_THRESHOLD        = 0.25   # Below this → eye is "closed"
EAR_CONSEC_FRAMES    = 20     # Frames eye must stay closed → drowsy alert

# ── PERCLOS (% Eye Closure over time) ───────
PERCLOS_WINDOW       = 100    # Rolling window (frames) for PERCLOS
PERCLOS_THRESHOLD    = 0.35   # >35% closed in window → severe drowsiness

# ── Head Pose (distraction) ─────────────────
YAW_THRESHOLD        = 25     # degrees left/right
PITCH_THRESHOLD      = 20     # degrees up/down
HEAD_CONSEC_FRAMES   = 20     # frames off-road before alert

# ── Alert levels ────────────────────────────
ALERT_LEVEL_1_FRAMES = 20     # Warning  (yellow)
ALERT_LEVEL_2_FRAMES = 50     # Danger   (red)

# ── Display ─────────────────────────────────
FRAME_WIDTH          = 960
FRAME_HEIGHT         = 540
SHOW_LANDMARKS       = True   # Set False to hide face mesh dots
SHOW_EAR_VALUE       = True
SHOW_HEAD_ANGLES     = True

# ── Sound ───────────────────────────────────
ALERT_SOUND_FILE     = "assets/alert.wav"
ALERT_SOUND_VOLUME   = 0.8    # 0.0 – 1.0

# ── MediaPipe Face Mesh ──────────────────────
MAX_NUM_FACES        = 1
REFINE_LANDMARKS     = True
MIN_DETECTION_CONF   = 0.7
MIN_TRACKING_CONF    = 0.7
