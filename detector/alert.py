"""
DriveSense — Alert Manager
Handles two-level alerts:
  Level 1 (WARNING)  — yellow overlay + beep
  Level 2 (DANGER)   — red overlay + loud alarm
"""

import cv2
import numpy as np
import time
import os
import threading
from utils.config import ALERT_LEVEL_1_FRAMES, ALERT_LEVEL_2_FRAMES, ALERT_SOUND_FILE, ALERT_SOUND_VOLUME

# Try to load pygame for audio; gracefully skip if not installed
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False


def _generate_beep_wav(path, frequency=880, duration=0.4, volume=0.5):
    """Generate a simple beep WAV file using only stdlib + numpy."""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)

    import wave, struct
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wave.pack('<' + 'h' * len(wave_data := wave), *wave_data))


def _make_alert_sound(path):
    """Create a WAV beep programmatically (no external audio file needed)."""
    import wave
    sample_rate = 44100
    duration    = 0.5
    freq        = 880.0

    t      = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (np.sin(2 * np.pi * freq * t) * ALERT_SOUND_VOLUME * 32767).astype(np.int16)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())


class AlertManager:
    """
    Tracks cumulative alert frames and manages sound + visual overlays.
    """

    LEVEL_NONE    = 0
    LEVEL_WARNING = 1   # drowsy or distracted
    LEVEL_DANGER  = 2   # prolonged drowsy or distracted

    def __init__(self):
        self.level           = self.LEVEL_NONE
        self._alert_frames   = 0
        self._sound_loaded   = False
        self._last_beep_time = 0.0
        self._beep_cooldown  = 2.0   # seconds between beeps

        self._load_sound()

    # ── Sound ──────────────────────────────────────────────────────────────

    def _load_sound(self):
        if not PYGAME_OK:
            return
        if not os.path.exists(ALERT_SOUND_FILE):
            _make_alert_sound(ALERT_SOUND_FILE)
        try:
            self._sound = pygame.mixer.Sound(ALERT_SOUND_FILE)
            self._sound.set_volume(ALERT_SOUND_VOLUME)
            self._sound_loaded = True
        except Exception as e:
            print(f"[AlertManager] Sound load failed: {e}")

    def _play_sound(self):
        if not self._sound_loaded:
            return
        now = time.time()
        if now - self._last_beep_time < self._beep_cooldown:
            return
        self._last_beep_time = now
        threading.Thread(target=self._sound.play, daemon=True).start()

    # ── Update ─────────────────────────────────────────────────────────────

    def update(self, drowsy: bool, distracted: bool):
        """
        Call every frame with current detection flags.
        Returns the current alert level (0, 1, or 2).
        """
        if drowsy or distracted:
            self._alert_frames += 1
        else:
            # Decay slowly so overlay doesn't flash off instantly
            self._alert_frames = max(0, self._alert_frames - 2)

        if self._alert_frames >= ALERT_LEVEL_2_FRAMES:
            self.level = self.LEVEL_DANGER
            self._beep_cooldown = 1.0
            self._play_sound()
        elif self._alert_frames >= ALERT_LEVEL_1_FRAMES:
            self.level = self.LEVEL_WARNING
            self._beep_cooldown = 2.0
            self._play_sound()
        else:
            self.level = self.LEVEL_NONE

        return self.level

    # ── Visual overlay ─────────────────────────────────────────────────────

    def draw_overlay(self, frame):
        """
        Draw a semi-transparent colour border + banner on the frame.
        """
        if self.level == self.LEVEL_NONE:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        if self.level == self.LEVEL_DANGER:
            border_color  = (0, 0, 220)       # red (BGR)
            banner_color  = (0, 0, 180)
            label         = "  !! DANGER — PULL OVER !!  "
            text_color    = (255, 255, 255)
        else:
            border_color  = (0, 200, 220)      # yellow-ish
            banner_color  = (0, 160, 200)
            label         = "  WARNING — STAY ALERT  "
            text_color    = (255, 255, 255)

        # Border rectangle
        thickness = 12
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), border_color, thickness)

        # Top banner
        cv2.rectangle(overlay, (0, 0), (w, 52), banner_color, -1)

        # Text on banner
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.85
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
        tx = (w - tw) // 2
        cv2.putText(overlay, label, (tx, 36), font, font_scale, text_color, 2, cv2.LINE_AA)

        # Blend
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def label(self):
        if self.level == self.LEVEL_DANGER:
            return "DANGER"
        if self.level == self.LEVEL_WARNING:
            return "WARNING"
        return "OK"

    def color_bgr(self):
        if self.level == self.LEVEL_DANGER:
            return (0, 0, 220)
        if self.level == self.LEVEL_WARNING:
            return (0, 200, 220)
        return (0, 200, 80)
