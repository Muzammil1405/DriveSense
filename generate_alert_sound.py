"""
Run this script once to generate the alert.wav file in the assets/ folder.
    python generate_alert_sound.py
"""
import numpy as np
import wave
import os

def generate(path="assets/alert.wav", freq=880, duration=0.5, volume=0.8, sample_rate=44100):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t      = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())

    print(f"[OK] Generated: {path}")

if __name__ == "__main__":
    generate()
