import os
import numpy as np
import soundfile as sf

def generate_chill_loop(filename, duration_seconds=30):
    sr = 44100
    t = np.linspace(0, duration_seconds, int(sr * duration_seconds), endpoint=False)
    
    # --- Bassline (simple sine wave, low frequency) ---
    bass_freq = 120  # Hz
    bass = 0.2 * np.sin(2 * np.pi * bass_freq * t)
    bass *= np.sin(2 * np.pi * 0.25 * t) * 0.5 + 0.5  # subtle volume modulation

    # --- Plucky synth melody (higher frequency) ---
    melody = np.zeros_like(t)
    note_times = np.arange(0, duration_seconds, 0.5)
    for start in note_times:
        idx_start = int(start * sr)
        idx_end = idx_start + int(0.2 * sr)
        if idx_end > len(t):
            idx_end = len(t)
        freq = np.random.choice([400, 500, 600, 700, 800])
        envelope = np.exp(-5 * np.linspace(0, 0.2, idx_end-idx_start))
        melody[idx_start:idx_end] += 0.3 * np.sin(2 * np.pi * freq * np.linspace(0, 0.2, idx_end-idx_start)) * envelope

    # --- Light percussive clicks ---
    percussion = np.zeros_like(t)
    click_times = np.arange(0, duration_seconds, 0.5)
    for ct in click_times:
        idx = int(ct * sr)
        if idx < len(t):
            percussion[idx:idx+100] += 0.1 * np.random.randn(min(100, len(t)-idx))  # short noise bursts

    # --- Subtle background pad ---
    pad = 0.05 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.05 * t)*0.5 + 0.5)

    # --- Combine layers ---
    track = bass + melody + percussion + pad
    track /= np.max(np.abs(track))  # normalize

    # Ensure output directory exists and save WAV file (16-bit PCM)
    out_dir = os.path.dirname(os.path.abspath(filename))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    sf.write(filename, track.astype(np.float32), sr, subtype='PCM_16')

if __name__ == "__main__":
    # Save into project assets/sounds relative to this file
    project_root = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(project_root, "assets", "sounds", "bubble_chill_loop.wav")
    generate_chill_loop(out_path, 30)