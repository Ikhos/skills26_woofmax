import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt, detrend

previous_bpm = None
previous_box = None
last_valid_bpm = None


def bandpass_filter(signal, fs, low=0.8, high=3.0, order=3):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def parabolic_interpolation(power, idx):
    if idx <= 0 or idx >= len(power) - 1:
        return idx

    alpha = power[idx - 1]
    beta = power[idx]
    gamma = power[idx + 1]

    denom = (alpha - 2 * beta + gamma)
    if denom == 0:
        return idx

    delta = 0.5 * (alpha - gamma) / denom
    return idx + delta


def analyze_vitals(cap, face_box, duration=12):
    global previous_bpm, previous_box, last_valid_bpm

    if face_box is None:
        return {"heart_rate": last_valid_bpm, "confidence": 0.1}

    # ---------------- Motion Reset ----------------
    if previous_box is not None:
        px1, py1, px2, py2 = previous_box
        x1, y1, x2, y2 = face_box
        movement = abs(x1 - px1) + abs(y1 - py1)

        if movement > 12:
            previous_bpm = None

    previous_box = face_box
    # ---------------------------------------------

    x1, y1, x2, y2 = face_box

    values_r, values_g, values_b = [], [], []
    timestamps = []

    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        h = y2 - y1
        w = x2 - x1

        fx1 = int(x1 + w * 0.35)
        fx2 = int(x1 + w * 0.65)
        fy1 = int(y1 + h * 0.12)
        fy2 = int(y1 + h * 0.30)

        roi = frame[fy1:fy2, fx1:fx2]
        if roi.size == 0:
            continue

        mean_color = np.mean(roi.reshape(-1, 3), axis=0)

        values_b.append(mean_color[0])
        values_g.append(mean_color[1])
        values_r.append(mean_color[2])
        timestamps.append(time.time())

    if len(values_r) < 60:
        return {"heart_rate": last_valid_bpm, "confidence": 0.2}

    # ---------- True FPS ----------
    timestamps = np.array(timestamps)
    total_time = timestamps[-1] - timestamps[0]
    if total_time <= 0:
        return {"heart_rate": last_valid_bpm, "confidence": 0.2}

    fps = len(timestamps) / total_time

    r = detrend(np.array(values_r))
    g = detrend(np.array(values_g))
    b = detrend(np.array(values_b))

    # Normalize
    r = (r - np.mean(r)) / (np.std(r) + 1e-6)
    g = (g - np.mean(g)) / (np.std(g) + 1e-6)
    b = (b - np.mean(b)) / (np.std(b) + 1e-6)

    # ---------- POS Projection ----------
    S1 = g - b
    S2 = g + b - 2 * r
    alpha = np.std(S1) / (np.std(S2) + 1e-6)
    H = S1 + alpha * S2

    try:
        filtered = bandpass_filter(H, fs=fps)
    except Exception:
        return {"heart_rate": last_valid_bpm, "confidence": 0.2}

    filtered *= np.hanning(len(filtered))

    # ---------- FFT ----------
    fft = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(len(filtered), d=1 / fps)
    power = np.abs(fft)

    valid = (freqs >= 0.8) & (freqs <= 3.0)

    if not np.any(valid):
        return {"heart_rate": last_valid_bpm, "confidence": 0.3}

    valid_power = power[valid]
    valid_freqs = freqs[valid]

    peak_idx = np.argmax(valid_power)

    refined_idx = parabolic_interpolation(valid_power, peak_idx)
    refined_freq = np.interp(
        refined_idx,
        np.arange(len(valid_freqs)),
        valid_freqs
    )

    bpm = refined_freq * 60

    # ---------- Harmonic Correction ----------
    if previous_bpm is not None:
        if bpm < 65 and previous_bpm > 75:
            bpm *= 2

    # ---------- Clamp Instead of Reject ----------
    bpm = max(45, min(180, bpm))

    # ---------- Confidence Scoring ----------
    sorted_power = np.sort(valid_power)
    if len(sorted_power) >= 2:
        dominance_ratio = sorted_power[-1] / (sorted_power[-2] + 1e-6)
    else:
        dominance_ratio = 1.0

    confidence = min(0.95, max(0.3, dominance_ratio / 2.0))

    # ---------- Temporal Smoothing ----------
    if previous_bpm is not None:
        if abs(bpm - previous_bpm) > 20:
            bpm = previous_bpm
            confidence *= 0.6
        else:
            bpm = 0.7 * previous_bpm + 0.3 * bpm

    previous_bpm = int(bpm)
    last_valid_bpm = int(bpm)

    return {
        "heart_rate": int(bpm),
        "confidence": round(confidence, 2)
    }