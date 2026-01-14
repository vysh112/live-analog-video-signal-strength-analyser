import cv2
import numpy as np

# ----------------------------
# Utility functions
# ----------------------------

def noise_level(frame_gray):
    """Estimate salt-and-pepper noise (isolated black/white pixels)."""
    lap = cv2.Laplacian(frame_gray, cv2.CV_64F)
    return lap.var()

def contrast_score(frame_gray):
    """Contrast = standard deviation of intensities."""
    return frame_gray.std()

def color_saturation(frame_bgr):
    """Average saturation in HSV space (0–255)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].mean()

def line_interference(frame_gray):
    """Detect repetitive horizontal/vertical line noise via normalized FFT energy."""
    f = np.fft.fft2(frame_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Horizontal and vertical frequency bands
    h_band = magnitude[magnitude.shape[0]//2 - 5 : magnitude.shape[0]//2 + 5, :]
    v_band = magnitude[:, magnitude.shape[1]//2 - 5 : magnitude.shape[1]//2 + 5]

    band_energy = h_band.sum() + v_band.sum()
    total_energy = magnitude.sum() + 1e-6  # avoid div by zero

    return (band_energy / total_energy) * 1000  # scaled ratio

def flicker_index(prev_frame_gray, frame_gray):
    """Frame-to-frame brightness change (flicker)."""
    if prev_frame_gray is None:
        return 0
    diff = cv2.absdiff(prev_frame_gray, frame_gray)
    return diff.mean()

def entropy_score(frame_gray):
    """Relative Shannon entropy of grayscale histogram."""
    hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))

    # Normalize entropy between 0 (flat) and ~8 (max spread for 8-bit)
    return (entropy / 8.0) * 100

# ----------------------------
# Normalization helper
# ----------------------------
def normalize(value, min_val, max_val, invert=False):
    """Map raw metric to 0–100 scale with clamping."""
    value = max(min_val, min(value, max_val))
    norm = (value - min_val) / (max_val - min_val) * 100
    return 100 - norm if invert else norm

# ----------------------------
# Signal Strength Aggregation
# ----------------------------
def compute_signal_strength(noise, contrast, saturation, lines, flicker, entropy):
    """Combine metrics into a 0–100 signal strength score."""

    # Normalize each metric (ranges tuned for analog FPV)
    noise_score = normalize(noise, 0, 5000, invert=True)       # less noise = better
    contrast_score = normalize(contrast, 5, 80)                # more contrast = better
    saturation_score = normalize(saturation, 0, 200)           # more color = better
    line_score = normalize(lines, 0, 5, invert=True)           # fewer lines = better
    flicker_score = normalize(flicker, 0, 20, invert=True)     # less flicker = better
    entropy_score_norm = normalize(entropy, 0, 100)            # higher entropy = better

    # Weighted average
    strength = (0.2 * noise_score +
                0.2 * contrast_score +
                0.15 * saturation_score +
                0.15 * line_score +
                0.15 * flicker_score +
                0.15 * entropy_score_norm)

    # Absolute feed loss check: grey static screen
    if saturation < 5 and entropy > 70:
        strength = min(strength, 10)

    return np.clip(strength, 0, 100)

# ----------------------------
# Main Video Processing
# ----------------------------

video_path = video_path = "C:\Vas Thesis Repo\CHAPTERS\Chapter 4\Weeganhasen Test\Analogue DVR\VID_0046.TS"
cap = cv2.VideoCapture(video_path)

prev_gray = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Metrics ---
    noise = noise_level(frame_gray)
    contrast = contrast_score(frame_gray)
    saturation = color_saturation(frame)
    lines = line_interference(frame_gray)
    flicker = flicker_index(prev_gray, frame_gray)
    entropy = entropy_score(frame_gray)

    # --- Signal Strength ---
    strength = compute_signal_strength(noise, contrast, saturation, lines, flicker, entropy)

    # --- Overlay on video ---
    overlay = frame.copy()

    # Traffic light coloring
    if strength > 50:
        color = (0, 255, 0)   # Green
    elif strength > 40:
        color = (0, 255, 255) # Yellow
    else:
        color = (0, 0, 255)   # Red

    cv2.putText(overlay, f"Signal Strength: {strength:.1f}%", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Show all metrics
    cv2.putText(overlay, f"Noise: {noise:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(overlay, f"Contrast: {contrast:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(overlay, f"Saturation: {saturation:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(overlay, f"Lines: {lines:.2f}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(overlay, f"Flicker: {flicker:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(overlay, f"Entropy: {entropy:.2f}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Warning overlay for near total loss
    if strength < 15:
        cv2.putText(overlay, "!!! SIGNAL LOST !!!", (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("FPV Quality Monitor", overlay)

    prev_gray = frame_gray.copy()

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):  # jump to 1 min
        cap.set(cv2.CAP_PROP_POS_MSEC, 60000)
    elif key == ord('2'):  # jump to 2 min
        cap.set(cv2.CAP_PROP_POS_MSEC, 120000)
    elif key == ord('3'):  # jump to 2 min
        cap.set(cv2.CAP_PROP_POS_MSEC, 60000*3)
    elif key == ord('4'):  # jump to 4 min
        cap.set(cv2.CAP_PROP_POS_MSEC, 60000*4)
    elif key == ord('5'):  # jump to 4.2 min
        cap.set(cv2.CAP_PROP_POS_MSEC, 60000*4.2)

cap.release()
cv2.destroyAllWindows()

