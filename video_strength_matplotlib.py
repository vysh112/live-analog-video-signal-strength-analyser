import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ----------------------------
# Utility functions
# ----------------------------
def noise_level(frame_gray):
    lap = cv2.Laplacian(frame_gray, cv2.CV_64F)
    return lap.var()

def contrast_score(frame_gray):
    return frame_gray.std()

def color_saturation(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].mean()

def line_interference(frame_gray):
    f = np.fft.fft2(frame_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h_band = magnitude[magnitude.shape[0]//2 - 5 : magnitude.shape[0]//2 + 5, :]
    v_band = magnitude[:, magnitude.shape[1]//2 - 5 : magnitude.shape[1]//2 + 5]
    band_energy = h_band.sum() + v_band.sum()
    total_energy = magnitude.sum() + 1e-6
    return (band_energy / total_energy) * 1000

def flicker_index(prev_frame_gray, frame_gray):
    if prev_frame_gray is None:
        return 0
    diff = cv2.absdiff(prev_frame_gray, frame_gray)
    return diff.mean()

def entropy_score(frame_gray):
    hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return (entropy / 8.0) * 100

def normalize(value, min_val, max_val, invert=False):
    value = max(min_val, min(value, max_val))
    norm = (value - min_val) / (max_val - min_val) * 100
    return 100 - norm if invert else norm

def compute_signal_strength(noise, contrast, saturation, lines, flicker, entropy):
    noise_score = normalize(noise, 0, 5000, invert=True)
    contrast_score_ = normalize(contrast, 5, 80)
    saturation_score = normalize(saturation, 0, 200)
    line_score = normalize(lines, 0, 5, invert=True)
    flicker_score = normalize(flicker, 0, 20, invert=True)
    entropy_score_norm = normalize(entropy, 0, 100)

    strength = (0.2 * noise_score +
                0.2 * contrast_score_ +
                0.15 * saturation_score +
                0.15 * line_score +
                0.15 * flicker_score +
                0.15 * entropy_score_norm)

    if saturation < 5 and entropy > 70:
        strength = min(strength, 10)
    return np.clip(strength, 0, 100)

# ----------------------------
# Video Inputs
# ----------------------------
video1 = "C:\Vas Thesis Repo\ERC Test Site\VID_0055.TS"
video2 = "C:\Vas Thesis Repo\ERC Test Site\VID_0056.TS"

cap1 = cv2.VideoCapture(video1)
cap2 = cv2.VideoCapture(video2)

prev1 = None
prev2 = None

frames = []
metrics1 = {"noise": [], "contrast": [], "saturation": [], "lines": [], "flicker": [], "entropy": [], "strength": []}
metrics2 = {"noise": [], "contrast": [], "saturation": [], "lines": [], "flicker": [], "entropy": [], "strength": []}

# --- Live Plot Setup ---
plt.ion()
fig, axs = plt.subplots(7, 1, figsize=(10, 14), sharex=True)

# Position matplotlib window based on backend
backend = matplotlib.get_backend()
print("Matplotlib backend:", backend)

if backend == 'Qt5Agg' or backend == 'QtAgg':
    fig.canvas.manager.window.move(800,0)
elif backend == 'TkAgg':
    fig.canvas.manager.window.wm_geometry("+800+100")
elif backend == 'WXAgg':
    fig.canvas.manager.window.SetPosition((800, 100))
else:
    print("Window positioning not implemented for backend:", backend)

titles = ["Noise", "Contrast", "Saturation", "Lines", "Flicker", "Entropy", "Signal Strength"]
plots = []
for ax, title in zip(axs, titles):
    ax.set_ylabel(title)
    p1, = ax.plot([], [], label='Video 1', color='blue')
    p2, = ax.plot([], [], label='Video 2', color='orange')
    ax.legend()
    plots.append((p1, p2))
axs[-1].set_xlabel("Frames")

# --- OpenCV Window Setup ---
cv2.namedWindow("FPV Dual Video Comparison", cv2.WINDOW_NORMAL)
cv2.moveWindow("FPV Dual Video Comparison", 0, 100)

frame_count = 0

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()

    if not ret1 or not ret2:
        break

    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # --- Video 1 metrics ---
    n1 = noise_level(g1)
    c1 = contrast_score(g1)
    s1 = color_saturation(f1)
    l1 = line_interference(g1)
    fl1 = flicker_index(prev1, g1)
    e1 = entropy_score(g1)
    st1 = compute_signal_strength(n1, c1, s1, l1, fl1, e1)

    # --- Video 2 metrics ---
    n2 = noise_level(g2)
    c2 = contrast_score(g2)
    s2 = color_saturation(f2)
    l2 = line_interference(g2)
    fl2 = flicker_index(prev2, g2)
    e2 = entropy_score(g2)
    st2 = compute_signal_strength(n2, c2, s2, l2, fl2, e2)

    # Append
    frames.append(frame_count)
    for k, v1, v2 in zip(metrics1.keys(),
                         [n1, c1, s1, l1, fl1, e1, st1],
                         [n2, c2, s2, l2, fl2, e2, st2]):
        metrics1[k].append(v1)
        metrics2[k].append(v2)

    # --- Update Live Plot every 1 frames ---
    if frame_count % 1 == 0:
        for (p1, p2), key in zip(plots, metrics1.keys()):
            p1.set_data(frames, metrics1[key])
            p2.set_data(frames, metrics2[key])
            ax = p1.axes
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.001)

    # --- Combine video previews ---
    h1, w1 = f1.shape[:2]
    h2, w2 = f2.shape[:2]
    scale = 0.5
    f1_resized = cv2.resize(f1, (int(w1*scale), int(h1*scale)))
    f2_resized = cv2.resize(f2, (int(w2*scale), int(h2*scale)))
    combined = np.hstack((f1_resized, f2_resized))

    # Overlay strengths
    cv2.putText(combined, f"Video1 Strength: {st1:.1f}%", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, f"Video2 Strength: {st2:.1f}%", (f1_resized.shape[1]+20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("FPV Dual Video Comparison", combined)

    prev1 = g1
    prev2 = g2
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# ----------------------------
# Final Plots
# ----------------------------
plt.ioff()
fig2, axs2 = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
for ax, title, key in zip(axs2, titles, metrics1.keys()):
    ax.plot(frames, metrics1[key], label="Relay Link", color='blue')
    ax.plot(frames, metrics2[key], label="Direct Link", color='orange')
    ax.set_ylabel(title)
    ax.legend()
axs2[-1].set_xlabel("Frames")
plt.tight_layout()
plt.show()
