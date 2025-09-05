# handSimple_realtime.py (robust backend + logging)

import os, sys, time
print("Starting script...")

# Pick a GUI backend that exists on your Mac
import matplotlib
for candidate in ("MacOSX", "TkAgg", "Qt5Agg"):
    try:
        matplotlib.use(candidate, force=True)
        print(f"Matplotlib backend set to: {candidate}")
        break
    except Exception as e:
        print(f"Backend {candidate} failed: {e}")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import numpy as np
from collections import deque

print("Importing MediaPipe...")
import mediapipe as mp

# ---------------- Tunables ----------------
REQUEST_720P = False
DISTURBANCE_AMPLITUDE_PX = 500
DISTURBANCE_FREQ_HZ = 0.1
HISTORY_SECONDS = 10.0
BUFFER_SIZE = 600
USE_HOLISTIC = True

# --------------- MediaPipe ----------------
print("Initializing MediaPipe model...")
if USE_HOLISTIC:
    mp_mod = mp.solutions.holistic
    model = mp_mod.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    def get_hand_landmarks(res):
        return getattr(res, "right_hand_landmarks", None)
else:
    mp_mod = mp.solutions.hands
    model = mp_mod.Hands(static_image_mode=False, model_complexity=1,
                         max_num_hands=1, min_detection_confidence=0.4,
                         min_tracking_confidence=0.5)
    def get_hand_landmarks(res):
        if getattr(res, "multi_hand_landmarks", None):
            return res.multi_hand_landmarks[0]
        return None

# ---------------- Camera ------------------
print("Opening camera...")
cap = cv2.VideoCapture(0)
if REQUEST_720P:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ok, frame = cap.read()
if not ok or frame is None:
    raise RuntimeError("Failed to read frame from webcam at startup")
frame_h, frame_w = frame.shape[:2]
print(f"Camera frame: {frame_w}x{frame_h}")

cx, cy = frame_w/2, frame_h/2
circle_radius = 0.1 * frame_w
target_radius = 1.1 * circle_radius
overlay_radius = int(circle_radius*0.5)

# ------------- Figures (guarded) ----------
plot_ok = True
try:
    fig_main, ax_main = plt.subplots(num="Cursor & Target")
    ax_main.set_facecolor('lightgray')
    ax_main.set_xlim(0, frame_w)
    ax_main.set_ylim(frame_h, 0)
    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_xmargin(0); ax_main.set_ymargin(0); ax_main.set_autoscale_on(False)

    target_circle = patches.Circle((cx, cy), target_radius, edgecolor='black', facecolor='none', linewidth=10)
    ax_main.add_patch(target_circle)
    circle = plt.Circle((cx, cy), circle_radius, color='red')
    ax_main.add_patch(circle)

    fig_ts, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, num="Live Signals")
    fig_ts.tight_layout()
    ax_top.set_title("Disturbance (red) & Hand X (blue)")
    ax_top.set_ylabel("Pixels"); ax_top.grid(True)
    line_disturb, = ax_top.plot([], [], '-', linewidth=1.6, label='Disturbance')
    line_hand,    = ax_top.plot([], [], '-', linewidth=1.2, label='Hand X')
    ax_top.legend(loc='upper right')

    ax_bot.set_title("Dot X Position (green)")
    ax_bot.set_ylabel("Pixels"); ax_bot.set_xlabel("Time (s)"); ax_bot.grid(True)
    line_dot, = ax_bot.plot([], [], '-', linewidth=1.6, label='Dot X')
    ax_bot.legend(loc='upper right')

except Exception as e:
    plot_ok = False
    print("Plotting setup failed; continuing with OpenCV only.")
    print("Matplotlib error:", repr(e))

# ----------- History buffers --------------
t_hist = deque(maxlen=BUFFER_SIZE)
dist_hist = deque(maxlen=BUFFER_SIZE)
hand_hist = deque(maxlen=BUFFER_SIZE)
dot_hist = deque(maxlen=BUFFER_SIZE)
start_time = time.perf_counter()

def get_hand_x_pixels(results, w):
    lm_container = get_hand_landmarks(results)
    if lm_container is None: return None
    landmarks = lm_container.landmark
    tips = {4, 8, 12, 16, 20}
    xs = [pt.x * w for i, pt in enumerate(landmarks) if i not in tips]
    return float(np.mean(xs)) if xs else None

# ------------- Main Loop ------------------
print("Entering main loop...")
plt.ion()
try:
    while True:
        ok, img = cap.read()
        if not ok or img is None:
            print("Camera read failed; exiting loop.")
            break

        h, w = img.shape[:2]
        if (w != frame_w) or (h != frame_h):
            frame_w, frame_h = w, h
            cx, cy = frame_w/2, frame_h/2
            circle_radius = 0.1 * frame_w
            target_radius = 1.1 * circle_radius
            overlay_radius = int(circle_radius*0.5)
            if plot_ok:
                ax_main.set_xlim(0, frame_w); ax_main.set_ylim(frame_h, 0)
                ax_main.set_aspect('equal', adjustable='box')
                target_circle.center = (cx, cy); target_circle.set_radius(target_radius)
                circle.set_radius(circle_radius)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = model.process(rgb)
        rgb.flags.writeable = True

        hand_x_raw = get_hand_x_pixels(results, frame_w)

        # OpenCV overlay for reference
        lm_container = get_hand_landmarks(results)
        if lm_container is not None:
            lms = lm_container.landmark
            tips = {4, 8, 12, 16, 20}
            xs = [int(p.x * frame_w) for i, p in enumerate(lms) if i not in tips]
            ys = [int(p.y * frame_h) for i, p in enumerate(lms) if i not in tips]
            if xs and ys:
                cv2.circle(img, (int(np.mean(xs)), int(np.mean(ys))), overlay_radius, (0,0,255), -1)

        # Time & disturbance
        t = time.perf_counter() - start_time
        disturbance = DISTURBANCE_AMPLITUDE_PX * np.sin(2*np.pi*DISTURBANCE_FREQ_HZ*t)
        base_x = (frame_w - hand_x_raw) if (hand_x_raw is not None) else cx
        margin = circle_radius
        dot_x = float(np.clip(base_x + disturbance, margin, frame_w - margin))

        if plot_ok:
            # Update cursor/target
            circle.center = (dot_x, cy)

            # Update plots
            t_hist.append(t)
            dist_hist.append(disturbance)
            hand_hist.append(base_x if hand_x_raw is not None else np.nan)
            dot_hist.append(dot_x)

            tmax = t_hist[-1]; tmin = max(0.0, tmax - HISTORY_SECONDS)
            ax_top.set_xlim(tmin, tmax); ax_bot.set_xlim(tmin, tmax)

            line_disturb.set_data(t_hist, dist_hist)
            line_hand.set_data(t_hist, hand_hist)
            line_dot.set_data(t_hist, dot_hist)

            for ax in (ax_top, ax_bot):
                ax.relim(); ax.autoscale_view()

            fig_main.canvas.draw_idle()
            fig_ts.canvas.draw_idle()
            fig_main.canvas.flush_events()
            fig_ts.canvas.flush_events()

        # OpenCV preview
        cv2.imshow("Webcam Feed", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit pressed.")
            break

        plt.pause(0.001)  # allow GUI to process events

except Exception as e:
    print("Exception in main loop:", repr(e))
finally:
    print("Cleaning up...")
    if hasattr(model, "close"):
        model.close()
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    if plot_ok:
        plt.show()
    print("Done.")
