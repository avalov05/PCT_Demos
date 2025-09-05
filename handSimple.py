import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from collections import deque
import time

# -----------------------------
# Disturbance parameters
# -----------------------------
DISTURBANCE_AMPLITUDE_PX = 500     # pixels
DISTURBANCE_FREQ_HZ = 0.1        # Hz (cycles per second)

# -----------------------------
# MediaPipe setup (Holistic)
# -----------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Open webcam (optionally ask for 1280x720)
# -----------------------------
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -----------------------------
# First frame
# -----------------------------
ret, frame = cap.read()
if not ret or frame is None:
    raise RuntimeError("Failed to read frame from webcam")
frame_height, frame_width = frame.shape[:2]

# Centers & sizes
centerX_coord = frame_width / 2
centerY_coord = frame_height / 2
circle_radius = 0.1 * frame_width            # plot circle radius (pixels)
circle_radius_vid = int(circle_radius * 0.5) # video overlay dot radius

# -----------------------------
# Figure A: main cursor/target
# -----------------------------
fig_main, ax_main = plt.subplots(num="Cursor & Target")
ax_main.set_facecolor('lightgray')
ax_main.set_xlim(0, frame_width)
ax_main.set_ylim(frame_height, 0)          # top-left origin to match image coords
ax_main.set_aspect('equal', adjustable='box')
ax_main.set_xmargin(0)
ax_main.set_ymargin(0)
ax_main.set_autoscale_on(False)

target_radius = circle_radius * 1.1
target_circle = patches.Circle(
    (centerX_coord, centerY_coord),
    radius=target_radius,
    edgecolor='black',
    facecolor='none',
    linewidth=10,
)
ax_main.add_patch(target_circle)

circle = plt.Circle((centerX_coord, centerY_coord), circle_radius, color='red')
ax_main.add_patch(circle)

# -----------------------------
# Figure B: time-series plots
# -----------------------------
fig_ts, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(9, 6), sharex=True, num="Live Signals"
)
fig_ts.tight_layout()

ax_top.set_title("Disturbance (red) & Hand X (blue)")
ax_top.set_ylabel("Pixels")
ax_top.grid(True)
line_disturb, = ax_top.plot([], [], '-', label='Disturbance', linewidth=1.5)
line_hand,    = ax_top.plot([], [], '-', label='Hand X', linewidth=1.2)
ax_top.legend(loc='upper right')

ax_bottom.set_title("Dot X Position (green)")
ax_bottom.set_ylabel("Pixels")
ax_bottom.set_xlabel("Time (s)")
ax_bottom.grid(True)
line_dot, = ax_bottom.plot([], [], '-', label='Dot X', linewidth=1.5)
ax_bottom.legend(loc='upper right')

# -----------------------------
# History buffers
# -----------------------------
BUFFER_SIZE = 300  # ~10s at 30 fps
time_hist     = deque(maxlen=BUFFER_SIZE)
disturb_hist  = deque(maxlen=BUFFER_SIZE)
hand_hist     = deque(maxlen=BUFFER_SIZE)
dot_hist      = deque(maxlen=BUFFER_SIZE)

start_time = time.perf_counter()

# -----------------------------
# Helpers
# -----------------------------
def get_hand_x_position(results, width):
    """Return average non-fingertip x (pixels) for the right hand, or None."""
    if getattr(results, "right_hand_landmarks", None):
        landmarks = results.right_hand_landmarks.landmark
        tips = {4, 8, 12, 16, 20}  # exclude fingertips to reduce jitter
        xs = [lm.x * width for idx, lm in enumerate(landmarks) if idx not in tips]
        if xs:
            return float(np.mean(xs))
    return None

# -----------------------------
# Animation update
# -----------------------------
def update(_frame_index):
    global frame_width, frame_height, centerX_coord, centerY_coord
    global circle_radius, target_circle

    ret, img = cap.read()
    if not ret or img is None:
        return (circle, line_disturb, line_hand, line_dot)

    height, width = img.shape[:2]

    # Keep axes & geometry in sync if the camera size changes
    if frame_width != width or frame_height != height:
        frame_width, frame_height = width, height
        centerX_coord, centerY_coord = frame_width / 2, frame_height / 2

        ax_main.set_xlim(0, frame_width)
        ax_main.set_ylim(frame_height, 0)
        ax_main.set_aspect('equal', adjustable='box')

        circle_radius = 0.1 * frame_width
        target_circle.set_radius(circle_radius * 1.1)
        target_circle.center = (centerX_coord, centerY_coord)

    # MediaPipe (RGB, no writing)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = holistic.process(img_rgb)
    img_rgb.flags.writeable = True

    # Hand X in pixels
    hand_x_raw = get_hand_x_position(results, width)

    # Draw average point on OpenCV view (unmirrored, for reference)
    if getattr(results, "right_hand_landmarks", None):
        landmarks = results.right_hand_landmarks.landmark
        tips = {4, 8, 12, 16, 20}
        xs = [int(lm.x * width) for idx, lm in enumerate(landmarks) if idx not in tips]
        ys = [int(lm.y * height) for idx, lm in enumerate(landmarks) if idx not in tips]
        if xs and ys:
            avg_x_img, avg_y_img = int(np.mean(xs)), int(np.mean(ys))
            cv2.circle(img, (avg_x_img, avg_y_img), int(circle_radius * 0.5), (0, 0, 255), -1)

    # Timebase & disturbance
    t = time.perf_counter() - start_time
    disturbance = DISTURBANCE_AMPLITUDE_PX * np.sin(2 * np.pi * DISTURBANCE_FREQ_HZ * t)

    # Base X for plot: mirror to match the visual "move-right means x increases"
    if hand_x_raw is not None:
        base_x = width - hand_x_raw
    else:
        base_x = centerX_coord

    # Apply disturbance; keep the red circle fully on-screen
    margin = circle_radius
    dot_x = float(np.clip(base_x + disturbance, margin, frame_width - margin))

    # Update circle on the main figure
    circle.center = (dot_x, centerY_coord)

    # Show webcam feed (separate window)
    cv2.imshow('Webcam Feed', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # graceful shutdown
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()
        plt.close('all')
        raise SystemExit

    # ---- Update time-series data & lines (Figure B) ----
    time_hist.append(t)
    disturb_hist.append(disturbance)
    hand_hist.append(base_x if hand_x_raw is not None else np.nan)
    dot_hist.append(dot_x)

    # Set data
    line_disturb.set_data(time_hist, disturb_hist)
    line_hand.set_data(time_hist, hand_hist)
    line_dot.set_data(time_hist, dot_hist)

    # Autoscale view to the newest data
    for ax in (ax_top, ax_bottom):
        ax.relim()
        ax.autoscale_view()

    # Return artists (blit=False lets us update both figures cleanly)
    return (circle, line_disturb, line_hand, line_dot)

# -----------------------------
# Run animation
# -----------------------------
# Use one animation (tie it to fig_main); set blit=False since we update two figures.
ani = FuncAnimation(fig_main, update, interval=30, blit=False, cache_frame_data=False)

plt.show()

# Cleanup if windows are closed without pressing 'q'
holistic.close()
cap.release()
cv2.destroyAllWindows()
