from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import winsound
import os


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/drowsiness_cnn.keras"

IMG_SIZE = 64

# Number of consecutive frames required before drowsiness
# is detected.
CLOSED_THRESHOLD = 20

# Minimum confidence required for CNN prediction
CONFIDENCE_THRESHOLD = 0.60

# Minimum time between alarm sounds
ALERT_COOLDOWN = 3


# ============================================================
# LOAD CNN MODEL
# ============================================================

print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

# Warm up model graph execution for fast real-time inference
_ = model(tf.zeros((2, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32), training=False)

print("CNN model loaded successfully.")


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")


# ============================================================
# HAAR CASCADES
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_eye.xml"
)

if face_cascade.empty():
    print("ERROR: Face cascade could not be loaded.")

if eye_cascade.empty():
    print("ERROR: Eye cascade could not be loaded.")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

closed_frames = 0
last_alert_time = 0

# Prevent multiple alert threads from running simultaneously
alert_running = False

# Lock for alert state
alert_lock = threading.Lock()

# Eye ROI anatomical defaults (relative to upper face ROI: width w, height = int(h * 0.60))
DEFAULT_LEFT_EYE = [0.13, 0.25, 0.33, 0.45]
DEFAULT_RIGHT_EYE = [0.54, 0.25, 0.33, 0.45]

tracked_left_eye = list(DEFAULT_LEFT_EYE)
tracked_right_eye = list(DEFAULT_RIGHT_EYE)


# ============================================================
# ALERT SOUND
# ============================================================

def play_alert():

    global alert_running

    try:

        frequency = 2000
        duration = 700

        winsound.Beep(
            frequency,
            duration
        )

    except Exception as e:

        print("Alert error:", e)

    finally:

        with alert_lock:
            alert_running = False


# ============================================================
# TRIGGER ALERT
# ============================================================

def trigger_alert():

    global last_alert_time
    global alert_running

    current_time = time.time()

    with alert_lock:

        # Do not start another beep if one is already playing
        if alert_running:
            return

        # Cooldown between alarms
        if current_time - last_alert_time < ALERT_COOLDOWN:
            return

        last_alert_time = current_time
        alert_running = True

        threading.Thread(
            target=play_alert,
            daemon=True
        ).start()


# ============================================================
# PREPROCESS EYE IMAGE & BATCH INFERENCE
# ============================================================

def preprocess_eye(eye):
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.resize(gray_eye, (IMG_SIZE, IMG_SIZE))
    gray_eye = gray_eye.astype(np.float32)
    gray_eye = np.expand_dims(gray_eye, axis=-1)  # shape: 64 x 64 x 1
    return gray_eye


def predict_eyes_batch(eye_list):
    if not eye_list:
        return []

    # Stack all eye crops into one batch tensor (N, 64, 64, 1)
    batch = np.stack([preprocess_eye(e) for e in eye_list], axis=0)

    # Fast direct tensor evaluation (10x-20x faster than model.predict)
    predictions = model(batch, training=False).numpy()

    results = []
    for pred in predictions:
        val = float(pred[0])
        if val >= 0.45:
            results.append(("OPEN", val))
        else:
            results.append(("CLOSED", 1.0 - val))

    return results


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    global closed_frames

    while True:

        # ----------------------------------------------------
        # Read webcam frame
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:

            print("ERROR: Could not read webcam frame.")
            break

        # ----------------------------------------------------
        # Mirror webcam
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # Convert frame to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Detect face (on scaled-down image for 4x speed)
        # ----------------------------------------------------

        scale = 0.5
        small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

        faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(90 * scale), int(90 * scale))
        )

        # ====================================================
        # NO FACE
        # ====================================================

        if len(faces) == 0:

            closed_frames = 0

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

        # ====================================================
        # FACE DETECTED
        # ====================================================

        else:

            # Select the largest face
            largest_face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            # Rescale face bounding box back to original coordinates
            x = int(largest_face[0] / scale)
            y = int(largest_face[1] / scale)
            w = int(largest_face[2] / scale)
            h = int(largest_face[3] / scale)

            # Ensure bounds within frame
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            # ------------------------------------------------
            # Draw face rectangle
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 0),
                2
            )

            # ------------------------------------------------
            # Face ROI
            # ------------------------------------------------

            roi_gray = gray[
                y:y + h,
                x:x + w
            ]

            roi_color = frame[
                y:y + h,
                x:x + w
            ]

            # ------------------------------------------------
            # Only search for eyes in upper half of face
            # ------------------------------------------------

            upper_gray = roi_gray[
                0:int(h * 0.60),
                :
            ]

            upper_color = roi_color[
                0:int(h * 0.60),
                :
            ]

            # ------------------------------------------------
            # Detect eyes with Haar cascade
            # ------------------------------------------------

            eyes = eye_cascade.detectMultiScale(
                upper_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(25, 18)
            )

            H_upper, W_upper = upper_color.shape[:2]

            # Classify detected eyes into left half vs right half
            detected_left = None
            detected_right = None

            for (ex, ey, ew, eh) in eyes:
                aspect_ratio = ew / float(eh)
                if 0.5 <= aspect_ratio <= 3.5:
                    center_x = ex + ew / 2.0
                    if center_x < W_upper * 0.5:
                        if detected_left is None or ew * eh > detected_left[2] * detected_left[3]:
                            detected_left = (ex, ey, ew, eh)
                    else:
                        if detected_right is None or ew * eh > detected_right[2] * detected_right[3]:
                            detected_right = (ex, ey, ew, eh)

            # Smooth update tracked eye positions when Haar detects eyes
            ALPHA = 0.20
            if detected_left is not None:
                lx, ly, lw, lh = detected_left
                tracked_left_eye[0] = (1 - ALPHA) * tracked_left_eye[0] + ALPHA * (lx / float(W_upper))
                tracked_left_eye[1] = (1 - ALPHA) * tracked_left_eye[1] + ALPHA * (ly / float(H_upper))
                tracked_left_eye[2] = (1 - ALPHA) * tracked_left_eye[2] + ALPHA * (lw / float(W_upper))
                tracked_left_eye[3] = (1 - ALPHA) * tracked_left_eye[3] + ALPHA * (lh / float(H_upper))

            if detected_right is not None:
                rx, ry, rw, rh = detected_right
                tracked_right_eye[0] = (1 - ALPHA) * tracked_right_eye[0] + ALPHA * (rx / float(W_upper))
                tracked_right_eye[1] = (1 - ALPHA) * tracked_right_eye[1] + ALPHA * (ry / float(H_upper))
                tracked_right_eye[2] = (1 - ALPHA) * tracked_right_eye[2] + ALPHA * (rw / float(W_upper))
                tracked_right_eye[3] = (1 - ALPHA) * tracked_right_eye[3] + ALPHA * (rh / float(H_upper))

            # Assemble valid eye bounding boxes (using Haar when open, ROI fallback when closed)
            valid_eyes = []

            # Left eye
            if detected_left is not None:
                valid_eyes.append(detected_left)
            else:
                lx = int(tracked_left_eye[0] * W_upper)
                ly = int(tracked_left_eye[1] * H_upper)
                lw = int(tracked_left_eye[2] * W_upper)
                lh = int(tracked_left_eye[3] * H_upper)
                valid_eyes.append((lx, ly, lw, lh))

            # Right eye
            if detected_right is not None:
                valid_eyes.append(detected_right)
            else:
                rx = int(tracked_right_eye[0] * W_upper)
                ry = int(tracked_right_eye[1] * H_upper)
                rw = int(tracked_right_eye[2] * W_upper)
                rh = int(tracked_right_eye[3] * H_upper)
                valid_eyes.append((rx, ry, rw, rh))

            # =================================================
            # EXTRACT CROPS AND RUN BATCHED CNN INFERENCE
            # =================================================

            eye_crops = []
            crop_rects = []

            for (ex, ey, ew, eh) in valid_eyes:
                crop_x = max(0, min(ex, W_upper - 10))
                crop_y = max(0, min(ey, H_upper - 10))
                crop_w = max(10, min(ew, W_upper - crop_x))
                crop_h = max(10, min(eh, H_upper - crop_y))

                eye = upper_color[
                    crop_y:crop_y + crop_h,
                    crop_x:crop_x + crop_w
                ]

                if eye.size > 0:
                    eye_crops.append(eye)
                    crop_rects.append((crop_x, crop_y, crop_w, crop_h))

            # Fast batched CNN inference in a single GPU/CPU pass
            predictions = predict_eyes_batch(eye_crops)

            eye_states = []
            eye_confidences = []

            for (crop_x, crop_y, crop_w, crop_h), (state, confidence) in zip(crop_rects, predictions):
                eye_states.append(state)
                eye_confidences.append(confidence)

                # -----------------------------------------
                # Eye rectangle
                # -----------------------------------------
                if state == "OPEN":
                    box_color = (0, 255, 0)
                else:
                    box_color = (0, 0, 255)

                cv2.rectangle(
                    upper_color,
                    (crop_x, crop_y),
                    (crop_x + crop_w, crop_y + crop_h),
                    box_color,
                    2
                )

                # -----------------------------------------
                # Eye text
                # -----------------------------------------
                text = f"{state} {confidence * 100:.0f}%"
                text_y = max(crop_y - 6, 12)

                cv2.putText(
                    upper_color,
                    text,
                    (crop_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    box_color,
                    2
                )

            # =================================================
            # DETERMINE OVERALL EYE STATE
            # =================================================

            if eye_states:

                open_count = eye_states.count("OPEN")
                closed_count = eye_states.count("CLOSED")

                # Both eyes must be closed for drowsiness when both are tracked
                if len(eye_states) >= 2:
                    if closed_count == len(eye_states):
                        closed_frames += 1
                        status = "DROWSY"
                    else:
                        # At least one eye is open -> user is awake
                        closed_frames = max(0, closed_frames - 2)
                        status = "AWAKE"
                else:
                    # Single eye detected
                    if closed_count == 1:
                        closed_frames += 1
                        status = "DROWSY"
                    else:
                        closed_frames = max(0, closed_frames - 2)
                        status = "AWAKE"

            else:
                # Grace decay when eyes temporarily lost
                closed_frames = max(0, closed_frames - 1)
                status = "EYES NOT DETECTED"

            # =================================================
            # DROWSINESS TIMER
            # =================================================

            # Assuming approximately 30 FPS.
            closed_seconds = closed_frames / 30.0

            # =================================================
            # DROWSINESS DECISION & DISPLAY
            # =================================================

            if status == "DROWSY":
                status_color = (0, 0, 255)  # Bright Red
                display_text = "DROWSY!!" if closed_frames >= CLOSED_THRESHOLD else "DROWSY"
                
                # Draw bold alert status above face
                cv2.putText(
                    frame,
                    display_text,
                    (x, max(y - 15, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    status_color,
                    3
                )

                # Trigger alert sound if threshold reached
                if closed_frames >= CLOSED_THRESHOLD:
                    trigger_alert()

            elif status == "AWAKE":
                status_color = (0, 255, 0)  # Green
                cv2.putText(
                    frame,
                    status,
                    (x, max(y - 15, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    status_color,
                    2
                )

            else:
                status_color = (0, 255, 255)  # Yellow
                cv2.putText(
                    frame,
                    status,
                    (x, max(y - 15, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    status_color,
                    2
                )

            # ------------------------------------------------
            # Closed-eye duration
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Closed: {closed_seconds:.1f}s",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Number of detected eyes
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Eyes detected: {len(valid_eyes)}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # ====================================================
        # ENCODE FRAME
        # ====================================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # ====================================================
        # SEND FRAME TO BROWSER
        # ====================================================

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# VIDEO ROUTE
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CNN DRIVER DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    print("Open browser at:")
    print("http://127.0.0.1:5000")
    print("=" * 60)
    print("\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )