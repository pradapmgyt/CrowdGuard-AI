import cv2
import time
import serial
from ultralytics import YOLO


# ============================================================
# CROWDGUARD AI - FINAL DETECTOR
# YOLOv8 + ByteTrack + ESP32 Laser + Buzzer
# ============================================================

# -------------------- SETTINGS --------------------

CAMERA_URL = "http://10.121.159.62:81/stream"  # ESP32-CAM URL
ESP32_PORT = "COM5"
BAUD_RATE = 115200

MODEL_PATH = "yolov8n.pt"

# Classroom
CLASSROOM_LENGTH = 10
CLASSROOM_WIDTH = 10
CLASSROOM_AREA = CLASSROOM_LENGTH * CLASSROOM_WIDTH  # 100 m²

# Final threshold
PEOPLE_PER_SQM = 1
CROWD_THRESHOLD = 1 # 100

# For testing, temporarily change to 1 if needed.
# CROWD_THRESHOLD = 1


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# CONNECT ESP32
# ============================================================

try:
    esp32 = serial.Serial(
        ESP32_PORT,
        BAUD_RATE,
        timeout=0.05
    )

    time.sleep(2)

    print("ESP32 connected on", ESP32_PORT)

except Exception as e:
    print("ESP32 connection failed:", e)
    esp32 = None


# ============================================================
# CONNECT CAMERA
# ============================================================

print("Connecting to ESP32-CAM...")

cap = cv2.VideoCapture(CAMERA_URL)

# Reduce OpenCV buffering
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except:
    pass

if not cap.isOpened():
    print("ERROR: Could not connect to ESP32-CAM")
    raise SystemExit

print("ESP32-CAM connected successfully.")


# ============================================================
# VARIABLES
# ============================================================

beam_count = 0
beam_status = 0

previous_people = 0

crowd_density = 0.0
density_rate = 0.0

previous_density = 0.0
previous_density_time = time.time()

system_status = "NORMAL"

last_command = ""

frame_count = 0
last_results = None

fps = 0
fps_counter = 0
fps_start_time = time.time()


# ============================================================
# SEND COMMAND TO ESP32
# ============================================================

def send_command(command):

    global last_command

    if esp32 is None:
        return

    # Send only when state changes.
    # This prevents serial flooding.
    if command != last_command:

        try:
            esp32.write((command + "\n").encode())

            print("ESP32 COMMAND:", command)

            last_command = command

        except Exception as e:
            print("ESP32 send error:", e)


# ============================================================
# DRAW TEXT
# ============================================================

def put_text(
    image,
    text,
    position,
    scale=0.6,
    thickness=2
):

    cv2.putText(
        image,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("======================================")
print(" CROWDGUARD AI STARTED")
print("======================================")
print("Camera :", CAMERA_URL)
print("ESP32  :", ESP32_PORT)
print("Area   :", CLASSROOM_AREA, "m²")
print("Threshold :", CROWD_THRESHOLD)
print("Press Q to exit.")
print()


while True:

    # ========================================================
    # READ ESP32 DATA
    # ========================================================

    if esp32 is not None:

        try:

            while esp32.in_waiting:

                line = esp32.readline().decode(
                    errors="ignore"
                ).strip()

                if line:

                    print("ESP32 DATA:", line)

                    # Expected:
                    # BEAM,1,5

                    if line.startswith("BEAM,"):

                        parts = line.split(",")

                        if len(parts) >= 3:

                            try:

                                beam_status = int(parts[1])
                                beam_count = int(parts[2])

                            except ValueError:
                                pass

        except Exception as e:

            print("Serial read error:", e)


    # ========================================================
    # READ CAMERA FRAME
    # ========================================================

    ret, frame = cap.read()

    if not ret:

        print("Camera frame lost. Reconnecting...")

        cap.release()

        time.sleep(0.5)

        cap = cv2.VideoCapture(CAMERA_URL)

        continue


    # ========================================================
    # FRAME COUNTER
    # ========================================================

    frame_count += 1


    # ========================================================
    # YOLO + BYTETRACK
    #
    # Run detection every 3rd frame.
    # Previous result is displayed between detections.
    # ========================================================

    if frame_count % 3 == 0:

        try:

            last_results = model.track(
                frame,
                imgsz=320,
                conf=0.25,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                verbose=False
            )

        except Exception as e:

            print("YOLO error:", e)


    # ========================================================
    # PEOPLE COUNT
    # ========================================================

    total_people = previous_people

    if last_results is not None:

        current_count = 0

        try:

            for result in last_results:

                if result.boxes is None:
                    continue

            for box in result.boxes:

                # Person class = 0
                if int(box.cls[0]) != 0:
                    continue

                confidence = float(box.conf[0])

                if confidence < 0.25:
                    continue

                current_count += 1

                # Draw YOLO bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Person label
                cv2.putText(
                    frame,
                    f"Person {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                    # Bounding boxes are intentionally NOT
                    # required for classroom-zone selection.

        except Exception as e:

            print("Detection processing error:", e)

        total_people = current_count

        previous_people = total_people


    # ========================================================
    # CROWD DENSITY
    #
    # Density = Number of people / Classroom area
    # ========================================================

    crowd_density = total_people / CLASSROOM_AREA


    # ========================================================
    # RATE OF CROWD DENSITY
    #
    # Rate = change in density / time
    #
    # Unit:
    # persons / m² / minute
    # ========================================================

    current_time = time.time()

    time_difference = current_time - previous_density_time

    if time_difference >= 1:

        density_rate = (
            (crowd_density - previous_density)
            / (time_difference / 60)
        )

        previous_density = crowd_density
        previous_density_time = current_time


    # ========================================================
    # CROWD STATUS
    # ========================================================

    # Laser count > 0 means the beam has detected
    # at least one crossing event.

    if total_people >= CROWD_THRESHOLD and beam_count > 0:

        system_status = "CRITICAL"

        send_command("ALERT")


    elif total_people >= max(1, int(CROWD_THRESHOLD * 0.7)):

        system_status = "WARNING"

        send_command("NORMAL")


    else:

        system_status = "NORMAL"

        send_command("NORMAL")


    # ========================================================
    # FPS
    # ========================================================

    fps_counter += 1

    elapsed = time.time() - fps_start_time

    if elapsed >= 1:

        fps = fps_counter / elapsed

        fps_counter = 0
        fps_start_time = time.time()


    # ========================================================
    # RESIZE CAMERA
    # ========================================================

    frame = cv2.resize(
        frame,
        (960, 720),
        interpolation=cv2.INTER_LINEAR
    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    dashboard = cv2.resize(
        frame,
        (1280, 720)
    )

    # Right-side dashboard panel
    panel_x = 960

    cv2.rectangle(
        dashboard,
        (panel_x, 0),
        (1280, 720),
        (35, 35, 35),
        -1
    )


    # ========================================================
    # TITLE
    # ========================================================

    put_text(
        dashboard,
        "CROWDGUARD AI",
        (985, 45),
        0.8,
        2
    )

    put_text(
        dashboard,
        "AI + IoT CROWD MONITORING",
        (985, 70),
        0.35,
        1
    )


    # ========================================================
    # PEOPLE
    # ========================================================

    put_text(
        dashboard,
        "PEOPLE DETECTED",
        (985, 115),
        0.45,
        1
    )

    put_text(
        dashboard,
        total_people,
        (985, 150),
        0.9,
        2
    )


    # ========================================================
    # CROWD DENSITY
    # ========================================================

    put_text(
        dashboard,
        "CROWD DENSITY",
        (985, 195),
        0.45,
        1
    )

    put_text(
        dashboard,
        f"{crowd_density:.3f} persons/m²",
        (985, 230),
        0.55,
        2
    )


    # ========================================================
    # DENSITY RATE
    # ========================================================

    put_text(
        dashboard,
        "DENSITY RATE",
        (985, 275),
        0.45,
        1
    )

    put_text(
        dashboard,
        f"{density_rate:.3f} /min",
        (985, 310),
        0.55,
        2
    )


    # ========================================================
    # LASER COUNT
    # ========================================================

    put_text(
        dashboard,
        "LASER COUNT",
        (985, 355),
        0.45,
        1
    )

    put_text(
        dashboard,
        beam_count,
        (985, 390),
        0.8,
        2
    )


    # ========================================================
    # THRESHOLD
    # ========================================================

    put_text(
        dashboard,
        "THRESHOLD",
        (985, 435),
        0.45,
        1
    )

    put_text(
        dashboard,
        f"{CROWD_THRESHOLD} people",
        (985, 470),
        0.55,
        2
    )


    # ========================================================
    # CLASSROOM AREA
    # ========================================================

    put_text(
        dashboard,
        "CLASSROOM AREA",
        (985, 515),
        0.45,
        1
    )

    put_text(
        dashboard,
        f"{CLASSROOM_AREA} m²",
        (985, 550),
        0.55,
        2
    )


    # ========================================================
    # FPS
    # ========================================================

    put_text(
        dashboard,
        "PROCESSING FPS",
        (985, 590),
        0.45,
        1
    )

    put_text(
        dashboard,
        f"{fps:.1f}",
        (985, 620),
        0.55,
        2
    )


    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    put_text(
        dashboard,
        "SYSTEM STATUS",
        (985, 660),
        0.45,
        1
    )

    put_text(
        dashboard,
        system_status,
        (1110, 660),
        0.55,
        2
    )
        # ========================================================
    # LARGE SYSTEM ALERT
    # ========================================================

    if system_status == "CRITICAL":

        cv2.rectangle(
            dashboard,
            (20, 20),
            (940, 85),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            dashboard,
            "!!! CROWD ALERT !!!",
            (230, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

    elif system_status == "WARNING":

        cv2.rectangle(
            dashboard,
            (20, 20),
            (940, 85),
            (0, 165, 255),
            -1
        )

        cv2.putText(
            dashboard,
            "!! CROWD WARNING !!",
            (225, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )


    # ========================================================
    # SHOW DASHBOARD
    # ========================================================

    cv2.imshow(
        "CrowdGuard AI",
        dashboard
    )


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

print("Stopping CrowdGuard AI...")

if esp32 is not None:

    try:
        esp32.write(b"NORMAL\n")
        time.sleep(0.2)
        esp32.close()

    except:
        pass


cap.release()

cv2.destroyAllWindows()

print("CrowdGuard AI stopped.")