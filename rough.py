import cv2
import numpy as np
import winsound
import serial
esp32 = serial.Serial("COM5", 115200, timeout=1)
beam_status = 0
beam_count = 0
print("ESP32 connected on COM5")
from ultralytics import YOLO
from datetime import datetime
import time
import threading
from utils.zone_selector import (
    mouse_callback,
    draw_zone,
    get_zone,
    reset_zone
)

# =====================================
# Load YOLO Model
# =====================================
print("Loading YOLOv8n...")
model = YOLO("yolov8n.pt")
print("Model Loaded Successfully!")

# =====================================
# Open Webcam
# =====================================
cap = cv2.VideoCapture("10.227.173.62:81/stream")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cv2.namedWindow("CrowdGuard AI")
cv2.setMouseCallback("CrowdGuard AI", mouse_callback)

if not cap.isOpened():
    print("Unable to open webcam")
    exit()
    # =====================================
# LATEST FRAME CAMERA THREAD
# =====================================

latest_frame = None
camera_running = True

def camera_reader():
    global latest_frame

    while camera_running:
        ret, new_frame = cap.read()

        if ret:
            latest_frame = new_frame

camera_thread = threading.Thread(target=camera_reader, daemon=True)
camera_thread.start()

# =====================================
# Colors
# =====================================
BLACK = (20, 20, 20)
DARK = (35, 35, 35)
WHITE = (255, 255, 255)
GREEN = (0, 255, 100)
YELLOW = (0, 220, 255)
RED = (0, 0, 255)
BLUE = (255, 120, 0)
GRAY = (160, 160, 160)

font = cv2.FONT_HERSHEY_SIMPLEX
# ============================================================
# CROWD THRESHOLD
# ============================================================

CROWD_THRESHOLD = 1

alarm_active = False
# ============================================================
# ALARM FUNCTION
# ============================================================

def play_alarm():

    while alarm_active:

        winsound.Beep(1200, 500)

        time.sleep(0.3)
# ============================================================
# FPS VARIABLES
# ============================================================

prev_time = time.time()
fps = 0

## ============================================================
# MAIN LOOP
# ============================================================

while True:

    if esp32.in_waiting:
        data = esp32.readline().decode(errors="ignore").strip()
        print("ESP32 DATA:", data)

        if data.startswith("BEAM,"):
            parts = data.split(",")

            if len(parts) == 3:
                try:
                    beam_status = int(parts[1])
                    beam_count = int(parts[2])
                except ValueError:
                    pass

    # ========================================================
    # READ FRAME
    # ========================================================

# ========================================================
# READ FRAME
# ========================================================

        if latest_frame is None:
         continue

    frame = latest_frame.copy()
    frame = cv2.resize(frame, (960, 720))

    dashboard = np.zeros((720, 1280, 3), dtype=np.uint8)
    dashboard[:, :960] = frame
    frame = dashboard






        # ========================================================
        # DRAW ZONE
        # ========================================================

    #draw_zone(frame)
        # ========================================================
        # MONITORING ZONE
        # ========================================================

    zone = get_zone()

        # Fixed classroom dimensions
    CLASSROOM_LENGTH = 10
    CLASSROOM_WIDTH = 10

        # Fixed classroom area
    CLASSROOM_AREA = CLASSROOM_LENGTH * CLASSROOM_WIDTH
    PEOPLE_PER_SQM = 1
    CROWD_THRESHOLD = 1

        # Convert selected zone points into NumPy array
    if len(zone) >= 3:

            zone_array = np.array(
                zone,
                dtype=np.int32
            )
        # ========================================================
        # INSTRUCTIONS
        # ========================================================

    cv2.putText(
            frame,
            "Left Click: Add Point | R: Reset Zone | Q: Quit",
            (20, 90),
            font,
            0.7,
            YELLOW,
            2
        )


        # ========================================================
        #  DETECTION
        # ========================================================
    frame_count = globals().get("frame_count", 0)
    frame_count += 1
    globals()["frame_count"] = frame_count

    if frame_count % 3 == 0:
        results = model(frame, imgsz=320, verbose=False)
    else:
        results = []
            # ========================================================
    # YOLO DETECTION + BYTETRACK
    # ========================================================

    frame_count = globals().get("frame_count", 0)
    frame_count += 1
    globals()["frame_count"] = frame_count

    last_results = globals().get("last_results", [])

    # Run YOLO + ByteTrack every 3rd frame
    if frame_count % 3 == 0:
        last_results = model.track(
            frame,
            imgsz=320,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

    globals()["last_results"] = last_results
    results = last_results

    # ========================================================
    # PEOPLE COUNT
    # ========================================================

    total_people = 0
    zone_people = 0
       

        # ========================================================
        # PROCESS YOLO RESULTS
        # ========================================================

    for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                # ------------------------------------------------
                # PERSON CLASS
                # ------------------------------------------------

                if int(box.cls[0]) != 0:
                    continue


                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                confidence = float(box.conf[0])

                if confidence < 0.5:
                    continue


                total_people += 1


                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                # ------------------------------------------------
                # TRACKING ID
                # ------------------------------------------------

                if box.id is not None:
                    person_id = int(box.id[0])
                else:
                    person_id = -1


                # =================================================
                # PERSON BOUNDING BOX
                # =================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    GREEN,
                    2
                )


                # =================================================
                # PERSON ID
                # =================================================

                label = f"ID: {person_id}"

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        max(0, y1 - 30)
                    ),
                    (
                        x1 + 100,
                        y1
                    ),
                    GREEN,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1 + 5,
                        max(20, y1 - 8)
                    ),
                    font,
                    0.55,
                    BLACK,
                    2
                )


                # =================================================
                # PERSON CENTER POINT
                # =================================================

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # Draw center point

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    RED,
                    -1
                )


                # =================================================
                # CHECK MONITORING ZONE
                # =================================================

                if len(zone) >= 3:

                    inside_zone = cv2.pointPolygonTest(
                        zone_array,
                        (center_x, center_y),
                        False
                    )

                    if inside_zone >= 0:

                        zone_people += 1

                        cv2.circle(
                            frame,
                            (center_x, center_y),
                            7,
                            YELLOW,
                            -1
                        )
        # ========================================================
        # CROWD THRESHOLD CHECK
        # ========================================================

    if total_people >= CROWD_THRESHOLD and beam_count > 0:

            crowd_status = "CRITICAL"
        

            if not alarm_active:

                alarm_active = True
                esp32.write(b"ALERT\n")

                alarm_thread = threading.Thread(
                    target=play_alarm,
                    daemon=True
                )

                alarm_thread.start()
    elif zone_people >= int(CROWD_THRESHOLD * 0.7):
            crowd_status = "WARNING"
            alarm_active = False
            esp32.write(b"NORMAL\n")

    else:
            crowd_status = "NORMAL"
            alarm_active = False
            esp32.write(b"NORMAL\n")

        # ========================================================
        # FPS CALCULATION
        # ========================================================

    current_time_fps = time.time()

    elapsed = current_time_fps - prev_time

    if elapsed > 0:
            fps = int(1 / elapsed)

    prev_time = current_time_fps

        # ========================================================
        # PROFESSIONAL TOP HEADER
        # ========================================================

    cv2.rectangle(
            frame,
            (0, 0),
            (1280, 70),
            BLACK,
            -1
        )


    cv2.circle(
            frame,
            (35, 35),
            8,
            GREEN,
            -1
        )


    cv2.putText(
            frame,
            "CROWDGUARD AI",
            (55, 45),
            font,
            1.0,
            WHITE,
            2
        )


    cv2.putText(
            frame,
            "INTELLIGENT CROWD MONITORING",
            (315, 43),
            font,
            0.5,
            GRAY,
            1
        )


        # LIVE indicator

    cv2.circle(
            frame,
            (885, 35),
            7,
            RED,
            -1
        )

    cv2.putText(
            frame,
            "LIVE",
            (900, 42),
            font,
            0.6,
            WHITE,
            2
        )


        # Date and time

    current_datetime = datetime.now().strftime(
            "%d-%m-%Y  %I:%M:%S %p"
        )

    cv2.putText(
            frame,
            current_datetime,
            (990, 42),
            font,
            0.55,
            GREEN,
            2
        )


        # ========================================================
        # RIGHT INFORMATION PANEL
        # ========================================================

    panel_x = 1000
    panel_y = 85
    panel_width = 260
    panel_height = 550


    cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (
                panel_x + panel_width,
                panel_y + panel_height
            ),
            DARK,
            -1
        )


        # Panel title

    cv2.putText(
            frame,
            "SYSTEM STATUS",
            (
                panel_x + 20,
                panel_y + 35
            ),
            font,
            0.65,
            WHITE,
            2
        )


    cv2.line(
            frame,
            (
                panel_x + 20,
                panel_y + 50
            ),
            (
                panel_x + panel_width - 20,
                panel_y + 50
            ),
            GRAY,
            1
        )


        # ========================================================
        # CAMERA STATUS
        # ========================================================

    y = panel_y + 90


    cv2.circle(
            frame,
            (
                panel_x + 30,
                y - 5
            ),
            6,
            GREEN,
            -1
        )


    cv2.putText(
            frame,
            "CAMERA",
            (
                panel_x + 50,
                y
            ),
            font,
            0.55,
            GRAY,
            1
        )


    cv2.putText(
            frame,
            "ONLINE",
            (
                panel_x + 50,
                y + 25
            ),
            font,
            0.65,
            GREEN,
            2
        )


        # ========================================================
        # AI MODEL STATUS
        # ========================================================

    y += 90


    cv2.circle(
            frame,
            (
                panel_x + 30,
                y - 5
            ),
            6,
            GREEN,
            -1
        )


    cv2.putText(
            frame,
            "AI MODEL",
            (
                panel_x + 50,
                y
            ),
            font,
            0.55,
            GRAY,
            1
        )


    cv2.putText(
            frame,
            "YOLOv8n",
            (
                panel_x + 50,
                y + 25
            ),
            font,
            0.65,
            WHITE,
            2
        )


        # ========================================================
        # PEOPLE DETECTED
        # ========================================================

    y += 100


    cv2.putText(
            frame,
            "PEOPLE DETECTED",
            (
                panel_x + 20,
                y
            ),
            font,
            0.5,
            GRAY,
            1
        )


    cv2.putText(
            frame,
            f"{total_people:02d}",
            (
                panel_x + 20,
                y + 55
            ),
            font,
            1.6,
            WHITE,
            3
        )


        # ========================================================
        # PEOPLE INSIDE ZONE
        # ========================================================

    y += 95


    cv2.putText(
            frame,
            "ZONE PEOPLE",
            (
                panel_x + 20,
                y
            ),
            font,
            0.5,
            GRAY,
            1
        )


    cv2.putText(
            frame,
            f"{zone_people:02d}",
            (
                panel_x + 20,
                y + 45
            ),
            font,
            1.2,
            YELLOW,
            2
        )
            # ========================================================
    # LASER BEAM COUNT
    # ========================================================

    y += 70

    cv2.putText(
        frame,
        "LASER COUNT",
        (
            panel_x + 20,
            y
        ),
        font,
        0.5,
        GRAY,
        1
    )

    cv2.putText(
        frame,
        f"{beam_count:02d}",
        (
            panel_x + 20,
            y + 40
        ),
        font,
        1.2,
        GREEN,
        2
    )
        # ========================================================
        # CROWD THRESHOLD
        # ========================================================

    y += 65

    cv2.putText(
            frame,
            "THRESHOLD",
            (
                panel_x + 20,
                y
            ),
            font,
            0.5,
            GRAY,
            1
        )

    cv2.putText(
            frame,
            str(CROWD_THRESHOLD),
            (
                panel_x + 20,
                y + 35
            ),
            font,
            0.9,
            WHITE,
            2
        )
        # ========================================================
        # CLASSROOM AREA
        # ========================================================

    y += 70

    cv2.putText(
            frame,
            "CLASSROOM AREA",
            (
                panel_x + 20,
                y
            ),
            font,
            0.5,
            GRAY,
            1
        )

    cv2.putText(
            frame,
            "100 m²",
            (
                panel_x + 20,
                y + 35
            ),
            font,
            0.9,
            WHITE,
            2
        )

        # ========================================================
        # FPS
        # ========================================================

    y += 45


    cv2.putText(
            frame,
            "PROCESSING FPS",
            (
                panel_x + 20,
                y
            ),
            font,
            0.5,
            GRAY,
            1
        )


    cv2.putText(
            frame,
            str(fps),
            (
                panel_x + 20,
                y + 45
            ),
            font,
            1.2,
            GREEN,
            2
        )


        # ========================================================
        # SYSTEM STATUS
        # ========================================================

    y += 55


    cv2.putText(
            frame,
            "SYSTEM STATUS",
            (
                panel_x + 20,
                y
            ),
            font,
            0.5,
            GRAY,
            1
        )


        # ========================================================
        # CROWD STATUS
        # ========================================================

    if crowd_status == "NORMAL":

            status_color = GREEN

    elif crowd_status == "WARNING":

            status_color = YELLOW

    else:

            status_color = RED


    cv2.putText(
            frame,
            crowd_status,
            (
                panel_x + 20,
                y + 30
            ),
            font,
            0.65,
            status_color,
            2
        )
            # ========================================================
        # ALARM STATUS
        # ========================================================

    if alarm_active:

            cv2.putText(
                frame,
                "ALARM ACTIVE",
                (
                    panel_x + 20,
                    y + 60
                ),
                font,
                0.55,
                RED,
                2
            )
            # ========================================================
        # CRITICAL ALERT
        # ========================================================

    if crowd_status == "CRITICAL":

            cv2.rectangle(
                frame,
                (250, 560),
                (850, 625),
                RED,
                -1
            )

            cv2.putText(
                frame,
                "!!! CROWD LIMIT EXCEEDED !!!",
                (295, 605),
                font,
                0.8,
                WHITE,
                2
            )

        # ========================================================
        # BOTTOM STATUS BAR
        # ========================================================

    cv2.rectangle(
            frame,
            (0, 650),
            (1280, 720),
            BLACK,
            -1
        )


    cv2.circle(
            frame,
            (35, 685),
            6,
            GREEN,
            -1
        )


    cv2.putText(
            frame,
            "SYSTEM ACTIVE",
            (50, 692),
            font,
            0.55,
            WHITE,
            2
        )


    cv2.putText(
            frame,
            "CAMERA 01",
            (300, 692),
            font,
            0.55,
            GRAY,
            1
        )


    cv2.putText(
            frame,
            "R  Reset Zone",
            (500, 692),
            font,
            0.55,
            WHITE,
            1
        )


    cv2.putText(
            frame,
            "Q  Quit",
            (680, 692),
            font,
            0.55,
            WHITE,
            1
        )


    cv2.putText(
            frame,
            "CrowdGuard AI v2.0",
            (1080, 692),
            font,
            0.5,
            BLUE,
            1
        )


        # ========================================================
        # SHOW WINDOW
        # ========================================================

    cv2.imshow(
            "CrowdGuard AI",
            frame
        )


        # ========================================================
        # KEYBOARD CONTROLS
        # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):

            reset_zone()

    elif key == ord("q"):

            break


        # ============================================================
        # CLEANUP
        # ============================================================

cap.release()

cv2.destroyAllWindows()