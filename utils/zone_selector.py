import cv2
import numpy as np


# ============================================================
# Store clicked points
# ============================================================

points = []


# ============================================================
# Mouse Callback
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        # Allow maximum 4 points
        if len(points) < 4:

            points.append((x, y))


# ============================================================
# Draw Zone
# ============================================================

def draw_zone(frame):

    global points


    # ========================================================
    # Draw Individual Points
    # ========================================================

    for point in points:

        cv2.circle(
            frame,
            point,
            6,
            (0, 255, 0),
            -1
        )


    # ========================================================
    # Draw Lines While Selecting
    # ========================================================

    if len(points) >= 2:

        pts = np.array(
            points,
            np.int32
        )

        pts = pts.reshape(
            (-1, 1, 2)
        )


        cv2.polylines(
            frame,
            [pts],
            False,
            (0, 255, 255),
            2
        )


    # ========================================================
    # Close Polygon After 4 Points
    # ========================================================

    if len(points) == 4:

        pts = np.array(
            points,
            np.int32
        )

        pts = pts.reshape(
            (-1, 1, 2)
        )


        cv2.polylines(
            frame,
            [pts],
            True,
            (0, 0, 255),
            3
        )


        # ----------------------------------------------------
        # Zone Label
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Monitoring Zone",
            (
                points[0][0],
                points[0][1] - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


# ============================================================
# Get Zone
# ============================================================

def get_zone():

    return points


# ============================================================
# Reset Zone
# ============================================================

def reset_zone():

    global points

    points.clear()