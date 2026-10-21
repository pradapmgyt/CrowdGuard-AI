import cv2

cap = cv2.VideoCapture("http://192.168.0.5:81/stream")

if not cap.isOpened():
    print("Camera failed")
    exit()

print("Camera connected")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame failed")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()