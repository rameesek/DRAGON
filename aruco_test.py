import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read from camera. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        print("Detected IDs:", ids)

    cv2.imshow('Aruco Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
