import cv2
import cv2.aruco as aruco

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Use the correct dictionary (based on marker generator site)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

# Define mapping from ID to room name
id_to_room = {
    1: "Kitchen",
    2: "Bedroom",
    3: "Bathroom",
    4: "Study Room"
}

print("🔍 Starting ArUco Detection... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            corner = corners[i]

            # Draw marker outline and center
            aruco.drawDetectedMarkers(frame, [corner], ids[i])

            cX = int(corner[0][:, 0].mean())
            cY = int(corner[0][:, 1].mean())
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

            # Get room name from ID
            room_name = id_to_room.get(marker_id, "Unknown")

            # Label on the video
            cv2.putText(frame, f"{room_name} (ID: {marker_id})", (cX - 40, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ✅ Print to terminal
            print(f"✅ Detected → ID: {marker_id}, Room: {room_name}")

    cv2.imshow("Aruco Detection - Room Labels", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
