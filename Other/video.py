import cv2
import mediapipe as mp
import csv
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utilities import draw_landmarks_on_image  # Assuming you already have this utility function

# STEP 1: Initialize the webcam and Mediapipe HandLandmarker object
camera = cv2.VideoCapture(0)  # 0 is usually the default camera

# STEP 2: Create a HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Prepare CSV file for saving landmarks
csv_file = open('hand_landmarks.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Write the header row for CSV
header = ['timestamp', 'hand', 'landmark_index', 'x', 'y', 'z']
csv_writer.writerow(header)

# STEP 4: Process the video stream frame by frame
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the frame to RGB as Mediapipe expects RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe image object from the frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # STEP 5: Detect hand landmarks from the frame
    detection_result = detector.detect(mp_image)

    # Get the current timestamp
    timestamp = time.time()

    # STEP 6: Save landmarks and timestamp to CSV
    if detection_result:
        for hand_id, hand_landmarks in enumerate(detection_result.hand_landmarks):
            for landmark in hand_landmarks:
                # Write the timestamp, hand id, landmark id, and (x, y, z) coordinates
                csv_writer.writerow([timestamp, hand_id, landmark, 
                                     landmark.x, landmark.y, landmark.z])

        # Visualize landmarks on the frame
        annotated_frame = draw_landmarks_on_image(frame, detection_result)
    else:
        annotated_frame = frame

    # Display the annotated frame
    cv2.imshow('Hand Landmarks', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# STEP 7: Release resources
camera.release()
csv_file.close()
cv2.destroyAllWindows()
