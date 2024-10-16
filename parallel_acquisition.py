import cv2
import mediapipe as mp
import csv
import time
import threading
import serial  # For acquiring data from COM port
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utilities import draw_landmarks_on_image  # Assuming you already have this utility function
import math

# Function to calculate Euclidean distance between two points
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 +
                     (landmark1.y - landmark2.y) ** 2 +
                     (landmark1.z - landmark2.z) ** 2)

# Function to determine if hand is open or closed
def is_hand_closed(landmarks):
    # Thumb tip = 4, Index finger tip = 8, Middle finger tip = 12, Ring finger tip = 16, Pinky tip = 20
    # Palm base = 0
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    palm_base = landmarks[0]

    pinky_mcp = landmarks[17]

    # Calculate distances between each fingertip and the palm base
    pinky_mcp_dist = calculate_distance(pinky_mcp, palm_base)

    thumb_dist = calculate_distance(thumb_tip, palm_base)
    index_dist = calculate_distance(index_tip, palm_base)
    middle_dist = calculate_distance(middle_tip, palm_base)
    ring_dist = calculate_distance(ring_tip, palm_base)
    pinky_dist = calculate_distance(pinky_tip, palm_base)


    # If all fingertips are closer to the palm, the hand is closed
    # if thumb_dist < 0.5 and index_dist < 0.5 and middle_dist < 0.5 and ring_dist < 0.5 and pinky_dist < 0.5:
    if middle_dist < pinky_mcp_dist:
        return True  # Hand is closed
    else:
        return False  # Hand is open

# Function to acquire signal from COM3
def acquire_signal(duration):
    ser = serial.Serial('COM4', 115200, timeout=1)  # Open COM3 with baud rate 9600
    ser.flush()  # Clear the serial buffer
    
    
    start_time = time.time()  # Record the start time
    
    with open('data/signal_data.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['timestamp', 'value1', 'value2']
        csv_writer.writerow(header)
        
        while time.time() - start_time < duration:  # Run for the specified duration
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                print(line)
                values = line.split(',')  # Assuming two values are comma-separated
                if len(values) == 2:
                    timestamp = time.time()
                    csv_writer.writerow([timestamp, values[0], values[1]])
                time.sleep(0.001)  # Sleep for 1ms for 1kHz signal frequency

# Function to acquire hand landmarks using webcam and detect hand gestures
def acquire_hand_landmarks(duration):
    # Initialize the webcam and Mediapipe HandLandmarker object
    camera = cv2.VideoCapture(0)  # 0 is usually the default camera

    # Create a HandLandmarker object
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Prepare CSV file for saving landmarks and gesture info
    csv_file = open('data/hand_landmarks.csv', 'w', newline='')
    csv_file_gestures = open('data/hand_gestures.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer_gestures = csv.writer(csv_file_gestures)

    # Write the header row for CSV
    header = ['timestamp', 'hand', 'landmark_index', 'x', 'y', 'z']
    csv_writer.writerow(header)
    csv_writer_gestures.writerow(['timestamp', 'hand',  'gesture'])

    start_time = time.time()  # Record the start time

    # Process the video stream frame by frame
    while camera.isOpened() and time.time() - start_time < duration:
        success, frame = camera.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame to RGB as Mediapipe expects RGB images
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe image object from the frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hand landmarks from the frame
        detection_result = detector.detect(mp_image)

        # Get the current timestamp
        timestamp = time.time()

        # Save landmarks and gesture information to CSV
        if detection_result:
            for hand_id, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Determine if the hand is open or closed
                if is_hand_closed(hand_landmarks):
                    gesture = "Closed"
                    print("Closed")
                else:
                    gesture = "Open"
                    print("Open")

                for landmark in hand_landmarks:
                    # Write the timestamp, hand id, landmark id, (x, y, z) coordinates, and gesture
                    csv_writer.writerow([timestamp, hand_id, landmark,
                                         landmark.x, landmark.y, landmark.z])
                    
                csv_writer_gestures.writerow([timestamp, hand_id, gesture])

            # Visualize landmarks on the frame
            annotated_frame = draw_landmarks_on_image(frame, detection_result)
        else:
            annotated_frame = frame

        # Display the annotated frame
        cv2.imshow('Hand Landmarks', annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    camera.release()
    csv_file.close()
    csv_file_gestures.close()
    cv2.destroyAllWindows()

# Main function to run both signal acquisition and hand landmark acquisition in parallel
def main():

    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    duration = 20  # Set the duration for acquisition to 2 minutes (120 seconds)

    # Create threads for signal acquisition and hand landmark acquisition
    signal_thread = threading.Thread(target=acquire_signal, args=(duration,))
    hand_thread = threading.Thread(target=acquire_hand_landmarks, args=(duration,))

    # Start both threads
    signal_thread.start()
    hand_thread.start()

    # Wait for both threads to complete
    signal_thread.join()
    hand_thread.join()

if __name__ == "__main__":
    main()
    import plot
