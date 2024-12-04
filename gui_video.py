import cv2
import time
import multiprocessing
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 +
                     (landmark1.y - landmark2.y) ** 2 +
                     (landmark1.z - landmark2.z) ** 2)

def is_hand_closed(landmarks):
    return calculate_distance(landmarks[12], landmarks[0]) < calculate_distance(landmarks[17], landmarks[0])

def acquire_hand_landmarks(duration, hand_queue):
    camera = cv2.VideoCapture(0)
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    start_time = time.time()
    while camera.isOpened() and time.time() - start_time < duration:
        success, frame = camera.read()
        if not success:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))
        timestamp = time.time()
        if detection_result:
            for hand_id, hand_landmarks in enumerate(detection_result.hand_landmarks):
                gesture = "Closed" if is_hand_closed(hand_landmarks) else "Open"
                hand_queue.put([timestamp, hand_id, gesture, hand_landmarks])
        time.sleep(0.001)
    camera.release()

def plot_hand_data(hand_queue, duration):
    hand_data = []
    start_time = time.time()
    while time.time() - start_time < duration:
        while not hand_queue.empty():
            hand_data.append(hand_queue.get())

        if hand_data:
            plt.clf()
            hand_df = pd.DataFrame(hand_data, columns=['timestamp', 'hand_id', 'gesture', 'landmarks'])
            gesture_mapping = {'Closed': 0, 'Open': 1}
            hand_df['gesture_num'] = hand_df['gesture'].map(gesture_mapping)
            plt.plot(hand_df['timestamp'], hand_df['gesture_num'], label='Gesture (Closed=0, Open=1)')
            plt.legend()
            plt.pause(0.001)

    plt.show()

def main():
    duration = 20
    hand_queue = multiprocessing.Queue()

    hand_process = multiprocessing.Process(target=acquire_hand_landmarks, args=(duration, hand_queue))
    plot_process = multiprocessing.Process(target=plot_hand_data, args=(hand_queue, duration))

    hand_process.start()
    plot_process.start()

    hand_process.join()
    plot_process.join()

if __name__ == "__main__":
    main()
