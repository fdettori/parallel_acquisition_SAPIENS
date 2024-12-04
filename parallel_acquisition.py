import cv2
import mediapipe as mp
import time
import multiprocessing
import serial
import pandas as pd
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import read_eeg

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 +
                     (landmark1.y - landmark2.y) ** 2 +
                     (landmark1.z - landmark2.z) ** 2)

def is_hand_closed(landmarks):
    return calculate_distance(landmarks[12], landmarks[0]) < calculate_distance(landmarks[17], landmarks[0])

def acquire_signal(duration, signal_queue):
    with serial.Serial('COM4', 115200, timeout=1) as ser:
        start_time = time.time()
        while time.time() - start_time < duration:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip().split(',')
                if len(line) == 2:
                    signal_queue.put([time.time(), *line])
                time.sleep(0.001)

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

def acquire_eeg_data(duration, eeg_queue):
    read_eeg.main(duration, eeg_queue)

def plot_data(signal_queue, hand_queue, eeg_queue, duration):
    signal_data = []
    hand_data = []
    eeg_data = []

    start_time = time.time()
    while time.time() - start_time < duration:
        while not signal_queue.empty():
            signal_data.append(signal_queue.get())
        while not hand_queue.empty():
            hand_data.append(hand_queue.get())
        while not eeg_queue.empty():
            eeg_data.append(eeg_queue.get())

        if signal_data or hand_data or eeg_data:
            plt.clf()
            if signal_data:
                signal_df = pd.DataFrame(signal_data, columns=['timestamp', 'value1', 'value2'])
                plt.subplot(3, 1, 1)
                plt.plot(signal_df['timestamp'], signal_df['value1'], label='Value 1')
                plt.plot(signal_df['timestamp'], signal_df['value2'], label='Value 2')
                plt.legend()
            if hand_data:
                hand_df = pd.DataFrame(hand_data, columns=['timestamp', 'hand_id', 'gesture', 'landmarks'])
                gesture_mapping = {'Closed': 0, 'Open': 1}
                hand_df['gesture_num'] = hand_df['gesture'].map(gesture_mapping)
                plt.subplot(3, 1, 2)
                plt.plot(hand_df['timestamp'], hand_df['gesture_num'], label='Gesture (Closed=0, Open=1)')
                plt.legend()
            if eeg_data:
                eeg_df = pd.DataFrame(eeg_data, columns=['timestamp', 'eeg_value'])
                plt.subplot(3, 1, 3)
                plt.plot(eeg_df['timestamp'], eeg_df['eeg_value'], label='EEG Value')
                plt.legend()
            plt.pause(0.001)

    plt.show()

def main():
    duration = 20
    signal_queue = multiprocessing.Queue()
    hand_queue = multiprocessing.Queue()
    eeg_queue = multiprocessing.Queue()

    processes = [
        multiprocessing.Process(target=acquire_signal, args=(duration, signal_queue)),
        multiprocessing.Process(target=acquire_hand_landmarks, args=(duration, hand_queue)),
        multiprocessing.Process(target=acquire_eeg_data, args=(duration, eeg_queue)),
        multiprocessing.Process(target=plot_data, args=(signal_queue, hand_queue, eeg_queue, duration))
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()