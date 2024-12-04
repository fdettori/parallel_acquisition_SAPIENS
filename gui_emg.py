import time
import multiprocessing
import serial
import pandas as pd
import matplotlib.pyplot as plt
import csv

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

def plot_signal_data(signal_queue, duration):
    signal_data = []
    start_time = time.time()
    while time.time() - start_time < duration:
        while not signal_queue.empty():
            signal_data.append(signal_queue.get())

        if signal_data:
            plt.clf()
            signal_df = pd.DataFrame(signal_data, columns=['timestamp', 'value1', 'value2'])
            plt.plot(signal_df['timestamp'], signal_df['value1'], label='Value 1')
            plt.plot(signal_df['timestamp'], signal_df['value2'], label='Value 2')
            plt.legend()
            plt.pause(0.001)

    plt.show()

def main():
    duration = 20

    signal_process = multiprocessing.Process(target=acquire_signal, args=(duration,))

    signal_process.start()

    signal_process.join()

if __name__ == "__main__":
    main()


