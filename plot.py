import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
signal_data = pd.read_csv('data/signal_data.csv')
hand_gestures = pd.read_csv('data/hand_gestures.csv')

# Plotting
fig, axs = plt.subplots(3, figsize=(10, 8), sharex=True)

# Plot value1
axs[0].plot(signal_data['timestamp'], signal_data['value1'], label='Value 1', color='blue')
axs[0].set_ylabel('Value 1')
axs[0].legend()

# Plot value2
axs[1].plot(signal_data['timestamp'], signal_data['value2'], label='Value 2', color='orange')
axs[1].set_ylabel('Value 2')
axs[1].legend()

# Map gestures to numeric values
gesture_mapping = {'Closed': 0, 'Open': 1}
hand_gestures['gesture_num'] = hand_gestures['gesture'].map(gesture_mapping)

# Plot gesture
axs[2].plot(hand_gestures['timestamp'], hand_gestures['gesture_num'], label='Gesture (Closed=0, Open=1)', color='green')
axs[2].set_ylabel('Gesture')
axs[2].legend()

# Shared X-axis label
plt.xlabel('Timestamp')

plt.tight_layout()
plt.show()
