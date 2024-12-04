import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Path to the CSV file
csv_file_path = 'c:/Users/VidaImre/OneDrive - Politecnico di Milano/University/hand/acquisition_git/parallel_acquisition_SAPIENS/hand_landmarks.csv'

# Read the CSV file into a DataFrame
landmarks_df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame
print(landmarks_df.head())

# Display the shape of the DataFrame
print(landmarks_df.shape)


import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Group by timestamp
grouped = landmarks_df.groupby('timestamp')

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Function to update the plot
def update(num, data, line):
    ax.clear()
    timestamp, group = data[num]
    ax.scatter(group['x'], group['y'], group['z'])
    
    # Color specific points red
    special_points = [0, 3, 4, 5, 9, 8, 12]
    for point in special_points:
        if point in group.index:
            ax.scatter(group.loc[point, 'x'], group.loc[point, 'y'], group.loc[point, 'z'], color='red')
    
    ax.set_title(f'Timestamp: {timestamp}')
    ax.set_xlim([landmarks_df['x'].min(), landmarks_df['x'].max()])
    ax.set_ylim([landmarks_df['y'].min(), landmarks_df['y'].max()])
    ax.set_zlim([landmarks_df['z'].min(), landmarks_df['z'].max()])

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(grouped), fargs=(list(grouped), None), interval=100)

# Display the plot
plt.show()