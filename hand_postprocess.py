import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def compute_angle(p1, p2, p3):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Create vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Compute the dot product and magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Compute the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Compute the angle in radians
    angle = np.arccos(cos_angle)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

# Define three points
point1 = [1, 0.4, 0]
point2 = [0.2, 0, 0]
point3 = [0, 1, 0]

# Compute the angle
angle = compute_angle(point1, point2, point3)
print(f"The angle between the vectors is {angle} degrees")
import matplotlib.pyplot as plt

# Plot the points and vectors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(*point1, color='r', label='Point 1')
ax.scatter(*point2, color='g', label='Point 2')
ax.scatter(*point3, color='b', label='Point 3')

# Plot vectors
ax.quiver(*point2, *(np.array(point1) - np.array(point2)), color='r', arrow_length_ratio=0.1)
ax.quiver(*point2, *(np.array(point3) - np.array(point2)), color='b', arrow_length_ratio=0.1)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set title
ax.set_title(f"The angle between the vectors is {angle:.2f} degrees")

# Show legend
ax.legend()

# Show plot
plt.show()