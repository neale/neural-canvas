import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep

def random_closed_spline_trajectory(n, num_control_points=4, degree=3, dimensions=8):
    # Generate random control points within [0, 10) range for 8 dimensions
    random_points = np.random.rand(num_control_points, dimensions) * 10

    # Ensure the spline is closed by duplicating the first point at the end
    closed_points = np.vstack((random_points, random_points[0]))

    # Use scipy's splprep to create a tck representation of the spline, with a periodic condition
    tck, u = splprep(closed_points.T, per=True, k=min(degree, num_control_points-1))

    # Sample n points along the spline
    u_new = np.linspace(0, 1, n)
    points = np.array(splev(u_new, tck, der=0)).T

    return points

# Test the function for 8 dimensions
n = 100  # number of points in the trajectory
trajectory = random_closed_spline_trajectory(n, dimensions=8)

# Plotting only the first two dimensions
plt.figure(figsize=(8, 5))
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-', color='blue')
plt.title('Projection of 8D Closed Spline Trajectory onto First Two Dimensions')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.axis('equal')
plt.show()
