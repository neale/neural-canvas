import torch
import numpy as np
import matplotlib.pyplot as plt

def slerp(pt1, pt2, n):
    # Ensure the points are normalized to lie on the unit sphere
    pt1_norm = pt1 / torch.norm(pt1, p=2, dim=1, keepdim=True)
    pt2_norm = pt2 / torch.norm(pt2, p=2, dim=1, keepdim=True)

    # Calculate the angle between the points
    dot = torch.clamp(torch.sum(pt1_norm * pt2_norm, dim=1), -1.0, 1.0)
    theta = torch.acos(dot)  # angle between input vectors
    
    # Create an array of angles from 0 to 2*pi with n points
    angles = torch.linspace(0, 2 * np.pi, n+1)[:-1]  # remove the last value to prevent duplicating the first point

    # Use SLERP formula to interpolate
    sin_theta = torch.sin(theta)
    slerp_points = []
    for angle in angles:
        alpha = torch.sin((1.0 - angle/theta) * theta) / sin_theta
        beta = torch.sin(angle/theta * theta) / sin_theta
        slerp_point = alpha * pt1_norm + beta * pt2_norm
        slerp_points.append(slerp_point)

    # Concatenate all interpolated points into a tensor
    return torch.cat(slerp_points, dim=0)

# Define two points
pt1 = torch.tensor([[1.0, 0.0, 0.0]])  # Point A on the unit sphere
pt2 = torch.tensor([[0.0, 1.0, 0.0]])  # Point B on the unit sphere

pt1 = torch.randn(1, 8)
pt2 = torch.randn(1, 8)
# Generate n interpolated points along the great circle
n = 100  # number of interpolated points
interpolated_points = slerp(pt1, pt2, n)

# For visualization purposes, we'll plot the 2D projection of these points
plt.figure(figsize=(6, 6))
plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], 'b.')
plt.scatter(pt1[:, 0], pt1[:, 1], color='r' )
plt.scatter(pt2[:, 0], pt2[:, 1], color='r' )

plt.axis('equal')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Great Circle Interpolation')
plt.grid(True)
plt.show()
