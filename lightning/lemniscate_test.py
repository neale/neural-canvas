import torch
import matplotlib.pyplot as plt
from neural_canvas.utils import utils

# Define a test function to plot the lemniscate interpolation path
@torch.no_grad()
def lemniscate_lerp(z1, z2, n, a=1):
    t_values = torch.linspace(0, 2 * torch.pi, n + 2)
    x = a * torch.cos(t_values) / (1 + torch.sin(t_values)**2)
    y = a * torch.cos(t_values) * torch.sin(t_values) / (1 + torch.sin(t_values)**2)
    
    # Normalize x, y to range between 0 and 1
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    
    # Reshape x and y to be compatible for broadcasting
    x = x.unsqueeze(-1)  # Adding a dimension for broadcasting
    y = y.unsqueeze(-1)  # Adding a dimension for broadcasting
    
    # Calculate the interpolated samples
    delta_x = z2['sample'] - z1['sample']
    x_scaled = z1['sample'] + delta_x * x
    y_scaled = z1['sample'] + delta_x * y
    
    # Combine x and y components
    samples = x_scaled + y_scaled * 1j  # Using complex numbers if that's acceptable

    # Prepare the states array
    states = []
    for i in range(n + 2):
        zx = {'sample': samples[i], 'sample_shape': z1['sample_shape']}
        states.append(zx)
    
    return states

def lemniscate_lerp_test(n, a=1):
    # Define two sample points in a 2D latent space
    z1 = {'sample': torch.tensor([0.0, 0.0]), 'sample_shape': (2,)}
    z2 = {'sample': torch.tensor([1.0, 1.0]), 'sample_shape': (2,)}
    
    # Generate the interpolation states
    states = lemniscate_lerp(z1, z2, n, a)
    # Extract x and y coordinates from the states for plotting
    x_coords = [state['sample'][0].item() for state in states]
    y_coords = [state['sample'][1].item() for state in states]

    # Plot the lemniscate path
    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, 'o-', label='Lemniscate path')
    plt.scatter([z1['sample'][0].item(), z2['sample'][0].item()], 
                [z1['sample'][1].item(), z2['sample'][1].item()], 
                color='red', label='Endpoints')
    plt.title('Lemniscate Interpolation Test')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Ensure the x and y scales are the same
    plt.show()

# Example usage of the test function with 10 interpolation points
lemniscate_lerp_test(10, a=0.5)
