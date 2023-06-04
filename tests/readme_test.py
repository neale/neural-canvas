import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_canvas.models.inrf import INRF2D

np.random.seed(42)
torch.manual_seed(42)

# Create a 2D implicit neural representation model
model = INRF2D()
# Generate the image given by the random INRF
size = (256, 256)
img = model.generate(output_shape=size)
#perturb the image by generating a random vector as an additional input
img2 = model.generate(output_shape=size, sample_latent=True)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img.squeeze())
axes[1].imshow(img2.squeeze())
axes[0].set_title('Gen image')
axes[1].set_title('Gen image with latent vector')
plt.show()
##
##

size = (256, 256)
latents = model.sample_latents(output_shape=size)  # random vector
fields = model.construct_fields(output_shape=size)  # XY coordinate inputs
img = model.generate(latents, fields)

size = (1024, 1024)
latents_lrg = model.sample_latents(output_shape=size, reuse_latents=latents)
fields_lrg = model.construct_fields(output_shape=size)
img_lrg = model.generate(latents_lrg, fields_lrg)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img.squeeze())
axes[1].imshow(img_lrg.squeeze())
axes[0].set_title('Gen image 256x256')
axes[1].set_title('Same image 1024x1024')
plt.show()
##
##

zoom_xy = (.25, .25) # zoom out -- xy zoom range is (0, inf)
pan_xy = (3, 3) # pan range is (-inf inf)
fields_zoom_pan = model.construct_fields(output_shape=size, zoom=zoom_xy, pan=pan_xy)
img_zoom_pan = model.generate(output_shape=size, latents=latents_lrg, fields=fields_zoom_pan)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_lrg.squeeze())
axes[1].imshow(img_zoom_pan.squeeze())
axes[0].set_title('Gen image 1024x1024')
axes[1].set_title(f'Zoom {zoom_xy}, Pan {pan_xy}')
plt.show()
##
##
model = INRF2D(graph_topology='WS') # init Watts-Strogatz graph
# is equivalent to 
model.init_map_fn(mlp_layer_width=32,
                  activations='random',
                  final_activation='tanh',
                  weight_init='normal',
                  num_graph_nodes=10,
                  graph_topology='WS',
                  weight_init_mean=0,
                  weight_init_std=3,)
model.init_map_weights()  # resample weights to get different outputs
img_ws = model.generate(output_shape=(256,256), sample_latent=True)

model = INRF2D(latent_dim=0)  # also try conv architecture
model.init_map_fn(conv_feature_map_size=64,
                  activations='random',
                  final_activation='tanh',
                  weight_init='normal',
                  input_encoding_dim=1,
                  graph_topology='conv',
                  weight_init_mean=0,
                  weight_init_std=3,)

img_conv = model.generate(output_shape=(256,256), sample_latent=False)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_ws.squeeze())
axes[1].imshow(img_conv.squeeze())
axes[0].set_title('WS Gen image')
axes[1].set_title('Conv Gen image')
plt.show()
##
##
from neural_canvas.utils import load_image_as_tensor
from neural_canvas.losses import LossModule

img = load_image_as_tensor('neural_canvas/assets/logo.jpg')[0]
model = INRF2D(device='cpu') # or 'cuda'
model.init_map_fn(activations='GELU',
                  weight_init='dip', 
                  graph_topology='conv', 
                  final_activation='tanh',
                  num_fourier_freqs=4,
                  input_encoding_dim=8) # better params for fitting
loss = LossModule(l1_alpha=1.0)
model.fit(img, loss=loss, n_iters=500)  # returns a implicit neural representation of the image

print ('INRF size', model.size)  # return size of neural representation
# >> 30083
print ('data size', np.prod(img.shape))
# >> 196608

# get original data
img_original = model.generate(output_shape=(256, 256), sample_latent=True)
print ('original size', img_original.shape)
# >> (1, 256, 256, 3)

img_super_res = model.generate(output_shape=(1024,1024), sample_latent=True) 
print ('super res size', img_super_res.shape)
# >> (1, 1024, 1024, 3)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_original.squeeze())
axes[1].imshow(img_super_res.squeeze())
axes[0].set_title('image fit 256x256')
axes[1].set_title('image fit 1024x1024')
plt.show()

##
##

from neural_canvas.models.inrf import INRF3D

# Create a 2D implicit neural representation model
model = INRF3D()
# Generate the image given by the random INRF
size = (256, 256, 256)
vol = model.generate(output_shape=size)
print ('volume size', vol.shape)
# We can do all the same transformations as with 2D INRFs
# Warning: 3D generation consumes large amounts of memory. With 64GB of RAM, 1500**3 is the max size
# `splits` generates the volume in parts to consume less memory
size = (512, 512, 512)
model = INRF3D(graph_topology='WS')
latents = model.sample_latents(output_shape=size)
fields = model.construct_fields(output_shape=size)
vol = model.generate(output_shape=size,
                     latents=latents,
                     fields=fields, 
                     splits=32)
print ('volume size', vol.shape)

##
##

import pyvista as pv
import numpy as np

model = INRF3D(graph_topology='WS')
size = (256, 256, 256)
rgb = model.generate(output_shape=size, splits=4)[0]
rgb = rgb.reshape(-1, 3)
print ('volume size', rgb.shape)
rgba = np.concatenate((rgb, np.ones((rgb.shape[0], 1), dtype=np.uint8)), 1)
# plot with pyvista
grid = pv.UniformGrid(dimensions=size)
p = pv.Plotter(notebook=False)
p.add_volume(grid, scalars=rgba)
p.show()