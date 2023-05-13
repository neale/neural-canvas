import matplotlib.pyplot as plt
import numpy as np
from neural_canvas.models.inrf import INRF2D

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