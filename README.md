
# Neural Canvas: creative deep learning through implicit data representations

<div align="center">
<img src="https://raw.githubusercontent.com/neale/neural-canvas/main/neural_canvas/assets/montage.png" alt="logo"></img>
</div>

# Overview

Neural canvas is a Python library that provides tools for working with implicit neural representations of data. This library is designed to make it easy for artists and researchers to experiment with various types of implicit neural representations and create stunning images and videos.

Implicit neural representations, also called [neural fields](https://neuralfields.cs.brown.edu/) are a powerful class of models that can learn to represent complex patterns in data. 
They work by learning to map a set of inputs to a continuous function that defines the implicit shape of the data. 
This resembles an optimization problem rather than machine learning, where machine learning techniques usually seek to find some configuration that generalizes to lots of data on some task. 
Examples of implicit neural representations include NERFs ([Neural Radiance Fields](https://arxiv.org/abs/2003.08934v2)) for 3D data, or CPPNs ([Compositional Pattern-Producing Networks](https://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/)) for 2D data. 

Implicit neural representation functions (INRFs) have a variety of uses, from data compression, to artistic generation. In pratice, INRF's a usually take some structural information, like (x, y) coordinates, and are optimized to fit a single example e.g. NERFs just learn to represent a single scene. Alternatively, random generation is interesting artistically, as 2D and 3D INRFs can be used to produce some really interesting looking artwork. 

Lets start with the most basic example

## Image generation with 2D INRFs

We can instantiate a 2D INRF with random parameters, by default the architecture is a 4 layer MLP. the INRF will transform _(x, y)_ coordinates and a radius input _r_ into an output image of arbitrary size. Because we use nonstandard activation functions in the MLP, the output image will contain different kinds of patterns, some examples are shown below.

<div align="center">
<img src="https://raw.githubusercontent.com/neale/neural-canvas/main/neural_canvas/assets/img_2d_gen.png" alt="gen_img"></img>
</div>

```python
from neural_canvas.models.indf import INRF2D

# Create a 2D implicit neural representation model
model = INRF2D()
# Generate the image given by the random INRF
size = (256, 256)
model.generate(output_size=size)
```
Importantly, the instantiated INRF is a neural representation of the output image, meaning that we can do things like modify the image size just by passing in a larger coordinate set

```python
size = (1024, 1024)
model.generate(output_shape=size)
```
Re-rendering at a higher resolution actually _adds_ detail, in contrast to traditional interpolation, we can use this fact to zoom in on our image. 
```python
size=(1024, 1024)
zoom_xy = (10, 10) # xy zoom range is (0, inf)z
model.generate(size=size, zoom=zoom_xy)
```
From this its clear that a random INRF is just an embedding of the INRF function into the input coordinate frame. We can change that function by using any of the 3 supported architectures

* MLP with random or specified nonlinearities
* 1x1 Convolutional stack with random or specified nonlinearities, and optional norms
* Random Watts-Strogatz graph with random activations 

Choose a different architecture quickly, or with more control
```python
model = INRF2D(graph_topology='ws') # init Watts-Strogatz graph
# is equivalent to 
model.init_map_fn(mlp_layer_width=32,
                  activations='random,
                  final_activation=None,
                  weight_init='normal',
                  num_graph_nodes=20,
                  graph_topology='ws',
                  weight_init_mean=0,
                  weight_init_std=3,)
```

We can also fit data if we want 

## Fitting INRFs to 2D data

We can utilize any of the INRF architectures toward fitting a 2D image. 

**Why would we want to do this?** 

Implicit data representations are cheap! There are less parameters in the neural networks used to represent the 2D and 3D data, than there are pixels or voxels in the data itself. 

Furthermore, the neural representation is flexible, and can be used to extend the data in a number of ways. 

For example, we can instantiate the following function to fit an image

```python3
from neural_canvas.functions import fit_image
from imagio import imread

img = imread('neural_canvas/assets/logo.png')
model = INRF2D()
neural_data = model.fit(img)  # returns a implicit neural representation of the image

print (neural_data.size())  # return size of neural representation
# >> 8547
print (img.size)
# >> 196608

# get original data
img_original = neural_data.data()
print (img_original.shape)
# >> (256, 256, 3)

img_super_res = neural_data.resize(1024) 
print (img_super_res.shape)
# >> (1024, 1024, 3)
```

### Positional Encodings

Sineusoidal encodings are not supported for `Conv` and `Linear` architectures!

* [Positional encodings work quite well for NERFs](https://arxiv.org/abs/2003.08934), so surely they would help here too.  
* `utils.positional_encodings.FourierEncoding` defines an alternating `Sin`, `Cos` encoding for a fixed number of frequencies. In practice this works quite a bit better for `INRFConvMap` architectures than any linear network, possibly due to the input channel concatenation. 
* See `examples/fit_2d_conf.yaml` for an example of fitting a target image utilizing these positional encodings.   

## 2D generators

Neural canvas provides a set of easy-to-use APIs for generating artwork with implicit neural representations. Here's an example of how to generate an image with a 2D Implicit Neural Representation Function:



## 3D generators

Neural canvas provides a set of easy-to-use APIs for generating artwork with implicit neural representations. Here's an example of how to generate an image with a 2D Implicit Neural Representation Function:

```python
from neural_canvas.models.indf import INDF2D
from neural_canvas.runners import indf_runner
from neural_canvas import imwrite

# Define the size of the image
width = 512
height = 512

# Create a 2D implicit neural representation model
model = INDF2D(width, height)
# Wrap model in a runner to allow generation and saving utilities
model = indf_runner(model, logdir='outputs/hello_canvas')
# Generate and save 10 images
model.generate(num=10)
```

### Render 3D volumetric data with PyVista and Fiji


## Contributions

Contributions are welcome! If you would like to contribute, please fork the project on GitHub and submit a pull request with your changes.
### Dependency Management

This project uses [Poetry](https://python-poetry.org/) to do environment management. If you want to develop on this project, the best first start is to use Poetry from the beginning. 

To install dependencies (including dev dependencies!) with Poetry:
```bash
poetry shell && poetry install 
```
You should now be able to run any and all code within the `dev` branch(es). 

### Linting

We primarily use [Black](https://black.readthedocs.io/en/stable/) for linting because it removes a lot of my strong opinions on the way code should look byu forcing everyone to adhere to a style we all partially agree on. 

To run the linter:
```bash
make lint
```
We will probably request that code by linted before a merged PR, but its not a critical thing.  

## License

Released under the MIT License. See the LICENSE file for more details.

---------------------------------------

Much of this readme generated with Chat-GPT4
