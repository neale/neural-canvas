<div align="center">
<img src="https://raw.githubusercontent.com/neale/neural-canvas/main/neural_canvas/assets/logo.jpg" alt="logo"></img>
</div>


# Neural Canvas: creative deep learning through implicit data representations

<div align="center">
<img src="https://raw.githubusercontent.com/neale/neural-canvas/main/neural_canvas/assets/montage.png" alt="logo"></img>
</div>

# Supported Models / Functions


# Usage

## Image generation with 2D INRFs

<div align="center">
<img src="https://raw.githubusercontent.com/neale/neural-canvas/main/neural_canvas/assets/img_2d_gen.png" alt="gen_img"></img>
</div>

## Fitting implicit functions to 2D data

We can utilize any of the INRF architectures toward fitting a 2D image. 

**Why would we want to do this?** 

Implicit data representations are cheap! There are less parameters in the neural networks used to represent the 2D and 3D data, than there are pixels or voxels in the data itself. 

Furthermore, the neural representation is flexible, and can be used to extend the data in a number of ways. 

For example, we can instantiate the following function to fit an image

```python3
from neural_canvas.functions import fit_image
from imagio import imread

img = imread('neural_canvas/assets/logo.png')
neural_data = fit_image(img)  # returns a implicit neural representation of the image

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
