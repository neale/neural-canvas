<div align="center">
<img src="https://raw.githubusercontent.com/neale/Inart/main/inart/assets/logo.jpg" alt="logo"></img>
</div>


# INArt: creative deep learning through implicit data representations
=======================================

The noise vector is not that important, its just a "view" into the output space.

The lack of variation between random vector outputs, when all else is fixed, I think has to do with the presence of squashing functions as activations. 

* Tanh
* Sigmoid
* Gaussian

5 or so random input vectors should be enough to get a glimpse of the space, and then try more if you like the style.

Importantly **There is not always something interesting in every configuration**

We can test this by generating randomly until we get a batch of outputs that are obviously interesting, and another batch that are more boring

Then perform a zoom test on both sets, yielding 10 videos in total. 

### Zoom Tests (Good)
![here](assets/zoomg1.mp4)
![here](assets/zoomg5.mp4)

### Zoom Tests (boring)
![here](assets/zoomb1.mp4)
![here](assets/zoomb5.mp4)

Its immediately clear with zooming that we're looking at the same scene, even though our inputs are randomly generated

For the boring test, we just have a network that not too interesting, and different views into its output space don't reveal any real structure. 

## Usage

INart provides a set of easy-to-use APIs for generating artwork with implicit neural representations. Here's an example of how to generate an image with a 2D Implicit Neural Representation Function z:
```python
import inart
from inart.models.indf import INDF2D
from inart.runners import indf_runner
from imageio import imwrite

# Define the size of the image
width = 512
height = 512

# Create a 2D implicit neural representation model
model = INDF2D(width, height)
# Wrap model in a runner to allow generation and saving utilities
model = indf_runner(model, logdir='outputs/hello_indf')
# Generate and save 10 images
model.generate(num=10)
```

## Contributions

Contributions are welcome! If you would like to contribute, please fork the project on GitHub and submit a pull request with your changes.

## License

Released under the MIT License. See the LICENSE file for more details.

---------------------------------------

Much of this readme generated with Chat-GPT4
