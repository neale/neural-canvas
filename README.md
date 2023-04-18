<div align="center">
<img src="https://raw.githubusercontent.com/neale/neural-canvas/main/neural_canvas/assets/logo.jpg" alt="logo"></img>
</div>


# Neural Canvas: creative deep learning through implicit data representations


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
