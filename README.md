# TORCH NDRF

## Notes

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

Implicit 
