# Deep Feature Image Re-creations

This script will re-create images from their feature representations from any layer in VGG-16. It works by  calculating the feature map for a given image up to the desired layer, then tries to match this feature representation by starting with an image of noise. The difference between the two feature maps is minimized by vanilla gradient descent.  

The script has several options so you can mess around and see how each of them affect the re-created image. The goal of this really is just to create beautiful images, but it is cool to see how deeper layers have less and less information about the original.

I update the image for 500 iterations of gradient descent. Every 10 iterations or so I apply a gaussian blur of size 3x3 and every 20 iterations I apply gaussian blur of size 7x7. I also clip the gradients and the pixels at 1.5 and 3 standard deviations respectively. This seems to make the images more interesting. I got the idea for that from [this paper](http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf).

## Results

I started pretty deep in the network because I wanted the images to look a lot different from the originals, but even this deep you can clearly see the objects. It isn't until you get to the fully connected layers that the shapes of the objects start to get fuzzy.

![alt text](https://i.imgur.com/zRnEJ0d.png)


## Usage
```
usage: main.py [-h] [--learning-rate LEARNING_RATE]
               [--layer-index LAYER_INDEX] [--image-std-clip IMAGE_STD_CLIP]
               [--grad-std-clip GRAD_STD_CLIP] [--verbose]
               images [images ...]

Re-create images from deep feature representations in VGG16

positional arguments:
  images

optional arguments:
  -h, --help            show this help message and exit
  --learning-rate LEARNING_RATE
                        the learning rate to update the image at
  --layer-index LAYER_INDEX
                        index of the layer in VGG16 from which to re-create
                        the image from
  --image-std-clip IMAGE_STD_CLIP
                        the number of standard deviations to clip the image at
                        after each update
  --grad-std-clip GRAD_STD_CLIP
                        the number of standard deviations to clip the gradient
                        at
  --verbose             view detailed status of the network
```


