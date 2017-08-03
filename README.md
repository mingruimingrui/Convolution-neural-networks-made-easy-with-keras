# Convolution neural networks made easy with keras
By Wang Ming Rui

> I wrote this article after watching [Andrej Karpathy's lecture](https://www.youtube.com/watch?v=AQirPKrAyDg&t=3200s) on YouTube and realized how easy it actually is to implement a basic deep learning model. This article is meant as a guide for people wishing to get into machine learning and deep learning models. Some will find the things covered here easier so feel free to speed through! If you do not consider yourself a highly-technical person, I try my best to keep things as simple as possible. Do remember to read every sentence and do multiple re-reads on parts that you do not fully understand to boost your understanding!

## Introduction
Image recognition is the task of taking an image and labelling it. For us humans, this is one of the first skills we learn from the moment we are born and is one that comes naturally and effortlessly. By the time we reach adulthood we are able to immediately recognize patterns and put labels onto objects we see. These skills to quickly identify images, generalized from prior knowledge, are ones that we do not share with our machines.

<p align="center"><img src="https://naushadsblog.files.wordpress.com/2014/01/pixel.gif", width="360"></p>
<p align="center">Fig 0.0 how a machine 'views' an image</p>

When a computer sees an image, it will see an array of pixel values, each between a range of 0 to 255. These values while meaningless to us are the only input available to a machine. No one knows how exactly we living beings process images but scientists today have figured out a technique to simulate this process, albeit at a basic level. We call this technique deep learning.

There are many good resources out there that teaches you how to build your very own deep learning model. In this guide, we shall focus on one of these models. It is perhaps one of the most revolutionary and fundamental model in recent times, a convolution neural network (or CNN for short).

## Before we get started
Though not necessary, some recommended prerequisites to this guide are:
- [python programming skills](https://learnpythonthehardway.org/)
- [basic understanding of machine learning concepts and intuition](https://www.coursera.org/learn/machine-learning)

The goal of this article is to allow anyone with coding abilities to create their own starter deep learning model.

## Contents
1. [Convolution neural networks, how it functions](#convolution-neural-networks-how-it-functions)
2. [Building your first model](#building-your-first-model)
3. [Visualizing your CNN](#visualizing-your-cnn)
4. [Improving your model](#improving-your-model)

## Convolution neural networks, how it functions

Image recognition used to be done using much simpler methods such as linear regression and comparison of similarities. The results were obviously not very good, even the simple task of recognizing hand-written alphabets proved difficult. Convolution neural networks (CNNs) are supposed to be a step up from what we traditionally do by offering a computationally cheap method of effectively simulating the neural activities of a human brain when it perceives images.

### CNNs explained
But first, let us understand what a convolution is without relating it to any of the brain stuff.

#### The mathematical part

<p align="center"><img src="/imgs/input-image-dimension.JPG", width="240"></p>
<p align="center">Fig 1.0 simplified depiction of a 32x32x3 image</p>

A typical input image will be broken down into it's individual pixel components. In the picture above, we have a 32x32 pixel image which has a R, G, and B value attached to each pixel, therefore a 32x32x3 input, Also known as an input with 32 height, 32 width, and 3 depth.

<p align="center"><img src="/imgs/filtering.JPG", width="360"></p>
<p align="center">Fig 1.1 applying a 3x3 filter</p>
<p align="center"><img src="/imgs/filtering-math.JPG", width="720"></p>
<p align="center">Fig 1.2 mathematics of filtering</p>

A CNN would then take a small 3x3 pixel chunk from the original image and transform it into a single figure in a process called filtering. This is achieved by multiplying a number to each of the pixel of the original image and summing it up. A simplified example of how the math is done is as described in the picture above. NOW STOP RIGHT HERE! Make sure you understand the mathematics of how to conduct filtering. Re-read the contents if you need to. As for how we arrive at this filter and why it is of the size 3x3, we will explain later in this article.

Since we are dealing with an image of depth 3 (number of colors), we need to imagine a 3x3x3 sized mini image being multiplied and summed up with another 3x3x3 filter. Then by adding another constant term, we will receive a single number result from this transformation.

<p align="center"><img src="/imgs/filtering-many-to-one.gif", width="360"></p>
<p align="center">Fig 1.3 filtering in action, original image is below</p>

This same filter will then be applied to every single possible 3x3 pixel on the original image. Notice that there are only 30x30 unique 3x3 squares on a 32x32 image, also remember that a filter will convert a 3x3 pixel image into a single image so the end result of applying a filter onto a 32x32x3 image will result in a 30x30x1 2nd 'image'.

#### The high-level explanation

What we are trying to do here is to detect the presence of simple patterns such as horizontal lines and color contrasts from the original image. The process as described above will output a single number. Typically this number will be either positive or negative. We can understand positive as the presence of a certain feature and negative as the absence of the feature.

<p align="center"><img src="/imgs/finding-horizontal-vertical.jpg", width="540"></p>
<p align="center">Fig 1.4 identifying vertical and horizontal lines in a picture of a face</p>

In the image above, a filter is applied to find vertical and horizontal lines and as we can see, in each of the pictures on the left, only the places where vertical lines are present will show up in white and likewise horizontal lines for the picture on the right.

Going by this idea we can think of filtering as a process of breaking down the original image into a list of presence of simplified structures. By knowing the presence of slanted lines and horizontal lines and other simple basic information, more interesting features such as eyes and nose and mouth then then be identified and if there is the presence of an eye and a mouth and a nose, then the classifier will have a pretty good certainty that the image at hand is probably a face. Basically that is what a CNN would do, by doing detective work on the abstract information that it is able to extract from the input image and through a somewhat logical thought process come to the deduction of the correct label to attach to a particular image.

Make sure that you have understood all that were covered previously because the next section is going to progress at a much faster rate. We are still not going to talk about how to calculate filters yet. First, let us finish up the mechanics of the CNN.

#### Back to the model

One filter would only be capable of finding a single simplified feature on the original image. Multiple filters can be applied to identify multiple features. Lets say on the original image, a total of 32 filters are applied on the input 32x32x3 image, and so then the end result will be a 30x30x32 'image'. It is no longer so much of an image but rather a collection of features extracted from the original image. A step by step explanation of how to do this is as follows,

1. generate a set of 32 filters of size 3x3x3 each
<!-- 2. take just a single filter and apply filter onto every single 3x3 chunk of the input image receive a 30x30x1 'image' in return. -->
2. place the 30x30x1 'image' aside and move onto the next filter
3. apply 2nd filter to input image and receive another 30x30x1 'image', stack this 'image' on top of the other 30x30x1 'image' to get a 30x30x2 'image'
4. repeat until all 32 filters are used.

The entire process of transforming an input from a 32x32x3 form to a 30x30x32 form is known as a single convolution layer. An entire CNN model is usually made up of multiple convolution layers and a classifier layer. Here is an example of how a typical CNN would look like.

<p align="center"><img src="/imgs/conv-layers.jpeg", width="720"></p>
<p align="center">Fig 1.5 structure of a typical CNN, here classifying a car</p>

The model would take an input from the left (here the image of a car). And the data will be transferred from the left side to the right, through each of the individual layers. Each layer would take the output of the previous layer as it's input and then produce a transformation on the image before passing it onto the next layer. There are probably a few terms that you might not understand at this point of time, but let us go through them one at a time:

- CONV: In the model in the picture, the first layer is a CONV layer. It is nothing new as CONV is just short form for convolution layer.

- RELU: The RELU layer (short for rectifier layer) is basically a transformation of all negative outputs of the previous layer into 0. As negative numbers would also contribute to the output of the next layer, 0 has a significance in the sense that it will not affect the results of the next layer. Looking back at the high-level definition of how a convolution works, negative numbers should mean the absence of a feature. 0 would fit that idea more concisely and that is the purpose of this layer. We will not change the values of the positive numbers as the magnitude of the positive number can help identify how closely the image represents a feature. The RELU layer will not transform the shape of it's input. If the input is of shape 30x30x32, the output would still be 30x30x32, except all the negatives are now 0s instead.

<p align="center"><img src="/imgs/max-pooling.jpeg", width="540"></p>
<p align="center">Fig 1.6 pooling on a 4x4 input</p>

- POOL: Image processing is a very computationally intensive process. To allow our algorithm to run at a decent speed while not compromising accuracy too heavily, we do a form of reduction on the image size in a technique called pooling. The image above shows how it is done. From each 2x2 square, we find the pixel with the largest value, retain it and throw away all the unused pixels we also do this for each depth layer (recall on the input image, it would be each color layer). Doing this transformation would essentially reduce the dimensions of the original image by half on height and another half on weight. Another reason why we wish to do this is to converge features of close proximity together such that more complex features can develop sooner.

The act of repeating the process of CONV RELU POOL would simulate the process of reinforcing the complexity of the features gathered from the original image.

- FC: After retrieving all of the advanced features from each image, we combine them together to classify the image to it's proper label. We do so in the fully connected layer.

<p align="center"><img src="/imgs/fully-connected-layer.JPG", width="540"></p>
<p align="center">Fig 1.7 A simple fully connected layer displaying probability outputs</p>

The fully connected layer, will take in all of the advanced features produced by the final convolution layer and output the probability for each label. Remember that the purpose of the convolution layers are to output the presence of advanced features such as eyes, mouth, or wings. By taking note of the presence of such features, the fully connected layer will do the last bit of detective work to determine the most suitable label to apply to each image. Mathematically, it works in the same way as filters do except this time, there's no 3x3 portions. Each 'filter' in this case will be the same size as the output layer from the final layer of convolution. There can however be multiple fully-connected-layers but only just as many as the number of label classes you have, the intuition being that you can calculate the confidence level of each individual class separately.

Do keep in mind, this is just a very basic understanding of what the fully connected layer seeks to accomplish. In actuality this layer can be much more complex but first, a much long awaited question should be answered.

### Where filter weights come from

> Short recap: Up to this current moment in time, your understanding of how CNNs work is that through a series of multiplications, summations and modifications, you are able to generate a prediction of some sort. Along the way, complex features that a computer would not normally be able to identify are extracted and turned into simple terms that it could, these terms represents weather a high level feature is present or not. This greatly simplifies the original problem of image identification into small simple steps that a computer can solve but there's just one mystery remains.

CNN is an algorithm that require some very specific parameters (we also call them weights) in the filter layers else the entire model would fail to function. We find these parameters using Mathematics.

The problem is this,

> _to find a set of parameters that allows the model to be as accurate at labelling images as possible_

To translate this into mathematics, let us first define a few terms,

<dl>
  <dt><img src="/imgs/x.JPG", width="30"></dt>
  <dd>Represents the original image</dd>

  <dt><img src="/imgs/y.JPG", width="30"></dt>
  <dd>Represents the actual label of the image</dd>

  <dt><img src="/imgs/y-hat.JPG", width="30"></dt>
  <dd>Represents the predicted label of the image</dd>

  <dt><img src="/imgs/y-hat.JPG", width="30"></dt>
  <dd>Represents the \`series of multiplications, summations and modifications\` the CNN makes on the the image to output you predicted value</dd>

  <dt><img src="/imgs/y-hat2.JPG", width="80"></dt>
  <dd></dd>
</dl>

Taking note of these definitions, we can also define our predicted y as follows,

> <p><img src="/imgs/y-hat2.JPG", width="80"></p>

When you take the predicted result and subtract it from our actual result, you get this back,

> <p><img src="/imgs/residual.JPG", width="80"></p>

One way of interpreting this is by viewing it as a measure of how far off the model is from the desired result (this measure is hereby called error). An error of 0 would mean that the model is spot on, 1 and -1 would mean that there are still improvements to be made. By averaging up the errors a CNN's predictions make on a set of images, you will be able to get a gauge of how well a set of parameters are doing. The greater the average error, the more inaccurate the predictions are, which prompts you to change the current set of parameters.

<p>Lets take the example of the case were we have 3 images. Suppose the errors of an algorithm trying to predict the actual labels of these images are 0, 1, and -1. If we sum up all these errors we should get the total error so 0 + 1 + (-1) = ... 0? Even if we average it out it would still be 0. <img src="/imgs/you-dont-say.jpg", width="40"></p>

That does not mean that the predictions the CNN made are all correct. The problem lies in the method error is accumulated. As there are both positive and negative errors, they will cancel each other out but thankfully simple modification will fix this. By squaring the errors you will force all errors to be positive.

> <p><img src="/imgs/summation-symbol.JPG", width="30">, this symbol just means summation. In the context below, it means for all images, sum up (the term inside)</p>

From this,

> <p><img src="/imgs/sum-residual-wrong.JPG", width="80"></p>

we will get this,

> <p><img src="/imgs/sum-residual-square.JPG", width="90"></p>

and of course to account for averaging,

> <p><img src="/imgs/average-squared-error.JPG", width="100"></p>

So errors of 0, 1, and -1 will sum up to be (0^2) + (1^2) + ((-1)^2) = 0 + 1 + 1 = 2. Averaging that out will give us 2/3. The smaller this figure is, the closer we are to the optimal set of parameters. Therefore minimizing this term would be the same as finding the optimal parameters for the CNN. Minimization also has a symbol for convenience.

> <p><img src="/imgs/cost-function.JPG", width="120"></p>

Then lets replace our predicted y.

> <p><img src="/imgs/cost-function2.JPG", width="120"></p>

Take note that here, x and y are both fixed based on the input images you have provided the CNN with. There only thing we can change to minimize this equation is A, the parameters of all the layers of filters in the CNN. If minimizing this equation also means to making the CNN more accurate, then that would be the same as solving the original English problem.

> _to find a set of parameters that allows the model to be as accurate at labelling images as possible_

The question of how we arrive at the optimal filter is still unanswered but to solve this,

> <p><img src="/imgs/cost-function2.JPG", width="120"> (also known as the cost function)</p>

there is an area of Mathematics dedicated to solving problems such as this called gradient descent.

### Gradient descent

Let us first plot a simple graph.

<p><img src="/imgs/graph1.jpg", width="320"></p>

Imagine this, you have a horizontal axis. This axis represents every single unique possible combination of parameters for the CNN, _A_, all mapped out onto a line. Each point on this axis represents a unique _A_.

The vertical axis represents the average error at that specific _A_ (the cost in terms of model inaccuracy therefore the name cost function). As one can expect, the average error will not be a linear function, rather it will be curved, like in the image above.

<p><img src="/imgs/graph2.jpg", width="320"></p>

Recall that minimizing this average error will result in a more accurate model. Therefore, the point where the curve dips lowest corresponds to the set of parameters which allows the model to perform best. The problem of finding this point can be solved using gradient descent. Sadly there is no simple way to explain how the process of gradient descent work without watering it down too much. In summary it goes a little something like this,

1. Start off at a randomly initialized A
2. Calculate the average error generated by some neighboring A and move to a neighbor with the lowest A.
3. Repeat this process multiple times until you reach the A with the lowest average error.

Those of you familiar with calculus should be able to recognize that solving this problem involves finding differential functions. But those who aren't don't have to worry too much as most deep learning libraries these days are capable of doing these math for you. Learning the math is tedious especially for people without prior mathematical knowledge however it is still useful and fundamental when building more complex algorithms and models. By the way, the full cost function (average error) would also contain a regularization term. [external materials TBI]

One thing of note is that we do not specify the objectives for each filter. That is because the filters usually adjust themselves to identify complex features. This isn't exactly surprising from a statistical standpoint. Eyes, nose, and mouth are usually very good indicators in face identification. Models are considered good if they are able to identify complex features.

<!-- ### Brief history
STASHED kept for reuse later

In the summer of 2012, ImageNet hosted it's annual Large Scale Visual Recognition Challenge to pit some of the world's most intelligent groups against one another. Alexnet was a deep CNN submitted in this competition. A good model classification error rate is typically around 25-26%. The second placed model in the competition was able to achieve an impressive 26.2% rate of error. Alexnet scored 15.3%. At over 10% better than the next best model, it signaled a new age in the field of computer vision.

<p align="center"><img src="/imgs/ILSVRC-2012-finalists.JPG", width="360"></p>

The basis of computer vision and CNNs were laid down in the early 1950s by Hubel and Wiesel when they studied the behavior of the neurons in the visual cortex of a cat as they moved an image across it's area of vision. During their studies, they observed that the image shown and it's orientation affects directly how the neurons fire and activates themselves. This lead to the deduction that our brain perceives information through the active and inactive states of the neurons that it is made up of.

<p align="center"><img src="/imgs/hubel-wiesel-experiment.jpg", width="360"></p> -->

<!-- ## Keras, deep learning simplified
- Sequential model
- Layer class
- optimizer

This section is going to be mainly for python coders since the library we will be using, Keras, operates supports only this language. Keras is built on top of some other very popular deep learning libraries such as TensorFlow. It acts as a wrapper to simplify the process of defining models and executing then. We shall go through a few key classes within the library before moving onto actually building a model.

### Sequential Models
By now you should have understood that CNN is a model made out of individual layers -->

## Building your first model
- dataset
- preprocessing
- model Building
- training

This section will mainly be for python coders since the library that will be used, Keras, only supports this language. Keras is built on top of some other very popular deep learning libraries such as TensorFlow. It acts as a wrapper to simplify the process of defining models and executing then. We shall get in more details later. I have also coded out the model in the file ```basic_model.py```. You can actually run all the codes in there without coding anything yourself but you will still need to fulfill all the dependencies listed below.

To run the model covered in this section, simple do the following,

1. Open up your console at whatever location you like and type ```git clone https://github.com/mingruimingrui/Convolution-neural-networks-made-easy-with-keras.git```
2. ```cd Convolution-neural-networks-made-easy-with-keras```
3. ```python basic_model.py```

### Depenencies
You will need the following software installed on your device of choice:
- Python 2/3 (I'm using Python 3.6)
- Numpy (for matrix manipulations and linear algebra)
- pathlib
- Matplotlib (optional)

Do also make sure that the dependencies you installed are suitable for the version of python you are working on. Matplotlib will only be used only later during model visualization.

### Dataset
The training set you will be using is the CIFAR-10 dataset. It is a collection of 60,000 32x32 pixel labelled images (meaning that each image is already labelled according what it is an image of). Loading the dataset is just the matter of 3 lines of codes (or 1 if you don't count importing).

```python
import numpy as np
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

Now you might have noticed that we have loaded this thing called ```X_train``` and ```y_train```. The X here simply means the collection of images and y is the collection of labels.

There are also ```X_train``` and ```X_test```. We call these training set and test set. We will build our model on the training set and test it's results on the test set. The reason why we wish to do thing this way is because we wish to ensure that our algorithm is capable of generalizing onto external data. It is possible to have a model which performs perfectly on a local dataset but fail completely on any outside datasets. We call this the case of overfitting.

Let us first visualize how data is stored in X_train,

```python
print(type(X_train))
>>> <class 'numpy.ndarray'>

print(X_train.shape)
>>> (50000, 32, 32, 3)
```

X_train is stored in a format known as a matrix in python, the Numpy library is a library for creating and manipulating matrix objects and a ```numpy.ndarray``` is the default matrix class. A matrix is relatively easy to understand. In the context of the example above, ```X_train``` can be viewed as a multi dimensional array. We know that the dataset is a collection of 32x32x3 images so ```X_train``` can be interpreted in the following format ```(image_index, height_index, width_index, rgb_index)```.

We can easily access individual images this way,

```python
img1 = X_train[0,:,:,:]
print(img1.shape)
>>> (32, 32, 3)

img1_reds = X_train[0,:,:,0]
print(img1_reds.shape)
>>> (32, 32)
```

We can also plot out the images using Matplotlib,

```python
from matplotlib.pyplot import plt

plt.imshow(img1)
plt.show()

img1_again = X_train[0]
plt.imshow(img1_again)
plt.show()
```

<p align="center"><img src="/imgs/frog.jpg", width="320"></p>
<p align="center">Fig 2.0 the image of the frog can clearly be seen plotted out</p>



## Visualizing your CNN
- activation based
- weight based
- result based
- external material

## Improving your model
Coming soon, I'm still tuning the model to get a right balance of scale and speed
Update, removed sparse encoding as even a 64x64 image require too long to process on a home desktop level computer.
Plans
- focus more on img augmentation / alternative optimization / model structure changes / batch wised training
