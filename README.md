# Convolution neural networks made easy with keras
By Wang Ming Rui

> I wrote this article after watching [Andrej Karpathy's lecture](https://www.youtube.com/watch?v=AQirPKrAyDg&t=3200s) on YouTube and realized how easy it actually is to implement a basic deep learning model. This article is meant as a guide for people wishing to get into machine learning and deep learning models. Some will find the things covered here easier so feel free to speed through! If you do not consider yourself a highly-technical person, I try my best to keep things as simple as possible. Do remember to read every sentence and do multiple re-reads on parts that you do not fully understand to boost your understanding!

## Introduction
Image recognition is the task of taking an image and labelling it. For us humans, this is one of the first skills we learn from the moment we are born and is one that comes naturally and effortlessly. By the time we reach adulthood we are able to immediately recognize patterns and put labels onto objects we see. These skills to quickly identify images, generalized from prior knowledge, are ones that we do not share with our machines.

<p align="center"><img src="https://naushadsblog.files.wordpress.com/2014/01/pixel.gif", width="360"></p>
<p align="center">Fig 0.0 how a machine 'views' an image</p>

When a computer sees an image, it will see an array of pixel values, each between a range of 0 to 255. These values while meaningless to us are the only input available to a machine. No one knows how exactly we living beings process images but scientists today have figured out a technique to simulate this process, albeit at a basic level. We call this technique deep learning.

There are many good resources out there that teaches you how to build your very own deep learning model. In this guide, we shall focus on one of these models. It is perhaps one of the most revolutionary and fundamental model in recent times, a convolution neural network (or CNN for short).

> Along the way, there are some sections listed this way. These are extra materials which will just be a little harder to understand but are there for completion sake. Feel free to skip over and review later if you are having difficulty understanding them at the point of time.

## Before we get started
Though not necessary, some recommended prerequisites to this guide are:
- [python programming skills](https://learnpythonthehardway.org/)
- [basic understanding of machine learning concepts and intuition](https://www.coursera.org/learn/machine-learning)

The goal of this article is to allow anyone with coding abilities to create their own starter deep learning model.

## Contents
1. [Convolution neural networks, how it functions](#convolution-neural-networks)
2. [Building your first model](#building-your-first-model)
3. [Visualizing your CNN](#visualizing-your-cnn)
4. [Improving your model](#improving-your-model)

## Convolution neural networks
Image recognition used to be done using much simpler methods such as linear regression and comparison of similarities. The results were obviously not very good, even the simple task of recognizing hand-written alphabets proved difficult. Convolution neural networks (CNNs) are supposed to be a step up from what we traditionally do by offering a computationally cheap method of loosely simulating the neural activities of a human brain when it perceives images.

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

Going by this idea we can think of filtering as a process of breaking down the original image into a list of presence of simplified structures. By knowing the presence of slanted lines and horizontal lines and other simple basic information, more interesting features such as eyes and nose and mouth then then be identified and if there is the presence of an eye and a mouth and a nose, then the classifier will have a pretty good certainty that the image at hand is probably a face. Basically that is what a CNN would do, by doing detective work on the abstract information that it is able to extract from the input image and through a somewhat logical thought process come to the deduction of the correct label to attach to a particular image. It won't exactly look for eyes or nose, but it would attempt to do something similar in an abstract manner.

Make sure that you have understood all that were covered previously because the next section is going to progress at a much faster rate. We are still not going to talk about how to calculate filters yet. First, let us finish up the mechanics of the CNN.

#### Back to the model

One filter would only be capable of finding a single simplified feature on the original image. Multiple filters can be applied to identify multiple features. Lets say on the original image, a total of 32 filters are applied on the input 32x32x3 image. One filter applied onto the image will result in a 30x30x1 output. So to apply 32 unique filters, you merely stack the outputs on top of one another to result in a 30x30x32 output.

The entire process of transforming an input from a 32x32x3 form to a 30x30x32 form is known as a single convolution layer. An entire CNN model is usually made up of multiple convolution layers and a classifier layer. Here is an example of how a typical CNN would look like.

<p align="center"><img src="/imgs/conv-layers.jpeg", width="720"></p>
<p align="center">Fig 1.5 structure of a typical CNN, here classifying a car</p>

The model would take an input from the left (here the image of a car). And the data will be transferred from the left side to the right, through each of the individual layers. Each layer would take the output of the previous layer as it's input and then produce a transformation on the image before passing it onto the next layer. There are probably a few terms that you might not understand at this point of time, but let us go through them one at a time:

- CONV: In the model in the picture, the first layer is a CONV layer. It is nothing new as CONV is just short form for convolution layer.

> There are of course convolution layers of different sizes and not just 3x3. Some models uses 7x7 and even 11x11 filters but larger filters also mean more parameters which means longer training time. Filters are also usually has odd lengths and are squares. This is so as to have some sort of center to take reference from. There is also another concept called strides. In the examples above we use strides of size 1. Stride 2 would mean starting from the top left most 3x3 section of the image, you move 2 pixels to the right before you apply your filter again, the same when you move downwards. Padding is another technique commonly used. Usually when filtering takes place, the original image would shrink. If you pad the original image with pixels of values of 0 around it's borders, you will effectively be able to maintain image size. ie 32x32 input 32x32 output (instead of 30x30). This is crucial if you wish to build deep learning models which contains more layers.

- RELU: The RELU layer (short for rectifier layer) is basically a transformation of all negative outputs of the previous layer into 0. As negative numbers would also contribute to the output of the next layer, 0 has a significance in the sense that it will not affect the results of the next layer. Looking back at the high-level definition of how a convolution works, negative numbers should mean the absence of a feature. 0 would fit that idea more concisely and that is the purpose of this layer. We will not change the values of the positive numbers as the magnitude of the positive number can help identify how closely the image represents a feature. The RELU layer will not transform the shape of it's input. If the input is of shape 30x30x32, the output would still be 30x30x32, except all the negatives are now 0s instead.

> In actual fact rectifiers are just a member of a larger family called activators, they all set out to achieve the same purpose as stated above. Another popular activation layer is the logistic activator, It transform it's inputs into a logistic distribution.

<p align="center"><img src="/imgs/max-pooling.jpeg", width="540"></p>
<p align="center">Fig 1.6 pooling on a 4x4 input</p>

- POOL: Image processing is a very computationally intensive process. To allow our algorithm to run at a decent speed while not compromising accuracy too heavily, we do a form of reduction on the image size in a technique called pooling. The image above shows how it is done. From each 2x2 square, we find the pixel with the largest value, retain it and throw away all the unused pixels we also do this for each depth layer (recall on the input image, it would be each color layer). Doing this transformation would essentially reduce the dimensions of the original image by half on height and another half on weight. Another reason why we wish to do this is to converge features of close proximity together such that more complex features can develop sooner.

> The pooling technique we describe here is called max-pooling because we are only taking the max of every 2x2 squares. There are also other pooling methods such as min pooling and mean pooling. But this is by far the most popular method of pooling. Pooling can also be of larger dimensions like 3x3 or 4x4 although it is unrecommended as image sizes will reduce too fast.

The act of repeating the process of CONV RELU POOL would simulate the process of identifying more complex features from the original image.

- FC: After retrieving all of the advanced features from each image, we combine them together to classify the image to it's proper label. We do so in the fully connected layer.

<p align="center"><img src="/imgs/fully-connected-layer.JPG", width="540"></p>
<p align="center">Fig 1.7 A simple fully connected layer displaying probability outputs</p>

The fully connected layer, will take in all of the advanced features produced by the final convolution layer and output the probability for each label. Remember that the purpose of the convolution layers are to output the presence of advanced features such as eyes, mouth, or wings. By taking note of the presence of such features, the fully connected layer will do the last bit of detective work to determine the most suitable label to apply to each image. Mathematically, it works in the same way as filters do except this time, there's no 3x3 portions. Each 'filter' in this case will be the same size as the output layer from the final layer of convolution. There can however be multiple fully-connected-layers but only just as many as the number of label classes you have, the intuition being that you can calculate the confidence level of each individual class separately.

Do keep in mind, this is just a very basic understanding of what the fully connected layer seeks to accomplish. In actuality this layer can be much more complex but first, a long awaited question should first be answered.

### Where filter weights come from

#### The objective statement

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

<p>Lets take the example of the case were we have 3 images. Suppose the errors of an algorithm trying to predict the actual labels of these images are 0, 1, and -1. If we sum up all these errors we should get the total error so 0 + 1 + (-1) = ... 0? Even if we average it out it would still be 0.
<!-- <img src="/imgs/you-dont-say.jpg", width="40"></p> -->

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

#### Gradient descent

Let us first plot a simple graph.

<p><img src="/imgs/graph1.jpg", width="320"></p>

Imagine this, you have a horizontal axis. This axis represents every single unique possible combination of parameters for the CNN, _A_, all mapped out onto a line. Each point on this axis represents a unique _A_.

The vertical axis represents the average error at that specific _A_ (the cost in terms of model inaccuracy therefore the name cost function). As one can expect, the average error will not be a linear function, rather it will be curved, like in the image above.

<p><img src="/imgs/graph2.jpg", width="320"></p>

Recall that minimizing this average error will result in a more accurate model. Therefore, the point where the curve dips lowest corresponds to the set of parameters which allows the model to perform best. The problem of finding this point can be solved using gradient descent. Sadly there is no simple way to explain how the process of gradient descent work without watering it down too much. In summary it goes a little something like this,

1. Start off at a randomly initialized A
2. Calculate the average error generated by some neighboring A and move to a neighbor with the lowest A.
3. Repeat this process multiple times until you reach the A with the lowest average error.

Those of you familiar with calculus should be able to recognize that solving this problem involves finding differential functions. But those who aren't don't have to worry too much as most deep learning libraries these days are capable of doing these math for you. Learning the math is tedious especially for people without prior mathematical knowledge however it is still useful and fundamental when building more complex algorithms and models. By the way, the full cost function (average error) would also contain a regularization term as well as some other sophistications depending on the problem at hand.

One thing of note is that we do not specify the objectives for each filter. That is because the filters usually adjust themselves to identify complex features. This isn't exactly surprising from a statistical standpoint. Eyes, nose, and mouth are usually very good indicators in face identification. Models are considered good if they are able to identify abstractions of such complex features.

If you intend to learn gradient descent from scratch, it might make sense to learn neural network basis while you're at it. There are some pretty good materials (some which are free) online. The most ones popular includes the [machine learning course on coursera](https://www.coursera.org/learn/machine-learning), [Learning From Data course by CalTech](https://www.edx.org/course/learning-data-introductory-machine-caltechx-cs1156x-0), and [many more](https://www.springboard.com/blog/machine-learning-online-courses/).

If you just wish to learn to do gradient descent and already have a decent mastery over calculus, then I would suggest watching [a video on the subject](https://www.youtube.com/watch?v=jc2IthslyzM).

<!-- ### Brief history
STASHED kept for reuse later

In the summer of 2012, ImageNet hosted it's annual Large Scale Visual Recognition Challenge to pit some of the world's most intelligent groups against one another. Alexnet was a deep CNN submitted in this competition. A good model classification error rate is typically around 25-26%. The second placed model in the competition was able to achieve an impressive 26.2% rate of error. Alexnet scored 15.3%. At over 10% better than the next best model, it signaled a new age in the field of computer vision.

<p align="center"><img src="/imgs/ILSVRC-2012-finalists.JPG", width="360"></p>

The basis of computer vision and CNNs were laid down in the early 1950s by Hubel and Wiesel when they studied the behavior of the neurons in the visual cortex of a cat as they moved an image across it's area of vision. During their studies, they observed that the image shown and it's orientation affects directly how the neurons fire and activates themselves. This lead to the deduction that our brain perceives information through the active and inactive states of the neurons that it is made up of.

<p align="center"><img src="/imgs/hubel-wiesel-experiment.jpg", width="360"></p> -->

## Building your first model
Having Python experience will help greatly in this section and general coding knowledge is a must. The library that will be used, Keras, only supports this language. Keras is built using some other very popular deep learning libraries such as TensorFlow and CNTK as a backend. It acts as a wrapper to simplify the process of defining models and executing then. We shall get in more details later. This section will have less explanation and more examples, coding's more of a 'go figure it out yourself' kind of thing. I have coded out the model in the file ```basic_model.py```. You can run it from there but where's the fun in that?

To run the model covered in this section, simply do the following,

1. Open up your console at the location you like and type ```git clone https://github.com/mingruimingrui/Convolution-neural-networks-made-easy-with-keras.git```
2. ```cd Convolution-neural-networks-made-easy-with-keras```
3. ```python basic_model.py```

#### Dependencies
You will  need the following software installed on your device of choice:
- Python 2/3 (I'm using Python 3.5)
- Numpy (for matrix manipulations and linear algebra)
- Keras (with your backend of choice, I'm using TensorFlow)
- pathlib (optional)
- Matplotlib (optional)
- Pandas (optional)

Do also make sure that the dependencies you installed are suitable for the version of python you are working on.

### Dataset
The training set you will be using is the CIFAR-10 dataset. It is a collection of 60,000 32x32 pixel images labelled to one of 10 different classes. Loading the dataset is just the matter of 3 lines of codes (or 1 if you don't count importing).

```python
import numpy as np
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

Now you might have noticed that we have loaded this thing called ```X_train``` and ```y_train```. The X here simply means the collection of images and y is the collection of labels.

There are also ```X_train``` and ```X_test```. We call these training set and test set. We will build our model on the training set and test it's results on the test set. The reason why we wish to do thing this way is because we wish to ensure that our algorithm is capable of generalizing onto external data. It is possible to have a model which performs perfectly on a local dataset but fail completely on any outside datasets. We call this the case of overfitting.

Let us first visualize how data is stored in ```X_train```,

```python
print(type(X_train))
>>> <class 'numpy.ndarray'>

print(X_train.shape) # method to identify shape(size) of numpy.ndarray also known as a matrix
>>> (50000, 32, 32, 3)
```

X data is stored in a format known as a matrix in python, the Numpy library is a library for creating and manipulating matrix objects and a ```numpy.ndarray``` is the default matrix class. A matrix is relatively easy to understand. In the context of the example above, ```X_train``` can be viewed as a multi dimensional array. We know that the dataset is a collection of 32x32x3 images so ```X_train``` can be interpreted in the following format ```(image_index, height_index, width_index, rgb_index)```. In other words, there are 50,000 images in ```X_train```.

We can easily access individual images this way,

```python
img1 = X_train[0, :, :, :] # : operator just means select all
print(img1.shape)
>>> (32, 32, 3)

img1_again = X_train[0] # another way of selecting images
print(img1_again.shape)
>>> (32, 32, 3)

img_exp = X_train[0:30] # selection of multiple images can be easily done this way
print(img_exp.shape)
>>> (30, 32, 32, 3)
```

We can also plot out the images using Matplotlib,

```python
from matplotlib.pyplot import plt

plt.imshow(img1)
plt.show()
```

<p align="center"><img src="/imgs/frog.jpg", width="18  0"></p>
<p align="center">Fig 2.0 the image of the frog can be seen plotted out</p>

y data is also stored in a matrix,

```python
print(y_train.shape)
>>> (50000, 1)

print(y_train[:5]) # here we look at the first 5 elements of y_train
>>> array([[6],
           [9],
           [9],
           [4],
           [1]], dtype=uint8)

print(y_train.min(), y_train.max())
>>> 0 9
```

As we expect, there are as many labels in ```y_train``` as images in ```X_train``` (50,000). Labels are integers range from 0 to 9 corresponding to the classes they represent. Here's a dictionary of what each integer represents.

```python
labels = {
	0: 'airplane',
  1: 'automobile',
  2: 'bird',
  3: 'cat',
  4: 'deer',
  5: 'dog',
  6: 'frog',
  7: 'horse',
  8: 'ship',
  9: 'truck'
} # remember img1 has label of 6, that corresponds to a frog
```

Lastly lets check the size of our test set, I did mention above that CIFAR-10 has 60,000 labelled images and the training set has 50,000 images. So just to be sure...

```python
print(X_train.shape)
>>> (10000, 32, 32, 3)

print(y_train.shape)
>>> (10000, 1)
```

Perfect lets move on.

### Preprocessing
Preprocessing is an important step in building machine learning algorithms. There are things that you can do on both your X and y. Here we will explore 2 preprocessing techniques, mean-normalization and binary encoding.

#### Mean-normalization
Image pixel values are usually of the datatype ```uint8``` which means an integer between the range of 0 to 255. If we make use of such large numbers in our models, there can be possibility of overflow (what happens when numbers get too big and the machine fails to compute correctly). To reduce possibility of overflow, we scale our original values down to a decimal between 0 and 1. Doing so is easy, we just have to divide every term by 255, the highest possible value.

```python
dtype_mult = 255.0
X_train = X_train.astype('float32') / dtype_mult
```

> There are actually many ways to do mean-normalization. Some scale to a range between -1 and 1. while others ensure that distribution is akin to a normal distribution of mean 0 and std 1. In other datasets where values can be of differing ranges, normalization is also done so that we will be able to select a suitable learning rate for gradient descent!

#### Binary encoding
We want to be able to generate a probability index of how likely an image is to belong to each different class. Therefore we make a separate prediction for each class. To do that we also have to do a modification to our y as demonstrated below!

```python
import keras

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)

print(y_train.shape)
>>> (50000, 10)

print(y_train[:5])
>>> array([[ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
```

As you can see, we basically transformed y_train into a binary code of is or is not. ```img1``` which is labelled as a frog has an original label value of 6. From the single value of 6 it has transformed into an array of 10 digits, 0s everywhere except for the 6th place which has a value of 1. It just means that it is not a airplane, not a automobile ... but is a frog.

### Model building
Now is time to define the model. before we declare the model, lets set out a clearly defined structure for our model before actually coding things out. We shall refer to the terminologies as defined in the [explanation of CNNs](#back-to-the-model).

#### Layer number
1. CONV1 3x3
2. RELU
3. CONV2 3x3
4. RELU
5. POOL1 2x2
6. CONV3 3x3
7. RELU
8. CONV4 3x3
9. RELU
10. POOL2 2x2
11. FC1 256
12. FC2 10 (as there as 10 classes)

Now that that's out of the way, here is how you define all this in code.

```python
model = Sequential() # our defined model functions in some sort of sequence, we use the Sequential class to initialize our model before adding the layers

# Here's how you add layers to your model
# Conv1 32 32 (3) => 30 30 (32)
model.add(Conv2D(32, (3, 3), input_shape=X_shape[1:])) # in layer 1 you need to specify input shape this is not needed in subsequent layers
model.add(Activation('relu'))
# Conv2 30 30 (32) => 28 28 (32)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# Pool1 28 28 (32) => 14 14 (32)
model.add(MaxPooling2D(pool_size=(2, 2))) # the CONV CONV POOL structure is popularized in during ImageNet 2014
model.add(Dropout(0.25)) # this thing called dropout is used to prevent overfitting

# Conv3 14 14 (32) => 12 12 (64)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
# Conv4 12 12 (64) => 6 6 (64)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
# Pool2 6 6 (64) => 3 3 (64)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# FC layers 3 3 (64) => 576
model.add(Flatten()) # to turn input into a 1 dimensional array
# Dense1 576 => 256
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Dense2 256 => 10
model.add(Dense(num_classes))
model.add(Activation('softmax')) # the softmax layer will scale all values down to between 0 and 1 which represents probability index
```

> The dropout layers works like this, choose a percentage of parameters randomly and discard them. Sounds counter intuitive but it works in ensuring that no parameter becomes overbearing on the entire model. Also it is a computationally cheap method to reduce overfitting. Do note that dropout layers do not activate during actual testing.

Then you also have to define your parameter optimization strategy.

```python
optimizer = keras.optimizers.Adam() # Adam is one of many gradient descent formulas and one of the most popular
```

> There are also other really good optimizers like RMSprop but for most cases Adam works well enough on it's own. You can attempt to change the learning rate and decay rate. But make sure you know how to conduct gradient descent before actually doing so!

Finally compile the model, simple as that.

```python
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
```

> Accuracy of class prediction model is how you are going to determine if the model is good or not so we use these loss and metrics. For more information visit (TO BE ADDED)

### Model training
Using the dataset we can calculate the set of suitable parameters, the process of finding those parameters is called training. Training of model cannot be simpler.

```python
nb_epoch = 200 # number of iterations to train on
batch_size = 128 # process entire image set by chunks of 128

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test)) # be wawrned that the entire model can take over 4 hours to train if you are not using GPU
```

> A batch size of 128 means to perform an iteration of gradient descent once on every 128 images. Gradient descent (or gradient update) is the most computationally intensive process in training CNNs but despite this it still makes sense to make more iterations of it. 128 is just about the right balance between training duration and frequency of gradient updates.

You can save and load models using these commands,

```python
# to save model architecture
outfile = open('./models/convnet_model.json', 'w') # of course you can specify your own file locations
json.dump(model.to_json(), outfile)
outfile.close()

# to save model weights
model.save_weights('./models/convnet_weights.h5')

# to load model architecture
infile = open('./models/convnet_model.json')
model = keras.models.model_from_json(json.load(infile))
infile.close()

# to load model weights
model.load_weights('./models/convnet_weights.h5')
```

Do note that in the ```basic_model.py``` script, the model weights are saved after each iteration. This way you will be able to continue training your model from where you left off even if you restart your Python. After training you should be able to achieve an accuracy of about 80%.

## Visualizing your CNN
An important skill to have is to be able to interpret models. Here we will cover 4 of such methods. Do note that I have used a deeper model (which requires longer training time) in the codes below as they generally give better visualization. You can load the model I used from ```./models/stashed/``` but it would be completely fine to use the model trained from the previous section.

The purpose of this section is focused around self-learning. Each section is going to be relatively unguided, only a basic intuition of what needs to be done is given the reason behind this is so that you can get right down to coding and researching the ways of implementation. If that isn't your cup of tea, then you can always just read through this and look at some of the pretty images I've plot out and run the codes I've done, I'll include the codes and how to run them below.

### Result based

On your console,
```
python visualize_results.py
```

It is not difficult to imagine how to visualize results based on how well a model performs but here are a list of things you can do,

- calculate model accuracy
- plotting out random images from the test set and printing the prediction made by the model
- plotting out wrongly predicted images
- plotting out a breakdown of wrongly predicted images

### Pixel-importance based

On your console,
```
python visualize_pixel_importance.py
```

You can also visualize which regions the model believes are important in making an accurate prediction. One way to do this is described in the steps below,

1. start with a correctly predicted image (it is important that it is correctly predicted since we know that the algorithm is probably capable of capturing it's key features)
2. remove a pixel or a section from the original image (I did by sections in ```visualize_pixel_importance.py```)
3. make predictions on the new image and see how much the removed aera contributed to making the correct prediction
4. plot out a heat map of how much each area contributes to making the prediction correct

<p align="center"><img src="/imgs/importance_1.jpeg", width="240"></p>
<p align="center">Fig 3.0 image of a dog, important areas shaded in red</p>

Depending on which pictures you used and the color scheme you used, you might end up with something like this. As you can see, important regions usually centered around the dogs ears, eyes and mouth. I apologies for the picture quality being like this the red parts are simply not coming out well. If anyone has any suggestion on making heat maps, please send me an email which can be found below!

### Activation based

On your console,
```
python visualize_hidden_layer_activation.py
```

After training your model, you can also attempt to visualize exactly what each filter is attempting to do. One method is through the construction of an input image which would maximize the output of a filter. In essence what this would achieve is the recreation of the feature that the filter gets most excited over (what the filter is attempting to find). Exactly how this is done is through gradient ascent (opposite of descent). Thankfully Keras can take care of the mathematics for us. A guide on how to do this along with some sample codes are available on [Keras's official blog](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html).

> Here you can also challenge yourself to learn gradient ascent and write your own algorithm to create these images

Here I have plotted out some images which would maximize the activation for 4 filters in each odd numbered convolution layer (this is done so as to save space and maintain objectivity).

<p align="center"><img src="/imgs/activation_1.jpeg", width="560"></p>
<p align="center">Fig 3.1 activation of convolution layer 1</p>

Layer 1:
- As we might expect, filters in layer 1 are looking for simple features. In this case, they are looking for unique colors.

<p align="center"><img src="/imgs/activation_3.jpeg", width="560"></p>
<p align="center">Fig 3.2 activation of convolution layer 3, more complex features are developing such as lines at different orientations</p>

Layer 3:
- Here is where things become more interesting. Filters above are attempting to detect lines of different tilt and colors.

<p align="center"><img src="/imgs/activation_5.jpeg", width="560"></p>
<p align="center">Fig 3.3 activation of convolution layer 5, filters can be seem attempting to find ball shapes</p>

Layer 5:
- A filter can clearly be seen built for the purpose of finding red balls, however from this point on features are starting to become too abstract to fully understand.

<p align="center"><img src="/imgs/activation_7.jpeg", width="560"></p>
<p align="center">Fig 3.4 activation of fully connected layer 1</p>

Layer 7:
- It is unclear what exactly these filters are attempting to look for as the level of abstraction is too high.


### Partial-output based

On your console,
```
python visualize_hidden_layer_output.py
```

Another way to visualize what filters are attempting to do is by plotting out the partial output after each convolution layer. The intuition is that partial outputs are the indicators for the presence of certain features (recall [the high-level explanation](#The-high-level-explanation)). After identifying a suitable image, all you have to do is to run the image through the layers one at a time and plot out those partial outputs.

Here we have an image of a truck, lets take a look at what each filter is attempting to detect.

<p align="center"><img src="/imgs/output_1.jpeg", width="180"></p>
<p align="center">Fig 3.5 original image of a truck</p>

<p align="center"><img src="/imgs/output_2.jpeg", width="560"></p>
<p align="center">Fig 3.6 partial output of convolution layer 2, high-levels of activation are colored in yellow</p>

Layer 1:
- We know from the previous visualization that this layer is attempting to locate colors. The filters that attempt to detect white are getting excited over the body of the truck while those which attempt to locate orange are excited over the head light.

<p align="center"><img src="/imgs/output_4.jpeg", width="560"></p>
<p align="center">Fig 3.7 partial output of convolution layer 3</p>

Layer 3:
- Here filters are getting excited over more complex features. Some filters appear to be detecting wheels and others seem to be attempting to find doors and windows.

Another thing to note is that partial outputs in convolution layer 3 is significantly smaller that those from convolution layer 1. This is due to the effects of pooling. Thus this method of visualization is suitable only for earlier layers as the deeper you go, the lower the resolution of the partial outputs.

## Improving your model

From the [basic model as defined earlier](#model-building) you would only be able to achieve a test accuracy of about 80%. Good models are capable of reaching as high as [95.5% accuracy](http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/). There are a few things that you can do to improve from your basic model.

### Learn from successful implementations of other CNNs
There are billions of different ways to build a CNN and it is not possible to explore all of them. But a good way to get a general grasp of what is expected to work and what isn't is through learning from [past implementations of successful CNNs](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html).

Practices such as multi CONVS then POOL are so successful that they are now generally industry practice. Another general consensus that was derived from history is that increasing model depth would also improve model accuracy.

### Collect more example images
A CNN is only capable of generalizing from images it has seen before. By increasing the number of example images, the CNN would have more experience in classifying more diverse sets of image. Collection of new example images however can sometimes be difficult due to the unavailability of free datasets. At other times, datasets can be of poor quality with tons of wrongly labelled examples, rendering them less useful.

### Use image augmentation
The dataset that has been used in this article contains only 60,000 unique images. excluding testing data, that leaves us with only 50,000 images. However from these 50,000 images, you can 'make' more images.

<p align="center"><img src="/imgs/cat.jpg", width="180"></p>
<p align="center"><img src="/imgs/cat2.jpg", width="180"></p>
<p align="center">Fig 4.0 an image of a cat, flipped on the vertical axis</p>

The two images above are not the same to a machine as they comprise of different sets of pixel values. By doing transformations such as this, we are able to 'expand' the size of the original training set. Other popular methods to expand the training set is through adding white noise to the original picture and contorting the image by zooming and shrinking.

### Apply sparse-encoding
Sparse-encoding techniques such as [sparse-coding](http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding) and [sparse PCA](https://en.wikipedia.org/wiki/Sparse_PCA) are recently discovered methods to boost model accuracy. In some sense, they are akin to Fourier transformations.

Sparse representations are effective for storing patterns and maximizing the independence of features this would lead to more pronounced identification of complex image features. An implementation of how scarcity can help CNNs can be seen in [this paper](https://arxiv.org/abs/1409.6070).

### Transfer Learning
Huge CNNs and large input images can take weeks on end to train. Luckily many world famous CNNs such as Google's Inception V3 and Microsoft's Resnet from the ImageNet competition, can be downloaded online and you can make use of them to generate your own models using some relatively computationally cheap methods. The idea is that the convolution layers have the purpose of sorting out the advanced features from the input images and that the fully connected layers have the job of making use of these advanced features to correctly predict the appropriate label for images. You can remove the fully connected layers and convert the images in your dataset into it's core features. You can then do classification through your own fully connected layers on these collected core features. This works because generally image features are pretty invariant. If a model is capable of detecting rough surfaces in one dataset, then it should also be capable of doing the same thing in another.

Now that you have learnt about the various ways of improving your model, why not take a moment to make your own improvements to the basic model. This section is left intentionally short as now is a great opportunity for you to begin reading other people's publications and researching to understand said publications. It is a tedious process but the most impactful ways to learn is not easy.

In ```improved_model.py``` I have introduced of 2 more layers of convolution as well as image augmentation to reach a accuracy of about 85% in 50 iterations. Try your best to beat this benchmark.

## Foreword
This ends the article. If you have read everything up till this point, I thank you from the bottom of my heart and wish that you have learnt something new. I hope that my explanation was sufficient but if there are any points to improve on or important points that I have left out, please email me at mingruimingrui@hotmail.com.

<div>
  <img src="/imgs/me.jpeg", width="240" align="left" />
  <h3 align="center">Wang Ming Rui</h3>
  <p align="center">I am a 3rd year student at the National University of Singapore. I'm a strong believer in hard work and constant learning. My hobbies include mathematics, going on road trips, and reading dank memes.</p>
</div>
