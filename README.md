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
1. [Convolution neural networks, the game changer](#convolution-neural-networks-the-game-changer)
2. [Keras, deep learning simplified](#keras-deep-learning-simplified)
3. [Building your first models](#building-your-first-models)
4. [Visualizing your CNN](#visualizing-your-CNN)
5. [Improving your model](#improving-your-model)

## Convolution neural networks, the game changer

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
2. take just a single filter and apply filter onto every single 3x3 chunk of the input image receive a 30x30x1 'image' in return.
3. place the 30x30x1 'image' aside and move onto the next filter
4. apply 2nd filter to input image and receive another 30x30x1 'image', stack this 'image' on top of the other 30x30x1 'image' to get a 30x30x2 'image'
5. repeat until all 32 filters are used.

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

The fully connected layer, will take in all of the features produced from the prior convolution layers and output the probability of the image being of each particular label. Remember that the purpose of the convolution layers are to output the presence of advanced features such as eyes, mouth, or wings. By taking note of the presence of such features, the fully connected layer will do the last bit of detective work to determine the most suitable label to apply to each image. Mathematically, it works in the same way as filters do except this time, there's no 3x3 portions. Each 'filter' in this case will be the same size as the output layer from the final layer of convolution. There can however be multiple 'filters' but just as many as the number of label classes you have, the intuition being that you can calculate the confidence level of each individual class separately.

Do keep in mind, this is just a very basic understanding of what the fully connected layer does. In actuality this layer can be much more complex but first, a much long awaited question should be answered.

### Where filter weights come from

> Short recap: Up to this current moment in time, your understanding of how CNNs work is that through a series of multiplications, summations and modifications, you are able to generate a prediction of some sort. Along the way, complex features that a computer would not normally be able to identify are extracted and turned into a simple term that it could, either a feature is present or it is not. This greatly simplifies the original problem of image identification into small simple steps that a computer can solve but there's just one mystery remains.

CNN is an algorithm that requires some very specific parameters (we also call them weights) in the filter layers else the entire model would fail to function. Some of you might not be comfortable to hear this but do not be alarmed! To solve this problem we will have to make use of Mathematics.

The problem is this,

> _find a set of parameters that allows the model to be as accurate at labelling images as possible_

To translate this into mathematics, let us first define a few terms,

<dl>
  <dt><img src="/imgs/x.JPG", width="30"></dt>
  <dd>Represents the original image</dd>

  <dt><img src="/imgs/y.JPG", width="30"></dt>
  <dd>Represents the actual label of the image</dd>

  <dt><img src="/imgs/y-hat.JPG", width="30"></dt>
  <dd>Represents the predicted label of the image</dd>
</dl>

When we take our predicted result and subtract it from our actual result, we get this back,

<p><img src="/imgs/residual.JPG", width="80"></p>

One way of interpreting this is by viewing it as a measure of how far off we are from our desired result (also called the error). An error of 0 would mean that we are spot on, 1 and -1 would mean that there are still improvements to be made. By averaging up the errors a CNN's predictions make on a set of images, we will be able to get a gauge of how well a set of parameters (for filters) are doing. The greater the average error, the more off our predictions are, which prompts us to change the parameters we are using.

<p>Lets take the example of the case were we have 3 images, the errors of an algorithm trying to predict the actual labels of these images are 0, 1, and -1. If we sum up all these errors we should get the total error so 0 + 1 + (-1) = ... 0? Even if we average it out it would still be 0. <img src="/imgs/you-dont-say.jpg", width="80"></p>

That does not mean that the CNN makes perfect predictions and obviously we have applied the wrong way of accumulating errors. A simple modification will fix this issue, by squaring our errors.

> <p><img src="/imgs/summation-symbol.JPG", width="30">, this symbol just means sum up all</p>

<!-- - POOL: POOL is also called the pooling layer. The main purpose of the pooling layer is to reduce the size of the input for the following layers.  -->

 <!-- and the benefits of doing so is manifold. Having a smaller image would mean faster processing time. Also by converging -->









<!-- ### Brief history
STASHED kept for reuse later

In the summer of 2012, ImageNet hosted it's annual Large Scale Visual Recognition Challenge to pit some of the world's most intelligent groups against one another. Alexnet was a deep CNN submitted in this competition. A good model classification error rate is typically around 25-26%. The second placed model in the competition was able to achieve an impressive 26.2% rate of error. Alexnet scored 15.3%. At over 10% better than the next best model, it signaled a new age in the field of computer vision.

<p align="center"><img src="/imgs/ILSVRC-2012-finalists.JPG", width="360"></p>

The basis of computer vision and CNNs were laid down in the early 1950s by Hubel and Wiesel when they studied the behavior of the neurons in the visual cortex of a cat as they moved an image across it's area of vision. During their studies, they observed that the image shown and it's orientation affects directly how the neurons fire and activates themselves. This lead to the deduction that our brain perceives information through the active and inactive states of the neurons that it is made up of.

<p align="center"><img src="/imgs/hubel-wiesel-experiment.jpg", width="360"></p> -->


### TBI

- CNNs explained
  - convolution
  - detective deduction by machine

- study materials
  - videos
  - textbooks
  - mathematics

## Keras, deep learning simplified
- Sequential model
- Layer class
- optimizer

## Building your first models
- dataset
- preprocessing
- model Building
- training

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
