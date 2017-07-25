# Convolution neural networks made easy with keras
By Wang Ming Rui

## Introduction
Image recognition is the task of taking an image and labelling it. For us humans, this is one of the first skills we learn from the moment we are born and is one that comes naturally and effortlessly. By the time we reach adulthood we are able to immediately recognize patterns and put labels onto objects we see. These skills to quickly identify images, generalized from prior knowledge, are ones that we do not share with our machines.

<p align="center"><img src="https://naushadsblog.files.wordpress.com/2014/01/pixel.gif", width="360"></p>

When a computer sees an image, it will see an array of pixel values, each between a range of 0 to 255. These values while meaningless to us are the only input available to a machine. No one knows how exactly we living beings processes images but scientists today have figured out a technique to simulate this process, albeit at a basic level. We call this technique deep learning.

There are many good resources out there that teaches you how to build your very own deep learning model but in this guide, we shall focus on one of the most revolutionary and fundamental model in recent times, called convolution neural network (or CNN for short).

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
- brief history
In the summer of 2012 ImageNet hosted it's annual Large Scale Visual Recognition Challenge to pit some of the world's most intelligent groups against one another. Alexnet was a deep convolution neural network model submitted in this competition. A good model classification error rate is typically around 25-26%. The second placed model in the competition was able to achieve an impressive 26.2% rate of error. Alexnet scored 15.3%. An unprecedented margin that is over 10% better than the next best model. That was the moment that many people believed marked the beginning of a new age in the field of computer vision.

The basis of computer vision and CNNs were laid down in the early 1950s by Hubel and Wiesel when they studied the behavior of the neurons in the visual cortex of a cat as they moved an image across it's area of vision. During their studies, they observed that the image shown and the orientation affects directly how the neurons fires and activates themselves.

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
