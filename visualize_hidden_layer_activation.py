import sys
import json
from pathlib import Path
from time import time
import math
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.datasets import cifar10

model_path = './models/stashed/convnet_model.json'
weight_path = './models/stashed/convnet_weights.h5'
dtype_mult = 255
num_classes = 10
X_shape = (-1,32,32,3)
layer_depths = ['conv2d_1','conv2d_2','conv2d_3','conv2d_4']

def load_model():
    if (Path('./models/convnet_improved_model.json').is_file() == False) | (Path('./models/convnet_improved_model.json').is_file() == False):
        sys.stdout.write('Please train model using basic_model.py first')
        sys.stdout.flush()
        raise SystemExit

    with open(model_path) as file:
        model = keras.models.model_from_json(json.load(file))
        file.close()

    model.load_weights(weight_path)

    return model

def get_layer_dict(model):
    return dict([(layer.name, layer) for layer in model.layers if (layer.name.find('dense') > -1) | (layer.name.find('conv') > -1)])

def deprocess_image(img):
    # normalize tensor: center on 0., ensure std is 0.1
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1

    # clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def plot_hidden_layer_activation(full_model, layer_model, num_plot=16):
    sys.stdout.write('Plotting for {}\n\n'.format(layer_model.name))
    sys.stdout.flush()
    start_time = time()
    _ = plt.suptitle(layer_model.name)

    # we shall only plot out 16(default) as there are too many filters to visualize
    sub_plot_height = math.ceil(np.sqrt(num_plot))
    layer_output = layer_model.output
    output_shape = layer_model.output_shape
    nb_filters = output_shape[len(output_shape) - 1]

    # here we need to conduct gradient acdent on each filter
    counter = 0
    for i in range(nb_filters):
        if counter < num_plot:
            # conv layers have different outputs than dense layers therefore different loss function sizes
            if layer_model.name.find('conv') != -1:
                loss = K.mean(layer_output[:,:,:,np.random.randint(nb_filters)])
            else:
                loss = K.mean(layer_output[:,np.random.randint(nb_filters)])

            # randomise initial input_img and calc gradient
            input_img = full_model.input#np.expand_dims(np.ones(X_shape[1:]), axis=0)
            grads = K.gradients(loss, input_img)[0]

            # normalize gradient
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img], [loss, grads])

            # we start from a gray image with some noise
            input_img_data = np.random.rand(1, X_shape[1], X_shape[2], X_shape[3]) * 0.1 + 0.5

            # run gradient ascdent for 20 steps
            for j in range(40):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value

            # deprocess_image and plot if found
            if loss_value > 0:
                img = deprocess_image(input_img_data[0])
                ax = plt.subplot(sub_plot_height, sub_plot_height, counter+1)
                _ = plt.axis('off')
                _ = ax.set_xticklabels([])
                _ = ax.set_yticklabels([])
                _ = ax.set_aspect('equal')
                _ = plt.imshow(img.squeeze(), cmap='inferno')

                counter += 1

    sys.stdout.write('Done in {}s\n\n'.format(round(time() - start_time, 2)))
    sys.stdout.flush()
    _ = plt.show()

def main():
    K.set_learning_phase(1)
    model = load_model()
    layer_dict = get_layer_dict(model)

    for layer_name in layer_dict:
        plot_hidden_layer_activation(model, layer_dict[layer_name])

if __name__ == "__main__":
    # execute only if run as a script
    main()
