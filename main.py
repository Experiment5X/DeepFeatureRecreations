import os
import cv2
import argparse
import numpy as np


def get_feature_vector(vgg_model, image, feature_layer_index):
    input_img = vgg_model.input
    feature_vector = vgg_model.layers[feature_layer_index].output

    feature_vector_func = K.function([input_img], [feature_vector])
    return feature_vector_func([image.reshape(1, 224, 224, 3)])[0]


def clip_by_std(x, deviations):
    x_min = np.mean(x) - deviations * np.std(x)
    x_max = np.mean(x) + deviations * np.std(x)

    return np.clip(x, x_min, x_max)


def create_from_features(vgg_model, features, feature_layer_index, learning_rate=0.1, grad_std_clip=1.5,
                         img_std_clip=3, verbose=False):
    input_img = vgg_model.input

    feature_vector = vgg_model.layers[feature_layer_index].output
    loss = K.sum(K.abs(feature_vector - features))

    gradient = K.gradients(loss, input_img)[0]
    iterate = K.function([input_img], [loss, gradient])

    start_image = np.random.rand(1, 224, 224, 3)
    for i in range(0, 500):
        l, grad = iterate([start_image])

        grad_clipped = clip_by_std(grad, grad_std_clip)
        grad_normalized = (grad_clipped - np.min(grad_clipped))
        grad_normalized /= np.max(grad_normalized)

        start_image += learning_rate * (-grad_normalized)
        start_image = clip_by_std(start_image, img_std_clip)

        # in the very beginning and at the very end don't blur the image
        if 25 < i < 490:
            if i % 5 == 0:
                start_image = cv2.GaussianBlur(start_image.reshape(224, 224, 3), (3, 3), 0, 0).reshape(1, 224, 224, 3)
            if i % 15 == 8:
                start_image = cv2.GaussianBlur(start_image.reshape(224, 224, 3), (7, 7), 0, 0).reshape(1, 224, 224, 3)

        # decay the learning rate
        if i == 400:
            i /= 10
        if i == 450:
            i /= 15

        if verbose:
            print('Epoch ' + str(i) + ': Loss = ' + str(l))

    # normalize to between 0 and 255
    start_image -= start_image.min()
    start_image /= start_image.max()
    start_image *= 255

    return start_image.reshape(224, 224, 3)


parser = argparse.ArgumentParser(description='Re-create images from deep feature representations in VGG16')
parser.add_argument('--learning-rate', action='store', type=float, default=0.1,
                    help='the learning rate to update the image at')
parser.add_argument('--layer-index', action='store', type=int, default=-3,
                    help='index of the layer in VGG16 from which to re-create the image from')
parser.add_argument('--image-std-clip', action='store', type=float, default=3,
                    help='the number of standard deviations to clip the image at after each update')
parser.add_argument('--grad-std-clip', action='store', type=float, default=1.5,
                    help='the number of standard deviations to clip the gradient at')
parser.add_argument('--verbose', action='store_true', default=False, help='view detailed status of the network')
parser.add_argument('images', nargs='+')
args = parser.parse_args()

# do this after parsing the arguments so the user doesn't have to wait forever to view the help
import keras.backend as K
from keras.applications import vgg16
model = vgg16.VGG16(weights='imagenet', include_top=True)
model.summary()

for input_image_file in args.images:
    image_base_name = os.path.splitext(os.path.basename(input_image_file))[0]

    input_image = cv2.imread(input_image_file)
    f = get_feature_vector(model, input_image, args.layer_index)
    print('[' + image_base_name + '] Calculated feature vector')

    re_created_image = create_from_features(model, f, args.layer_index, learning_rate=args.learning_rate,
                                            grad_std_clip=args.grad_std_clip, img_std_clip=args.image_std_clip,
                                            verbose=args.verbose)

    save_file_name = image_base_name + '_recreated.png'
    cv2.imwrite(save_file_name, re_created_image)
    print('[' + image_base_name + '] Wrote re-created image')




