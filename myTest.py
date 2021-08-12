import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display


import discriminator
import generator


def load(image_file):
    if tf.is_tensor(image_file):
        path = image_file.numpy().decode('utf-8')
    else:
        path = image_file

    input_image = tf.io.read_file(image_file)
    input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.cast(input_image, tf.float32)

    path = path.replace('input', 'output')
    path = path.replace('jpg', 'png')
    image_output = tf.io.read_file(path)
    image_output = tf.image.decode_jpeg(image_output)
    image_output = tf.cast(image_output, tf.float32)

    return input_image, image_output


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


BUFFER_SIZE = 464
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

test = os.path.join(os.path.dirname(__file__), 'dataset\\originaleSquared\\train\\input\\ALT_170925_522.jpg')
img_input, img_output = load(test)

img_input, img_output = random_jitter(img_input, img_output)


dirname = os.path.dirname(__file__)
pathInputTrain = os.path.join(dirname, 'dataset\\originaleSquared\\train\\input\\*.jpg')

train_dataset = tf.data.Dataset.list_files(pathInputTrain)
train_dataset = train_dataset.map(lambda x: tf.py_function(load_image_train, [x], [tf.float32, tf.float32]),
                                  num_parallel_calls=tf.data.AUTOTUNE)

# train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

pathInputVal = os.path.join(dirname, 'dataset\\originaleSquared\\val\\input\\*.jpg')
test_dataset = tf.data.Dataset.list_files(pathInputVal)
test_dataset = test_dataset.map(lambda x: tf.py_function(load_image_test, [x], [tf.float32, tf.float32]))
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3




generator = generator.Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

gen_output = generator(img_input[tf.newaxis, ...], training=False)


LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)






discriminator = discriminator.Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

disc_out = discriminator([img_input[tf.newaxis, ...], gen_output], training=False)






generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)
