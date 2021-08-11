import tensorflow as tf
import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt

def loadImages(input_image):
    # il metodo legge le immagini e le converte in tensori unit8
    imageInput = tf.io.read_file(input_image)
    imageInput = tf.image.decode_jpeg(imageInput)

    oldname = os.path.basename(str(input_image))
    filename = oldname[0:len(oldname)-3] + 'png'
    output_image = str(input_image).replace('JPEGImages','Contorni')
    output_image = output_image.replace(oldname, filename)

    outputImage = tf.io.read_file(output_image)
    outputImage = tf.image.decode_png(outputImage)

#   conversione immagini
    imageInput = tf.cast(imageInput, tf.float32)
    outputImage = tf.cast(outputImage, tf.float32)

    return imageInput, outputImage

BATCH_SIZE = 1
IMG_WIDTH = 256
IMW_HEIGHT = 256
OUTPUT_CHANNELS = 3
BUFFER_SIZE = 400 # The facade training set consist of 400 images
LAMBDA = 100


def resize(input_image, output_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output_image = tf.image.resize(output_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, output_image


# Normalizing the images to [-1, 1]
def normalize(input_image, output_image):
  input_image = (input_image / 127.5) - 1
  output_image = (output_image / 127.5) - 1

  return input_image, output_image

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)





train_dataset = tf.data.Dataset.list_files("C:\\Users\\carra\\Desktop\\dataset\\JPEGImages\\*.jpg")
train_dataset = train_dataset.map(loadImages,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


inp, re = loadImages("C:\\Users\\carra\\Desktop\\dataset\\JPEGImages\\ALEX_20190615_0042.jpg")
inp, re = resize(inp, re, IMG_WIDTH, IMW_HEIGHT)
inp, re = normalize(inp, re)
plt.figure()
plt.imshow(inp)
plt.figure()
plt.imshow(re)
generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.figure()
plt.imshow(gen_output[0, ...]/255.0)
plt.show()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss
