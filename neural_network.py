import tensorflow as tf
import os
import generator
import discriminator
import time
import datetime
from IPython import display

from matplotlib import pyplot as plt

BATCH_SIZE = 1
IMG_WIDTH = 512
IMW_HEIGHT = 512
BUFFER_SIZE = 400  # The facade training set consist of 400 images


def loadImages(input_image):
    # il metodo legge le immagini e le converte in tensori unit8
    imageInput = tf.io.read_file(input_image)
    imageInput = tf.image.decode_jpeg(imageInput)

    oldname = os.path.basename(str(input_image))
    filename = oldname[0:len(oldname) - 3] + 'png'
    if str(input_image).find('JPEGRotazione') !=  -1:
        output_image = str(input_image).replace('JPEGRotazione', 'ContorniRotazione')
    else:
        output_image = str(input_image).replace('JPEGImages', 'Contorni')
    output_image = output_image.replace(oldname, filename)

    outputImage = tf.io.read_file(output_image)
    outputImage = tf.image.decode_png(outputImage)

    #   conversione immagini
    imageInput = tf.cast(imageInput, tf.float32)
    outputImage = tf.cast(outputImage, tf.float32)

    return imageInput, outputImage


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




def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


train_dataset = tf.data.Dataset.list_files("C:\\Users\\carra\\Desktop\\dataset\\JPEGRotazione\\*.jpg")
train_dataset = train_dataset.map(loadImages,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files("C:\\Users\\carra\\Desktop\\dataset\\JPEGImages\\*.jpg")
test_dataset = test_dataset.map(loadImages,
                                num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

inp, re = loadImages("C:\\Users\\carra\\Desktop\\dataset\\JPEGImages\\ALEX_20190615_0042.jpg")
inp, re = resize(inp, re, IMG_WIDTH, IMW_HEIGHT)
inp, re = normalize(inp, re)
plt.figure()
plt.imshow(inp)
plt.figure()
plt.imshow(re)
generator = generator.Generator()

gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.figure()
plt.imshow(gen_output[0, ...] / 255.0)
plt.show()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

discriminator = discriminator.Discriminator()
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()
plt.show()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# training
log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator.discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)


def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if step % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_target)
            print(f"Step: {step // 1000}k")

        train_step(input_image, target, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


fit(train_dataset, test_dataset, steps=40000)
