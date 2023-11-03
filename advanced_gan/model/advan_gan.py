import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import wandb
import os
import time
import pandas as pd
import cv2
from PIL import Image

# Set up Weights and Biases
wandb.init(project="Advan_DCGAN", job_type="training")

# Hyperparameters
batch_size = 64
latent_dim = 100
num_epochs = 100
low_resolution = (32, 32)
high_resolution = (400, 400)
image_channels = 3  # Assuming full-color images
buffer_size = 60000

# Load JPG
file_path = []
train_path = '/pokemon_jpg/'
for path in os.listdir(train_path):
    if '.jpg' in path:
        file_path.append(os.path.join(train_path, path))

new_path = file_path
images = [np.array((Image.open(path)).resize((128, 128))) for path in new_path]

for i in range(len(images)):
    images[i] = ((images[i] - images[i].min()) / (255 - images[i].min()))
    images = np.array(images)

train_data=images

train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size).batch(batch_size)

# Define your generator model
def build_generator(latent_dim, output_resolution):
    model = keras.Sequential([
        layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding='valid', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(True),

        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(True),

        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(True),

        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(True),

        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.Activation('tanh')
    ])
    return model

def build_discriminator(input_resolution):
    model = keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False, input_shape=(*input_resolution, 3)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False),
    ])
    return model

# Loss function and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training step
@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, image_batch):
    # Your training step logic here
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(image_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Create a directory to save generated images
os.makedirs("generated_images", exist_ok=True)

# Initialize lists to track accuracy and loss
epoch_discriminator_losses = []
epoch_generator_losses = []

generator = build_generator(latent_dim, high_resolution)
discriminator = build_discriminator(high_resolution)

# Training loop
for epoch in range(num_epochs):
    start = time.time()
    for image_batch in train_dataset:
        train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, image_batch)

    # Initialize generated_images here
noise = tf.random.normal([batch_size, latent_dim])
generated_images = generator(noise, training=False)

    # Generate and save images
if (epoch + 1) % 10 == 0:  # Adjust the interval as needed
        generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

        for i in range(batch_size):
            image = generated_images[i].numpy()
            image = (image * 255).astype(np.uint8)
            filename = f"generated_images/epoch_{epoch + 1}_image_{i + 1}.png"
            keras.preprocessing.image.save_img(filename, image)

    # Calculate and log generator and discriminator losses
        generator_loss_value = generator_loss(discriminator(generated_images, training=False)).numpy()
        discriminator_loss_value = discriminator_loss(
        discriminator(image_batch, training=False), discriminator(generated_images, training=False)
    ).numpy()
        wandb.log({"generator_loss": generator_loss_value, "discriminator_loss": discriminator_loss_value, "Epoch: ": epoch + 1, "Time: ": time.time() - start})

# Log generated images to WandB
generated_image_array = (generated_images * 255).numpy().astype(np.uint8)
wandb.log({"Generated Images (Epoch {})".format(epoch + 1): [wandb.Image(img) for img in generated_image_array]})

# Finish the WandB run
wandb.finish()
