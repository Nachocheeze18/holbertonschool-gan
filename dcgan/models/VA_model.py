import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import wandb
import os
import time
# Set up Weights and Biases
wandb.init(project="dcgan_mnist", name="VA_DCGAN", job_type="training")

# Hyperparameters
batch_size = 64
latent_dim = 100
num_epochs = 100
image_size = 28
image_channels = 1
num_examples_to_generate = 16

# Define your generator model
def build_generator(latent_dim):
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(7 * 7 * 512),  # Increased number of filters from 256 to 512
        layers.Reshape((7, 7, 512)),
        layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False), #From 128 to 256
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False), #From 64 to 128
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        # Added an extra Conv2DTranspose layer with more filters
    ])
    return model


# Define your discriminator model
def build_discriminator():
    model = keras.Sequential([
        layers.Input(shape=(image_size, image_size, image_channels)),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),  # Increased the number of filters from 64 to 128
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),  # Increased the number of filters from 128 to 256
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
        # Added an extra Conv2D layer with more filters
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

# Load and preprocess the MNIST dataset
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], image_size, image_size, image_channels).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

# Create a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(batch_size)

# Create a directory to save generated images
os.makedirs("generated_images", exist_ok=True)

# Initialize lists to track accuracy and loss
epoch_discriminator_losses = []
epoch_generator_losses = []

# Initialize the generator and discriminator models
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Training loop
for epoch in range(num_epochs):
    start = time.time()
    for image_batch in train_dataset:
        train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, image_batch)

    # Initialize generated_images here
    generated_images = generator(tf.random.normal([num_examples_to_generate, latent_dim]), training=False)

    # Generate and save images
    if (epoch + 1) % 10 == 0:  # Adjust the interval as needed
        generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

        for i in range(num_examples_to_generate):
            image = generated_images[i].numpy()
            image = (image * 255).astype(np.uint8)
            image = np.squeeze(image, axis=-1)
            image = np.expand_dims(image, axis=-1)
            filename = f"generated_images/epoch_{epoch + 1}_image_{i + 1}.png"
            keras.preprocessing.image.save_img(filename, image)

    # Calculate and log generator and discriminator losses
    generator_loss_value = generator_loss(discriminator(generated_images, training=False)).numpy()
    discriminator_loss_value = discriminator_loss(discriminator(image_batch, training=False), discriminator(generated_images, training=False)).numpy()
    wandb.log({"generator_loss": generator_loss_value, "discriminator_loss": discriminator_loss_value, "Epoch: ": epoch + 1, "Time: ": time.time() - start})

    # Log generated images to WandB
generated_image_array = (generated_images * 255).numpy().astype(np.uint8)
wandb.log({"Generated Images (Epoch {})".format(epoch + 1): [wandb.Image(img) for img in generated_image_array]})
generated_image_array = (generated_images * 255).numpy().astype(np.uint8)

# Finish the WandB run
wandb.finish()

generator_loss