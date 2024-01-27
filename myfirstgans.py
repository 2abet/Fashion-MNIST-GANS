# Load libraries
import tensorflow as tf

#Limit GPU memory usage
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

for gpu in gpus:
    print(gpu)

# Import the rest of the libraries
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import numpy as np
# Load the dataset
ds = tfds.load('fashion_mnist', split='train')

ds.as_numpy_iterator().next()['label']

# Visualize the dataset and Build the input pipeline
# Do data transformations
dataiterator = ds.as_numpy_iterator() # Create an iterator to go through the dataset

# Getting data out of the pipeline
dataiterator.next()['image']

# Visualize the dataset
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# Show the first 4 images
for i in range(4):
    sample = dataiterator.next() # Get the next batch
    ax[i].imshow(np.squeeze(sample['image'])) # Show the image
    ax[i].title.set_text(sample['label']) # Append the label to the title

# Scale the images between 0 and 1
def scale_images(data):
    image = data['image'] 
    return image / 255


# Reload the dataset
ds = tfds.load('fashion_mnist', split='train')
# Apply the transformation
ds = ds.map(scale_images)
# Cache the dataset
ds = ds.cache()
# Shuffle the dataset
ds = ds.shuffle(60000)
# Batch the dataset into 128 images per sample
ds = ds.batch(128)
# Reduces the likelihood of bottlenecking
ds = ds.prefetch(64)

ds.as_numpy_iterator().next().shape

# Build the Neural Network

# Import Modelling components
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, LeakyReLU, UpSampling2D


# Build the Generator
def build_generator():
    model = Sequential()
    # Takes in a random value and reshapes it to a 7x7x128
    # Beginnings of a generated image
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))

    # Upsample block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=5, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Upsample block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=5, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Convolutional block 1
    model.add(Conv2D(128, kernel_size=4, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Convolutional block 2
    model.add(Conv2D(128, kernel_size=4, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Convolutional layer to generate the image
    model.add(Conv2D(1, kernel_size=4, padding='same', activation='sigmoid'))

    return model

generator = build_generator()
generator.summary()

# Generate a random fashion image
img = generator.predict(np.random.randn(4, 128, 1))
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# Show the first 4 images
for idx, img in enumerate(img):
    ax[idx].imshow(np.squeeze(img)) # Show the image
    ax[idx].title.set_text(idx) # Append the label to the title

# Build the Discriminator
def build_discriminator():
    model = Sequential()
    
    # Convolutional block 1
    model.add(Conv2D(32, kernel_size=5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    # Convolutional block 2
    model.add(Conv2D(64, kernel_size=5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    # Convolutional block 3
    model.add(Conv2D(128, kernel_size=5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    # Convolutional block 4
    model.add(Conv2D(256, kernel_size=5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    # Flatten the image
    model.add(Flatten())
    model.add(Dropout(0.4))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = build_discriminator()
discriminator.summary()


img = generator.predict(np.random.randn(4, 128, 1))
img.shape

discriminator.predict(img)

# Construct Training Loop

# Setup Losses and Optimizers
# Adam optimizer for both networks
from tensorflow.keras.optimizers import Adam
# Binary cross entropy loss for both networks
from tensorflow.keras.losses import BinaryCrossentropy

# Learning rate for the generator and discriminator optimizer
g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

# Build Subclass Models
# Import the base Model class to subclass training steps
from tensorflow.keras.models import Model

class FashionGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create the generator and discriminator networks
        self.generator = generator
        self.discriminator = discriminator
        
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        # Compile with base class
        super().compile(*args, **kwargs)

        # Create optimizers and losses attributes
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        # Get the data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real images and fake images through the discriminator
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create the labels for the real and fake images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Add some noise to the TRUE outputs
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate the discriminator loss
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # Apply backpropagation - NN learns from its mistakes
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate fake images
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)

            # Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)

            # Calculate the generator loss
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels) # Generator wants to fool the discriminator

        # Apply backpropagation
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        # Return the losses as a dictionary
        return {'d_loss': total_d_loss, 'g_loss': total_g_loss}


# Create instance of subclassed model
fashgan = FashionGAN(generator, discriminator)

# Compile the model
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

# Build Callbacks
# Import dependencies
import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

class GANMonitor(Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        # Randomly generate images
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim, 1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()

        # Save images
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))

# Train the model
hist = fashgan.fit(ds, epochs=2000, callbacks=[GANMonitor()])

# Visualize the training process
plt.plot(hist.history['g_loss'], label='Generator loss')
plt.plot(hist.history['d_loss'], label='Discriminator loss')
plt.legend()
plt.show()


# Save the Generator
generator.save('generator.h5')
discriminator.save('discriminator.h5')

# Load the Generator
from tensorflow.keras.models import load_model

generator.load_weights('generator.h5')
imgs = generator.predict(np.random.randn(16, 128, 1))

# Visualize the generated images
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 20))
for i in range(4):
    for j in range(4):
        ax[i, j].imshow(np.squeeze(imgs[i * 4 + j]))
        ax[i, j].axis('off')