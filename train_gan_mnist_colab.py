# Train a GAN on MNIST using TensorFlow (NO pretrained model)
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(-1, 28, 28, 1)

# Generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, input_shape=(100,)),
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Dense(512),
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Dense(28*28, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512),
        layers.LeakyReLU(),
        layers.Dense(256),
        layers.LeakyReLU(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# GAN = generator + discriminator
z = layers.Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
gan = tf.keras.Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training
epochs = 10000
batch_size = 64

for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]

    noise = np.random.normal(0, 1, (batch_size, 100))
    gen_imgs = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
    
    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, D loss: {(d_loss_real + d_loss_fake)/2}, G loss: {g_loss}")
        generator.save(f"generator_epoch_{epoch}.h5")
