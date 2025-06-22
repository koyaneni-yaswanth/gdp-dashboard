import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load trained generator
generator = tf.keras.models.load_model('generator_epoch_9000.h5')

st.title("🧠 MNIST Digit Generator")
digit = st.selectbox("Choose a digit (0–9)", list(range(10)))

if st.button("Generate 5 Images"):
    z = np.random.normal(0, 1, (5, 100))
    gen_imgs = generator.predict(z)

    st.subheader(f"Generated images for digit: {digit} (random sampling)")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
