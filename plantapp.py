#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TFSMLayer


# In[4]:


# Load the pre-trained CNN model
# model = tf.keras.models.load_model('MobileNetV2_augment_30epochs', compile=False)
#model = tf.keras.layers.TFSMLayer(MobileNetV2_augment_30epochs, call_endpoint='serving_default')

# Path to the directory containing the SavedModel
model_path = 'MobileNetV2_random'

# Load the SavedModel
model = tf.saved_model.load(model_path)

# Load the labels from labels.npy
class_names = np.load('random_labels.npy')

# Streamlit 
st.title('Image Classification and Identification Website')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, [224, 224], method=tf.image.ResizeMethod.BICUBIC)

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Assuming the signature key is 'serving_default'
    infer = model.signatures["serving_default"]
    output = infer(tf.constant(img_array))
    predictions = output[list(output.keys())[0]]
    predicted_label_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_label = class_names[predicted_label_index]

    # Display the predicted label
    st.write(f"Prediction: {predicted_label}")




