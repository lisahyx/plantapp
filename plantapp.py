import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Path to the directory containing the SavedModel
model_path = 'MobileNetV2_random'

# Load the SavedModel
model = tf.saved_model.load(model_path)

# Load the labels from labels.npy
class_names = np.load('unique_labels.npy')

# Streamlit 
st.title('Tropical Plant Classification and Identification Web App')

# Introduction
st.write("This web app allows you to upload images of tropical plants and receive predictions from a pre-trained MobileNetV2 deep learning model. Once you upload an image, the model will provide you with its prediction along with the probability scores for each class.")

# Display the list of supported tropical plants
st.write(f"The tropical plants supported are: {', '.join(class_names)}")

# Link to test images
st.write("You can find sample test images [here](https://drive.google.com/drive/folders/1FL8a_rlJ_TG-SJURA7HQ-SYspWLTnujG?usp=drive_link).")

st.write("")

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
    st.markdown(f"### Predicted Label: {predicted_label}")
    
    st.write("")
    
    # Convert probabilities to percentages
    probabilities_percent = [f"{prob * 100:.2f}%" for prob in predictions[0]]

    # Create a table to display prediction probabilities
    prob_table = {"Label Name": class_names, "Probability": probabilities_percent}
    st.markdown(f"#### Predicted Probabilities:")
    st.table(prob_table)
