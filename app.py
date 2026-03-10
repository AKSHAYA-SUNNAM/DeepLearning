import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detector")

st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to detect tumor")

model = load_model("model/cnn_model.h5")
IMG_SIZE = 224

uploaded_file = st.file_uploader("Choose MRI image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        pred = model.predict(img_array)[0][0]
        confidence = round(pred * 100, 2)

        if pred > 0.5:
            st.error(f"Tumor Detected (Confidence: {confidence}%)")
        else:
            st.success(f"No Tumor Detected (Confidence: {100-confidence}%)")
