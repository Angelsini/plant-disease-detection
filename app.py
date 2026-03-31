import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gradcam import get_gradcam

st.title("🌿 Plant Disease Detection")

# Load model
model = load_model("model.h5", compile=False)

# Load class names
with open("classes.json") as f:
    class_names = json.load(f)

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write("### 🌱 Prediction:", class_names[class_index])
    st.write("### 🔢 Confidence:", round(confidence * 100, 2), "%")

    heatmap = get_gradcam(model, img_array)

    st.write("### 🔍 Model Focus")
    st.image(heatmap, width=200)