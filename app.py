import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", layout="wide")

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

model = load_model("mask_detection_model.h5", compile=False)

def predict_mask(image):
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = img_normalized.reshape(1, 128, 128, 3)
    prediction = model.predict(img_input)[0]
    label = np.argmax(prediction)
    return prediction, label

st.title("Face Mask Detection System")

if st.session_state.uploaded_file is None:
    st.write("Upload an image to check whether the person is wearing a mask.")

left_col, right_col = st.columns(2)

with left_col:
    if st.session_state.uploaded_file is None:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "png", "jpeg"]
        )
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.rerun()
    else:
        image = Image.open(st.session_state.uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("❌ Remove Image"):
            st.session_state.uploaded_file = None
            st.rerun()

with right_col:
    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file).convert("RGB")
        prediction, label = predict_mask(image)
        labels = ["Without Mask", "With Mask"]
        values = [prediction[0] * 100, prediction[1] * 100]
        st.subheader("Prediction Output")
        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 100)
        st.pyplot(fig)
        if label == 1:
            st.success("Person is wearing a mask.")
        else:
            st.error("Person is not wearing a mask.")
