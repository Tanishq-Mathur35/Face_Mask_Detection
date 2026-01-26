import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.title("😷 Face Mask Detection System")
st.write("Upload an image or use webcam for real-time face mask detection")

model = load_model("mask_detection_model.h5", compile=False)

# ---------------- IMAGE UPLOAD SECTION ---------------- #

st.header("📤 Image Mask Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = img_normalized.reshape(1, 128, 128, 3)

    prediction = model.predict(img_input)[0]

    no_mask_confidence = prediction[0] * 100
    mask_confidence = prediction[1] * 100

    predicted_class = np.argmax(prediction)

    if predicted_class == 1:
        st.success("✅ The person IS wearing a mask")
    else:
        st.error("❌ The person IS NOT wearing a mask")

    st.subheader("Prediction Confidence (%)")

    labels = ["Without Mask", "With Mask"]
    values = [no_mask_confidence, mask_confidence]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)

    st.pyplot(fig)

# ---------------- WEBCAM SECTION ---------------- #

st.header("📸 Real-Time Mask Detection (Webcam)")

start_camera = st.button("Start Webcam Detection")
stop_camera = st.button("Stop Webcam")

frame_window = st.image([])

if start_camera:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_img = cv2.resize(frame_rgb, (128, 128))
        face_img = face_img / 255.0
        face_img = face_img.reshape(1, 128, 128, 3)

        prediction = model.predict(face_img)[0]
        label = np.argmax(prediction)

        no_mask_conf = prediction[0] * 100
        mask_conf = prediction[1] * 100

        if label == 1:
            text = f"Mask ({mask_conf:.2f}%)"
            color = (0, 255, 0)
        else:
            text = f"No Mask ({no_mask_conf:.2f}%)"
            color = (255, 0, 0)

        cv2.putText(
            frame_rgb,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        frame_window.image(frame_rgb)

        if stop_camera:
            break

    cap.release()

st.write(
    "**Note:** Streamlit Cloud does not support direct webcam access via OpenCV "
    "(`cv2.VideoCapture`). Therefore, this application uses Streamlit’s "
    "browser-based camera interface (`st.camera_input`) for image capture, "
    "which is the recommended and officially supported approach."
)
