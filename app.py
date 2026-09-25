import os
from pathlib import Path

import boto3
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/drowsiness_model.h5"))
S3_BUCKET = os.getenv("MODEL_S3_BUCKET", "driver-drowsiness-model-madhu-2026")
S3_KEY = os.getenv("MODEL_S3_KEY", "drowsiness_model.h5")

def ensure_model():
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    st.info("Downloading the trained model from AWS S3...")
    boto3.client("s3").download_file(S3_BUCKET, S3_KEY, str(MODEL_PATH))

@st.cache_resource
def load_model():
    ensure_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
CLASS_NAMES = ["Yawn", "No Yawn", "Eyes Closed", "Eyes Open"]
DROWSY_CLASSES = {"Yawn", "Eyes Closed"}

st.title("Driver Drowsiness Detection 🚗💤")
uploaded_file = st.file_uploader("Upload a driver image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Unable to read the uploaded image.")
        st.stop()

    st.image(img, caption="Uploaded Image", use_container_width=True)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))

    prediction = model.predict(img_resized, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])
    class_name = CLASS_NAMES[predicted_index]
    label = "Drowsy" if class_name in DROWSY_CLASSES else "Alert"

    st.write(f"Prediction: **{class_name}**")
    st.metric(label="Model Confidence", value=f"{confidence:.2%}")
    st.write(f"Overall status: **{label}**")

    if label == "Drowsy":
        st.error("⚠️ Driver appears **Drowsy**. Immediate attention required!")
    else:
        st.success("✅ Driver appears **Alert**.")
