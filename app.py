import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# ✅ Load your trained model at the start
model = tf.keras.models.load_model("drowsiness_model.h5")

st.title("Driver Drowsiness Detection 🚗💤")

# Upload image
uploaded_file = st.file_uploader("Upload a driver image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ✅ Preprocess correctly for model input (224x224, grayscale)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))  # shape (1,224,224,1)

    # Predict
    prediction = model.predict(img_resized)
    confidence = float(prediction[0][0])

    # Label based on threshold
    label = "Drowsy" if confidence > 0.5 else "Alert"

    # Display results with more detail
    st.write(f"Prediction: **{label}**")
    st.metric(label="Drowsiness Probability", value=f"{confidence:.2%}")

    if label == "Drowsy":
        st.error("⚠️ Driver appears **Drowsy**. Immediate attention required!")
    else:
        st.success("✅ Driver appears **Alert**.")
