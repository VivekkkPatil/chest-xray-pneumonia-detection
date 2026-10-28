
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat

st.set_page_config(page_title="Chest X-Ray Pneumonia Detection", page_icon="🩺")
st.title("Chest X-Ray Pneumonia Detection")

st.warning(
    "Educational demonstration only. Upload a frontal chest X-ray. "
    "This model is not validated for medical diagnosis."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/simple_model.keras")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        width, height = image.size

        # Reject extremely small or unusually shaped images.
        if width < 100 or height < 100:
            st.error("Image is too small. Upload a clear chest X-ray.")
            st.stop()

        ratio = width / height
        if ratio < 0.5 or ratio > 2.0:
            st.error("Image dimensions are unsuitable. Upload a chest X-ray.")
            st.stop()

        # Basic filter for images with very little variation.
        grayscale = image.convert("L")
        std_dev = ImageStat.Stat(grayscale).stddev[0]
        if std_dev < 10:
            st.error("This image does not appear suitable. Upload a clear chest X-ray.")
            st.stop()

        st.image(image, caption="Uploaded image", width=350)

        if st.button("Predict"):
            resized = image.resize((128, 128))
            img_array = np.array(resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = float(model.predict(img_array, verbose=0)[0][0])

            if prediction >= 0.5:
                result = "PNEUMONIA"
                confidence = prediction * 100
            else:
                result = "NORMAL"
                confidence = (1 - prediction) * 100

            st.subheader(f"Model prediction: {result}")
            st.write(f"Model score: {confidence:.2f}%")
            st.caption(
                "This score is not a verified probability of disease. "
                "The model may produce incorrect predictions, especially "
                "for images outside its training data."
            )

    except Exception:
        st.error("Could not process this file. Please upload a valid JPG or PNG image.")

st.divider()
st.caption("For educational use only—not for clinical diagnosis.")
