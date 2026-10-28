
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Chest X-Ray Diagnosis",
    page_icon="🩺"
)

st.title("Chest X-Ray Pneumonia Detection")

st.write("Upload a chest X-ray image to predict Normal or Pneumonia.")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/simple_model.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-Ray", width=350)

    if st.button("Predict"):

        # Preprocess image
        resized = image.resize((128, 128))
        img_array = np.array(resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Rescaling is already included inside the CNN
        prediction = float(model.predict(img_array, verbose=0)[0][0])

        if prediction >= 0.5:
            result = "PNEUMONIA"
            confidence = prediction * 100
        else:
            result = "NORMAL"
            confidence = (1 - prediction) * 100

        st.subheader(f"Prediction: {result}")
        st.write(f"Model confidence score: {confidence:.2f}%")

        st.progress(int(confidence))

st.divider()
st.caption(
    "Educational project only. This model is not validated "
    "for clinical diagnosis."
)
