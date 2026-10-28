
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="🩺"
)

st.title("🩺 Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray to check the model's prediction.")

st.warning(
    "Educational use only. This application is not validated for "
    "medical diagnosis. Predictions may be incorrect."
)


@st.cache_resource
def load_pneumonia_model():
    return tf.keras.models.load_model("models/simple_model.keras")


@st.cache_resource
def load_image_validator():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


try:
    pneumonia_model = load_pneumonia_model()
    image_validator = load_image_validator()
except Exception as e:
    st.error("Could not load the required models.")
    st.caption(f"Technical details: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", width=350)

        if st.button("Check and Predict"):
            with st.spinner("Checking image type..."):
                labels = [
                    "a chest X-ray radiograph showing human lungs",
                    "a photograph of a street",
                    "a photograph of a wall or building",
                    "a landscape photograph",
                    "a photograph of a person",
                    "a photograph of an everyday object"
                ]

                results = image_validator(
                    image,
                    candidate_labels=labels
                )

                best = results[0]
                xray_result = next(
                    (item for item in results
                     if item["label"] == labels[0]),
                    None
                )

                # Require the X-ray label to rank first and pass
                # a screening threshold.
                if (
                    best["label"] != labels[0]
                    or xray_result is None
                    or xray_result["score"] < 0.35
                ):
                    st.error(
                        "This image could not be verified as a chest "
                        "X-ray. Please upload a clear chest X-ray."
                    )
                    st.stop()

            st.success("Image passed the initial screening.")

            with st.spinner("Analyzing image..."):
                resized = image.resize((128, 128))
                img_array = np.array(
                    resized, dtype=np.float32
                )
                img_array = np.expand_dims(img_array, axis=0)

                prediction = float(
                    pneumonia_model.predict(
                        img_array, verbose=0
                    )[0][0]
                )

            if prediction >= 0.5:
                result = "PNEUMONIA"
                score = prediction * 100
            else:
                result = "NORMAL"
                score = (1 - prediction) * 100

            st.subheader(f"Model prediction: {result}")
            st.write(f"Model score: {score:.2f}%")
            st.caption(
                "This score is not a clinically validated probability. "
                "The image screening step can also make mistakes."
            )

    except Exception as e:
        st.error("Could not process the image.")
        st.caption(f"Technical details: {e}")

st.divider()
st.caption(
    "Educational demonstration only. Not intended for clinical use."
)
