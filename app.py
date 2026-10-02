import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Page settings
st.set_page_config(
    page_title="Brain MRI AI Assistant",
    page_icon="🧠",
    layout="centered"
)

# Model settings
MODEL_URL = "https://github.com/yasaswi7517/brain-mri-ai/releases/latest/download/resnet50_brain_mri_final.keras"
MODEL_PATH = "/tmp/resnet50_brain_mri_final.keras"


# Download and load model
@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading AI model..."):

            response = requests.get(
                MODEL_URL,
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# Class names
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# Title
st.title("🧠 Brain MRI AI Assistant")

st.write(
    "Upload a Brain MRI image to get an AI-based prediction."
)

st.warning(
    "⚠️ This AI result is for educational/research purposes only "
    "and is not a medical diagnosis."
)


# Upload image
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )


    # Prediction button
    if st.button("🔍 Predict"):

        with st.spinner("Analyzing MRI image..."):

            # Resize
            img = image.resize((224, 224))

            # Convert to array
            img_array = np.array(img)

            # Add batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # ResNet50 preprocessing
            img_array = preprocess_input(img_array)

            # Prediction
            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction[0]
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = (
                prediction[0][predicted_index] * 100
            )


        # Result
        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )


        # Show probabilities
        st.subheader("Class Probabilities")

        for name, probability in zip(
            class_names,
            prediction[0]
        ):

            st.write(
                f"**{name}:** "
                f"{probability * 100:.2f}%"
            )
