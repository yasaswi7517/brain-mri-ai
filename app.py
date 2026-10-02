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

# Model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("resnet50_brain_mri.keras")

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
st.write("Upload a Brain MRI image to get an AI-based prediction.")

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
            img_array = np.expand_dims(img_array, axis=0)

            # ResNet50 preprocessing
            img_array = preprocess_input(img_array)

            # Prediction
            prediction = model.predict(img_array, verbose=0)

            predicted_index = np.argmax(prediction[0])
            predicted_class = class_names[predicted_index]
            confidence = prediction[0][predicted_index] * 100

        # Result
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Show all probabilities
        st.subheader("Class Probabilities")

        for name, probability in zip(class_names, prediction[0]):
            st.write(
                f"**{name}:** {probability * 100:.2f}%"
            )
