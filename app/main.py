import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Define working directory and paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/Plant_Disease_Prediction_InceptionV3.h5"
class_indices_path = f"{working_dir}/blood_group_class_indices.json"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices = json.load(open(class_indices_path))

# Function to Load and Preprocess the Image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Blood Group from an Image
def predict_blood_group(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.set_page_config(page_title="Blood Group Detector", layout="wide")
st.title("🩸 Blood Group Detector")
st.markdown("Upload an image of a blood smear, and the model will predict the blood group. 🩸")

# Sidebar for additional information
with st.sidebar:
    st.header("📝 How to Use")
    st.write("""
    1. Upload a clear `.bmp` image of a blood smear.
    2. Click the **Classify** button to get the prediction.
    3. The result will display the detected blood group.
    """)
    st.info("Supported format: BMP")

# Uploading the image
uploaded_image = st.file_uploader("Upload a blood smear image (.bmp):", type=["bmp"])

if uploaded_image is not None:
    # Display the uploaded image with resizing using PIL
    image = Image.open(uploaded_image)
    
    # Resize the image manually (width=300, height=250)
    image_resized = image.resize((300, 250))

    # Two-column layout for the interface
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_resized, caption="Uploaded Image")  # Display resized image

    with col2:
        st.subheader("🔍 Prediction Results")
        if st.button("Classify"):
            # Predict and display the blood group
            prediction = predict_blood_group(model, image, class_indices)
            st.success(f"**Prediction:** {prediction}")
        else:
            st.write("Click **Classify** to predict the blood group.")

# Footer
st.markdown("---")
st.markdown("Developed by **Anandu P G** with ❤️ and TensorFlow")
