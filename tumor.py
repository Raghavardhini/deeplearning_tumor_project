import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r"Z:\resume\organ_tumor_segmentation_unet___2.h5")  # Make sure to update the path

# Function to preprocess the image
def preprocess_image(image):
    # Ensure image is in RGB format
    image = image.convert("RGB")
    # Resize image to the input shape expected by the model
    image = image.resize((150, 150))  # Update if your model expects a different size
    image_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_tumor(img_array):
    prediction = model.predict(img_array)
    index = np.argmax(prediction)  # Get the index of the highest probability
    tumor_types = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
    return tumor_types[index]

# Streamlit Web App Configuration
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

# Sidebar
st.sidebar.image(r"C:\Users\Jayaraj.V\Downloads\boys img.jpg", width=150)  # Replace with your logo URL or file path
st.sidebar.title("About the App")
st.sidebar.write("""
    This app detects brain tumors from MRI scan images. Upload an image and get the prediction for potential tumor types.
    
    **Supported Tumor Types:**
    - Glioma Tumor
    - Meningioma Tumor
    - No Tumor
    - Pituitary Tumor
""")

# Function to display treatment recommendations
def display_treatment_recommendations(tumor_type):
    if tumor_type == "Glioma Tumor":
        st.subheader("Glioma Tumor - Treatment Options")
        st.write("""
        **Siddha**: Uses detoxifying herbs like Vellai Milagu chooranam and Karunai Kudineer, emphasizing natural treatments for symptom management.
        
        **Ayurveda**: Includes Ashwagandha, Brahmi, and Guduchi for their neuroprotective properties. Panchakarma therapy is often recommended.
        
        **Hospitals in Tamil Nadu**:
        - Apollo Specialty Cancer Hospital, Chennai: Specializes in neurosurgical treatments.
        - Christian Medical College, Vellore: Known for neurology and radiotherapy.
        """)
        
    elif tumor_type == "Meningioma Tumor":
        st.subheader("Meningioma Tumor - Treatment Options")
        st.write("""
        **Siddha**: Herbal remedies like Chandraprabha Vati and Amukkura Choornam are used to manage symptoms.
        
        **Ayurveda**: Herbs like Kanchanar Guggulu and Shilajit have anti-tumor properties. Ayurvedic dietary recommendations are also provided.
        
        **Hospitals in Tamil Nadu**:
        - MIOT International, Chennai: Offers specialized neurology and neurosurgery facilities.
        - Government Kilpauk Medical College, Chennai: Provides affordable neurology and oncology services.
        """)

    elif tumor_type == "Pituitary Tumor":
        st.subheader("Pituitary Tumor - Treatment Options")
        st.write("""
        **Siddha**: Neerkovai Mathirai and lifestyle adjustments are common, focusing on hormonal balance and symptom relief.
        
        **Ayurveda**: Shatavari, Brahmi, and Vacha help balance the endocrine system. Other treatments may include lifestyle and dietary changes.
        
        **Hospitals in Tamil Nadu**:
        - Sri Ramachandra Medical Center, Chennai: Known for endocrinology and neurosurgery.
        - Madras Medical Mission, Chennai: Offers specialized neurology and endocrinology services.
        """)

# Header and Main Body Styling
st.markdown("""
    <style>
    body {
        background-color: #f0f4f7;
        color: #333;
        font-family: 'Roboto', sans-serif;
    }
    .header {
        font-size: 3em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .subheader {
        font-size: 1.5em;
        text-align: center;
        color: #34495e;
        font-weight: 400;
    }
    .prediction {
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        color: #e74c3c;
        padding: 15px;
        background-color: rgba(231, 76, 60, 0.1);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="header">Brain Tumor Detection App 🧠</p>', unsafe_allow_html=True)

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and make prediction
    img_array = preprocess_image(image)
    result = predict_tumor(img_array)
    
    # Display the prediction
    st.markdown(f'<p class="prediction">Prediction: {result}</p>', unsafe_allow_html=True)

    # Display treatment recommendations
    display_treatment_recommendations(result)

else:
    st.info("Please upload an MRI scan for tumor detection.")
