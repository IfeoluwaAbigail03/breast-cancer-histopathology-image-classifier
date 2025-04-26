import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import os
import requests
import tempfile

# 1. Define Custom Layer (MUST match training exactly)
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

# 2. Robust Model Loading (from Google Drive)
@st.cache_resource
def load_model():
    MODEL_URL = "https://drive.google.com/uc?export=download&id=1r-8OcG1ggLx5KnNLudPbMJbWw1pc9z_l"
    
    # Download model
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
    except Exception as e:
        st.error(f"❌ Failed to download model: {str(e)}")
        return None

    # Save to a temporary file
    temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
    temp_model_file.write(response.content)
    temp_model_file.flush()

    # Try to load model
    try:
        model = tf.keras.models.load_model(
            temp_model_file.name,
            custom_objects={'L2Normalize': L2Normalize},
            compile=False
        )
        st.success("✅ Model downloaded and loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# 3. Streamlit UI Configuration
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# 4. App Interface
st.title("🔬 Breast Cancer Classification")
st.write("Upload a mammogram image for analysis")

# 5. Load Model
model = load_model()
if model is None:
    st.error("""
    Critical Error: Could not load model.
    Please verify:
    1. Google Drive link is correct
    2. Model file is not corrupted
    3. TensorFlow versions match
    """)
    st.stop()

# 6. Image Processing
def preprocess_image(upload):
    try:
        img = Image.open(upload).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0).astype(np.float32)
    except Exception as e:
        st.error(f"❌ Image processing failed: {str(e)}")
        return None

# 7. Prediction Interface
upload = st.file_uploader(
    "Choose a mammogram image...", 
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=False
)

if upload is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(upload, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            try:
                img_array = preprocess_image(upload)
                if img_array is None:
                    st.stop()
                
                pred = model.predict(img_array)
                benign_prob = pred[0][0]
                malignant_prob = pred[0][1]
                
                # Results Display
                st.subheader("Diagnostic Results")
                
                if malignant_prob > 0.5:
                    st.error(f"""
                    🚨 **Malignant Detected**  
                    Confidence: {malignant_prob:.2%}
                    """, icon="⚠️")
                else:
                    st.success(f"""
                    ✅ **Benign**  
                    Confidence: {benign_prob:.2%}
                    """, icon="✔️")
                
                # Visualizations
                st.progress(int(malignant_prob * 100))
                st.metric(
                    "Malignancy Probability", 
                    f"{malignant_prob:.2%}",
                    delta=f"{(malignant_prob-benign_prob)*100:+.2f}%",
                    delta_color="inverse"
                )
                
                # Confidence breakdown
                with st.expander("Detailed Confidence Scores"):
                    st.write(f"Benign: {benign_prob:.4f}")
                    st.write(f"Malignant: {malignant_prob:.4f}")
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# 8. Footer
st.markdown("---")
st.caption("""
**Clinical Note:** This AI-assisted tool provides preliminary analysis only.  
Final diagnosis must be made by a qualified radiologist.  
Model version: breast_cancer_classifier_FIXED.keras
""")