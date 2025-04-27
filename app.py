import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import tempfile
import os
import requests
from io import BytesIO
import zipfile
import hashlib

# ======================
# 1. MODEL CONFIGURATION
# ======================
MODEL_CONFIG = {
    "drive_id": "1r-8OcG1ggLx5KnNLudPbMJbWw1pc9z_l",
    "expected_hash": "a1b2c3d4...",  # SHA256 of your model file
    "input_shape": (128, 128, 3),
    "class_names": ["Benign", "Malignant"]
}

# =====================
# 2. CUSTOM LAYERS
# =====================
class L2Normalize(tf.keras.layers.Layer):
    """Custom layer that must match training exactly"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

# =====================
# 3. MODEL LOADING
# =====================
@st.cache_resource
def load_model():
    """Triple-redundant model loading with verification"""
    # Try multiple sources in order
    sources = [
        {
            "name": "Google Drive",
            "url": f"https://drive.google.com/uc?id={MODEL_CONFIG['drive_id']}",
            "type": "gdown"
        },
        {
            "name": "GitHub Mirror",
            "url": "https://github.com/yourusername/repo/releases/download/v1.0/model.keras",
            "type": "direct"
        }
    ]

    for source in sources:
        try:
            with st.spinner(f"Loading model from {source['name']}..."):
                # Create temp directory
                temp_dir = tempfile.mkdtemp()
                model_path = os.path.join(temp_dir, "model.keras")
                
                # Download based on source type
                if source["type"] == "gdown":
                    gdown.download(source["url"], model_path, quiet=True)
                else:  # direct download
                    response = requests.get(source["url"], stream=True)
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                # Verify download
                if not os.path.exists(model_path):
                    raise FileNotFoundError("File not downloaded")
                if os.path.getsize(model_path) < 1024:
                    raise ValueError("File too small (likely corrupted)")
                
                # Verify hash (optional but recommended)
                if MODEL_CONFIG["expected_hash"]:
                    with open(model_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    if file_hash != MODEL_CONFIG["expected_hash"]:
                        raise ValueError("Model hash mismatch - file may be corrupted")
                
                # Handle ZIP if needed
                if zipfile.is_zipfile(model_path):
                    with zipfile.ZipFile(model_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    model_path = os.path.join(temp_dir, zip_ref.namelist()[0])
                
                # Load model
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'L2Normalize': L2Normalize},
                    compile=False
                )
                
                # Verify model structure
                if model.input_shape[1:] != MODEL_CONFIG["input_shape"]:
                    raise ValueError("Model input shape mismatch")
                
                st.success(f"✅ Model loaded from {source['name']}")
                return model

        except Exception as e:
            st.warning(f"⚠️ {source['name']} failed: {str(e)}")
            continue

    # If all sources fail
    st.error("""
    ❌ All model sources failed. Required actions:
    1. Verify Google Drive file is shared publicly (Anyone with link)
    2. Check the file is valid (.keras or .h5 format)
    3. Try manual download: https://drive.google.com/uc?id={MODEL_CONFIG['drive_id']}
    """)
    return None

# =====================
# 4. STREAMLIT UI
# =====================
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# Title and description
st.title("🔬 Breast Cancer Classification")
st.markdown("""
Upload a mammogram image for AI-assisted analysis.  
*Note: This tool provides preliminary results only.*
""")

# =====================
# 5. MODEL LOADING
# =====================
model = load_model()
if model is None:
    st.stop()  # Don't proceed if model fails

# =====================
# 6. IMAGE PROCESSING
# =====================
def preprocess_image(uploaded_file):
    """Convert uploaded file to model-ready array"""
    try:
        # Read image
        img = Image.open(uploaded_file).convert('RGB')
        
        # Resize and normalize
        img = img.resize(MODEL_CONFIG["input_shape"][:2])
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0).astype(np.float32)
    
    except Exception as e:
        st.error(f"❌ Image processing failed: {str(e)}")
        return None

# =====================
# 7. PREDICTION LOGIC
# =====================
def predict(image_array):
    """Run model prediction and format results"""
    try:
        # Get predictions
        preds = model.predict(image_array, verbose=0)[0]
        
        # Convert to percentages
        results = {
            "class": MODEL_CONFIG["class_names"][np.argmax(preds)],
            "confidence": float(np.max(preds)),
            "breakdown": {
                name: float(pred) 
                for name, pred in zip(MODEL_CONFIG["class_names"], preds)
            }
        }
        return results
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None

# =====================
# 8. MAIN INTERFACE
# =====================
upload = st.file_uploader(
    "Choose mammogram image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if upload:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(upload, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            # Preprocess
            img_array = preprocess_image(upload)
            if img_array is None:
                st.stop()
            
            # Predict
            results = predict(img_array)
            if not results:
                st.stop()
            
            # Display results
            st.subheader("Analysis Results")
            
            if results["class"] == "Malignant":
                st.error(f"""
                ## 🚨 Suspicious Findings
                **Classification:** {results['class']}  
                **Confidence:** {results['confidence']:.1%}
                """)
            else:
                st.success(f"""
                ## ✅ Normal Findings
                **Classification:** {results['class']}  
                **Confidence:** {results['confidence']:.1%}
                """)
            
            # Confidence meter
            st.progress(int(results["confidence"] * 100))
            
            # Detailed breakdown
            with st.expander("Detailed Probabilities"):
                for name, prob in results["breakdown"].items():
                    st.write(f"{name}: {prob:.3f}")

# =====================
# 9. FOOTER
# =====================
st.markdown("---")
st.caption(f"""
*Model version: {MODEL_CONFIG['drive_id']} | TensorFlow {tf.__version__}*  
**Disclaimer:** This AI tool provides preliminary analysis only. Always consult a qualified radiologist for diagnosis.
""")