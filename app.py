import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import tempfile
import os
import shutil

# ======================
# 1. CUSTOM LAYER DEFINITION
# ======================
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

# ======================
# 2. MODEL CONFIGURATION
# ======================
MODEL_CONFIG = {
    "drive_id": "1r-8OcG1ggLx5KnNLudPbMJbWw1pc9z_l",
    "expected_size": 265411495,  # Exact size from your test
    "input_shape": (128, 128, 3),
    "class_names": ["Benign", "Malignant"]
}

# =====================
# 3. MODEL LOADING
# =====================
@st.cache_resource
def load_model():
    try:
        # Setup temp directory
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "model.keras")
        
        # Download model
        url = f"https://drive.google.com/uc?id={MODEL_CONFIG['drive_id']}"
        gdown.download(url, model_path, quiet=False)
        
        # Verify download
        if not os.path.exists(model_path):
            raise FileNotFoundError("File not downloaded")
            
        if os.path.getsize(model_path) != MODEL_CONFIG["expected_size"]:
            raise ValueError("File size mismatch - possibly corrupted")
        
        # Load model
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'L2Normalize': L2Normalize},
            compile=False
        )
        
        # Verify model structure
        if model.input.shape[1:] != MODEL_CONFIG["input_shape"]:
            raise ValueError("Model input shape mismatch")
            
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Troubleshooting:
        1. Verify the file is accessible: {url}
        2. Check TensorFlow version matches training environment
        3. Try converting model to .h5 format
        """)
        return None
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# =====================
# 4. STREAMLIT APP
# =====================
def main():
    # Configure page
    st.set_page_config(
        page_title="Breast Cancer Classifier",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Breast Cancer Classification")
    st.write("Upload a mammogram image for analysis")
    
    # Load model
    model = load_model()
    if not model:
        st.stop()
    
    # File uploader
    upload = st.file_uploader(
        "Choose mammogram image", 
        type=["jpg", "jpeg", "png"]
    )
    
    if upload:
        cols = st.columns(2)
        
        with cols[0]:
            st.image(upload, caption="Uploaded Image", use_column_width=True)
        
        with cols[1]:
            with st.spinner("Analyzing..."):
                try:
                    # Preprocess image
                    img = Image.open(upload).convert('RGB')
                    img = img.resize(MODEL_CONFIG["input_shape"][:2])
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
                    
                    # Predict
                    pred = model.predict(img_array, verbose=0)[0]
                    benign_prob = pred[0]
                    malignant_prob = pred[1]
                    
                    # Display results
                    if malignant_prob > 0.5:
                        st.error(f"""
                        ## 🚨 Suspicious Findings
                        **Malignant Probability:** {malignant_prob:.1%}
                        """)
                    else:
                        st.success(f"""
                        ## ✅ Normal Findings
                        **Benign Probability:** {benign_prob:.1%}
                        """)
                    
                    # Confidence meter
                    st.progress(int(malignant_prob * 100))
                    
                    # Detailed breakdown
                    with st.expander("Detailed Results"):
                        st.write(f"Benign: {benign_prob:.4f}")
                        st.write(f"Malignant: {malignant_prob:.4f}")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    # Configure TensorFlow to be quiet
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    main()