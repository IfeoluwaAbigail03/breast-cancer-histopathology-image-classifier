import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import os
import shutil
import requests

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
    "model_url": "https://raw.githubusercontent.com/IfeoluwaAbigail03/breast-cancer-histopathology-image-classifier/master/breast_cancer_classifier_FIXED.keras",
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
        response = requests.get(MODEL_CONFIG['model_url'], stream=True)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception("Failed to download model. Check the URL.")
        
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
        1. Verify the model URL is correct: {MODEL_CONFIG['model_url']}
        2. Check TensorFlow version matches training environment
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