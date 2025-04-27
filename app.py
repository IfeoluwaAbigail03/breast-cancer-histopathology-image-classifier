import streamlit as st
import tensorflow as tf
import requests
import tempfile
import os
from PIL import Image
import numpy as np

# ======================
# 1. CUSTOM LAYER DEFINITION
# ======================
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

# ======================
# 2. GITHUB MODEL LOADER
# ======================
@st.cache_resource
def load_model():
    """Robust model loader with GitHub LFS support"""
    try:
        # GitHub configuration
        MODEL_URL = "https://github.com/IfeoluwaAbigail03/breast-cancer-histopathology-image-classifier/raw/master/breast_cancer_classifier_FIXED.keras"
        EXPECTED_SIZE = 265411495  # From your earlier test
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "model.keras")
        
        # Download with LFS headers
        headers = {
            "Accept": "application/octet-stream",
            "User-Agent": "Streamlit-App"
        }
        
        with st.spinner("Downloading model from GitHub..."):
            response = requests.get(MODEL_URL, headers=headers, stream=True)
            response.raise_for_status()
            
            # Save with progress
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Verify download
        if os.path.getsize(model_path) < EXPECTED_SIZE * 0.9:
            raise ValueError("Downloaded file too small - might be LFS pointer")
        
        # Load model
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'L2Normalize': L2Normalize},
            compile=False
        )
        
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Troubleshooting:
        1. Verify the file exists at: {MODEL_URL}
        2. Check Git LFS is tracking the file
        3. Try manual download to confirm
        """)
        return None
    finally:
        # Clean up temp files
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            os.rmdir(temp_dir)

# ======================
# 3. STREAMLIT APP
# ======================
def main():
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
    
    # Image processing
    def preprocess_image(upload):
        try:
            img = Image.open(upload).convert('RGB')
            img = img.resize((128, 128))  # Match your model's input shape
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0).astype(np.float32)
        except Exception as e:
            st.error(f"Image processing failed: {str(e)}")
            return None
    
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
                    img_array = preprocess_image(upload)
                    if img_array is None:
                        st.stop()
                    
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
                    
                    # Confidence visualization
                    st.progress(int(malignant_prob * 100))
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    # Configure TensorFlow to be quiet
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    main()