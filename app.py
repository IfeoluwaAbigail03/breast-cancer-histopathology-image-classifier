import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import tempfile
import os
from tensorflow.keras.utils import get_file

# 1. Define Custom Layer (MUST match training exactly)
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

# 2. Robust Model Loading (with multiple fallbacks)
@st.cache_resource
def load_model():
    MODEL_URL = "https://drive.google.com/uc?id=1r-8OcG1ggLx5KnNLudPbMJbWw1pc9z_l"
    MODEL_NAME = "breast_cancer_model.keras"
    
    try:
        # Option 1: Use get_file (most reliable)
        model_path = get_file(
            MODEL_NAME,
            MODEL_URL,
            cache_subdir='models',
            extract=False  # Important for .keras files
        )
        
        # Verify download
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Download failed - no file at {model_path}")
        if os.path.getsize(model_path) < 1024:  # At least 1KB
            raise ValueError("Downloaded file is too small (possibly corrupted)")
        
        # Load with custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'L2Normalize': L2Normalize},
            compile=False
        )
        st.toast("✅ Model loaded successfully!", icon="✔️")
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ Primary loading method failed: {str(e)}
        Attempting fallback method...
        """)
        
        try:
            # Option 2: Manual download to temp directory
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, MODEL_NAME)
            
            gdown.download(MODEL_URL, model_path, quiet=True)
            
            if not os.path.exists(model_path):
                raise RuntimeError("Fallback download failed")
                
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'L2Normalize': L2Normalize},
                compile=False
            )
            st.toast("⚠️ Model loaded via fallback method", icon="⚠️")
            return model
            
        except Exception as fallback_error:
            st.error(f"""
            ❌ Critical Error: All loading methods failed
            Last error: {str(fallback_error)}
            
            Required actions:
            1. Verify the Google Drive link is accessible
            2. Check file is valid .keras format
            3. Try manual download: {MODEL_URL}
            """)
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

# 5. Load Model (with progress)
with st.spinner("Loading AI model..."):
    model = load_model()
    
if model is None:
    st.error("""
    ## Critical Error: Model Not Loaded
    Please verify:
    - Google Drive link is publicly accessible
    - File is proper .keras format
    - TensorFlow version matches training environment
    """)
    st.stop()

# 6. Image Processing
def preprocess_image(upload):
    try:
        img = Image.open(upload).convert('RGB')
        img = img.resize((128, 128))  # Must match model input shape
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0).astype(np.float32)
    except Exception as e:
        st.error(f"❌ Image processing failed: {str(e)}")
        return None

# 7. Prediction Interface
upload = st.file_uploader(
    "Choose a mammogram image...", 
    type=["jpg", "png", "jpeg"],
    help="Upload a clear mammogram scan in JPG, PNG or JPEG format"
)

if upload is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(upload, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            try:
                img_array = preprocess_image(upload)
                if img_array is None:
                    st.stop()
                
                # Prediction
                pred = model.predict(img_array, verbose=0)
                benign_prob = pred[0][0]
                malignant_prob = pred[0][1]
                
                # Results Display
                st.subheader("Diagnostic Results")
                
                if malignant_prob > 0.5:
                    st.error(f"""
                    ## 🚨 Suspicious Findings
                    **Malignant Probability:** {malignant_prob:.2%}  
                    **Recommendation:** Urgent specialist consultation
                    """)
                else:
                    st.success(f"""
                    ## ✅ Normal Findings
                    **Benign Probability:** {benign_prob:.2%}  
                    **Recommendation:** Routine follow-up
                    """)
                
                # Confidence visualization
                st.progress(int(malignant_prob * 100))
                st.metric(
                    "Malignancy Confidence", 
                    f"{malignant_prob:.2%}",
                    delta=f"{(malignant_prob-0.5)*100:+.2f}% from decision threshold",
                    delta_color="inverse"
                )
                
            except Exception as e:
                st.error(f"""
                ❌ Analysis Failed
                Error: {str(e)}
                Please try another image or contact support
                """)

# 8. Footer
st.markdown("---")
st.caption("""
**Clinical Disclaimer:**  
This AI tool provides preliminary analysis only. Final diagnosis must be made by a qualified radiologist.  
Model version: breast_cancer_v2.1.keras | TF {tf.__version__}
""")