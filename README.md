# Live Demo: https://breast-cancer-histopathology-image-classifier-nfm69tmfoqeii47a.streamlit.app/
# Breast Cancer Classifier App 

This is a Streamlit web application that classifies breast histology images as benign or malignant using a deep learning model trained on the BreakHis dataset.

## About
Upload a mammogram or histological image.

The AI model analyzes the image and predicts if it shows benign or malignant tissue.

Built with TensorFlow, Keras, and Streamlit.

The histological images used to train the model were sourced from the BreakHis dataset (Breast Cancer Histopathological Database).

## How to Use
Click on "Upload" to select an image file (formats: .jpg, .png, .jpeg).

View prediction results and detailed confidence scores.

## Tech Stack
- Streamlit
- TensorFlow
- Keras
- NumPy
- Pillow
- Requests

## Installation
To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py```


📚 Acknowledgments
BreakHis Dataset: BreakHis Official Site

Streamlit Community


