import cv2
import pytesseract
import streamlit as st
from PIL import Image
import numpy as np

# Streamlit app setup
st.title('OCR with Tesseract')

# File uploader to upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image with PIL
    img = Image.open(uploaded_file)

    # Convert the image to an OpenCV-compatible format (numpy array)
    img_cv = np.array(img)

    # Convert the image from RGB (PIL) to BGR (OpenCV)
    img_cv = img_cv[:, :, ::-1]

    # Convert to grayscale
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Increase contrast by applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray_img)

    # Apply adaptive thresholding (better for varying light conditions)
    adaptive_thresh_img = cv2.adaptiveThreshold(contrast_img, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)

    # Denoise the image
    denoised_img = cv2.fastNlMeansDenoising(adaptive_thresh_img, None, 30, 7, 21)

    # Show the preprocessed image
    st.image(denoised_img, caption="Processed Image", use_column_width=True, channels="GRAY")

    # Run OCR on the processed image with custom PSM setting (single block of text)
    custom_config = r'--psm 6'  # Use psm 6 for a block of text
    text = pytesseract.image_to_string(denoised_img, config=custom_config)

    # Display the extracted text
    st.subheader("Extracted Text:")
    st.text_area("Extracted Text", text, height=200)
