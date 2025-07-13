import os
import re
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import base64

# ============ Set background image ============
def set_background():
    image_path = r"C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\App\background.jpg"
    with open(image_path, "rb") as img_file:
        bg_img = img_file.read()
    b64_img = base64.b64encode(bg_img).decode()
    bg_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

set_background()

# ============ Set all text to bold Times New Roman, black, and centered for title ============ 
st.markdown("""
    <style>
    html, body, .stApp, [class*="css"]  {
        color: black !important;
        font-weight: bolder !important;  /* More bold */
        font-size: 20px !important;      /* Larger font size */
        font-family: "Times New Roman", Times, serif !important;  /* Font change */
    }
    h1, h2, h3, h4, h5, h6 {
        font-weight: bolder !important;  /* More bold for headers */
        font-size: 30px !important;      /* Larger size for headers */
    }
    .stText, .stMarkdown, .stSubheader, .stInfo {
        font-weight: bolder !important;  /* Ensures all text elements are bold */
        font-size: 20px !important;      /* Uniform text size */
    }
    .stImage {
        font-weight: bolder !important;  /* Bolder font for image captions */
        color: black !important;         /* Black color for captions */
    }
    .stTitle {
        text-align: center !important;   /* Center-align the title */
        font-family: "Times New Roman", Times, serif !important; /* Font for title */
        font-weight: bolder !important;  /* Bolder title */
        color: red !important;           /* Red title text */
        background-color: black !important; /* Black background for title */
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ============ Configuration ============ 
MODEL_PATH = r"C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\unet_model.h5"
LABELS_CSV = r"C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\Metadata\Label Data.csv"

MASK_DIRS = [
    r'C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\Sliced\Cirrhosis_T1_3D\test_masks',
    r'C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\Sliced\Cirrhosis_T1_3D\train_masks',
    r'C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\Sliced\Cirrhosis_T1_3D\valid_masks',
    r'C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\Sliced\Cirrhosis_T2_3D\test_masks',
    r'C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\Sliced\Cirrhosis_T2_3D\train_masks',
    r'C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\Sliced\Cirrhosis_T2_3D\valid_masks',
    r'C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\Sliced\Healthy_subjects\T1_W_Healthy\train_masks',
    r'C:\Users\Name\Documents\STUDIES\Liver Cirrhosis\Sliced\Healthy_subjects\T2_W_Healthy\train_masks'
]

CLASS_LABELS = ['Cirrhosis_T1', 'Cirrhosis_T2', 'Healthy_T1', 'Healthy_T2']
EVALUATION_MAP = {0: "Healthy Liver", 1: "Early Stage Cirrhosis", 2: "Compensated Cirrhosis", 3: "Decompensated Cirrhosis"}
GENDER_MAP = {1: "Female", 2: "Male"}

# ============ Load model and metadata ============ 
model = load_model(MODEL_PATH)
df_labels = pd.read_csv(LABELS_CSV) if os.path.exists(LABELS_CSV) else pd.DataFrame()

def get_metadata_by_filename(filename):
    match = re.match(r"^(\d+)", filename)
    if match and not df_labels.empty:
        patient_id = int(match.group(1))
        row = df_labels[df_labels["Patient ID"] == patient_id]
        if not row.empty:
            return row.iloc[0]
    return None

def find_mask_by_filename(filename):
    for mask_dir in MASK_DIRS:
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            return mask_path
    return None

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def overlay_mask_exact(image, mask, alpha=0.4):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_bin = (mask > 0).astype(np.uint8)
    green = np.zeros_like(image, dtype=np.uint8)
    green[:, :] = (0, 255, 0)
    mask_rgb = np.stack([mask_bin]*3, axis=-1)
    blended = image.copy()
    blended[mask_rgb == 1] = cv2.addWeighted(image, 1 - alpha, green, alpha, 0)[mask_rgb == 1]
    return blended

# ============ Streamlit UI ============ 
st.markdown("<h1 class='stTitle'> LIVER CIRRHOSIS DETECTION AND CLASSIFICATION </h1>", unsafe_allow_html=True)

# Add the "ABOUT CIRRHOSIS" link below the title
st.markdown("""
    <h2 style="text-align: center; font-weight: bolder; font-family: 'Times New Roman', Times, serif; color: black;">
        <a href="https://my.clevelandclinic.org/health/diseases/15572-cirrhosis-of-the-liver" target="_blank" style="color: blue; text-decoration: none;">
            About Cirrhosis ↗
        </a>
    </h2>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an MRI image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())

    image = cv2.imread("temp_image.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed = preprocess_image("temp_image.jpg")
    os.remove("temp_image.jpg")

    pred = model.predict(preprocessed)
    predicted_class = np.argmax(pred)
    predicted_label = CLASS_LABELS[predicted_class]

    filename = uploaded_file.name
    mask_path = find_mask_by_filename(filename)
    metadata = get_metadata_by_filename(filename)

    st.subheader("DETECTION")
    # Removed Predicted Class from output
    # st.write(f"**Predicted Class:** `{predicted_label}`")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    with col2:
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            overlay = overlay_mask_exact(image_rgb, binary_mask)
            st.image(binary_mask, caption="Predicted Mask", use_container_width=True)
        else:
            st.warning("⚠️ No corresponding mask found for this image.")

    with col3:
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            overlay = overlay_mask_exact(image_rgb, binary_mask)
            st.image(overlay, caption="Segmented Liver", use_container_width=True)
        else:
            st.warning("⚠️ No corresponding mask found for this image.")

    if metadata is not None:
        gender_code = int(metadata['Gender']) if pd.notna(metadata['Gender']) else None
        gender = GENDER_MAP.get(gender_code, "Unknown")
        evaluation = EVALUATION_MAP.get(int(metadata['Radiological Evaluation']), "Unknown") if pd.notna(metadata['Radiological Evaluation']) else "N/A"

        st.subheader("Patient Summary")
        st.markdown(f"""
        - **Patient ID**: {metadata['Patient ID']}
        - **Age**: {metadata['Age']}
        - **Gender**: {gender}
        - **Evaluation**: {evaluation}
        """)
    else:
        st.subheader("Patient Summary")
        st.info("No metadata available for this image.")

# Display the final message "Caring for your liver is caring for your life"
st.markdown("""
    <h3 style="text-align: center; font-weight: bold; color: green; font-family: 'Times New Roman'; background-color: black; padding: 10px;">
        Caring for your liver is caring for your life
    </h3>
""", unsafe_allow_html=True)
