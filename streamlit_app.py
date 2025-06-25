import streamlit as st
from PIL import Image

from ultralytics import YOLO

@st.cache_resource  # So it doesn't reload every time
def load_model():
    return YOLO("best.pt")  # Make sure best.pt is in your project folder

model = load_model()

# Set page config
st.set_page_config(
   page_title="Del Monte Product Identifier",
   page_icon="🍍",
   layout="centered"
)


# Inject custom CSS
st.markdown("""
   <style>
   /* Style the file uploader label */
   .custom-upload-label {
       font-size: 18px;
       color: #007A33;  /* Del Monte green */
       font-weight: 600;
       font-family: Optima, sans-serif;
       margin-bottom: 16px;
       display: block;
   }


   /* OUTER WRAPPER */
.stFileUploader {
    background-color: #F4F1DE;
    padding: 20px;
    border-radius: 16px;
    border: 3px dashed #007A33;
    width: 100%;
    max-width: 100% !important;
    margin: auto;
    box-shadow: 0px 3px 12px rgba(0, 0, 0, 0.1);
}

/* INNER YELLOW BOX */
section[data-testid="stFileUploader"] > div > div {
    background-color: #FFD700;
    padding: 40px;
    border-radius: 12px;
    min-height: 200px;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* PREVENT INTERNAL LABEL FROM SHRINKING UP */
section[data-testid="stFileUploader"] label {
    font-size: 1.2rem;
    font-weight: 600;
    color: #000000;
    width: 100%;
    max-width: 100%;
    text-align: center;
}
       .app-header {
           font-family: Optima;
           font-size: 2.5rem;
           font-weight: 700;
           color: #4E342E;
           margin-bottom: 0.5rem;
       }
       .app-subheader {
           font-size: 1.1rem;
           font-weight: 600;
           color: #555;
           margin-bottom: 2rem;
       }
       .stButton>button {
           background-color: #FFD700;
           color: white;
           font-weight: bold;
           border-radius: 10px;
       }
       .top-green-bar {
           width: 100%;
           height: 60px;
           background-color: #0A6B3E;
           position: fixed;
           top: 0;
           left: 0;
           z-index: 9999;
       }
       .css-18e3th9 {
           padding-top: 70px;
       }
   </style>
""", unsafe_allow_html=True)


# Top green bar
st.markdown('<div class="top-green-bar"></div>', unsafe_allow_html=True)


# Side-by-side logo and header
col1, col2 = st.columns([1, 5])


with col1:
   st.image("del_monte_logo.png", width=80)


with col2:
   st.markdown("<div class='app-header'>Del Monte Product Identifier</div>", unsafe_allow_html=True)


# Subheader
st.markdown("<div class='app-subheader'>Welcome to Del Monte’s Product Identifier! Simply upload a photo of your product, and our model will recognize it for you within seconds.</div>", unsafe_allow_html=True)


# Manually add the styled label
st.markdown('<label class="custom-upload-label">📸 Upload your photo below:</label>', unsafe_allow_html=True)


# File uploader
image_file = st.file_uploader("", type=["png", "jpg", "jpeg"])


# PREDICTION LOGIC
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporarily for YOLO input
    image.save("temp.jpg")

    st.info("Analyzing image...")
    results = model.predict("temp.jpg", save=True)  # Runs prediction and saves result image

    # Load and show result image
    st.image("runs/detect/predict/image0.jpg", caption="Prediction", use_column_width=True)
