import streamlit as st
import requests
from PIL import Image
import io
import base64
import os

# FastAPI endpoint
API_URL = "https://9b89a4025884.ngrok-free.app/predict/"  # Change this to your deployed API URL if needed

# Page configuration
st.set_page_config(
    page_title="Del Monte Product Identifier",
    page_icon="🍍",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.custom-upload-label {
    font-size: 18px;
    color: #007A33;
    font-weight: 600;
    font-family: Optima, sans-serif;
    margin-bottom: 16px;
    display: block;
}
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
section[data-testid="stFileUploader"] label {
    font-size: 1.2rem;
    font-weight: 600;
    color: #000000;
    width: 100%;
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

# Green header bar
st.markdown('<div class="top-green-bar"></div>', unsafe_allow_html=True)

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("del_monte_logo.png", width=80)
with col2:
    st.markdown("<div class='app-header'>Del Monte Product Identifier</div>", unsafe_allow_html=True)

# Subheader
st.markdown(
    "<div class='app-subheader'>Welcome to Del Monte’s Product Identifier!<br>Upload a photo of your product, and our model will recognize it for you within seconds.</div>",
    unsafe_allow_html=True
)

# Upload UI
st.markdown('<label class="custom-upload-label">📸 Upload your photo below:</label>', unsafe_allow_html=True)
image_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to bytes for API
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=image.format)
    img_bytes.seek(0)

    with st.spinner("Analyzing image via API..."):
        try:
            # Send the image to FastAPI
            response = requests.post(
                API_URL,
                files={"file": (image_file.name, img_bytes, f"image/{image.format.lower()}")}
            )

            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])

                if predictions:
                    # Build messages
                    messages = []
                    for pred in predictions:
                        label = pred["label"].replace("_", " ").title()
                        conf = round(pred["confidence"] * 100)
                        messages.append(f"✅ We’re <strong>{conf}%</strong> sure this is <strong>{label}</strong>")
                    message_html = "<br>".join(messages)

                    st.markdown(
                        f"""
                        <div style='
                            background-color: #E8F5E9;
                            border-left: 6px solid #007A33;
                            padding: 20px;
                            border-radius: 12px;
                            font-family: Optima, sans-serif;
                            font-size: 1.3rem;
                            text-align: center;
                            color: #2E7D32;
                            margin-top: 20px;
                        '>
                            {message_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Shopping cart button
                    def get_base64_image(image_path):
                        with open(image_path, "rb") as img_file:
                            return base64.b64encode(img_file.read()).decode()

                    shopping_cart_base64 = get_base64_image("shopping_cart.png")  # Ensure this image exists
                    buy_url = "https://www.delmonte.com/where-to-buy"

                    st.markdown(
                        f"""
                        <div style='text-align: center; margin-top: 20px;'>
                            <a href="{buy_url}" target="_blank" style="text-decoration: none;">
                                <button style="
                                    background-color: #FFD700;
                                    color: #FFFFFF;
                                    border: none;
                                    padding: 12px 24px;
                                    font-size: 1.2rem;
                                    font-weight: bold;
                                    border-radius: 8px;
                                    cursor: pointer;
                                    display: inline-flex;
                                    align-items: center;
                                    justify-content: center;
                                    gap: 10px;
                                    transition: background-color 0.3s ease;
                                " 
                                onmouseover="this.style.backgroundColor='#E6C200'"
                                onmouseout="this.style.backgroundColor='#FFD700'">
                                    <img src="data:image/png;base64,{shopping_cart_base64}" style="width: 20px; height: 20px; vertical-align: middle;"/>
                                    Get it now
                                </button>
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div style='
                            background-color: #FFD700;
                            border-left: 6px solid #FFB300;
                            padding: 20px;
                            border-radius: 12px;
                            font-family: Optima, sans-serif;
                            font-size: 1.2rem;
                            color: #BF360C;
                            margin-top: 20px;
                        '>
                            ⚠️ We couldn’t detect any recognizable Del Monte products in this image. Please make sure the entire product is clearly visible and try again.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the API. Make sure the FastAPI server is running.")
