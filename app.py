import streamlit as st
import requests
from PIL import Image

# ==============================
# AZURE CUSTOM VISION DETAILS
# ==============================

PREDICTION_URL = "https://fishclassification1234-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/1db802c4-5241-4a09-ac5e-7c4e37445e25/classify/iterations/Iteration1/image"
PREDICTION_KEY = "AnvlI15qdsZ9RYjhgG2beTkzbkwTHomctWizxX87RK53KdXXhBPQJQQJ99CDACYeBjFXJ3w3AAAIACOGRQTr"

headers = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream"
}

# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Fish Disease Detection", layout="centered")

st.title("🐟 Fish Disease Detection System")
st.write("Upload a fish image to detect disease using Azure AI.")

uploaded_file = st.file_uploader(
    "Upload Fish Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Disease"):
        with st.spinner("Analyzing image..."):
            image_bytes = uploaded_file.getvalue()
            response = requests.post(
                PREDICTION_URL,
                headers=headers,
                data=image_bytes
            )

        if response.status_code == 200:
            predictions = response.json()["predictions"]

            st.subheader("📊 Prediction Results")

            for pred in predictions:
                disease = pred["tagName"]
                confidence = round(pred["probability"] * 100, 2)
                st.write(f"**{disease}** → {confidence}%")

            best_prediction = max(predictions, key=lambda x: x["probability"])
            st.success(
                f"✅ Detected Disease: **{best_prediction['tagName']}** "
                f"({round(best_prediction['probability'] * 100, 2)}%)"
            )
        else:
            st.error("❌ Error connecting to Azure Custom Vision")
