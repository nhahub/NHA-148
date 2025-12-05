import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://192.168.1.13:5000"   # Flask API URL
LABELS = ["Casual", "Ethnic", "Formal", "Sports"]

bg_url = "https://thumbs.dreamstime.com/b/people-shopping-clothes-store-retail-scene-vector-design-generative-ai-browsing-showcasing-consumer-culture-fashion-373824979.jpg?w=992"
# Set background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Multimodal Classifier (Image + Text)")

def show_probability_bars(probs, labels):
    import streamlit as st

    st.subheader("Class Probabilities")

    for label, p in zip(labels, probs):
        st.write(f"**{label} — {p*100:.2f}%**")
        st.progress(float(p))


# ================================
#   MULTIMODAL PREDICTION
# ================================

multi_image = st.file_uploader("Upload image for multimodal", type=["jpg","jpeg","png"])
multi_text = st.text_area("Enter description text for multimodal")

if st.button("Predict"):
    if not multi_image:
        st.error("Please upload an image.")
    elif not multi_text.strip():
        st.error("Please enter text.")
    else:
        try:
            files = {
                "file": (multi_image.name, multi_image.getvalue(), multi_image.type)
            }
            data = {"text": multi_text}

            response = requests.post(
                f"{API_URL}/predict_multimodal",
                files=files,
                data=data
            )

            result = response.json()
            pred_idx = result["prediction"]

            st.success(f"Final Prediction: **{LABELS[pred_idx]}**")
            show_probability_bars(result["probabilities"], LABELS)

        except Exception as e:
            st.error("Error contacting multimodal API.")
            st.error(str(e))
