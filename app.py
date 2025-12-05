from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import torch
import cv2
import joblib
import traceback

app = Flask(__name__)

# =====================================================
#                 LOAD MODELS
# =====================================================
model = tf.keras.models.load_model("model.h5")

# Load Label Encoder
label_encoder = joblib.load("label_encoder.joblib")

# =====================================================
#                 LOAD BERT
# =====================================================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()


# =====================================================
#                 IMAGE PREPROCESSING
# =====================================================
def preprocess_image_api(img):

    # Ensure numpy array (already BGR)
    img = np.array(img)

    # If grayscale
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # If RGBA
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # Resize
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # Normalize
    img = img.astype("float32") / 255.0

    return np.expand_dims(img, axis=0)



# =====================================================
#                TEXT EMBEDDING
# =====================================================
def embed_text_api(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return np.expand_dims(vec.astype("float32"), axis=0)


# =====================================================
#               MULTIMODAL
# =====================================================
@app.route("/predict_multimodal", methods=["POST"])
def predict_multimodal():
    try:
        # --------- IMAGE PART ---------
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["file"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_input = preprocess_image_api(img)

        # --------- TEXT PART ---------
        text = request.form.get("text", "")
        if len(text.strip()) == 0:
            return jsonify({"error": "Text is empty"}), 400

        text_vec = embed_text_api(text)

        # --------- PREDICT ---------
        preds = model.predict([text_vec, image_input])

        cls = int(np.argmax(preds))
        probs = preds[0].tolist()
        label = label_encoder.inverse_transform([cls])[0]

        return jsonify({
            "prediction": cls,
            "label": label,
            "probabilities": probs
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =====================================================
#               RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
