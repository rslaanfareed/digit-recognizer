import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import cv2

# Page config
st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")

st.title("✏️ Handwritten Digit Recognizer")
st.markdown("Draw a digit (0–9) below and the model will predict it!")

# Load model (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

# Drawing canvas
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw here:")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction:")
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        
        # Check if canvas has any drawing
        if img[:, :, :3].sum() > 0:
            # Convert to grayscale and resize to 28x28
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            img_input = img_resized.reshape(1, 28, 28, 1).astype("float32") / 255.0

            # Predict
            predictions = model.predict(img_input, verbose=0)[0]
            predicted = np.argmax(predictions)
            confidence = predictions[predicted] * 100

            st.markdown(f"### 🔢 Predicted: **{predicted}**")
            st.markdown(f"Confidence: **{confidence:.1f}%**")

            # Bar chart of all probabilities
            st.markdown("**Probability distribution:**")
            prob_dict = {str(i): float(predictions[i]) for i in range(10)}
            st.bar_chart(prob_dict)
        else:
            st.info("Start drawing to see the prediction!")
    else:
        st.info("Start drawing to see the prediction!")

st.markdown("---")
if st.button("🗑️ Clear Canvas"):
    st.rerun()