import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2

# Page config
st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")

st.title("Handwritten Digit Recognizer")
st.markdown("Draw a digit (0–9) below and the model will predict it!")

# Load model (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

# Preprocessing — centers and crops digit like MNIST
def preprocess(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    
    # Find bounding box of drawn digit
    coords = cv2.findNonZero(gray)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    cropped = gray[y:y+h, x:x+w]
    
    # Make it square by padding the shorter side
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    # Add border padding (like MNIST has)
    pad = int(size * 0.3)
    square = cv2.copyMakeBorder(square, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=0)
    
    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    return resized.reshape(1, 28, 28, 1).astype("float32") / 255.0

# Layout
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
        
        img_input = preprocess(img)
        
        if img_input is not None:
            predictions = model.predict(img_input, verbose=0)[0]
            predicted = np.argmax(predictions)
            confidence = predictions[predicted] * 100

            st.markdown(f"###Predicted: **{predicted}**")
            st.markdown(f"Confidence: **{confidence:.1f}%**")

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
