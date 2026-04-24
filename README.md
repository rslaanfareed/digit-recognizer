# Handwritten Digit Recognizer

A web-based deep learning application that recognizes handwritten digits in real time. Draw any digit from 0 to 9 using your mouse, and the model instantly predicts what you drew — along with a full confidence breakdown.

**Live Demo:** [digit-recognizer-arslanfareed.streamlit.app](https://digit-recognizer-arslanfareed.streamlit.app)

---

## Overview

This project combines a Convolutional Neural Network trained on the MNIST dataset with an interactive Streamlit frontend. The model processes your drawing the same way MNIST digits are formatted, cropped, centered, padded, and resized to 28x28, which significantly improves real-world prediction accuracy compared to naive implementations.

---

## Features

- Freehand drawing canvas directly in the browser
- Real-time digit prediction as you draw
- Confidence score and full probability distribution chart for all 10 digits
- Smart preprocessing pipeline that mirrors MNIST formatting
- Lightweight deployment with no GPU required

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras (CNN) |
| Dataset | MNIST (70,000 handwritten digits) |
| Frontend | Streamlit |
| Drawing | streamlit-drawable-canvas |
| Image Processing | OpenCV, NumPy, Pillow |
| Deployment | Streamlit Community Cloud |

---

## Model Architecture

The CNN was trained on the full MNIST dataset and achieves over 99% accuracy on the test set.

```
Input (28x28x1)
    Conv2D(32, 3x3, relu)
    MaxPooling2D(2x2)
    Conv2D(64, 3x3, relu)
    MaxPooling2D(2x2)
    Conv2D(64, 3x3, relu)
    Flatten
    Dense(64, relu)
    Dropout(0.5)
    Dense(10, softmax)
```

---

## Preprocessing Pipeline

Raw canvas drawings look very different from MNIST digits. To bridge that gap, each drawing goes through:

1. Grayscale conversion
2. Tight bounding box crop around the drawn digit
3. Square padding to preserve aspect ratio
4. 30% border padding to replicate MNIST whitespace
5. Bicubic resize to 28x28
6. Pixel normalization to [0, 1]

This pipeline is what makes the predictions reliable on free-form mouse drawings.

---

## Run Locally

```bash
git clone https://github.com/rslaanfareed/digit-recognizer.git
cd digit-recognizer
pip install -r requirements.txt
streamlit run app.py
```

> The trained model file `mnist_model.h5` must be present in the project root. To retrain it, run the training script in Google Colab or locally with TensorFlow installed.

---

## Project Structure

```
digit-recognizer/
├── app.py               # Streamlit application
├── train_model.py       # CNN training script
├── mnist_model.h5       # Trained model weights
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python version pin for Streamlit Cloud
└── README.md
```

---

## Limitations

- The model is trained on MNIST which consists of relatively thin, centered digits. Very thick or unusually styled handwriting may still confuse it occasionally.
- Performance depends on how centered and proportionate the drawing is on the canvas.

---





**Arslan Fareed**
[github.com/rslaanfareed](https://github.com/rslaanfareed)
