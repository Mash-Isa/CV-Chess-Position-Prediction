import streamlit as st
import tensorflow as tf
import numpy as np
from skimage import io, transform
from keras.models import load_model
from skimage.util.shape import view_as_blocks
from helpers import *

# Load the saved model
model = load_model("model.h5")

# Function to preprocess the input image
def process_image_app(image):
    downsample_size = 200
    square_size = int(downsample_size / 8)
    img_read = transform.resize(image, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)

# Function to predict the chess position from the image
def predict_chess_position(image):
    processed_image = process_image_app(image)
    prediction = model.predict(processed_image).argmax(axis=1).reshape(-1, 8, 8)
    fen_label = fen_from_onehot(prediction[0])
    return fen_label

def main():
    st.title("Chess Position Prediction")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = io.imread(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        if st.button("Predict Chess Position"):
            predicted_fen = predict_chess_position(image)
            st.success(f"Predicted Chess Position: {predicted_fen}")

if __name__ == "__main__":
    main()
