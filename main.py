import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

def classify_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: File {img_path} not found.")
        return

    # Load pre-trained MobileNetV2 model
    model = MobileNetV2(weights='imagenet')

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)
    
    # Decode and print top 3 predictions
    print('Predicted top 3:')
    for i, (imagenet_id, label, score) in enumerate(decode_predictions(preds, top=3)[0]):
        print(f"{i+1}: {label} ({score:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_image>")
    else:
        classify_image(sys.argv[1])