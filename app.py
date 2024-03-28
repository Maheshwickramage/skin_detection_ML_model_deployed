from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("D:/ML_deployment/new_model/skin_detection_modeltemplate/skin_detection.h5")

# Define skin types mapping
skin_types = {
    0: "Dry",
    1: "Normal",
    2: "Oily",
    3: "Combination"
}

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html', prediction=None)

@app.route('/', methods=['POST'])
def predict():
    # Receive the image file from the user
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Perform prediction
    oilyness_level, skin_type = detect_oilyness(image_path)

    return render_template('index.html', prediction=f'Oilyness Level: {oilyness_level}, Skin Type: {skin_type}')

def detect_oilyness(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    oilyness_level = np.argmax(prediction)
    skin_type = skin_types[oilyness_level]

    return oilyness_level, skin_type

if __name__ == '__main__':
    app.run(port=3000, debug=True)
