from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf

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

# Define image paths or URLs for each recommendation
recommendations = {
    "Oily": "/static/oily.jpg",
    "Combination": "/static/combination.jpg",
    "Normal": "/static/normal.jpg",
    "Dry": "/static/dry.jpg",
}

# Function to recommend face wash based on skin type
def recommend_face_wash(skin_type):
    if skin_type == "Oily":
        return "You better use Himalaya Purifying Neem Face Wash or Himalaya Moisturizing Aloe Vera Face Wash", recommendations["Oily"]
    elif skin_type == "Combination":
        return "Himalaya Moisturizing Aloe Vera Face Wash", recommendations["Combination"]
    elif skin_type == "Normal":
        return "Himalaya Natural Glow Kesar Face Wash", recommendations["Normal"]
    elif skin_type == "Dry":
        return "Himalaya Deep Cleanse Balancing Face Wash (Neem and Turmeric) or Himalaya Gentle Hydrating Face Wash", recommendations["Dry"]
    else:
        return "Unknown skin type", None

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html', prediction=None, image=None, recommendation=None, recommendation_image=None)
    elif request.method == 'POST':
        # Receive the image file from the user
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        # Perform prediction
        oilyness_level, skin_type = detect_oilyness(image_path)

        # Recommend face wash based on skin type
        face_wash_recommendation, recommendation_image = recommend_face_wash(skin_type)

        return render_template('index.html', prediction=f'Oilyness Level: {oilyness_level}, Skin Type: {skin_type}',
                               image=imagefile.filename, recommendation=face_wash_recommendation, recommendation_image=recommendation_image)

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
