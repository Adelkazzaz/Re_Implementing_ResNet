from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load the pre-trained model
model = load_model('./models/brain_tumor_detector_v1.keras')

# Initialize Flask app
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file:
        # Open the image file and preprocess it
        img = Image.open(file)
        img = img.resize((224, 224))  # Resize image to match model input size
        img = np.array(img) / 255.0   # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Get model predictions
        predictions = model.predict(img)
        class_idx = np.argmax(predictions[0])  # Get index of the class with the highest probability

        # Define class names (update with your model's actual classes)
        class_names = ['Glioma', 'Menin-Gioma', 'No-Tumor', 'Pituitary']

        # Return the prediction result
        predicted_class = class_names[class_idx]
        
        return render_template('result.html', prediction=predicted_class)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
# http://127.0.0.1:5000