from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from transformers import CamembertTokenizer, TFCamembertForSequenceClassification
import tensorflow as tf
import numpy as np
import os
import uuid
from datetime import datetime
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Load the image classification model with proper architecture
try:
    # Create the same model architecture as training
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
    image_model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    # Load the trained weights
    image_model.load_weights("wildfire_model.h5")
    print("Wildfire detection model loaded successfully")
except Exception as e:
    print(f"Failed to load wildfire model: {e}")
    image_model = None

# Define class labels
class_labels = ['fire', 'not_fire']

def preprocess_image(image_path):
    """Preprocess image exactly as done during training"""
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if image_model is None:
            return jsonify({'error': 'Image model not loaded properly'}), 500

        if 'imagefile' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        imagefile = request.files['imagefile']
        if imagefile.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Create uploads directory if it doesn't exist
        image_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(image_dir, exist_ok=True)

        # Save image with unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        file_extension = os.path.splitext(imagefile.filename)[1].lower()
        
        # Validate file extension
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            return jsonify({'error': 'Invalid file format'}), 400

        unique_filename = f"{timestamp}_{unique_id}{file_extension}"
        image_path = os.path.join(image_dir, unique_filename)
        imagefile.save(image_path)

        # Preprocess and predict
        img_array = preprocess_image(image_path)
        if img_array is None:
            return jsonify({'error': 'Error preprocessing image'}), 500

        predictions = image_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class] * 100)
        
        result = {
            'class': class_labels[predicted_class],
            'confidence': f"{confidence:.2f}%",
            'prediction': f"{class_labels[predicted_class]} ({confidence:.2f}%)"
        }

        # Clean up - remove uploaded file
        os.remove(image_path)

        # Return both JSON and template response based on request type
        if request.headers.get('Accept') == 'application/json':
            return jsonify(result)
        return render_template('index.html', prediction=result['prediction'])

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)