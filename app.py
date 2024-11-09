from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import CamembertTokenizer, TFCamembertForSequenceClassification
import tensorflow as tf
import numpy as np
import os
import uuid
from datetime import datetime
import requests

app = Flask(__name__)

# Load the image classification model
image_model = load_model("wildfire_model.h5",compile=False)  
class_labels = ['fire', 'not_fire']  # Adjust based on your classes

# Load the text sentiment model
text_model_name = "jplu/tf-camembert-base"
tokenizer = CamembertTokenizer.from_pretrained(text_model_name)
text_model = TFCamembertForSequenceClassification.from_pretrained(text_model_name, num_labels=2)
text_model.load_weights("camembert_weights.hdf5")

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust size if needed
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Text encoding function
def encode_single_review(review, tokenizer, max_length=400):
    encoded = tokenizer(review, max_length=max_length, truncation=True, padding='max_length', return_tensors='tf')
    return {"input_ids": encoded['input_ids'], "attention_mask": encoded['attention_mask']}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    imagefile = request.files['imagefile']
    
    # Save image with a unique name
    image_dir = './uploads'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex
    file_extension = imagefile.filename.split('.')[-1]
    unique_filename = f"{timestamp}_{unique_id}.{file_extension}"
    image_path = os.path.join(image_dir, unique_filename)
    imagefile.save(image_path)
    
    # Preprocess and predict
    img_array = preprocess_image(image_path)
    predictions = image_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    classification = f"{class_labels[predicted_class]} ({predictions[0][predicted_class] * 100:.2f}%)"
    
    return render_template('index.html', prediction=classification)

@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    inputs = encode_single_review(text, tokenizer)
    outputs = text_model(inputs)
    predicted_class = tf.argmax(outputs.logits, axis=1).numpy()[0]
    sentiment = "positive" if predicted_class == 1 else "negative"

    return jsonify({'text': text, 'sentiment': sentiment})

@app.route('/llama_predict', methods=['POST'])
def llama_predict():
    data = request.json
    description = data.get('description', '')

    if not description:
        return jsonify({'error': 'No description provided'}), 400

    response = requests.post(
        "https://tight-union-6bba.m-boukandoura.workers.dev/",
        json={"description": description}
    )

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({'error': 'Error analyzing with LLaMA'}), response.status_code

if __name__ == '__main__':
    app.run( debug=True)
