'''from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('my_final_model.h5')

@app.route('/', methods=['GET'])
def index():
    # Render the upload form
    return render_template('upload.html')

def preprocess_image(image):
    """Preprocesses an image for model input."""
    image = image.convert('RGB')
    image = image.resize((176, 176))  # Adjust according to your model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values if your model expects normalization
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the image with PIL
    img = Image.open(file)
    img_array = preprocess_image(img)

    # Make a prediction
    prediction = model.predict(img_array)
    # Assuming your model outputs a one-hot encoded vector
    classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']  # Adjust as per your model
    prediction = np.argmax(prediction, axis=1)
    predicted_class = classes[prediction[0]]

    # Return the prediction
    #print('predicted Disease Stage: {predicted_class}')
    return f'Predicted Disease Stage: {predicted_class}'

if __name__ == '__main__':
    app.run(debug=True)'''
    
    
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('my_final_model.h5')

@app.route('/', methods=['GET'])
def index():
    # Render the upload form
    return render_template('upload.html')

def preprocess_image(image):
    """Preprocesses an image for model input."""
    image = image.convert('RGB')
    image = image.resize((176, 176))  # Adjust according to your model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values if your model expects normalization
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image with PIL
    img = Image.open(file)
    img_array = preprocess_image(img)

    # Make a prediction
    prediction = model.predict(img_array)
    # Assuming your model outputs a one-hot encoded vector
    classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']  # Adjust as per your model
    prediction = np.argmax(prediction, axis=1)
    predicted_class = classes[prediction[0]]

    # Return the prediction as a simple string
    # For better practice and future flexibility, you might want to return JSON
    return jsonify({'prediction': f'Predicted Disease Stage: {predicted_class}'})

if __name__ == '__main__':
    app.run(debug=True)

