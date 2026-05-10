from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from Prediction.ImagePrediction import predict_image, create_product_list

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize products once
PRODUCTS = create_product_list()

@app.route('/all_products')
def all_products():
    return jsonify({'products': PRODUCTS})

@app.route('/filter_products', methods=['GET'])
def filter_products():
    category = request.args.get('category')
    if not category:
        return jsonify({'error': 'Category parameter is missing'})

    filtered_products = [product for product in PRODUCTS if product['category'] == category]
    return jsonify({'filtered_products': filtered_products})

@app.route('/')
def index():
    return render_template('index.html', products=PRODUCTS)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the file temporarily
    img_path = 'temp_image.jpg'
    file.save(img_path)
    prediction = predict_image(img_path)
    
    filtered_products = [product for product in PRODUCTS if product['category'].strip().lower() == prediction.strip().lower()]
    return jsonify({'filtered_products': filtered_products})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

