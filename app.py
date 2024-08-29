
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from Prediction.ImagePrediction import *
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

products = create_product_list()

@app.route('/all_products')
def all_products():
    return jsonify({'products': products})

@app.route('/filter_products', methods=['GET'])
def filter_products():
    category = request.args.get('category')
    if not category:
        return jsonify({'error': 'Category parameter is missing'})

    # Implement the logic to filter products by the given category
    filtered_products = [product for product in products if product['category'] == category]

    return jsonify({'filtered_products': filtered_products})

@app.route('/')
def index():
    return render_template('index.html', products=products)

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
    prediction =predict_image(img_path)
    print(f'prediction {prediction}')
    for product in products:
        print(product['category'])


    filtered_products=[]
    for product in products:
         if product['category'].strip().lower() == prediction.strip().lower():
            filtered_products.append(product)
    print(filtered_products)
    return jsonify({'filtered_products': filtered_products})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

