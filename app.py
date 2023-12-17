from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

model = VGG16(weights='imagenet', include_top=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

products = [
    {'name': 'Blue Canvas Shoes', 'price': 500, 'category': 'Footwear', 'image': '1/product1.jpg'},
    {'name': 'White Sports Shoes', 'price': 700, 'category': 'Footwear', 'image': '1/product2.jpg'},
    {'name': 'Black Formal Lace Shoes', 'price': 600, 'category': 'Footwear', 'image': '1/1.jpg'},
    {'name': 'Brown Formal Shoes', 'price': 650, 'category': 'Footwear', 'image': '1/2.jpg'},
    {'name': 'White Football Shoes', 'price': 750, 'category': 'Footwear', 'image': '1/3.jpg'},
    {'name': 'Brown Formal Shoes', 'price': 600, 'category': 'Footwear', 'image': '1/4.jpg'},
    {'name': 'Black Sports Shoes', 'price': 750, 'category': 'Footwear', 'image': '1/5.jpg'}
    # Add more product data as needed
]

@app.route('/all_products')
def all_products():
    return jsonify({'products': products})

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    # Flatten the features and ensure they have the same length
    return features.flatten()[:25088]  # Assuming the original shape is (1, 7, 7, 512)

def define_product_features(product):
    # This is a hypothetical function and should be replaced with your logic
    # For example, you might extract information from the product name, description, etc.
    # and combine it with image features
    product_name_features = [hash(product['name']) % 100]  # Hypothetical feature from product name
    product_description_features = [len(product.get('description', ''))]  # Hypothetical feature from product description

    # Extract image features
    img_path = os.path.join('static', product['image'])  # Assuming images are in the 'static' folder
    image_features = extract_features(img_path)

    # Ensure all features have the same length
    product_name_features = np.array(product_name_features).flatten()[:len(image_features)]
    product_description_features = np.array(product_description_features).flatten()[:len(image_features)]

    # Combine different features into a single feature vector
    combined_features = np.concatenate([image_features, product_name_features, product_description_features])

    return combined_features

def calculate_similarity(image_features, product_features):
    # Print the shapes for troubleshooting
    print("Image Features Shape:", image_features.shape)
    print("Product Features Shape:", product_features.shape)
    
    # Implement your similarity calculation here
    dot_product = np.dot(image_features, product_features)
    norm_image = np.linalg.norm(image_features)
    norm_product = np.linalg.norm(product_features)

    if norm_image == 0 or norm_product == 0:
        return 0  # Avoid division by zero

    similarity = dot_product / (norm_image * norm_product)
    return similarity

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

    # Extract features from the image
    image_features = extract_features(img_path)

    # Add features to each product
    for product in products:
        product['features'] = define_product_features(product)  # Replace with the actual features for each product

    # Assuming you have a similarity function for comparing features
    # This function is hypothetical and should be replaced with your similarity logic

    for product in products:
        product_features = product.get('features', None)
        if product_features is not None:
            similarity = calculate_similarity(image_features, product_features)
            product['similarity'] = similarity

    # Sort products by similarity
    filtered_products = sorted(products, key=lambda x: x.get('similarity', 0), reverse=True)

    # Return the filtered products as a JSON response
    return jsonify({'filtered_products': filtered_products})

if __name__ == '__main__':
    app.run(debug=True)
