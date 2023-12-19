from keras.models import load_model
import cv2
import numpy as np
import os

def predict_image( image_path):
    model_path = "C:\\Users\LENOVO\Documents\python_prog\PYTHON-IMAGE-SEARCH-PROJECT-\Prediction\converted_keras\keras_model.h5"
    labels_path = "C:\\Users\LENOVO\Documents\python_prog\PYTHON-IMAGE-SEARCH-PROJECT-\Prediction\converted_keras\labels.txt"
    # Load the model
    model = load_model(model_path, compile=False)

    # Load the labels
    class_names = open(labels_path, "r").readlines()

    # Read the image from the specified path
    image = cv2.imread(image_path)

    # Resize the image into (224-height, 224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)



    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    os.remove(image_path)
    # return prediction and confidence score
    return class_name[2:]


def create_product_list():
    root_folder='C:\\Users\LENOVO\Documents\python_prog\PYTHON-IMAGE-SEARCH-PROJECT-\static'
    product_list = []

    # Iterate through each subdirectory in the root folder
    for category_folder in os.listdir(root_folder):
        category_path = os.path.join(root_folder, category_folder)

        # Check if the item is a directory
        if os.path.isdir(category_path):
            # Get a list of image files in the category folder
            image_files = [file for file in os.listdir(category_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

            # Iterate through image files in the category folder
            for i, image_filename in enumerate(image_files, start=1):
                image_path = os.path.join(category_folder, image_filename)

                # Create a product dictionary and append it to the list
                product = {
                    'name': f'Product {i}',
                    'price': 500,  # Set the price as needed
                    'category': category_folder,
                    'image': f'{category_folder}/{image_filename}'
                }
                product_list.append(product)

    return product_list
