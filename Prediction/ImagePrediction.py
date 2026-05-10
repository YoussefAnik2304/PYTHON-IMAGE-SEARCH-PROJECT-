# Global paths
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "converted_keras/model.tflite")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "converted_keras/labels.txt")

# Try to use tflite-runtime for production, fallback to tensorflow for local
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        print("Error: Neither tflite-runtime nor tensorflow is installed.")

# Global Interpreter and labels (Loaded once at startup)
print(f"Loading TFLite model from {TFLITE_MODEL_PATH}...")
INTERPRETER = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
INTERPRETER.allocate_tensors()

# Get input and output details
INPUT_DETAILS = INTERPRETER.get_input_details()
OUTPUT_DETAILS = INTERPRETER.get_output_details()

CLASS_NAMES = open(LABELS_PATH, "r").readlines()

def predict_image(image_path):
    # Read and resize image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Preprocess image
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Set input tensor
    INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], image)

    # Run inference
    INTERPRETER.invoke()

    # Get output tensor
    prediction = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])
    
    index = np.argmax(prediction)
    class_name = CLASS_NAMES[index]
    
    if os.path.exists(image_path):
        os.remove(image_path)
        
    return class_name[2:]


def create_product_list():
    root_folder='./static'
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
