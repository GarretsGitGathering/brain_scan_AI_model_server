from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np

# Load DenseNet121 without the top layers, with pre-trained weights
densenet = DenseNet121(
    weights="imagenet",  # Use 'imagenet' for pre-trained weights
    include_top=False,
    input_shape=(128, 128, 3)
)

def preprocess_image(img_input, target_size=(128, 128)):
    """
    Preprocess an image for DenseNet121 model.

    Parameters:
    img_input (str or PIL.Image.Image): Path to the image file or PIL image object.
    target_size (tuple): Desired image size.

    Returns:
    numpy.ndarray: Preprocessed image array ready for prediction.
    """
    if isinstance(img_input, str):
        # Load and resize the image from path
        img = load_img(img_input, target_size=target_size)
    elif isinstance(img_input, Image.Image):
        # Resize the PIL image directly
        img = img_input.resize(target_size)
    else:
        raise ValueError("img_input must be a file path or PIL image object")

    # Convert the image to an array
    img_array = img_to_array(img)

    # If the image is grayscale, add the channel dimension
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    elif img_array.shape[-1] != 3:
        raise ValueError("The image must have either 1 or 3 channels")

    # Expand dimensions to match the shape (1, height, width, channels) for batch size
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image input
    img_array = preprocess_input(img_array)

    return img_array

def create_model():
    model = models.Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(6, activation="sigmoid"))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001)
    )
    return model

def train_model(model, train_data, epochs=10, batch_size=32):
    model.fit(train_data, epochs=epochs, batch_size=batch_size)
    model.save_weights('model.h5')

def predict(model, img_input):
    processed_image = preprocess_image(img_input)
    classification = model.predict(processed_image)
    predicted_classes = (classification > 0.5).astype(int)
    
    response = {
        'any': int(predicted_classes[0, 0]),
        'epidural': int(predicted_classes[0, 1]),
        'intraparenchymal': int(predicted_classes[0, 2]),
        'intraventricular': int(predicted_classes[0, 3]),
        'subarachnoid': int(predicted_classes[0, 4]),
        'subdural': int(predicted_classes[0, 5])
    }
    return response

if __name__ == "__main__":
    # Load an example image for prediction
    image_path = "ID_0005db660.png"
    image = Image.open(image_path)
    
    # Create the model
    model = create_model()
    model.summary()
    
    # Load model weights if they exist
    try:
        model.load_weights('model.h5')
    except OSError:
        print("Model weights not found. Training from scratch.")
    
    # Prepare training data (example using ImageDataGenerator)
    train_datagen = ImageDataGenerator(rescale=0.255)
    
    # Uncomment and set the path to your training data
    # train_data = train_datagen.flow_from_directory(
    #     'path_to_train_data',
    #     target_size=(128, 128),
    #     batch_size=32,
    #     class_mode='categorical'
    # )
    
    # Uncomment to train the model
    # train_model(model, train_data)
    
    # Predict with the model
    prediction = predict(model, image)
    print(prediction)
