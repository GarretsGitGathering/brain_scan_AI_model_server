from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd
import os

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

    # Path to the CSV and image folder
    csv_file = 'dataset/stage_1_train.csv'
    image_folder = 'dataset/stage_1_train_images'

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract unique IDs and labels
    df['image_id'] = df['ID'].apply(lambda x: x.split('_')[1])
    df['label'] = df['ID'].apply(lambda x: x.split('_')[2])

    # Pivot table to reshape the dataframe
    df_pivot = df.pivot_table(index='image_id', columns='label', values='Label', aggfunc='first').reset_index()

    # Create the 'filepath' column
    df_pivot['filepath'] = df_pivot['image_id'].apply(lambda x: os.path.join(image_folder, f"ID_{x}.png"))

    # Validate image files
    def validate_image(filepath):
        try:
            with Image.open(filepath) as img:
                img.verify()  # Verify that it is an image
            return True
        except (IOError, SyntaxError):
            return False

    df_pivot['valid_image'] = df_pivot['filepath'].apply(validate_image)
    df_pivot = df_pivot[df_pivot['valid_image']]

    # Debugging: Check the first few file paths
    print(df_pivot['filepath'].head())

    # Ensure the 'filepath' column is created and the files exist
    df_pivot['filepath'] = df_pivot['filepath'].apply(lambda x: x if os.path.exists(x) else None)

    # Debugging: Count the number of valid file paths
    print(f"Number of valid file paths: {df_pivot['filepath'].notna().sum()}")

    df_pivot = df_pivot.dropna(subset=['filepath'])

    # Ensure the label columns are of type int
    for label in ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']:
        df_pivot[label] = df_pivot[label].astype(int)

    # Ensure the 'filepath' column is created
    if 'filepath' not in df_pivot.columns:
        raise KeyError("The 'filepath' column was not created correctly")

    # Prepare training data (example using ImageDataGenerator)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # Split for validation data
    )

    train_data = train_datagen.flow_from_dataframe(
        dataframe=df_pivot,
        x_col='filepath',
        y_col=['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'],
        target_size=(128, 128),
        batch_size=32,
        class_mode='raw',  # Using 'raw' for multi-label classification
        subset='training'
    )

    validation_data = train_datagen.flow_from_dataframe(
        dataframe=df_pivot,
        x_col='filepath',
        y_col=['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'],
        target_size=(128, 128),
        batch_size=32,
        class_mode='raw',
        subset='validation'
    )

    # Create the model
    model = create_model()
    model.summary()

    # Load model weights if they exist
    try:
        model.load_weights('model.h5')
    except OSError:
        print("Model weights not found. Training from scratch.")
    
    # Uncomment to train the model
    train_model(model, train_data)
    
    # Predict with the model
    prediction = predict(model, image)
    print(prediction)
