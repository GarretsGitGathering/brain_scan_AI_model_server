from keras import layers # High-level neural netwrok api
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
import numpy as np


densenet = DenseNet121(
    weights="DenseNet-BC-121-32-no-top.h5",
    include_top=False,
    input_shape=(128, 128, 3)       # 128x128, RGB 
)


### Process the image so it will be scalable with the model ###     example 3D numpy array: {{a1, a4, a7}, {a2, a5, a8}, {a3, a6, a9}}
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')    # convert RGBA image to RGB
    image = image.resize((128, 128))    # resize the image to fit the model\
    image = np.array(image)             # convert image to numpy array (matrix)
    image = image / 255.0               # convert range of values to 0-100%, because the model expects this
    image = np.expand_dims(image, axis=0)   # add batch dimension                                   !!! RESEARCH !!!!
    return image


### Create Our Instance of the DenseNET model ###
def create_model():
    model = Sequential()        # initializing a model with empty weights. ex: {{0,0,0}, {0,0,0}, {0,0,0}}
    model.add(densenet)         # add densenet model
    model.add(layers.GlobalAveragePooling2D())      # Global average pooling layer
    model.add(layers.Dropout(0.5))              # Initialize Dropout layer to prevent overfitting   !!! RESEARCH !!!!
    model.add(layers.Dense(6, activation="sigmoid"))    # the output layer with sigmoid activation  !!! RESEARCH !!!

    model.compile(
        loss='binary_crossentropy', # cross-entropy loss 
        optimizer=Adam(lr=0.001)    # Adam Optimizer with learning rate 0.001                       !!! RESEARCH !!!
    )

    return model

### Train the DenseNET model ###
def train_model(model, train_data, epochs=10, batch_size=32):
    model.fit(train_data, epochs=epochs, batch_size=batch_size)

    ### WILL SAVE OUR TRAINED MODEL WEIGHTS TO model.h5 ###
    model.save_weights('model.h5')

### Predict with the DenseNET model ###
def predict(model, image):
    processed_image = preprocess_image(image)       # preprocess our image

    classification = model.predict(processed_image)           # predict classification of image

    # assuming the model outputs a binary prediction for each class
    # adjust this if the model output is different
    predicted_classes = (classification > .5).astype(int)   # convert probabilities to binary predicions

    response = {
        'any': int(predicted_classes[0,0]),
        'epidural': int(predicted_classes[0,1]),
        'intraparenchymal': int(predicted_classes[0,2]),    # intraparenchymal  -> 2
        'intraventricular': int(predicted_classes[0,3]),    # intraventricular -> 3
        'subarachnoid': int(predicted_classes[0, 4]),
        'subdural': int(predicted_classes[0, 5])
    }

    return response


if __name__ == "__main__":

    image = Image.open("ID_0005db660.png")

    ### CREATE MODEL ###
    model = create_model()
    model.summary()                     # display a summary of the model (output the weights)
    model.load_weights('model.h5')      # load saved weights into our custom DenseNet AI model

    ### TRAIN MODEL ###
    train_data = ""
    train_model(model, train_data)

    ### PREDICT WITH THE MODEL ###
    prediction = predict(model, image)