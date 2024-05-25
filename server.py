from flask import Flask, request, jsonify
from PIL import Image
import io # library for handling binary data
import base64

from engine import predict, create_model, train_model

# create our model
model = create_model()

app = Flask(__name__)

@app.route('/')
def waahtever():
    return "empty page"

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if JSON data is present in the request
    if not request.is_json:
        return jsonify({'error': 'No JSON data received'}), 400

    # Get JSON data from the request
    json_data = request.get_json()

    # Check if 'file' key is present in the JSON data
    if 'file' not in json_data:
        return jsonify({'error': 'No file key in JSON data'}), 400

    # Get the base64 encoded image data from the 'file' key
    base64_encoded_image = json_data['file']

    try:
        # Decode the base64 encoded image data
        image_data = base64.b64decode(base64_encoded_image)

        # Convert the image data into an Image object
        image = Image.open(io.BytesIO(image_data))

        # Perform prediction with the model using the image
        prediction = predict(model, image)

        # Return the prediction as JSON response
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/train', methods=['POST'])
def train():
    
    data = request['data']          # request will look like this: {"data": ** all of the train data **}

    train_model(model, data)

    return "good job, you trained the model!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5050')




# DATA AT THE BOTTOM OF A POST REQUEST TO AN INSTAGRAM SIGN IN PAGE:

# {"username": "garret_is_cool", "password": "supersecretpassword", "captcha": "giugl4"} -> {KEY: VALUE}