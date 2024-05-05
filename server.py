from flask import Flask, request, jsonify
from PIL import Image
import io # library for handling binary data

from engine import predict, create_model, train_model

# create our model
model = create_model()

app = Flask(__name__)

@app.route('/')
def waahtever():
    return "empty page"

@app.route('/analyze', methods=['POST'])
def analyze():                                     # THE BOTTOM PART OF OUR HTTP REQUEST:   {'file': ***ENCODED IMAGE***}
    #check if a file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    # grabbing the file from the POST request received from our server
    file = request.files['file']

    if file:
        # open the image after converting from a byte array to an image object
        image = Image.open(io.BytesIO(file.read()))

        #create the prediction with the model
        prediction = predict(model, image)

        return jsonify(prediction)
    
@app.route('/train', methods=['POST'])
def train():
    
    data = request['data']          # request will look like this: {"data": ** all of the train data **}

    train_model(model, data)

    return "good job, you trained the model!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5050')




# DATA AT THE BOTTOM OF A POST REQUEST TO AN INSTAGRAM SIGN IN PAGE:

# {"username": "garret_is_cool", "password": "supersecretpassword", "captcha": "giugl4"} -> {KEY: VALUE}