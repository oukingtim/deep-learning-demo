# Import flask which will handle our communication with the frontend
# Also import few other libraries
from flask import Flask, render_template, request,jsonify

import numpy as np

from tensorflow import keras

# from load import *

# Initialize flask app
app = Flask(__name__)

# Path to our saved model
model = keras.models.load_model('./model/model.h5')
model.summary()

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/mnist", methods=['post'])
def mnist():
    # print(request.json)
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 28, 28, 1)
    out = model.predict(input)
    # response = np.argmax(out, axis=1)
    return jsonify(out.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
