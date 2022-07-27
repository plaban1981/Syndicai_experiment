import json
from flask import Flask, request
from syndicai import PythonPredictor
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)


@app.route('/')
def hello():
    """ Main page of the app. """
    return "Hello World!"


@app.route('/predict')
def predict():
    """ Return JSON serializable output from the model """
    print(request.args.to_dict())
    variance = float(request.args.get("variance"))
    skewness = float(request.args.get("skewness"))
    curtosis = float(request.args.get("curtosis"))
    entropy = float(request.args.get("entropy"))
    payload = np.array([[variance],[skewness],[curtosis],[entropy]])
    print(payload)
    classifier = PythonPredictor("")
    return classifier.predict(payload)


#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=5000)
