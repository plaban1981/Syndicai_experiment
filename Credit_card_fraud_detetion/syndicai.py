import requests
import json
import pickle
import numpy as np


class PythonPredictor:
    def __init__(self, model):
        pickle_in = open("classifier.pkl","rb")
        model = pickle.load(pickle_in)
        self.model = model

    def predict(self, payload):
        #data = requests.get(payload["url"]).content
        prediction = self.model.predict(payload)
        print(prediction[0])
        labels = {0 : "Original Bank Note",1:"Fraud Bank Note"}
      
        return "Hello The answer is "+labels[prediction[0]]
