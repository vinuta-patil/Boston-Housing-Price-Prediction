# -*- coding: utf-8 -*-
"""
@author: anantSinghCross
"""
import flask
import json
import numpy as np
import joblib
from flask import Flask, render_template, request
from keras.models import model_from_json

app = Flask(__name__)

@app.route("/")
@app.route("/bostonindex")
def index():
	return flask.render_template('bostonIndex.html')



@app.route("/predict", methods=['POST'])
def make_predictions():
    if request.method == 'POST':
        a = request.form.get('crim')
        b = request.form.get('zn')
        c = request.form.get('indus')
        d = request.form.get('chas')
        e = request.form.get('nox')
        f = request.form.get('rm')
        g = request.form.get('age')
        h = request.form.get('dis')
        i = request.form.get('rad')
        j = request.form.get('tax')
        k = request.form.get('ptratio')
        l = request.form.get('b')
        m = request.form.get('lstat')

        # Convert inputs to float before passing them to the model
        X = np.array([[float(a), float(b), float(c), float(d), float(e), float(f), 
                       float(g), float(h), float(i), float(j), float(k), float(l), float(m)]])

        # Make prediction
        pred = loaded_model.predict(X)

        return flask.render_template('predictPage.html', response=pred[0][0])


if __name__ == '__main__':
    json_file = open("model.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.weights.h5")

    app.run(host='0.0.0.0', port=8001, debug=True)
