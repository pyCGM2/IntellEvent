import os
import sys

from flask import Flask, request, jsonify
import tensorflow as tf #conda install tensorflow==2.3.0
import numpy as np #conda install numpy==1.18.5

# pkgPath = "C:\\Users\\fleboeuf\\Documents\\Programmation\\git folders\\intellevent(fork)"
# if pkgPath not in sys.path: sys.path.append(pkgPath)

import intellevent

def main():
    app = Flask(__name__)
    # Upon starting the 'server.py', both models will be automatically loaded
    # this will ensure faster predictions

    app.fc_model = tf.keras.models.load_model(intellevent.DATA_TRAINED_MODEL_PATH+ "FC_velo_model.h5")
    app.fo_model = tf.keras.models.load_model(intellevent.DATA_TRAINED_MODEL_PATH+"FO_velo_model.h5")


    @app.route('/predict_fc', methods=['POST'])
    def predict_fc():
        # Get data from 'request.py', makes the predictions and returns the probabilities
        data = request.get_json(force=True)
        fc_prediction = app.fc_model(np.array(data['traj']), training=False)
        
        return jsonify(fc_prediction.numpy().tolist())


    @app.route('/predict_fo', methods=['POST'])
    def predict_fo():
        # Get data from 'request.py', makes the predictions and returns the probabilities
        data = request.get_json(force=True)
        fo_prediction = app.fo_model(np.array(data['traj']), training=False)
        return jsonify(fo_prediction.numpy().tolist())


    app.run(port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
    #app.run(port=5000, debug=False, threaded=True)