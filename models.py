import math
import time
import redis
from flask import Flask
from flask import request, current_app, abort
from functools import wraps
from flask import current_app
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from sklearn import metrics
import tensorflow as tf
import tensorflow_hub as hub
import json
import pickle
import urllib
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
from tensorflow.python.data import Dataset

class Model(object):

    def tensorflowPredict(self, description):
        
        descriptions = []
        descriptions.append(description)
        embedded_text_feature_column = hub.text_embedding_column(
            key="name_and_description",
            module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

        estimator = tf.estimator.DNNClassifier(
            hidden_units=[500, 100],
            model_dir='tensorflowmodel',
            feature_columns=[embedded_text_feature_column],
            n_classes=2,
            optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.05))

        predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn({"name_and_description": np.array(descriptions).astype(np.str)}, shuffle=False)

        results = estimator.predict(
            predict_input_fn
        )
        # for result in results:
            # print(result['class_ids'][0])
            # top_2 = result['probabilities'].argsort()[-2:][::-1]
            # for genre in top_2:
            #     print('result ' + ': ' + str(round(result['probabilities'][genre] * 100, 2)) + '%')
            # print('')
            # print(result)
        # predictions = np.array([item['class_ids'][0] for item in results])
        predictions = np.array([item ['class_ids'][0]for item in results])

        return "Prediction: {}".format(str(predictions))

model = Model()
