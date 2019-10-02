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

def allwave_predict():
    embedded_text_feature_column = hub.text_embedding_column(
        key="name_and_description",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        model_dir='tensorflowmodel',
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.05))
    raw_test = [
    ]

    predict_input_fn = tf.estimator.inputs.numpy_input_fn({"name_and_description": np.array(raw_test).astype(np.str)}, shuffle=False)

    results = estimator.predict(
        predict_input_fn
    )

    for result in results:
        print(result['class_ids'][0])

if __name__ == '__main__':
    allwave_predict()
