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

def movie_trainer_script():
    #import data from csv
    data = pd.read_csv("data/imdb/imdb_0_1.csv", sep=",")
    #shuffle the data and split into train and test sets
    shuffledData = data.sample(frac=1).reset_index(drop=True)
    msk = np.random.rand(len(shuffledData)) < 0.8
    trainData = shuffledData[msk]
    testData = shuffledData[~msk]

    # Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        trainData, trainData["sentiment"], num_epochs=None, shuffle=True)

    # # Prediction on the whole training set.
    predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        trainData, trainData["sentiment"], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        testData, testData["sentiment"], shuffle=False)

    embedded_text_feature_column = hub.text_embedding_column(
        key="review",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        model_dir='tensorflowmodel',
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.05))

    estimator.train(input_fn=train_input_fn, steps=1000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Test set accuracy: {accuracy}".format(**test_eval_result))

    return 'Model Trained'

if __name__ == '__main__':
    movie_trainer_script()
