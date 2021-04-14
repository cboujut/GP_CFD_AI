import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def NN_coeff_error(AoA, ca1, ca2, ce1, ce2):
    trained_model = tf.keras.models.load_model('dnn_model')
    inp = np.array([[AoA, ca1, ca2, ce1, ce2]])
    df = pd.DataFrame(inp)
    return trained_model.predict(df).flatten()