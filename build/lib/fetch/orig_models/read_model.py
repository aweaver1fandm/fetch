import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import glob

for model in glob.glob('*.h5'):
    print(f"*****************************************************")
    print(f"Model definition file: {model}")
    saved_model = load_model(model)
    saved_model.summary()
