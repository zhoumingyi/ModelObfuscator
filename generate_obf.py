import os
import sys
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from model_assembler import *

from utils.utils import *
import time

def generate_obf_model(model_path):

    # --------------------------------------------------
    # generate random data
    # --------------------------------------------------
    inputs = generate_random_data(model_path, batch_size=1000)[0]
    # print(inputs[0].shape)
    x = tf.constant(np.expand_dims(inputs[0], 0), dtype=tf.float32)

    # --------------------------------------------------
    # assemble the obfuscated model
    # --------------------------------------------------
    interpreter = tf.lite.Interpreter(
    model_path, experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()

    model_assembler(interpreter)
    from tf_model import create_model
    create_model(x)
