import os
import sys
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from model_assembler import *
from generate_obf import *

from utils.utils import *
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='fruit', help='name of the model')
opt = parser.parse_args()

model_path = './tflite_model/'
model_name = opt.model_name + '.tflite'

def model_inference(interpreter, inputs):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    for i in range(len(inputs)):
        interpreter.set_tensor(input_details[0]["index"], np.expand_dims(inputs[i], 0))
        interpreter.invoke()
        if i == 0:
            output = interpreter.get_tensor(output_details[0]['index'])
        else:
            output = np.concatenate((output, interpreter.get_tensor(output_details[0]['index'])), axis=0)
    return output

def model_test(model_path):

    # --------------------------------------------------
    # generate random data
    # --------------------------------------------------
    inputs = generate_random_data(model_path, batch_size=100)[0]

    # --------------------------------------------------
    # get the output of the obfuscated model
    # --------------------------------------------------
    interpreter = tf.lite.Interpreter(
    'obf_model.tflite', experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()

    time_start=time.time()
    output_obf = model_inference(interpreter, inputs)
    time_end=time.time()
    print('obf time cost: ',time_end-time_start,'s')
    gc.collect()

    # --------------------------------------------------
    # get the output of the original model
    # --------------------------------------------------
    interpreter = tf.lite.Interpreter(
    model_path, experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()

    time_start=time.time()
    output_ori = model_inference(interpreter, inputs)
    time_end=time.time()
    print('ori time cost: ',time_end-time_start,'s')
    gc.collect()

    print('obfuscation error:', (output_obf.squeeze()-output_ori.squeeze()).mean())

# --------------------------------------------------
# test the obfuscated model
# --------------------------------------------------
model_test(os.path.join(model_path, model_name))
