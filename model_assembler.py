import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import json
import numpy as np
import tensorflow as tf
import fileinput
from utils.utils import *
# from tf_model import *
# model_path = '/data/mingyi/code/obf_tf/tflite_model/mobilenet_v1_0.75_224_1_default_1.tflite'
# def model_assembler(input, model_json, interpreter):


def model_assembler(interpreter, json_path='./ObfusedModel.json'):
    with open(json_path,'r') as f:
        model_json_f = f.read()
    model_json = json.loads(model_json_f)

    input_id = interpreter.get_input_details()[0]['index']
    output_id = [idx['index'] for idx in interpreter.get_output_details()]

    OpList = model_json['oplist']
    OpIDList =[]
    for op in model_json['oplist']:
        OpIDList.append(op['ObfusedId'])

    model_file = './tf_model.py'
    with fileinput.input(files=model_file, inplace=True) as f:
        del_sign = False
        for line in f:
            if 'add the so file above' in line:
                del_sign = False
            elif 'add the data flow above' in line:
                del_sign = False
            if 'del_here' in line:
                print(line, end="")
                del_sign = True
            if not del_sign:
                print(line, end="")


    def get_inout_string(inout_list):
        input_string = []
        for i in range(len(inout_list)):
            # print(inout_list[i])
            input_string.append('out_' + str(inout_list[i]))
        return str(input_string).strip('[').strip(']').replace('\'', '')


    get_tensor= []
    get_tensor.append(input_id)
    with fileinput.input(files=model_file, inplace=True) as f:
        for line in f:
            if 'add the so file above' in line:
                for i in range(len(OpIDList)):
                    print('op_%s = tf.load_op_library(\'./tf_output_file/%s.so\')' % (i, OpIDList[i]))
            elif 'add the data flow above' in line:
                print('        out_%s = x' % (input_id))
                while(len(OpIDList)):
                    for i in range(len(OpIDList)):
                        if set(OpList[i]['input']) <= set(get_tensor):
                            print(('        %s = op_%s.%s(%s)' % (get_inout_string(OpList[i]['output']), OpList[i]['sign'], OpList[i]['ObfusedId'], get_inout_string(OpList[i]['input']))))
                            for j in range(len(OpList[i]['output'])):
                                get_tensor.append(OpList[i]['output'][j])
                            del OpIDList[i]
                            del OpList[i]
                            break
                print('        return %s' % (get_inout_string(output_id)))
            print(line, end="")

