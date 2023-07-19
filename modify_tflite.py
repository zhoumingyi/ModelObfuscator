import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import json
import fileinput


with open('ObfusedModel.json','r') as f:
    obfuscation_json_f = f.read()
obfuscation_json = json.loads(obfuscation_json_f)

# with open('obf_model.json', "r", encoding="utf-8") as f1:
quant_min = 0.0
quant_max = 0.0
quant_scale = 0.0
quant_zero_point = 0
dict_lenth = 0

previous_line = ""
next_line = ""
with fileinput.input(files='obf_model.json', inplace=True) as f:
    for line in f:
        if 'name:' in line and '          buffer:' in previous_line:
            for inputs in obfuscation_json['inputs']:
                if inputs['name'] in line:
                    print("          type: \"" + inputs['type']+ "\",")
            for outputs in obfuscation_json['outputs']:
                if outputs['name'] in line:
                    print("          type: \"" + outputs['type']+"\",")
            for oplist in obfuscation_json['oplist']:
                if (str.upper(oplist['ObfusedId'][0])+oplist['ObfusedId'][1:]) in line:
                    print("          type: \"" + oplist['type'] + "\",")
            print(previous_line, end="")
        if 'buffer:' not in line or '          ],' not in previous_line:                 
            print(line, end="")

        if 'quantization:' in line:
            for inputs in obfuscation_json['inputs']:
                if inputs['name'] in previous_line:
                    if len(inputs['quantization']) == 2:
                        # print("          \"quantization\": {")
                        print("            min: [")
                        print("              ", inputs['quantization']['min'][0])
                        print("            ],")
                        print("            max: [")
                        print("              ", inputs['quantization']['max'][0])
                        print("            ]")
                    elif len(inputs['quantization']) == 4:
                        # print("          \"quantization\": {")
                        print("            min: [")
                        print("              ", inputs['quantization']['min'][0])
                        print("            ],")
                        print("            max: [")
                        print("              ", inputs['quantization']['max'][0])
                        print("            ],")
                        print("            scale: [")
                        print("              ", inputs['quantization']['scale'][0])
                        print("            ],")
                        print("            zero_point: [")
                        print("              ", inputs['quantization']['zero_point'][0])
                        print("            ]")
            for outputs in obfuscation_json['outputs']:
                if outputs['name'] in previous_line:
                    if len(outputs['quantization']) == 2:
                        # print("          \"quantization\": {")
                        print("            min: [")
                        print("              ", outputs['quantization']['min'][0])
                        print("            ],")
                        print("            max: [")
                        print("              ", outputs['quantization']['max'][0])
                        print("            ]")
                    elif len(outputs['quantization']) == 4:
                        # print("          \"quantization\": {")
                        print("            min: [")
                        print("              ", outputs['quantization']['min'][0])
                        print("            ],")
                        print("            max: [")
                        print("              ", outputs['quantization']['max'][0])
                        print("            ],")
                        print("            scale: [")
                        print("              ", outputs['quantization']['scale'][0])
                        print("            ],")
                        print("            zero_point: [")
                        print("              ", outputs['quantization']['zero_point'][0])
                        print("            ]")
            for oplist in obfuscation_json['oplist']:
                if (str.upper(oplist['ObfusedId'][0])+oplist['ObfusedId'][1:]) in previous_line:
                    if len(oplist['quantization']) == 2:
                        # print("          \"quantization\": {")
                        print("            min: [")
                        print("              ", oplist['quantization']['min'][0])
                        print("            ],")
                        print("            max: [")
                        print("              ", oplist['quantization']['max'][0])
                        print("            ]")
                    elif len(oplist['quantization']) == 4:
                        # print("          \"quantization\": {")
                        print("            min: [")
                        print("              ", oplist['quantization']['min'][0])
                        print("            ],")
                        print("            max: [")
                        print("              ", oplist['quantization']['max'][0])
                        print("            ],")
                        print("            scale: [")
                        print("              ", oplist['quantization']['scale'][0])
                        print("            ],")
                        print("            zero_point: [")
                        print("              ", oplist['quantization']['zero_point'][0])
                        print("            ]")
        previous_line = line
        # i+=1



