import re,os
import random
import string
import fileinput
# import tensorflow as tf
import numpy as np
import json
tfl_source_path = './tfl_source_file/'
tfl_output_path = './tfl_output_file/'
tf_source_path = './tf_source_file/'
tf_output_path = './tf_output_file/'
tfl_build_path = './tensorflow-2.9.1/tensorflow/lite/kernels/'
register_file = './tensorflow-2.9.1/tensorflow/lite/kernels/register.cc'
build_file = './tensorflow-2.9.1/tensorflow/lite/kernels/BUILD'

def remove_dir(filepath, del_build=False):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path) and f != '.gitignore':
            os.remove(file_path)
            if del_build:
                os.remove(os.path.join(tfl_build_path, f))

def conv_activation_parser(activation_name):
    if activation_name == 'RELU':
        return 'kTfLiteActRelu'
    elif activation_name == 'RELU6':
        return 'kTfLiteActRelu6'
    elif activation_name == 'TANH':
        return 'kTfLiteActTanh'
    elif activation_name == 'RELU_N1_TO_1':
        return 'kTfLiteActReluN1To1'
    elif activation_name == 'NONE':
        return 'kTfLiteActNone'
    else:
        raise TypeError('Activation type ' + activation_name + ' not supported by conv layers')

def conv_padding_parser(padding_name):
    if padding_name == 'SAME':
        return 'kTfLitePaddingSame'
    elif padding_name == 'VALID':
        return 'kTfLitePaddingValid'
    else:
        raise TypeError('padding type' + padding_name + ' not supported by conv layers')

def conv_filter_type_parser(filter_type):
    if filter_type == 'float32':
        return 'kTfLiteFloat32', 'float'
    elif filter_type == 'int8':
        return 'kTfLiteInt8', 'int8_t'
    elif filter_type == 'uint8':
        return 'kTfLiteUInt8', 'uint8_t'
    elif filter_type == 'int16':
        return 'kTfLiteInt16', 'int16_t'
    elif filter_type == 'int32':
        return 'kTfLiteInt32', 'int32_t'
    else:
        raise TypeError('filter type ' + filter_type + ' not supported by conv layers')

def weights_format_parser(weights_format):
    if weights_format == 'DEFAULT':
        return 'kTfLiteFullyConnectedWeightsFormatDefault'
    else:
        return 'kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8'

def get_attributes_params(op, interpreter):
    kwargs = {}
    if op['builtin_options_type'] == 'Conv2DOptions' or op['builtin_options_type'] == 'DepthwiseConv2DOptions':
        # print(op['inputs'][2])
        # load the stride
        try:
            stride_w = op['builtin_options']['stride_w']
        except:
            kwargs['stride_width='] = 'stride_width=1'
        else:
            kwargs['stride_width='] = 'stride_width=' + str(stride_w)
        try:
            stride_h = op['builtin_options']['stride_h']
        except:
            kwargs['stride_height='] = 'stride_height=1'
        else:
            kwargs['stride_height='] = 'stride_height=' + str(stride_h)
        # load the dilation_
        try:
            dilation_w_factor = op['builtin_options']['dilation_w_factor']
        except:
            kwargs['dilation_width_factor='] = 'dilation_width_factor=1'
        else:
            kwargs['dilation_width_factor='] = 'dilation_width_factor=' + str(dilation_w_factor)
        try:
            dilation_h_factor = op['builtin_options']['dilation_h_factor']
        except:
            kwargs['dilation_height_factor='] = 'dilation_height_factor=1'
        else:
            kwargs['dilation_height_factor='] = 'dilation_height_factor=' + str(dilation_h_factor)
        # load the activation
        try:
            fused_activation_function = op['builtin_options']['fused_activation_function']
        except:
            kwargs['activation='] = 'activation=kTfLiteActNone'
            print("Warning: no activation function found")
        else:
            kwargs['activation='] = 'activation=' + conv_activation_parser(fused_activation_function)
        # load the padding
        try:
            padding = op['builtin_options']['padding']
        except:
            kwargs['paddings='] = 'paddings=kTfLitePaddingSame'
            print("Warning: no paddings found and using default padding SAME")
        else:
            kwargs['paddings='] = 'paddings=' + conv_padding_parser(padding)
        if len(op['inputs']) > 2:
            kwargs['has_conv_bias='] = 'has_conv_bias=true'
        else:
            kwargs['has_conv_bias='] = 'has_conv_bias=false'
            kwargs['const TfLiteType bias_type=;'] = ' '
            kwargs['const int bias_dims_size=;'] = ' '
            kwargs['const int32_t bias_dims_raw=;'] = ' '
            kwargs['float bias_raw=;'] = ' '

        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op['inputs'][1]:
                filter_tensor = interpreter.get_tensor(tensor_details["index"])
                filter_item_num = filter_tensor.size
                filter_input_channel = filter_tensor.shape[3]
                filter_output_channel = filter_tensor.shape[0]
                filter_height = filter_tensor.shape[1]
                filter_width = filter_tensor.shape[2]
                filter_dims_raw = '{' + str(filter_output_channel) + ',' + str(filter_height)  + ',' + str(filter_width)  + ',' + str(filter_input_channel) +'}'
                filter_dims_size = len(filter_tensor.shape)
                tflite_type, type_str = conv_filter_type_parser(filter_tensor.dtype)
                quantization_filter = tensor_details['quantization']
                kwargs['filter_dims_size='] = 'filter_dims_size=' + str(filter_dims_size)
                kwargs['filter_dims_raw='] = 'filter_dims_raw[' + str(filter_dims_size) + ']=' + filter_dims_raw
                kwargs['filter_type='] = 'filter_type=' + tflite_type
                kwargs['filter_raw='] = type_str + ' filter_raw[' + str(filter_item_num) + ']=' + '{' + str(filter_tensor.flatten('C').tolist()).strip('[').strip(']') + '}'
                kwargs['scale_filter='] = 'scale_filter=' + str(quantization_filter[0])
                kwargs['zero_point_filter='] = 'zero_point_filter=' + str(quantization_filter[1])
                kwargs['filter_tensor_data=filter_raw'] = type_str + '* filter_tensor_data=filter_raw'

            elif tensor_details['index'] == op['inputs'][2]:
                bias_tensor = interpreter.get_tensor(tensor_details["index"])
                bias_item_num = bias_tensor.size
                bias_channel = bias_tensor.shape[0]
                bias_dims_raw = '{' + str(bias_channel) + '}'
                bias_dims_size = len(bias_tensor.shape)
                tflite_type, type_str = conv_filter_type_parser(bias_tensor.dtype)
                quantization_bias = tensor_details['quantization']
                kwargs['bias_type='] = 'bias_type=' + tflite_type
                kwargs['bias_dims_size='] = 'bias_dims_size=' + str(bias_dims_size)
                kwargs['bias_dims_raw='] = 'bias_dims_raw[' + str(bias_dims_size) + ']=' + bias_dims_raw
                kwargs['bias_raw='] = type_str + ' bias_raw[' + str(bias_item_num) + ']=' + '{' + str(bias_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['scale_bias='] = 'scale_bias=' + str(quantization_bias[0])
                kwargs['zero_point_bias='] = 'zero_point_bias=' + str(quantization_bias[1])
                kwargs['bias_tensor_data=bias_raw'] = type_str + '* bias_tensor_data=bias_raw'
    elif op['builtin_options_type'] == 'AveragePool2DOptions' or op['builtin_options_type'] == 'MaxPool2DOptions':
        try:
            stride_w = op['builtin_options']['stride_w']
        except:
            kwargs['stride_width='] = 'stride_width=1'
        else:
            kwargs['stride_width='] = 'stride_width=' + str(stride_w)
        try:
            stride_h = op['builtin_options']['stride_h']
        except:
            kwargs['stride_height='] = 'stride_height=1'
        else:
            kwargs['stride_height='] = 'stride_height=' + str(stride_h)
        # load the dilation_
        try:
            filter_height = op['builtin_options']['filter_height']
        except:
            # kwargs['filter_height='] = 'filter_height=2'
            raise ValueError("no filter_height found")
        else:
            kwargs['filter_height='] = 'filter_height=' + str(filter_height)
        try:
            filter_width = op['builtin_options']['filter_width']
        except:
            # kwargs['filter_width='] = 'filter_width=1'
            raise ValueError("no filter_width found")
        else:
            kwargs['filter_width='] = 'filter_width=' + str(filter_width)
        # load the activation
        try:
            fused_activation_function = op['builtin_options']['fused_activation_function']
        except:
            kwargs['activation='] = 'activation=kTfLiteActNone'
            print("Warning: no activation function found")
        else:
            kwargs['activation='] = 'activation=' + conv_activation_parser(fused_activation_function)
        # load the padding
        try:
            padding = op['builtin_options']['padding']
        except:
            kwargs['paddings='] = 'paddings=kTfLitePaddingSame'
            print("Warning: no paddings found and using default padding SAME")
        else:
            kwargs['paddings='] = 'paddings=' + conv_padding_parser(padding)
    elif op['builtin_options_type'] == 'SqueezeOptions':
        try:
            squeeze_dims = op['builtin_options']['squeeze_dims']
        except:
            raise ValueError("no squeeze_dims found")
        else:
            kwargs['squeeze_dim='] = 'squeeze_dim[8]=' + '{' + str(squeeze_dims).strip('[').strip(']') + '}'
            kwargs['num_squeeze_dim='] = 'num_squeeze_dim=' + str(len(squeeze_dims))
    elif op['builtin_options_type'] == 'SoftmaxOptions':
        try:
            beta = op['builtin_options']['beta']
        except:
            kwargs['beta='] = 'beta=1.0'
        else:
            kwargs['beta='] = 'beta=' + str(beta)

    elif op['builtin_options_type'] == 'FullyConnectedOptions':
        try:
            activation = op['builtin_options']['fused_activation_function']
        except:
            kwargs['activation='] = 'activation=kTfLiteActNone'
            print("Warning: no activation function found")
        else:
            kwargs['activation='] = 'activation=' + conv_activation_parser(activation)
        try:
            weights_format = op['builtin_options']['weights_format']
        except:
            kwargs['weights_format='] = 'weights_format=kTfLiteFullyConnectedWeightsFormatDefault'
            print("Warning: no weights_format function found")
        else:
            kwargs['weights_format='] = 'weights_format=' + weights_format_parser(weights_format)
        try:
            asymmetric_quantize_inputs = op['builtin_options']['asymmetric_quantize_inputs']
        except:
            kwargs['asymmetric_quantize_inputs='] = 'asymmetric_quantize_inputs=false'
            print("Warning: no asymmetric_quantize_inputs function found")
        else:
            kwargs['asymmetric_quantize_inputs='] = 'asymmetric_quantize_inputs=' + str.lower(str(asymmetric_quantize_inputs))
        try:
            keep_num_dims = op['builtin_options']['keep_num_dims']
        except:
            kwargs['keep_num_dims='] = 'keep_num_dims=false'
            print("Warning: no keep_num_dims function found")
        else:
            kwargs['keep_num_dims='] = 'keep_num_dims=' + str.lower(str(keep_num_dims))
        if len(op['inputs']) > 2:
            kwargs['has_conv_bias='] = 'has_conv_bias=true'
        else:
            kwargs['has_conv_bias='] = 'has_conv_bias=false'
            kwargs['const TfLiteType bias_type=;'] = ' '
            kwargs['const int bias_dims_size=;'] = ' '
            kwargs['const int32_t bias_dims_raw=;'] = ' '
            kwargs['float bias_raw=;'] = ' '
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op['inputs'][1]:
                filter_tensor = interpreter.get_tensor(tensor_details["index"])
                filter_item_num = filter_tensor.size
                filter_input_channel = filter_tensor.shape[1]
                filter_output_channel = filter_tensor.shape[0]
                filter_dims_raw = '{' + str(filter_output_channel) + ',' + str(filter_input_channel) +'}'
                filter_dims_size = len(filter_tensor.shape)
                tflite_type, type_str = conv_filter_type_parser(filter_tensor.dtype)
                quantization_filter = tensor_details['quantization']
                kwargs['filter_dims_size='] = 'filter_dims_size=' + str(filter_dims_size)
                kwargs['filter_dims_raw='] = 'filter_dims_raw[' + str(filter_dims_size) + ']=' + filter_dims_raw
                kwargs['filter_type='] = 'filter_type=' + tflite_type
                kwargs['filter_raw='] = type_str + ' filter_raw[' + str(filter_item_num) + ']=' + '{' + str(filter_tensor.flatten('C').tolist()).strip('[').strip(']') + '}'
                kwargs['scale_filter='] = 'scale_filter=' + str(quantization_filter[0])
                kwargs['zero_point_filter='] = 'zero_point_filter=' + str(quantization_filter[1])
                kwargs['filter_tensor_data=filter_raw'] = type_str + '* filter_tensor_data=filter_raw'

            elif tensor_details['index'] == op['inputs'][2]:
                bias_tensor = interpreter.get_tensor(tensor_details["index"])
                bias_item_num = bias_tensor.size
                bias_channel = bias_tensor.shape[0]
                bias_dims_raw = '{' + str(bias_channel) + '}'
                bias_dims_size = len(bias_tensor.shape)
                tflite_type, type_str = conv_filter_type_parser(bias_tensor.dtype)
                quantization_bias = tensor_details['quantization']
                kwargs['bias_type='] = 'bias_type=' + tflite_type
                kwargs['bias_dims_size='] = 'bias_dims_size=' + str(bias_dims_size)
                kwargs['bias_dims_raw='] = 'bias_dims_raw[' + str(bias_dims_size) + ']=' + bias_dims_raw
                kwargs['bias_raw='] = type_str + ' bias_raw[' + str(bias_item_num) + ']=' + '{' + str(bias_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['scale_bias='] = 'scale_bias=' + str(quantization_bias[0])
                kwargs['zero_point_bias='] = 'zero_point_bias=' + str(quantization_bias[1])
                kwargs['bias_tensor_data=bias_raw'] = type_str + '* bias_tensor_data=bias_raw'
    elif op['builtin_options_type'] == 'AddOptions':
        try:
            activation = op['builtin_options']['fused_activation_function']
        except:
            kwargs['activation='] = 'activation=kTfLiteActNone'
            print("Warning: no activation function found")
        else:
            kwargs['activation='] = 'activation=' + conv_activation_parser(activation)
        try:
            pot_scale_int16 = op['builtin_options']['pot_scale_int16']
        except:
            kwargs['pot_scale_int16='] = 'pot_scale_int16=true'
            print("Warning: no pot_scale_int16 function found")
        else:
            kwargs['pot_scale_int16='] = 'pot_scale_int16=' + str.lower(str(pot_scale_int16))
    
    elif op['builtin_options_type'] == 'ReducerOptions':
        try:
            keep_dims = op['builtin_options']['keep_dims']
        except:
            kwargs['keep_dims='] = 'keep_dims=true'
            print("Warning: no keep_dims of mean_op found")
        else:
            kwargs['keep_dims='] = 'keep_dims=' + str.lower(str(keep_dims))
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op['inputs'][1]:
                axis_tensor = interpreter.get_tensor(tensor_details["index"])
                axis_item_num = axis_tensor.size
                kwargs['axis_input='] = 'axis_input[' + str(axis_item_num) + ']=' + '{' + str(axis_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['axis_size='] = 'axis_size=' + str(axis_item_num)
    elif op['builtin_options_type'] == 'ReshapeOptions':
        # try:
        #     keep_dims = op['builtin_options']['keep_dims']
        # except:
        #     kwargs['keep_dims='] = 'keep_dims=true'
        #     print("Warning: no pot_scale_int16 function found")
        # else:
        #     kwargs['keep_dims='] = 'keep_dims=' + str.lower(str(keep_dims))
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op['inputs'][1]:
                shape_tensor = interpreter.get_tensor(tensor_details["index"]).squeeze()
                shape_item_num = shape_tensor.size
                kwargs['shape='] = 'shape' + '{' + str(shape_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['shape_size='] = 'shape_size=' + str(shape_item_num)
    elif op['builtin_options_type'] == 'ConcatenationOptions':
        try:
            axis = op['builtin_options']['axis']
        except:
            kwargs['axis='] = 'axis=3'
            print("Warning: no axis of concat_op found")
        else:
            kwargs['axis='] = 'axis=' + str(axis)
    elif op['builtin_options_type'] == 'ResizeBilinearOptions':
        try:
            align_corners = op['builtin_options']['align_corners']
        except:
            kwargs['align_corners='] = 'align_corners=false'
            print("Warning: no pot_scale_int16 function found")
        else:
            kwargs['align_corners='] = 'align_corners=' + str.lower(str(align_corners))
        try:
            half_pixel_centers = op['builtin_options']['half_pixel_centers']
        except:
            kwargs['half_pixel_centers='] = 'half_pixel_centers=true'
            print("Warning: no pot_scale_int16 function found")
        else:
            kwargs['half_pixel_centers='] = 'half_pixel_centers=' + str.lower(str(half_pixel_centers))
        try:
            new_width = op['builtin_options']['new_width']
        except:
            kwargs['new_width='] = 'new_width=0'
        else:
            kwargs['new_width='] = 'new_width=' + str(new_width)
        try:
            new_height = op['builtin_options']['new_height']
        except:
            kwargs['new_height='] = 'new_height=0'
        else:
            kwargs['new_height='] = 'new_height=' + str(new_height)
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op['inputs'][1]:
                size_tensor = interpreter.get_tensor(tensor_details["index"]).squeeze()
                size_item_num = size_tensor.size
                size_dims_raw = '{' + str(size_tensor.shape).strip('(').strip(')') + '}'
                size_dims_size = len(size_tensor.shape)
                kwargs['size_raw='] = 'size_raw[' + str(size_item_num) + ']=' + '{' + str(size_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['size_dims_size='] = 'size_dims_size=' + str(size_dims_size)
                kwargs['size_dims_raw='] = 'size_dims_raw[' + str(size_dims_size) + ']=' + size_dims_raw
    # elif op['builtin_options_type'] == 'ObfOptions':
    #     continue
    return kwargs

def code_generator(op, kwargs, tfl_filelist, tf_filelist, input_details, jsontext, op_sign, inout_list):
    random_str = ''.join(random.choice(string.ascii_lowercase) for _ in range(6))
    jsontext['oplist'].append({'ObfusedId':random_str, 'OpName':op['builtin_options_type'], 'input': op['inputs'], 'output': op['outputs'], 'sign': op_sign})
    with open('./oplist.txt', "a", encoding="utf-8") as f:
        f.write(random_str + '.  ' + op['builtin_options_type'] + '\n')
    input_num = 0
    for i in op['inputs']:
        if i in inout_list:
            input_num += 1
    for i in range(len(tfl_filelist)):
        if op['builtin_options_type'] == os.path.splitext(tfl_filelist[i])[0]:

            with open(os.path.join(tfl_source_path,tfl_filelist[i]), "r", encoding="utf-8") as f1,open(os.path.join(tfl_output_path,("%s.cc" % random_str)), "w", encoding="utf-8") as f2:
                for line in f1:
                    find_key = False
                    for key in kwargs:
                        if key in line:
                            f2.write(re.sub(key,kwargs[key],line))
                            del kwargs[key]
                            find_key = True
                            break
                    if not find_key and 'randopname' in line:
                        f2.write(re.sub('randopname',random_str,line))
                    elif not find_key:
                        f2.write(line)
            os.system("cp %s " % (os.path.join(tfl_output_path,("%s.cc" % random_str))) + " %s" % (os.path.join(tfl_build_path,("%s.cc" % random_str))))
    for i in range(len(tf_filelist)):
        # if op['builtin_options_type'] == os.path.splitext(tf_filelist[i])[0]:

        with open('./tf_source_file/obfuscation_op.cc', "r", encoding="utf-8") as f1,open(os.path.join(tf_output_path,("%s.cc" % random_str)), "w", encoding="utf-8") as f2:
            for line in f1:
                if 'Randopname' in line:
                    f2.write(re.sub('Randopname',str.upper(random_str[0])+random_str[1:],line))
                elif 'output_shape=c->MakeShape({})' in line:
                    f2.write(re.sub('{}',('{%s}' % (str(input_details).strip('[').strip(']'))), line))
                else:
                    if ('// .Input("input') in line:
                        for i in range(input_num):
                            if ('// .Input("input%s: T")' % str(i)) in line:
                                f2.write(re.sub('// ', '', line))
                    else:
                        f2.write(line)
        with open(os.path.join(tf_source_path,'compile.sh'), "r", encoding="utf-8") as f1,open(os.path.join(tf_output_path,("%s.sh" % random_str)), "w", encoding="utf-8") as f2:
            for line in f1:
                if 'randopname' in line:
                    f2.write(re.sub('randopname',tf_output_path+random_str,line))
                else:
                    f2.write(line)
    with fileinput.input(files=register_file, inplace=True) as f:
        for line in f:
            if 'add_cus_here' in line:
                # f.write("  AddCustom(\"%s\"," % (str.upper(random_str[0])+random_str[1:]) + " tflite::ops::custom::Register_%s());" % random_str)
                print("  AddCustom(\"%s\"," % (str.upper(random_str[0])+random_str[1:]) + " tflite::ops::custom::Register_%s());" % random_str)
            elif 'add_rig_here' in line:
                print("TfLiteRegistration* Register_%s();" % random_str)
            print(line, end="")

    with fileinput.input(files=build_file, inplace=True) as f:
        for line in f:
            if 'add_cus_here' in line:
                print("    \"%s.cc\"," % random_str)
            print(line, end="")

    os.system("bash ./tf_output_file/%s.sh" % random_str)

def del_previous_file(register_file, build_file):
    with fileinput.input(files=register_file, inplace=True) as f:
        del_sign = False
        for line in f:
            if 'add_cus_here' in line:
                del_sign = False
            elif 'add_rig_here' in line:
                del_sign = False
            if 'del_here' in line:
                print(line, end="")
                del_sign = True
            if not del_sign:
                print(line, end="")

    with fileinput.input(files=build_file, inplace=True) as f:
        del_sign = False
        for line in f:
            if 'add_cus_here' in line:
                del_sign = False
            if 'del_here' in line:
                print(line, end="")
                del_sign = True
            if not del_sign:
                print(line, end="")
    remove_dir(tfl_output_path, del_build=True)
    remove_dir(tf_output_path)

def correct_json(interpreter, model_json):
    op_details = interpreter._get_ops_details()
    json_op_details = model_json['subgraphs'][0]["operators"]
    for i in range(len(json_op_details)):
        # print(json_op_details[i])
        if 'builtin_options_type' in json_op_details[i]:
            if json_op_details[i]['builtin_options_type'] == 'Pool2DOptions':
                if op_details[i]['op_name'] == 'AVERAGE_POOL_2D':
                    json_op_details[i]['builtin_options_type'] = 'AveragePool2DOptions'
                elif op_details[i]['op_name'] == 'MAX_POOL_2D':
                    json_op_details[i]['builtin_options_type'] = 'MaxPool2DOptions'
        else:
            if interpreter._get_ops_details()[i]['op_name'] == 'RESHAPE':
                json_op_details[i]['builtin_options_type'] = 'ReshapeOptions'
                print("Warning: didn't find the builtin_options_type for this op")
            elif interpreter._get_ops_details()[i]['op_name'] == 'LOGISTIC':
                json_op_details[i]['builtin_options_type'] = 'LogisticOptions'
                print("Warning: didn't find the builtin_options_type for this op")
            elif interpreter._get_ops_details()[i]['op_name'] == 'RELU':
                json_op_details[i]['builtin_options_type'] = 'ReluOptions'
                print("Warning: didn't find the builtin_options_type for this op")
    return json_op_details


def lib_generator(model_json, interpreter, inout_list):
    tfl_filelist = os.listdir(tfl_source_path)
    tf_filelist = os.listdir(tf_source_path)
    op_sign = 0
    file = open('./oplist.txt', 'w').close()
    del_previous_file(register_file, build_file)
    input_details = (interpreter.get_input_details())[0]['shape'].astype(np.int32).tolist()

    jsontext = {'oplist':[]}
    json_op_details = correct_json(interpreter, model_json)
    # print(json_op_details)
    for op in json_op_details:
        # print(op)
        kwargs = get_attributes_params(op, interpreter)
        code_generator(op, kwargs, tfl_filelist, tf_filelist, input_details, jsontext, op_sign, inout_list)
        # print(kwargs)
        op_sign = op_sign + 1
        # if op_sign == 37:
        #     break
    return jsontext

