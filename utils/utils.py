import os
import numpy
import flatbuffers
import torch
# from onnx2pytorch.convert.layer import *
# import onnx
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

# def OnnxWeights2Torch(params):
#     return torch.from_numpy(numpy_helper.to_array(params))

# def OnnxWeights2Numpy(params):
#     return numpy_helper.to_array(params)

# def Torch2OnnxWeights(params):
#     return numpy_helper.from_array(params.numpy())
 
def OutputsOffset(subgraph, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
    if o != 0:
        a = subgraph._tab.Vector(o)
        return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
    return 0
 
def buffer_change_output_tensor_to(model_buffer, new_tensor_i):
    
    root = schema_fb.Model.GetRootAsModel(model_buffer, 0)
    output_tensor_index_offset = OutputsOffset(root.Subgraphs(0), 0)
    
    # Flatbuffer scalars are stored in little-endian.
    new_tensor_i_bytes = bytes([
    new_tensor_i & 0x000000FF, \
    (new_tensor_i & 0x0000FF00) >> 8, \
    (new_tensor_i & 0x00FF0000) >> 16, \
    (new_tensor_i & 0xFF000000) >> 24 \
    ])
    # Replace the 4 bytes corresponding to the first output tensor index
    return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]

def generate_random_data(model_path, batch_size=1):
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    model = tf.lite.Interpreter(model_content=model_buffer)
    input_details = model.get_input_details()
    input_tensors = [] 
    for i in range(len(input_details)):
        # print(input_details[i]['dtype'])
        shape_input = input_details[i]['shape'].astype(numpy.int32).tolist()
        shape_input[0] = batch_size
        print(shape_input)
        if input_details[i]['dtype'] == numpy.uint8:
            inputs = torch.randint(low=0, high=255, size=tuple(shape_input)).to(torch.uint8).numpy()
            print("Data type of this model is uint8")
        elif input_details[i]['dtype'] == numpy.float32:
            inputs = torch.randn(tuple(shape_input)).numpy()
            print("Data type of this model is float32")
        input_tensors.append(inputs)
    return input_tensors

def test_tflite_results(model_path, inputs, inter_out=None):
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    interpreter = tf.lite.Interpreter(model_content=model_buffer)
    for details in interpreter.get_tensor_details():
        # print(details)
        if details['name'] == inter_out:
            idx = details['index']
    if inter_out is not None:
        model_buffer = buffer_change_output_tensor_to(model_buffer, idx)
        # with open('./tflite_model/modified.tflite', 'wb') as f:
        #     f.write(model_buffer)

    model = tf.lite.Interpreter(model_content=model_buffer, experimental_preserve_all_tensors=True)
    model.allocate_tensors()
    input_details = model.get_input_details()
    print(input_details)
    for i in range(len(input_details)):
        # shape_input = tuple(input_details[i]['shape'].astype(np.int32).tolist())
        output_details = model.get_output_details()
        model.resize_tensor_input(input_details[i]['index'],input_details[i]['shape'])
        model.set_tensor(input_details[i]['index'], inputs[i])
        print(inputs[i].shape)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data, output_details

def TfliteToOnnx(path):
    filelist = os.listdir(path)
    for i in range(len(filelist)):
        if os.path.splitext(filelist[i])[1] != ('.tflite') and os.path.splitext(filelist[i])[1] != ('.lite'):
            os.system("rm " + path + filelist[i])
        else:
            os.system("python -m tf2onnx.convert --opset 13 --tflite " +  path + filelist[i] +
             " --output " + "out_model/" + os.path.splitext(filelist[i])[0] + ".onnx")