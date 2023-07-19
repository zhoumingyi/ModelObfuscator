import tensorflow as tf
import numpy as np
import torch
import time

# input_data_tf = np.array(np.random.random_sample([32, 1, 224, 224, 3]), dtype=np.float32)
# input_data = torch.rand(32, 1, 224, 224, 3)
class tf_inference():
    def __init__(self, model):
        # self.input = input
        self.model = model
        # interpreter = self.model
        # input_details = self.model.get_input_details()
        # output_details = self.model.get_output_details()

    def query(self, inputs, expand=99, dtype='uint8'):
        inputs = inputs.cpu().numpy()
        # print(inputs.shape)
        results = torch.from_numpy(self.inference(inputs, expand=expand, dtype=dtype))
        # print(results.size())
        return results

    def inference(self, inputs, expand, dtype):
        dim = inputs.shape[0]
        # results = np.array(np.random.random_sample([dim, class_num]), dtype=np.uint8)
        # results = []
        # for i in range(dim):
        #     results.append(self.andoid_model(inputs[i], expand=expand, dtype=dtype))
        results = self.andoid_model(inputs, expand=expand, dtype=dtype)
        # return list(results)
        return results

    def andoid_model(self, input, expand, dtype):

        # Load TFLite model and allocate tensors
        # interpreter = tf.lite.Interpreter(model_path="/datasata/mingyi/ondevice/DL_models/fine_tuned/mobilenet.letgo.v1_1.0_224_quant.v7.tflite")
        interpreter = self.model

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        # print(str(input_details))
        output_details = interpreter.get_output_details()
        # print(str(output_details))
        # Test model on input data
        if expand == 99:
            # input_data = input.squeeze()
            input_data = input
        else:
            input_data = np.expand_dims(input.squeeze(), axis=expand)
        if dtype == 'uint8':
            input_data = input_data.astype(np.uint8)
        elif dtype == 'int32':
            input_data = input_data.astype(np.int32)
        elif dtype == 'float32':
            input_data = input_data.astype(np.float32)
        # print(input_data.shape)
        # print(input_data.max(), input_data.min())
        # interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.resize_tensor_input(input_details[0]['index'],[input_data.shape[0], 224, 224, 3])
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        # interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = np.squeeze(output_data)
        result = (result / 255.0)
        if dtype == 'uint8':
            result = result.astype(np.float32)
        # print(result.sum())
        return result

# # results = tf_inference(input_data)
# # t1 = time.time()
# model = tf.lite.Interpreter(model_path="./models/fine_tuned/fruit_graph.tflite")
# input_details = model.get_input_details()
# # print(str(input_details))
# # output_details = model.get_output_details()
# # print(str(output_details))
# input_data = torch.rand(100, 224, 224, 3)
# tf_model = tf_inference(model)
# outputs = tf_model.query(input_data, expand=99, dtype='float32')
# print(outputs)
# print('Time taken: %f' % (time.time() - t1))
# print(outputs[3])

# outputs = tf_model.query(input_data[3], expand=0, dtype='float32')
# print(outputs.sum())
