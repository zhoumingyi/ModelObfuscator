/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/leaky_relu.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/gelu.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/log_softmax.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/prelu.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#if __aarch64__ && __clang__
#include <arm_neon.h>
#endif

namespace tflite {
namespace ops {
namespace custom {
namespace randopname {

enum KernelType {
  kReference,
  kGenericOptimized,
  kFixedPointOptimized,
};

struct OpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
  uint8_t table[256] = {0};
};

struct ReluOpData : public OpData {
  int32_t output_multiplier = 0;
  int output_shift = 0;
};


template <typename T>
void QuantizedReluX(float act_min, float act_max, const TfLiteTensor* input,
                    TfLiteTensor* output, const ReluOpData* data) {
  ReluParams params;
  params.quantized_activation_min =
      std::max(static_cast<int32_t>(std::numeric_limits<T>::min()),
               output->params.zero_point +
                   static_cast<int32>(roundf(act_min / output->params.scale)));
  params.quantized_activation_max =
      act_max == std::numeric_limits<float>::infinity()
          ? static_cast<int32_t>(std::numeric_limits<T>::max())
          : std::min(
                static_cast<int32_t>(std::numeric_limits<T>::max()),
                output->params.zero_point +
                    static_cast<int32>(roundf(act_max / output->params.scale)));
  params.input_offset = input->params.zero_point;
  params.output_offset = output->params.zero_point;
  params.output_multiplier = data->output_multiplier;
  params.output_shift = data->output_shift;
  optimized_ops::ReluX(params, GetTensorShape(input), GetTensorData<T>(input),
                       GetTensorShape(output), GetTensorData<T>(output));
}

void* ReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  return new ReluOpData;
}

void ReluFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<ReluOpData*>(buffer);
}

TfLiteStatus ReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
//   TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8 ||
      input->type == kTfLiteInt16) {
    double real_multiplier = input->params.scale / output->params.scale;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  }

  if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
      optimized_ops::Relu(GetTensorShape(input), GetTensorData<float>(input),
                          GetTensorShape(output), GetTensorData<float>(output));
    } break;
    // TODO(renjieliu): We may revisit the quantization calculation logic,
    // the unbounded upper limit is actually hard to quantize.
    case kTfLiteUInt8: {
      QuantizedReluX<uint8_t>(0.0f, std::numeric_limits<float>::infinity(),
                              input, output, data);
    } break;
    case kTfLiteInt8: {
      QuantizedReluX<int8_t>(0.0f, std::numeric_limits<float>::infinity(),
                             input, output, data);
    } break;
    case kTfLiteInt16: {
      QuantizedReluX<int16_t>(0.0f, std::numeric_limits<float>::infinity(),
                              input, output, data);
    } break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8, int8 and int16 are supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Relu1Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
      optimized_ops::Relu1(GetTensorShape(input), GetTensorData<float>(input),
                           GetTensorShape(output),
                           GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      QuantizedReluX<uint8_t>(-1.0f, 1.0f, input, output, data);
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      QuantizedReluX<int8_t>(-1, 1, input, output, data);
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8, int8 supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration* Register_randopname() {
  static TfLiteRegistration r = {randopname::ReluInit, randopname::ReluFree,
                                 randopname::ReluPrepare,
                                 randopname::ReluEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite