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

template <typename T>
void PopulateLookupTable(struct OpData* data, const TfLiteTensor* input,
                         TfLiteTensor* output,
                         const std::function<float(float)>& transform) {
  static_assert(sizeof(T) == 1, "Lookup table valid only for 8bit");
  const float inverse_scale = 1 / output->params.scale;
  int32_t maxval = std::numeric_limits<T>::max();
  int32_t minval = std::numeric_limits<T>::min();
  for (int32_t val = minval; val <= maxval; ++val) {
    const float dequantized =
        input->params.scale * (val - input->params.zero_point);
    const float transformed = transform(dequantized);
    const float rescaled = std::round(transformed * inverse_scale);
    const int32_t quantized =
        static_cast<int32_t>(rescaled + output->params.zero_point);
    data->table[static_cast<uint8_t>(static_cast<T>(val))] =
        static_cast<uint8_t>(
            static_cast<T>(std::max(std::min(maxval, quantized), minval)));
  }
}

// TODO(b/143696793): move this to optimized_ops.
void EvalUsingLookupTable(struct OpData* data, const TfLiteTensor* input,
                          TfLiteTensor* output) {
  const int size =
      MatchingFlatSize(GetTensorShape(input), GetTensorShape(output));
  uint8_t* output_data = GetTensorData<uint8_t>(output);
  const uint8_t* input_data = GetTensorData<uint8_t>(input);
  int i = 0;
#if __aarch64__ && __clang__
  // This code uses ARM64-only instructions.
  // TODO(b/143709993): Port to ARMv7

  // Load the tables into registers. (4*4 128-bit registers)
  uint8x16x4_t table[4];
  table[0] = vld1q_u8_x4(data->table + 16 * 4 * 0);
  table[1] = vld1q_u8_x4(data->table + 16 * 4 * 1);
  table[2] = vld1q_u8_x4(data->table + 16 * 4 * 2);
  table[3] = vld1q_u8_x4(data->table + 16 * 4 * 3);

  // Vectorized loop; process uint8x16_t (16 elements) at a time.
  constexpr int vectorized_16_loop_step = 16;
  const int vectorized_16_loop_end =
      size / vectorized_16_loop_step * vectorized_16_loop_step;
  for (; i < vectorized_16_loop_end; i += vectorized_16_loop_step) {
    uint8x16_t input = vld1q_u8(input_data + i);
    uint8x16_t output = optimized_ops::aarch64_lookup_vector(table, input);
    vst1q_u8(output_data + i, output);
  }
  // Postamble and non-ARM64 code: simple for loop.
#endif
  for (; i < size; ++i) {
    output_data[i] = data->table[input_data[i]];
  }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

template <KernelType kernel_type>
TfLiteStatus SigmoidPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

//   TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (kernel_type == kFixedPointOptimized) {
    if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
      if (input->type == kTfLiteUInt8) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                          std::numeric_limits<uint8_t>::min());
      }
      if (input->type == kTfLiteInt8) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                          std::numeric_limits<int8_t>::min());
      }
      TF_LITE_ENSURE(context, output->params.scale == 1. / 256);

      static constexpr int kInputIntegerBits = 4;

      const double input_real_multiplier =
          input->params.scale *
          static_cast<double>(1 << (15 - kInputIntegerBits));

      const double q =
          std::frexp(input_real_multiplier, &data->input_left_shift);
      auto q_fixed = static_cast<int32_t>(TfLiteRound(q * (1LL << 15)));
      data->input_multiplier = static_cast<int16_t>(q_fixed);

      int16_t input_range_radius =
          CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 15);
      data->input_range_radius = input_range_radius;
    }
  }

  if (kernel_type == kGenericOptimized || kernel_type == kReference) {
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE(context, output->params.scale == 1. / 256);
      PopulateLookupTable<uint8_t>(data, input, output, [](float value) {
        return 1.0f / (1.0f + std::exp(-value));
      });
    } else if (input->type == kTfLiteInt8) {
      TF_LITE_ENSURE(context, output->params.scale == 1. / 256);
      PopulateLookupTable<int8_t>(data, input, output, [](float value) {
        return 1.0f / (1.0f + std::exp(-value));
      });
    } else if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE(context, output->params.scale == 1. / 32768);
      TF_LITE_ENSURE(context, output->params.zero_point == 0);
    }
  }

  if (input->type == kTfLiteInt16) {
    static constexpr int kInputIntegerBits = 3;
    static constexpr int kOutputFractionalBits = 15;

    // See comments in TanhPrepare about requiring zero_point==0
    // and a power-of-two ("POT") scale.

    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    int input_scale_log2_rounded;
    bool param_scale_pot =
        CheckedLog2(input->params.scale, &input_scale_log2_rounded);

    data->input_left_shift =
        (15 - kInputIntegerBits) + input_scale_log2_rounded;
    param_scale_pot &= (data->input_left_shift == 0);

    if (!param_scale_pot) {
      // Calculate multiplier to change input scale to 1/(3*4096)
      // as required by the table lookup.
      // In this scaling +/-2^17 represents +/-10.7
      double multiplier = input->params.scale * 4096.0 * 3.0;

      data->input_left_shift = 0;

      while (multiplier <= 32767.0 / 2.0 && data->input_left_shift <= 30) {
        data->input_left_shift++;
        multiplier = multiplier * 2.0;
      }

      data->input_multiplier = static_cast<int32_t>(multiplier);
    }

    int output_scale_log2_rounded;
    TF_LITE_ENSURE(
        context, CheckedLog2(output->params.scale, &output_scale_log2_rounded));
    TF_LITE_ENSURE_EQ(context, output_scale_log2_rounded,
                      -kOutputFractionalBits);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

// Sigmoid is also know as "Logistic".
template <KernelType kernel_type>
TfLiteStatus SigmoidEval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  switch (input->type) {
    case kTfLiteFloat32: {
      if (kernel_type == kReference) {
        reference_ops::Logistic(
            GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        optimized_ops::Logistic(
            GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      break;
    }
    case kTfLiteInt16: {
      LogisticParams params;
      if (kernel_type == kReference || (data->input_multiplier > 0)) {
        const int size =
            MatchingFlatSize(GetTensorShape(input), GetTensorShape(output));

        reference_integer_ops::Logistic(
            data->input_multiplier, data->input_left_shift, size,
            GetTensorData<int16_t>(input), GetTensorData<int16_t>(output));
      } else {
        optimized_ops::Logistic(
            params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<int16_t>(output));
      }
      break;
    }
    case kTfLiteUInt8: {
      if (kernel_type == kFixedPointOptimized) {
        LogisticParams params;
        params.input_zero_point = input->params.zero_point;
        params.input_range_radius = data->input_range_radius;
        params.input_multiplier = data->input_multiplier;
        params.input_left_shift = data->input_left_shift;
        optimized_ops::Logistic16bitPrecision(
            params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      } else {
        EvalUsingLookupTable(data, input, output);
      }
      break;
    }
    case kTfLiteInt8: {
      if (kernel_type == kFixedPointOptimized) {
        LogisticParams params;
        params.input_zero_point = input->params.zero_point;
        params.input_range_radius = data->input_range_radius;
        params.input_multiplier = data->input_multiplier;
        params.input_left_shift = data->input_left_shift;
        optimized_ops::Logistic16bitPrecision(
            params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<int8_t>(output));
      } else {
        EvalUsingLookupTable(data, input, output);
      }
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8, int16 and int8 are supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace 

TfLiteRegistration* Register_randopname_REF() {
  static TfLiteRegistration r = {
      randopname::Init, randopname::Free,
      randopname::SigmoidPrepare<randopname::kReference>,
      randopname::SigmoidEval<randopname::kReference>};
  return &r;
}

TfLiteRegistration* Register_randopname_GENERIC_OPT() {
  static TfLiteRegistration r = {
      randopname::Init, randopname::Free,
      randopname::SigmoidPrepare<randopname::kGenericOptimized>,
      randopname::SigmoidEval<randopname::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_randopname() {
  // TODO(b/134622898): Switch over from the LUT optimized method to the fixed
  // point optimized method when typical Android hardware performs better on
  // the latter one.
  return Register_randopname_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

