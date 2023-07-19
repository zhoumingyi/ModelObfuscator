#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include <stddef.h>
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace tflite {
namespace ops {
namespace custom {
namespace randopname {

enum KernelType {
  kReference,
  kGenericOptimized,
  kFixedPointOptimized,
};

const float beta=;

struct OpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
  uint8_t table[256] = {0};
};

struct SoftmaxOpData {
  struct SoftmaxParams params = {};
  float table[256];
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
  uint8_t uint8_table1[256];
  uint8_t uint8_table2[256];
#endif
  static constexpr int kInt16LUTArraySize = lut_size<int16_t>();
  int16_t exp_lut[kInt16LUTArraySize];  // int16 LUT for exp(x), where x uniform
                                        // distributed between [-10.0 , 0.0]
  int16_t one_over_one_plus_x_lut[kInt16LUTArraySize];  // int16 LUT for 1 /
                                                        // (1 + x), where x
                                                        // uniform distributed
                                                        // between [0.0 , 1.0]
};

void* SoftmaxInit(TfLiteContext* context, const char* buffer, size_t length) {
  return new SoftmaxOpData;
}

void SoftmaxFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<SoftmaxOpData*>(buffer);
}

template <KernelType kernel_type>
TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  SoftmaxOpData* data = reinterpret_cast<SoftmaxOpData*>(node->user_data);

  // TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
    TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 256,
                        (0.001f * 1.f / 256));
  } else if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 32768,
                        (0.001f * 1.f / 32768));
  }

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    if (kernel_type == kReference) {
      const int kScaledDiffIntegerBits = 5;
      int input_left_shift;
      tflite::PreprocessSoftmaxScaling(
          static_cast<double>(beta),
          static_cast<double>(input->params.scale), kScaledDiffIntegerBits,
          &data->params.input_multiplier, &input_left_shift);
      data->params.input_left_shift = input_left_shift;
      data->params.diff_min =
          -1.0 * tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                              input_left_shift);
    } else {
      switch (output->type) {
        case kTfLiteUInt8:
        case kTfLiteInt8:
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
          // Only apply when both input & output are uint8/int8 & build with
          // clang on aarch64.
          // TODO(b/143709993): Port to ARMv7 and other platforms.
          data->params.uint8_table1 = data->uint8_table1;
          data->params.uint8_table2 = data->uint8_table2;
          optimized_ops::PopulateSoftmaxUInt8LookupTable(
              &data->params, input->params.scale, beta);
          break;
#endif
        case kTfLiteInt16:
        default:
          data->params.table = data->table;
          optimized_ops::PopulateSoftmaxLookupTable(
              &data->params, input->params.scale, beta);
      }

      data->params.zero_point = output->params.zero_point;
      data->params.scale = output->params.scale;
    }
  } else if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    data->params.exp_lut = data->exp_lut;
    // exp LUT only used on nagative values
    // we consider exp(-10.0) is insignificant to accumulation
    gen_lut<double, int16_t, int16_t>(
        [](double value) { return std::exp(value); }, -10.0, 0.0, -1.0, 1.0,
        data->params.exp_lut);
    data->params.one_over_one_plus_x_lut = data->one_over_one_plus_x_lut;
    gen_lut<double, int16_t, int16_t>(
        [](double value) { return 1.0 / (1.0 + value); }, 0.0, 1.0, -1.0, 1.0,
        data->params.one_over_one_plus_x_lut);
    data->params.zero_point = output->params.zero_point;
    data->params.scale = output->params.scale;

    double input_scale_beta_rescale =
        input->params.scale * beta /
        (10.0 / 65535.0);  // scale the input_diff such that [-65535, 0]
                           // correspond to [-10.0, 0.0]
    QuantizeMultiplier(input_scale_beta_rescale, &data->params.input_multiplier,
                       &data->params.input_left_shift);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus SoftmaxFloat(TfLiteContext* context, const TfLiteTensor* input,
                          TfLiteTensor* output, TfLiteSoftmaxParams* params,
                          KernelType kernel_type = kGenericOptimized) {
  SoftmaxParams op_params;
  op_params.beta = beta;
  if (kernel_type == kReference) {
    reference_ops::Softmax(op_params, GetTensorShape(input),
                           GetTensorData<float>(input), GetTensorShape(output),
                           GetTensorData<float>(output));
  } else {
    optimized_ops::Softmax(op_params, GetTensorShape(input),
                           GetTensorData<float>(input), GetTensorShape(output),
                           GetTensorData<float>(output),
                           CpuBackendContext::GetFromContext(context));
  }
  return kTfLiteOk;
}

template <typename In, typename Out>
TfLiteStatus SoftmaxQuantized(TfLiteContext* context, const TfLiteTensor* input,
                              TfLiteTensor* output, SoftmaxOpData* data,
                              KernelType kernel_type = kGenericOptimized) {
  if (kernel_type == kReference) {
    reference_ops::Softmax(data->params, GetTensorShape(input),
                           GetTensorData<In>(input), GetTensorShape(output),
                           GetTensorData<Out>(output));
  } else {
    optimized_ops::Softmax(data->params, GetTensorShape(input),
                           GetTensorData<In>(input), GetTensorShape(output),
                           GetTensorData<Out>(output));
  }
  return kTfLiteOk;
}

template <>
TfLiteStatus SoftmaxQuantized<int8_t, int8_t>(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              TfLiteTensor* output,
                                              SoftmaxOpData* data,
                                              KernelType kernel_type) {
  if (kernel_type == kReference) {
    reference_ops::Softmax(data->params, GetTensorShape(input),
                           GetTensorData<int8_t>(input), GetTensorShape(output),
                           GetTensorData<int8_t>(output));
  } else {
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
  optimized_ops::SoftmaxInt8LUT(
      data->params, GetTensorShape(input), GetTensorData<int8_t>(input),
      GetTensorShape(output), GetTensorData<int8_t>(output));
#else
  optimized_ops::Softmax(data->params, GetTensorShape(input),
                         GetTensorData<int8_t>(input), GetTensorShape(output),
                         GetTensorData<int8_t>(output));
#endif
  }
  return kTfLiteOk;
}

template <>
TfLiteStatus SoftmaxQuantized<uint8_t, uint8_t>(TfLiteContext* context,
                                                const TfLiteTensor* input,
                                                TfLiteTensor* output,
                                                SoftmaxOpData* data,
                                                KernelType kernel_type) {
  if (kernel_type == kReference) {
    reference_ops::Softmax(
        data->params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  } else {
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
  optimized_ops::SoftmaxInt8LUT(
      data->params, GetTensorShape(input), GetTensorData<uint8_t>(input),
      GetTensorShape(output), GetTensorData<uint8_t>(output));
#else
  optimized_ops::Softmax(data->params, GetTensorShape(input),
                         GetTensorData<uint8_t>(input), GetTensorShape(output),
                         GetTensorData<uint8_t>(output));
#endif
  }
  return kTfLiteOk;
}

template <>
TfLiteStatus SoftmaxQuantized<int16, int16>(TfLiteContext* context,
                                            const TfLiteTensor* input,
                                            TfLiteTensor* output,
                                            SoftmaxOpData* data,
                                            KernelType kernel_type) {
  if (NumDimensions(input) >= 1 && NumDimensions(input) <= 4) {
    reference_ops::SoftmaxInt16(
        data->params, GetTensorShape(input), GetTensorData<int16_t>(input),
        GetTensorShape(output), GetTensorData<int16_t>(output));
    return kTfLiteOk;
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Only 1D, 2D, 3D and 4D tensors supported for int16 "
                       "input with int16 output, got %dD.",
                       NumDimensions(input));
    return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  SoftmaxOpData* data = reinterpret_cast<SoftmaxOpData*>(node->user_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  switch (input->type) {
    case kTfLiteFloat32: {
      return SoftmaxFloat(context, input, output, params, kernel_type);
    }
    case kTfLiteUInt8: {
      switch (output->type) {
        case kTfLiteUInt8:
          return SoftmaxQuantized<uint8_t, uint8_t>(context, input, output,
                                                    data, kernel_type);
        case kTfLiteInt16:
          return SoftmaxQuantized<uint8_t, int16_t>(context, input, output,
                                                    data, kernel_type);
        default:
          TF_LITE_KERNEL_LOG(context,
                             "Only uint8_t and int16_t outputs are supported "
                             "with uint8_t inputs currently, got %s.",
                             TfLiteTypeGetName(output->type));
          return kTfLiteError;
      }
    }
    case kTfLiteInt8: {
      switch (output->type) {
        case kTfLiteInt8:
          return SoftmaxQuantized<int8_t, int8_t>(context, input, output, data,
                                                  kernel_type);
        case kTfLiteInt16:
          return SoftmaxQuantized<int8_t, int16_t>(context, input, output, data,
                                                   kernel_type);
        default:
          TF_LITE_KERNEL_LOG(context,
                             "Only int8_t and int16_t outputs are supported "
                             "with int8_t inputs currently, got %s.",
                             TfLiteTypeGetName(output->type));
          return kTfLiteError;
      }
    }
    case kTfLiteInt16: {
      return SoftmaxQuantized<int16_t, int16_t>(context, input, output, data,
                                                kernel_type);
    }

    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8_t, Int8_t, Int16_t are supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration* Register_randopname_REF() {
  static TfLiteRegistration r = {
      randopname::SoftmaxInit, randopname::SoftmaxFree,
      randopname::SoftmaxPrepare<randopname::kReference>,
      randopname::SoftmaxEval<randopname::kReference>};
  return &r;
}

TfLiteRegistration* Register_randopname() {
  static TfLiteRegistration r = {
      randopname::SoftmaxInit, randopname::SoftmaxFree,
      randopname::SoftmaxPrepare<randopname::kGenericOptimized>,
      randopname::SoftmaxEval<randopname::kGenericOptimized>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
