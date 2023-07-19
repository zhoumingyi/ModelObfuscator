#define EIGEN_USE_THREADS

// #include "tensorflow/core/kernels/bias_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/core/kernels/redux_functor.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/tensor_format.h"
using namespace tensorflow;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {
using shape_inference::ShapeHandle;
// using shape_inference::DimensionHandle;
// using shape_inference::InferenceContext;
// --------------------------------------------------------------------------

Status ObfuscatedShape(shape_inference::InferenceContext* c) {
  ShapeHandle output_shape=c->MakeShape({});

  c->set_output(0, output_shape);
  return Status::OK();
}

REGISTER_OP("Randopname")
    .Attr("T: {half, bfloat16, float, double, int32}")
    // .Input("input0: T")
    // .Input("input1: T")
    // .Input("input2: T")
    // .Input("input3: T")
    // .Input("input4: T")
    // .Input("input5: T")
    // .Input("input6: T")
    // .Input("input7: T")
    // .Input("input8: T")
    // .Input("input9: T")
    .Output("output: T")
    .SetShapeFn(ObfuscatedShape);

}  // namespace

template <typename Device, typename T>
class ObfuscatedOp : public OpKernel {
 public:
  explicit ObfuscatedOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    // if (context->GetAttr("data_format", &data_format).ok()) {
    //   OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
    //               errors::InvalidArgument("Invalid data format"));
    // } else {
    //   data_format_ = FORMAT_NHWC;
    // }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
  }
//  private:
//   TensorFormat data_format_;
};

#define REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Randopname").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      ObfuscatedOp<CPUDevice, type>); 

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow
