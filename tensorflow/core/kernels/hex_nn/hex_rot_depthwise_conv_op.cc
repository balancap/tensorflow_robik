/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "hex_rot_depthwise_conv_op.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// In depthwise convolution, one input is convolved into depth_multipler
// outputs and the outputs don't need to be reduced again like what regular
// convolution does.
//  However, the way to apply filters to inputs is exactly the same as the
// regular convolution. Please refer to the regular convolution kernels for
// more details.

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Extern template instantiated in conv_ops.cc.
extern template class LaunchConv2DOp<CPUDevice, float>;

#if GOOGLE_CUDA

// Extern template instantiated in depthwise_conv_op_gpu.cc.
// extern template struct LaunchHexRotDepthwiseConvOp<GPUDevice, Eigen::half>;
extern template struct LaunchHexRotDepthwiseConvOp<GPUDevice, float>;
extern template struct LaunchHexRotDepthwiseConvOp<GPUDevice, double>;
// Extern template instantiated in conv_ops.cc.
extern template class LaunchConv2DOp<GPUDevice, float>;

#endif

template <typename Device, typename T>
class HexRotDepthwiseConv2dNativeOp : public OpKernel {
 public:
  explicit HexRotDepthwiseConv2dNativeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    stride_ = GetTensorDim(strides_, data_format_, 'H');
    const int64 stride_w = GetTensorDim(strides_, data_format_, 'W');
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');

    OP_REQUIRES(context, stride_ == stride_w,
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

    // For special case when in_depth == 1.
    use_cudnn_ = CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);
    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, depth_multiplier]
    const Tensor& filter = context->input(1);
    // Input rotation tensor, with the following dimensions.
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& rotation = context->input(2);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));
    // in_depth for input and filter must match.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(
        context, in_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(2)));

    // The last dimension for filter is depth multiplier.
    const int32 depth_multiplier = filter.dim_size(3);
    // The output depth is input depth x depth multipler
    const int32 out_depth = in_depth * depth_multiplier;

    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int32 input_rows = static_cast<int32>(input_rows_raw);
    const int32 filter_rows = filter.dim_size(0);

    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int32 input_cols = static_cast<int32>(input_cols_raw);
    const int32 filter_cols = filter.dim_size(1);

    // The first dimension for input is batch.
    const int32 batch = input.dim_size(0);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_,
                                         padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);
    OP_REQUIRES(
        context,
        (!std::is_same<Device, GPUDevice>::value ||
         FastBoundsCheck(out_shape.num_elements(),
                         std::numeric_limits<int32>::max())),
        errors::InvalidArgument("Output elements too large for GPU kernel"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "HexRotDepthwiseConv2dNative: "
            << " Input: [" << batch << ", " << input_rows << ", " << input_cols
            << ", " << in_depth << "]; Filter: [" << filter_rows << ", "
            << filter_cols << ", " << in_depth << ", " << depth_multiplier
            << "]; stride = " << stride_ << ", pad_rows = " << pad_rows
            << ", pad_cols = " << pad_cols << ", output: [" << batch << ", "
            << out_rows << ", " << out_cols << ", " << out_depth << "]"
            << ", GPU device: " << std::is_same<Device, GPUDevice>::value;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }
    HexRotDepthwiseArgs args;
    args.batch = batch;
    args.in_rows = input_rows;
    args.in_cols = input_cols;
    args.in_depth = in_depth;
    args.filter_rows = filter_rows;
    args.filter_cols = filter_cols;
    args.depth_multiplier = depth_multiplier;
    args.stride = stride_;
    args.pad_rows = pad_rows;
    args.pad_cols = pad_cols;
    args.out_rows = out_rows;
    args.out_cols = out_cols;
    args.out_depth = out_depth;

    auto input_ptr = input.template flat<T>().data();
    auto filter_ptr = filter.template flat<T>().data();
    auto rotation_ptr = rotation.template flat<T>().data();
    auto output_ptr = output->template flat<T>().data();

    LaunchHexRotDepthwiseConvOp<Device, T>()(
        context, args, input_ptr, filter_ptr, rotation_ptr, output_ptr, data_format_);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  int64 stride_;  // in height/width dimension.

  // For the case in_depth == 1.
  LaunchConv2DOp<Device, T> launcher_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(HexRotDepthwiseConv2dNativeOp);
};

/** CPU kernels no implemented yet!
 * Weird BUG: CPU kernel called even if GPU kernel exists?
 */
#define REGISTER_CPU_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("HexRotDepthwiseConv2dNative").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      HexRotDepthwiseConv2dNativeOp<CPUDevice, T>);
// TF_CALL_half(REGISTER_CPU_KERNEL);
// TF_CALL_float(REGISTER_CPU_KERNEL);
#if !defined(PLATFORM_WINDOWS) || !defined(_DEBUG)
// TF_CALL_double(REGISTER_CPU_KERNEL);
#endif

#if GOOGLE_CUDA
// REGISTER_KERNEL_BUILDER(
//     Name("HexRotDepthwiseConv2dNative").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
//     HexRotDepthwiseConv2dNativeOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("HexRotDepthwiseConv2dNative").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    HexRotDepthwiseConv2dNativeOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("HexRotDepthwiseConv2dNative").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    HexRotDepthwiseConv2dNativeOp<GPUDevice, double>);
#endif

}  // namespace tensorflow
