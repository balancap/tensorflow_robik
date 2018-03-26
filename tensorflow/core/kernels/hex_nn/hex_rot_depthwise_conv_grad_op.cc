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

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"

// #include "tensorflow/core/kernels/depthwise_conv_op.h"
#include "hex_rot_depthwise_conv_op.h"

#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// Gradient operations for depthwise convolution.

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Common code between the two backward pass kernels: verifies that the
// dimensions all match and extract the padded rows and columns.
#define EXTRACT_AND_VERIFY_DIMENSIONS(label)                                   \
  const Tensor& out_backprop = context->input(2);                              \
  OP_REQUIRES(                                                                 \
      context, input_shape.dims() == 4,                                        \
      errors::InvalidArgument(label, ": input must be 4-dimensional"));        \
  OP_REQUIRES(                                                                 \
      context, filter_shape.dims() == 4,                                       \
      errors::InvalidArgument(label, ": filter must be 4-dimensional"));       \
  OP_REQUIRES(                                                                 \
      context, out_backprop.dims() == 4,                                       \
      errors::InvalidArgument(label, ": out_backprop must be 4-dimensional")); \
  const int64 batch = input_shape.dim_size(0);                                 \
  OP_REQUIRES(                                                                 \
      context, batch == out_backprop.dim_size(0),                              \
      errors::InvalidArgument(                                                 \
          label, ": input and out_backprop must have the same batch size"));   \
  const int64 input_rows_raw = GetTensorDim(input_shape, data_format_, 'H');   \
  OP_REQUIRES(                                                                 \
      context,                                                                 \
      FastBoundsCheck(input_rows_raw, std::numeric_limits<int32>::max()),      \
      errors::InvalidArgument("Input rows too large"));                        \
  const int32 input_rows = static_cast<int32>(input_rows_raw);                 \
  const int64 input_cols_raw = GetTensorDim(input_shape, data_format_, 'W');   \
  OP_REQUIRES(                                                                 \
      context,                                                                 \
      FastBoundsCheck(input_cols_raw, std::numeric_limits<int32>::max()),      \
      errors::InvalidArgument("Input cols too large"));                        \
  const int32 input_cols = static_cast<int32>(input_cols_raw);                 \
  const int64 filter_rows = filter_shape.dim_size(0);                          \
  const int64 filter_cols = filter_shape.dim_size(1);                          \
  const int64 output_rows_raw =                                                \
      GetTensorDim(out_backprop.shape(), data_format_, 'H');                   \
  OP_REQUIRES(                                                                 \
      context,                                                                 \
      FastBoundsCheck(output_rows_raw, std::numeric_limits<int32>::max()),     \
      errors::InvalidArgument("Output rows too large"));                       \
  const int32 output_rows = static_cast<int32>(output_rows_raw);               \
  const int64 output_cols_raw =                                                \
      GetTensorDim(out_backprop.shape(), data_format_, 'W');                   \
  OP_REQUIRES(                                                                 \
      context,                                                                 \
      FastBoundsCheck(output_cols_raw, std::numeric_limits<int32>::max()),     \
      errors::InvalidArgument("Output cols too large"));                       \
  const int32 output_cols = static_cast<int32>(output_cols_raw);               \
  const int64 in_depth = GetTensorDim(input_shape, data_format_, 'C');         \
  OP_REQUIRES(context, in_depth == filter_shape.dim_size(2),                   \
              errors::InvalidArgument(                                         \
                  label, ": input and filter must have the same in_depth"));   \
  const int64 depth_multiplier = filter_shape.dim_size(3);                     \
  const int64 out_depth_raw =                                                  \
      GetTensorDim(out_backprop.shape(), data_format_, 'C');                   \
  OP_REQUIRES(                                                                 \
      context,                                                                 \
      FastBoundsCheck(out_depth_raw, std::numeric_limits<int32>::max()),       \
      errors::InvalidArgument("Output depth too large"));                      \
  const int32 out_depth = static_cast<int32>(out_depth_raw);                   \
  OP_REQUIRES(                                                                 \
      context, (depth_multiplier * in_depth) == out_depth,                     \
      errors::InvalidArgument(                                                 \
          label, ": depth_multiplier * in_depth not equal to out_depth"));     \
  const auto stride = stride_;                                                 \
  int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;                \
  OP_REQUIRES_OK(context,                                                      \
                 GetWindowedOutputSize(input_rows, filter_rows, stride,        \
                                       padding_, &out_rows, &pad_rows));       \
  OP_REQUIRES_OK(context,                                                      \
                 GetWindowedOutputSize(input_cols, filter_cols, stride,        \
                                       padding_, &out_cols, &pad_cols));       \
  OP_REQUIRES(                                                                 \
      context, output_rows == out_rows,                                        \
      errors::InvalidArgument(                                                 \
          label, ": Number of rows of out_backprop doesn't match computed: ",  \
          "actual = ", output_rows, ", computed = ", out_rows));               \
  OP_REQUIRES(                                                                 \
      context, output_cols == out_cols,                                        \
      errors::InvalidArgument(                                                 \
          label, ": Number of cols of out_backprop doesn't match computed: ",  \
          "actual = ", output_cols, ", computed = ", out_cols));               \
  HexRotDepthwiseArgs args;                                                          \
  args.batch = batch;                                                          \
  args.in_rows = input_rows;                                                   \
  args.in_cols = input_cols;                                                   \
  args.in_depth = in_depth;                                                    \
  args.filter_rows = filter_rows;                                              \
  args.filter_cols = filter_cols;                                              \
  args.depth_multiplier = depth_multiplier;                                    \
  args.stride = stride;                                                        \
  args.pad_rows = pad_rows;                                                    \
  args.pad_cols = pad_cols;                                                    \
  args.out_rows = out_rows;                                                    \
  args.out_cols = out_cols;                                                    \
  args.out_depth = out_depth;                                                  \
  VLOG(2) << "HexRotDepthwiseConv2d: " << label << " Input: [" << batch << ", "      \
          << input_rows << ", " << input_cols << ", " << in_depth              \
          << "]; Filter: [" << filter_rows << ", " << filter_cols << ", "      \
          << in_depth << ", " << depth_multiplier << "]; stride = " << stride  \
          << ", pad_rows = " << pad_rows << ", pad_cols = " << pad_cols        \
          << ", output: [" << batch << ", " << out_rows << ", " << out_cols    \
          << ", " << out_depth << "]";


// -------------------------------------------------------------------------- //
// Backward Input conv2d kernel.
// -------------------------------------------------------------------------- //
#if GOOGLE_CUDA

// extern template struct LaunchHexRotDepthwiseConvBackpropInputOp<GPUDevice, Eigen::half>;
extern template struct LaunchHexRotDepthwiseConvBackpropInputOp<GPUDevice, float>;
extern template struct LaunchHexRotDepthwiseConvBackpropInputOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA

// Kernel to compute the input backprop for depthwise convolution.
template <typename Device, class T>
class HexRotDepthwiseConv2dNativeBackpropInputOp : public OpKernel {
 public:
  explicit HexRotDepthwiseConv2dNativeBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));

    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

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
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& rotation = context->input(2);

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2dBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    const int32* in_sizes_data = input_sizes.template flat<int32>().data();
    for (int i = 0; i < input_sizes.NumElements(); ++i) {
      OP_REQUIRES(context, in_sizes_data[i] >= 0,
                  errors::InvalidArgument("Dimension ", i,
                                          " of input_sizes must be >= 0"));
      input_shape.AddDim(in_sizes_data[i]);
    }
    const TensorShape& filter_shape = filter.shape();
    EXTRACT_AND_VERIFY_DIMENSIONS("HexRotDepthwiseConv2dBackpropInput");
    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input_shape, &in_backprop));
    auto out_backprop_ptr = out_backprop.template flat<T>().data();
    auto filter_ptr = filter.template flat<T>().data();
    auto rotation_ptr = rotation.template flat<T>().data();
    auto in_backprop_ptr = in_backprop->template flat<T>().data();

    // If there is nothing to compute, return.
    if (input_shape.num_elements() == 0) {
      return;
    }
    LaunchHexRotDepthwiseConvBackpropInputOp<Device, T>()(
        context, args, out_backprop_ptr, filter_ptr, rotation_ptr, in_backprop_ptr,
        data_format_);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
  int64 stride_;

  TF_DISALLOW_COPY_AND_ASSIGN(HexRotDepthwiseConv2dNativeBackpropInputOp);
};

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("HexRotDepthwiseConv2dNativeBackpropInput")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T")
        .HostMemory("input_sizes"),
    HexRotDepthwiseConv2dNativeBackpropInputOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("HexRotDepthwiseConv2dNativeBackpropInput")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T")
        .HostMemory("input_sizes"),
    HexRotDepthwiseConv2dNativeBackpropInputOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

// -------------------------------------------------------------------------- //
// Backward Input conv2d kernel.
// -------------------------------------------------------------------------- //
template <typename Device, typename T>
struct LaunchHexRotDepthwiseConvBackpropFilterOp;

#if GOOGLE_CUDA

// extern template struct LaunchHexRotDepthwiseConvBackpropFilterOp<GPUDevice, Eigen::half>;
extern template struct LaunchHexRotDepthwiseConvBackpropFilterOp<GPUDevice, float>;
extern template struct LaunchHexRotDepthwiseConvBackpropFilterOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA

// Kernel to compute the filter backprop for depthwise convolution.
template <typename Device, class T>
class HexRotDepthwiseConv2dNativeBackpropFilterOp : public OpKernel {
 public:
  explicit HexRotDepthwiseConv2dNativeBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));

    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

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
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    const Tensor& rotation = context->input(2);

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2dBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    TensorShape filter_shape;
    const int32* filter_sizes_data = filter_sizes.template flat<int32>().data();
    for (int i = 0; i < filter_sizes.NumElements(); ++i) {
      OP_REQUIRES(context, filter_sizes_data[i] >= 0,
                  errors::InvalidArgument("Dimension ", i,
                                          " of filter_sizes must be >= 0"));
      filter_shape.AddDim(filter_sizes_data[i]);
    }
    const TensorShape& input_shape = input.shape();

    EXTRACT_AND_VERIFY_DIMENSIONS("HexRotDepthwiseConv2dBackpropFilter");
    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {1}, 0, filter_shape, &filter_backprop));

    auto out_backprop_ptr = out_backprop.template flat<T>().data();
    auto input_ptr = input.template flat<T>().data();
    auto rotation_ptr = rotation.template flat<T>().data();
    auto filter_backprop_ptr = filter_backprop->template flat<T>().data();

    // If there is nothing to compute, return.
    if (filter_shape.num_elements() == 0) {
      return;
    }
    LaunchHexRotDepthwiseConvBackpropFilterOp<Device, T>()(
        context, args, out_backprop_ptr, input_ptr, rotation_ptr, filter_backprop_ptr,
        data_format_);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
  int64 stride_;

  TF_DISALLOW_COPY_AND_ASSIGN(HexRotDepthwiseConv2dNativeBackpropFilterOp);
};

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("HexRotDepthwiseConv2dNativeBackpropFilter")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T")
        .HostMemory("filter_sizes"),
    HexRotDepthwiseConv2dNativeBackpropFilterOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("HexRotDepthwiseConv2dNativeBackpropFilter")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T")
        .HostMemory("filter_sizes"),
    HexRotDepthwiseConv2dNativeBackpropFilterOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

// -------------------------------------------------------------------------- //
// Backward Rotation conv2d kernel.
// -------------------------------------------------------------------------- //
#if GOOGLE_CUDA

// extern template struct LaunchHexRotDepthwiseConvBackpropRotationOp<GPUDevice, Eigen::half>;
extern template struct LaunchHexRotDepthwiseConvBackpropRotationOp<GPUDevice, float>;
extern template struct LaunchHexRotDepthwiseConvBackpropRotationOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA

// Kernel to compute the rotation backprop for depthwise convolution.
template <typename Device, class T>
class HexRotDepthwiseConv2dNativeBackpropRotationOp : public OpKernel {
 public:
  explicit HexRotDepthwiseConv2dNativeBackpropRotationOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));

    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

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
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& rotation = context->input(2);

    // OP_REQUIRES(
    //     context, TensorShapeUtils::IsVector(rot_sizes.shape()),
    //     errors::InvalidArgument(
    //         "Conv2dBackpropRotation: rot_sizes input must be 1-dim, not ",
    //         rot_sizes.dims()));
    // TensorShape rot_shape;
    // const int32* rot_sizes_data = rot_sizes.template flat<int32>().data();
    // for (int i = 0; i < rot_sizes.NumElements(); ++i) {
    //   OP_REQUIRES(context, rot_sizes_data[i] >= 0,
    //               errors::InvalidArgument("Dimension ", i,
    //                                       " of rot_sizes must be >= 0"));
    //   rot_shape.AddDim(rot_sizes_data[i]);
    // }
    const TensorShape& input_shape = input.shape();
    const TensorShape& filter_shape = filter.shape();
    const TensorShape& rot_shape = rotation.shape();
    EXTRACT_AND_VERIFY_DIMENSIONS("HexRotDepthwiseConv2dBackpropRotation");

    Tensor* rot_backprop = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, rot_shape, &rot_backprop));
    auto out_backprop_ptr = out_backprop.template flat<T>().data();
    auto input_ptr = input.template flat<T>().data();
    auto filter_ptr = filter.template flat<T>().data();
    auto rotation_ptr = rotation.template flat<T>().data();
    auto rot_backprop_ptr = rot_backprop->template flat<T>().data();

    // If there is nothing to compute, return.
    if (rot_shape.num_elements() == 0) {
      return;
    }
    LaunchHexRotDepthwiseConvBackpropRotationOp<Device, T>()(
        context, args, out_backprop_ptr, input_ptr, filter_ptr, rotation_ptr, rot_backprop_ptr,
        data_format_);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
  int64 stride_;

  TF_DISALLOW_COPY_AND_ASSIGN(HexRotDepthwiseConv2dNativeBackpropRotationOp);
};

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("HexRotDepthwiseConv2dNativeBackpropRotation")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T")
        .HostMemory("input_sizes"),
    HexRotDepthwiseConv2dNativeBackpropRotationOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("HexRotDepthwiseConv2dNativeBackpropRotation")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T")
        .HostMemory("input_sizes"),
    HexRotDepthwiseConv2dNativeBackpropRotationOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
