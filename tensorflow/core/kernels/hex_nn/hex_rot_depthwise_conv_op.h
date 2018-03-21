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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

struct HexRotDepthwiseArgs {
  // Input layer dimensions
  int batch;
  int in_rows;
  int in_cols;
  int in_depth;
  int filter_rows;
  int filter_cols;
  int depth_multiplier;
  int stride;
  int pad_rows;
  int pad_cols;

  // Output layer dimensions
  int out_rows;
  int out_cols;
  int out_depth;

  HexRotDepthwiseArgs()
      : batch(0),
        in_rows(0),
        in_cols(0),
        in_depth(0),
        filter_rows(0),
        filter_cols(0),
        depth_multiplier(0),
        stride(0),
        pad_rows(0),
        pad_cols(0),
        out_rows(0),
        out_cols(0),
        out_depth(0) {}
};

// Forward declaration.
class OpKernelContext;

template <typename Device, typename T>
struct LaunchHexRotDepthwiseConvOp {
  void operator()(OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
                  const T* input, const T* filter, const T* rotation, T* output,
                  TensorFormat data_format);
};
template <typename Device, typename T>
struct LaunchHexRotDepthwiseConvBackpropInputOp {
  void operator()(OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
                  const T* out_backprop, const T* filter, const T* rotation, T* in_backprop,
                  TensorFormat data_format);
};
template <typename Device, typename T>
struct LaunchHexRotDepthwiseConvBackpropFilterOp {
  void operator()(OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
                  const T* out_backprop, const T* input, const T* rotation, T* filter_backprop,
                  TensorFormat data_format);
};
template <typename Device, typename T>
struct LaunchHexRotDepthwiseConvBackpropRotationOp {
  void operator()(OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
                  const T* out_backprop, const T* input, const T* filter, T* rot_backprop,
                  TensorFormat data_format);
};

#if GOOGLE_CUDA
template <typename T>
struct LaunchHexRotDepthwiseConvOp<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, const HexRotDepthwiseArgs args,
                  const T* input, const T* filter, const T* rotation, T* output,
                  TensorFormat data_format);
};
template <typename T>
struct LaunchHexRotDepthwiseConvBackpropInputOp<Eigen::GpuDevice, T> {
  void operator()(class OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
                  const T* out_backprop, const T* filter, const T* rotation, T* in_backprop,
                  TensorFormat data_format);
};
template <typename T>
struct LaunchHexRotDepthwiseConvBackpropFilterOp<Eigen::GpuDevice, T> {
  void operator()(class OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
                  const T* out_backprop, const T* input, const T* rotation, T* filter_backprop,
                  TensorFormat data_format);
};
template <typename T>
struct LaunchHexRotDepthwiseConvBackpropRotationOp<Eigen::GpuDevice, T> {
  void operator()(class OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
                  const T* out_backprop, const T* input, const T* filter, T* rot_backprop,
                  TensorFormat data_format);
};

#endif

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_H_
