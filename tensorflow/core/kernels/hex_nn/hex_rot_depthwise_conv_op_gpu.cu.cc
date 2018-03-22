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
#define GOOGLE_CUDA 1

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"

#include "hex_rot_depthwise_conv_op.h"

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"
#include "external/cub_archive/cub/util_ptx.cuh"

#if !defined(_MSC_VER)
#define UNROLL _Pragma("unroll")
#define NOUNROLL _Pragma("nounroll")
#else
#define UNROLL
#define NOUNROLL
#endif

namespace tensorflow {

using Eigen::GpuDevice;

// -------------------------------------------------------------------------- //
// Depthwise conv2d parameters.
// -------------------------------------------------------------------------- //

// TheHexRotDepthwiseConv2DGPUKernels perform either forward or backprop input
// convolution depending on a template argument of this enum.
enum HexRotDepthwiseConv2DDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };


// -------------------------------------------------------------------------- //
// Look-up tables for hexagonal filters and coordinates.
// -------------------------------------------------------------------------- //
#define MAX_NUM_ELEMENTS 19
/** Number of elements per circle of the filter. */
__constant__ int NUM_ELEMENTS_RADIUS[4] = {1, 6, 12, 18};
/** Filter table, started with top-left corner. */
__constant__ int ELEMENTS_GRAD[MAX_NUM_ELEMENTS] = {
  0,
  4, 5, 6, 1, 2, 3,
  13, 14, 15, 16, 17, 18, 7, 8, 9, 10, 11, 12
};
__constant__ int INPUT_DELTA_ROWS[2][MAX_NUM_ELEMENTS] = {
  {
    0,
    -1, -1, 0, 1, 1, 0,
    -2, -2, -2, -1, 0, 1, 2, 2, 2, 1, 0, -1
  },
  {
    0,
    -1, -1, 0, 1, 1, 0,
    -2, -2, -2, -1, 0, 1, 2, 2, 2, 1, 0, -1
  },
};
__constant__ int INPUT_DELTA_COLS[2][MAX_NUM_ELEMENTS] = {
  {
    0,
    -1, 0, 1, 0, -1, -1,
    -1, 0, 1, 1, 2, 1, 1, 0, -1, -2, -2, -2
  },
  {
    0,
    0, 1, 1, 1, 0, -1,
    -1, 0, 1, 2, 2, 2, 1, 0, -1, -1, -2, -1
  },
};

// -------------------------------------------------------------------------- //
// Hexagonal tiling, implementation of convolution kernels.
// -------------------------------------------------------------------------- //
/** A Cuda kernel to compute the depthwise convolution forward pass in NHWC format.
 */
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(1024, 2)
    HexRotDepthwiseConv2DGPUKernelNHWC(const HexRotDepthwiseArgs args,
                                       const T* input,
                                       const T* filter,
                                       const T* rotation,
                                       T* output,
                                       int num_outputs) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_cols =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;
  // Filter radius
  const int radius = filter_cols / 2;

  CUDA_1D_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    const int OD = thread_id % out_depth;
    const int OC = (thread_id / out_depth) % out_cols;
    const int OR = (thread_id / out_depth / out_cols) % out_rows;
    const int OB = thread_id / out_depth / out_cols / out_rows;
    // Compute the input depth and the index of depth multiplier.
    const int in_d = OD / depth_multiplier;
    const int multiplier = OD % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int input_row_start = OR * stride - pad_rows;
    const int input_col_start = OC * stride - pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;
    // Input center coordinates.
    const int input_row_center = input_row_start + radius;
    const int input_col_center = input_col_start + radius;
    const int input_row_sign = input_row_center % 2;
    const int out_row_sign = OR % 2;

    T sum = static_cast<T>(0);
    const int input_offset_temp = in_rows * OB;
    // Full implementation only for stride == 1. TODO: merge everything.
    if (stride == 1) {
      if (input_row_start >= 0 && input_col_start >= 0 &&
          input_row_end < in_rows && input_col_end < in_cols) {
        // Loop on filter radius.
        int f_idx = 0;
        UNROLL for (int r = 0 ; r <= radius ; ++r) {
          UNROLL for (int idx = 0 ; idx < NUM_ELEMENTS_RADIUS[r] ; ++idx) {
            // Input coordinates.
            const int in_r = input_row_center + INPUT_DELTA_ROWS[input_row_sign][f_idx];
            const int in_c = input_col_center + INPUT_DELTA_COLS[input_row_sign][f_idx];

            const int input_offset =
                in_d + in_depth * (in_c + in_cols * (in_r + input_offset_temp));
            const int filter_offset =
                multiplier +
                depth_multiplier * (in_d + in_depth * f_idx);
            sum += ldg(input + input_offset) * ldg(filter + filter_offset);
            // Update filter index.
            ++f_idx;
          }
        }
        // UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        //   const int in_r = input_row_start + f_r;
        //   const int filter_offset_temp = filter_cols * f_r;
        //   UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
        //     const int in_c = input_col_start + f_c;

        //     const int input_offset =
        //         in_d + in_depth * (in_c + in_cols * (in_r + input_offset_temp));
        //     const int filter_offset =
        //         multiplier +
        //         depth_multiplier * (in_d + in_depth * (f_c + filter_offset_temp));
        //     sum += ldg(input + input_offset) * ldg(filter + filter_offset);
        //   }
        // }
      }
      else {
        // Loop on filter radius.
        int f_idx = 0;
        UNROLL for (int r = 0 ; r <= radius ; ++r) {
          UNROLL for (int idx = 0 ; idx < NUM_ELEMENTS_RADIUS[r] ; ++idx) {
            // Input coordinates.
            const int in_r = input_row_center + INPUT_DELTA_ROWS[input_row_sign][f_idx];
            const int in_c = input_col_center + INPUT_DELTA_COLS[input_row_sign][f_idx];
            if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
              const int input_offset =
                  in_d + in_depth * (in_c + in_cols * (in_r + input_offset_temp));
              const int filter_offset =
                  multiplier +
                  depth_multiplier * (in_d + in_depth * f_idx);
              sum += ldg(input + input_offset) * ldg(filter + filter_offset);
            }
            // Update filter index.
            ++f_idx;
          }
        }
      }
    }
    else {
      // Stride > 1. Only downsizing for now! TODO: everything else!
      const int in_r = input_row_center;
      const int in_c = input_col_center + out_row_sign * (stride - 1);
      // Zero padding?
      if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
        const int input_offset =
                  in_d + in_depth * (in_c + in_cols * (in_r + input_offset_temp));
        sum = ldg(input + input_offset);
      }
      // TODO: symmetric padding for outside values? Not loosing as much information.
    }
    // output[thread_id] = static_cast<T>(0);
    output[thread_id] = sum;
  }
}

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. input.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    HexRotDepthwiseConv2DBackpropInputGPUKernelNHWC(const HexRotDepthwiseArgs args,
                                                    const T* out_backprop,
                                                    const T* filter,
                                                    const T* rotation,
                                                    T* in_backprop,
                                                    int num_in_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_cols =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  // Filter radius
  const int radius = filter_cols / 2;
  CUDA_1D_KERNEL_LOOP(thread_id, num_in_backprop) {
    // Compute the indexes of this thread in the output.
    const int in_d = thread_id % in_depth;
    const int in_c = (thread_id / in_depth) % in_cols;
    const int in_r = (thread_id / in_depth / in_cols) % in_rows;
    const int b = thread_id / in_depth / in_cols / in_rows;
    // Depth / Cols / Rows / Batch.
    const int OD = thread_id % out_depth;
    const int OC = (thread_id / out_depth) % out_cols;
    const int OR = (thread_id / out_depth / out_cols) % out_rows;
    const int OB = thread_id / out_depth / out_cols / out_rows;
    // Compute the input depth and the index of depth multiplier.
    // const int in_d = OD / depth_multiplier;
    const int multiplier = OD % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int input_row_start = OR * stride - pad_rows;
    const int input_col_start = OC * stride - pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;
    // Input center coordinates.
    const int input_row_center = input_row_start + radius;
    const int input_col_center = input_col_start + radius;
    const int input_row_sign = input_row_center % 2;
    const int out_row_sign = OR % 2;

    T sum = static_cast<T>(0);
    const int input_offset_temp = in_rows * OB;
    // Full implementation only for stride == 1. TODO: merge everything.
    if (stride == 1) {
      if (input_row_start >= 0 && input_col_start >= 0 &&
          input_row_end < in_rows && input_col_end < in_cols) {

        // Loop on filter radius.
        int f_idx = 0;
        UNROLL for (int r = 0 ; r <= radius ; ++r) {
          UNROLL for (int idx = 0 ; idx < NUM_ELEMENTS_RADIUS[r] ; ++idx) {
            // Input coordinates.
            const int in_r = input_row_center + INPUT_DELTA_ROWS[input_row_sign][f_idx];
            const int in_c = input_col_center + INPUT_DELTA_COLS[input_row_sign][f_idx];

            const int input_offset =
                in_d + in_depth * (in_c + in_cols * (in_r + input_offset_temp));
            const int filter_offset =
                multiplier + depth_multiplier * (in_d + in_depth * ELEMENTS_GRAD[f_idx]);
            sum += ldg(out_backprop + input_offset) * ldg(filter + filter_offset);
            // Update filter index.
            ++f_idx;
          }
        }
      }
      else {
        // Loop on filter radius.
        int f_idx = 0;
        UNROLL for (int r = 0 ; r <= radius ; ++r) {
          UNROLL for (int idx = 0 ; idx < NUM_ELEMENTS_RADIUS[r] ; ++idx) {
            // Input coordinates.
            const int in_r = input_row_center + INPUT_DELTA_ROWS[input_row_sign][f_idx];
            const int in_c = input_col_center + INPUT_DELTA_COLS[input_row_sign][f_idx];
            if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
              const int input_offset =
                  in_d + in_depth * (in_c + in_cols * (in_r + input_offset_temp));
              const int filter_offset =
                  multiplier + depth_multiplier * (in_d + in_depth * ELEMENTS_GRAD[f_idx]);
              sum += ldg(out_backprop + input_offset) * ldg(filter + filter_offset);
            }
            // Update filter index.
            ++f_idx;
          }
        }
      }
    }
    else {
      // stride > 1: very basic downscaling. TODO: the rest!
      if (in_r % stride == 0) {
        const int out_row = in_r / stride;
        const int out_row_sign = out_row % 2;
        const int in_col_delta = (in_c % stride);

        if (out_row_sign == in_col_delta) {
          const int out_col = in_c / stride;
          const int out_offset =
                  in_d + in_depth * (out_col + out_cols * (out_row + out_rows * b));
          sum = ldg(out_backprop + out_offset);
        }
      }
    }
    in_backprop[thread_id] = sum;


    //     // Compute the indexes of this thread in the output.
    //     const int in_d = thread_id % in_depth;
    //     const int in_c = (thread_id / in_depth) % in_cols;
    //     const int in_r = (thread_id / in_depth / in_cols) % in_rows;
    //     const int b = thread_id / in_depth / in_cols / in_rows;

    //     // Output coordinates...
    //     const int o_row_start = in_c * stride - pad_rows;
    //     const int o_col_start = in_c * stride - pad_cols;
    //     const int o_row_end = o_row_start + filter_rows;
    //     const int o_col_end = o_col_start + filter_cols;
    //     // Center coordinates.
    //     const int o_row_center = input_row_start + radius;
    //     const int o_col_center = input_col_start + radius;
    //     const int o_row_sign = input_row_center % 2;

    //     T sum = static_cast<T>(0);
    //     // ELEMENTS_GRAD
    //     const int out_r_start =
    //         tf_max<int>(0, (in_r - filter_rows + pad_rows + stride) / stride);
    //     const int out_r_end = tf_min(out_rows - 1, (in_r + pad_rows) / stride);
    //     const int out_c_start =
    //         tf_max(0, (in_c - filter_cols + pad_cols + stride) / stride);
    //     const int out_c_end = tf_min(out_cols - 1, (in_c + pad_cols) / stride);

    //     NOUNROLL for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
    //       const int f_r = in_r + pad_rows - out_r * stride;
    //       NOUNROLL for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
    //         const int f_c = in_c + pad_cols - out_c * stride;
    //         int filter_offset =
    //             depth_multiplier * (in_d + in_depth * (f_c + filter_cols * f_r));
    //         const int out_backprop_offset =
    //             out_depth * out_c + out_depth * out_cols * (out_r + out_rows * b);
    // #pragma unroll 6
    //         for (int i = 0; i < depth_multiplier; ++i) {
    //           sum += ldg(out_backprop + out_backprop_offset +
    //                      in_d * depth_multiplier + i) *
    //                  ldg(filter + filter_offset + i);
    //         }
    //       }
    //     }
    //     const int in_backprop_offset =
    //         in_d + in_depth * (in_c + in_cols * (in_r + in_rows * b));
    //     in_backprop[in_backprop_offset] = sum;
  }
}

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    HexRotDepthwiseConv2DBackpropFilterGPUKernelNHWC(const HexRotDepthwiseArgs args,
                                                     const T* out_backprop,
                                                     const T* input,
                                                     const T* rotation,
                                                     T* filter_backprop,
                                                     int num_out_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_cols =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  // Stride > 1: zero gradient. TODO: real implementation!
  if (stride > 1) {
    return;
  }

  const int radius = filter_cols / 2;
  CUDA_1D_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int out_d = thread_id % out_depth;
    const int out_c = (thread_id / out_depth) % out_cols;
    const int out_r = (thread_id / out_depth / out_cols) % out_rows;
    const int b = thread_id / out_depth / out_cols / out_rows;
    // Compute the input depth and the index of depth multiplier.
    const int in_d = out_d / depth_multiplier;
    const int dm = out_d % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_r_start = out_r * stride - pad_rows;
    const int in_c_start = out_c * stride - pad_cols;
    const int in_r_end = in_r_start + filter_rows;
    const int in_c_end = in_c_start + filter_cols;
    // Input center coordinates.
    const int input_row_center = in_r_start + radius;
    const int input_col_center = in_c_start + radius;
    const int out_row_sign = out_r % 2;

    const int out_backprop_offset =
        out_d + out_depth * (out_c + out_cols * (out_r + out_rows * b));
    const T out_bp = ldg(out_backprop + out_backprop_offset);
    if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
        in_c_end < in_cols) {

      // Loop on filter radius.
      int f_idx = 0;
      UNROLL for (int r = 0 ; r <= radius ; ++r) {
        UNROLL for (int idx = 0 ; idx < NUM_ELEMENTS_RADIUS[r] ; ++idx) {
          // Input coordinates.
          const int in_r = input_row_center + INPUT_DELTA_ROWS[out_row_sign][f_idx];
          const int in_c = input_col_center + INPUT_DELTA_COLS[out_row_sign][f_idx];

          const int input_offset =
              in_d + in_depth * (in_c + in_cols * (in_r + in_rows * b));
          T partial_sum = ldg(input + input_offset) * out_bp;
          T* addr = filter_backprop + (dm + depth_multiplier * (in_d + in_depth * f_idx));
          CudaAtomicAdd(addr, partial_sum);
          // Update filter index.
          ++f_idx;
        }
      }
      // UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
      //   // Avoid repeated computation.
      //   UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
      //     const int in_r = in_r_start + f_r;
      //     const int in_c = in_c_start + f_c;

      //     const int input_offset = in_d + in_depth * (in_c + in_cols * (in_r + in_rows * b));
      //     T partial_sum = ldg(input + input_offset) * out_bp;
      //     T* addr = filter_backprop +
      //               (dm + depth_multiplier *
      //                         (in_d + in_depth * (f_c + filter_cols * f_r)));
      //     CudaAtomicAdd(addr, partial_sum);
      //   }
      // }
    }
    else {
      int f_idx = 0;
      UNROLL for (int r = 0 ; r <= radius ; ++r) {
        UNROLL for (int idx = 0 ; idx < NUM_ELEMENTS_RADIUS[r] ; ++idx) {
          // Input coordinates.
          const int in_r = input_row_center + INPUT_DELTA_ROWS[out_row_sign][f_idx];
          const int in_c = input_col_center + INPUT_DELTA_COLS[out_row_sign][f_idx];

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int input_offset =
                in_d + in_depth * (in_c + in_cols * (in_r + in_rows * b));
            T partial_sum = ldg(input + input_offset) * out_bp;
            T* addr = filter_backprop +
                      (dm + depth_multiplier * (in_d + in_depth * f_idx));
            CudaAtomicAdd(addr, partial_sum);
          }
          // Update filter index.
          ++f_idx;
        }
      }
      // Potentially many threads can add to the same address so we have
      // to use atomic add here.
      // TODO(jmchen): If atomic add turns out to be slow, we can:
      // 1. allocate multiple buffers for the gradients (one for each
      // example in a batch, for example). This can reduce the
      // contention on the destination; 2. Have each thread compute one
      // gradient for an element in the filters. This should work well
      // when the input depth is big and filter size is not too small.
    }
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    HexRotDepthwiseConv2DBackpropRotationGPUKernelNHWC(const HexRotDepthwiseArgs args,
                                                    const T* out_backprop,
                                                    const T* input,
                                                    const T* filter,
                                                    T* rot_backprop,
                                                    int num_rot_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_cols =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;
}


// -------------------------------------------------------------------------- //
// -------------------------------------------------------------------------- //
// TensorFlow stuff...
// -------------------------------------------------------------------------- //
// -------------------------------------------------------------------------- //


// -------------------------------------------------------------------------- //
// Forward conv2d kernel.
// -------------------------------------------------------------------------- //
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
void LaunchHexRotDepthwiseConv2dGPU(const GpuDevice& d, const HexRotDepthwiseArgs args,
                              const T* input, const T* filter, const T* rotation, T* output,
                              TensorFormat data_format) {
  void (*kernel)(const HexRotDepthwiseArgs, const T*, const T*, const T*, T*, int);
  if (data_format == FORMAT_NHWC) {
    kernel =
        HexRotDepthwiseConv2DGPUKernelNHWC<T, kKnownFilterWidth, kKnownFilterHeight,
                                     kKnownDepthMultiplier>;
  } else {
    assert(false && "Incorrect data format not NHWC.");
    return;
  }
  const int num_outputs =
      args.batch * args.out_rows * args.out_cols * args.out_depth;
  CudaLaunchConfig config = GetCudaLaunchConfig(num_outputs, d, kernel, 0, 0);
  // The compile-time constant version runs faster with a single block.
  const int max_block_count = kKnownFilterWidth < 0 || kKnownFilterHeight < 0 ||
                                      kKnownDepthMultiplier < 0
                                  ? std::numeric_limits<int>::max()
                                  : d.getNumCudaMultiProcessors();
  kernel<<<std::min(max_block_count, config.block_count),
           config.thread_per_block, 0, d.stream()>>>(args, input, filter, rotation,
                                                     output, num_outputs);
}
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchHexRotDepthwiseConv2dGPU(const GpuDevice& d, const HexRotDepthwiseArgs args,
                              const T* input, const T* filter, const T* rotation, T* output,
                              TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    LaunchHexRotDepthwiseConv2dGPU<T, kKnownFilterWidth, kKnownFilterHeight, 1>(
        d, args, input, filter, rotation, output, data_format);
  } else {
    LaunchHexRotDepthwiseConv2dGPU<T, kKnownFilterWidth, kKnownFilterHeight, -1>(
        d, args, input, filter, rotation, output, data_format);
  }
}
// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
void LaunchHexRotDepthwiseConvOp<GPUDevice, T>::operator()(OpKernelContext* ctx,
                                                     const HexRotDepthwiseArgs args,
                                                     const T* input,
                                                     const T* filter,
                                                     const T* rotation,
                                                     T* output,
                                                     TensorFormat data_format) {
  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  if (args.filter_rows == 3 && args.filter_cols == 3) {
    LaunchHexRotDepthwiseConv2dGPU<T, 3, 3>(d, args, input, filter, rotation, output,
                                      data_format);
  }
  else if (args.filter_rows == 5 && args.filter_cols == 5) {
    LaunchHexRotDepthwiseConv2dGPU<T, 5, 5>(d, args, input, filter, rotation, output,
                                      data_format);
  }
  else {
    LaunchHexRotDepthwiseConv2dGPU<T, -1, -1>(d, args, input, filter, rotation, output,
                                        data_format);
  }
  auto stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream->ok(),
              errors::Internal(
                  "Launch of gpu kernel for HexRotDepthwiseConv2DGPULaunch failed"));
}
template struct LaunchHexRotDepthwiseConvOp<GPUDevice, Eigen::half>;
template struct LaunchHexRotDepthwiseConvOp<GPUDevice, float>;
template struct LaunchHexRotDepthwiseConvOp<GPUDevice, double>;


// -------------------------------------------------------------------------- //
// Backward Input conv2d kernel.
// -------------------------------------------------------------------------- //
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
void LaunchHexRotDepthwiseConv2dBackpropInputGPU(const GpuDevice& d,
                                           const HexRotDepthwiseArgs args,
                                           const T* out_backprop,
                                           const T* filter,
                                           const T* rotation,
                                           T* in_backprop,
                                           TensorFormat data_format) {
  void (*kernel)(const HexRotDepthwiseArgs, const T*, const T*, const T*, T*, int);
  if (data_format == FORMAT_NHWC) {
    kernel = HexRotDepthwiseConv2DBackpropInputGPUKernelNHWC<
        T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
  } else {
    assert(false && "Incorrect data format not NHWC.");
    return;
  }
  const int num_in_backprop =
      args.batch * args.in_rows * args.in_cols * args.in_depth;
  CudaLaunchConfig config =
      GetCudaLaunchConfig(num_in_backprop, d, kernel, 0, 0);
  kernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      args, out_backprop, filter, rotation, in_backprop, num_in_backprop);
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchHexRotDepthwiseConv2dBackpropInputGPU(const GpuDevice& d,
                                           const HexRotDepthwiseArgs args,
                                           const T* out_backprop,
                                           const T* filter,
                                           const T* rotation,
                                           T* in_backprop,
                                           TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    LaunchHexRotDepthwiseConv2dBackpropInputGPU<T, kKnownFilterWidth,
                                          kKnownFilterHeight, 1>(
        d, args, out_backprop, filter, rotation, in_backprop, data_format);
  } else {
    LaunchHexRotDepthwiseConv2dBackpropInputGPU<
        T, kKnownFilterWidth, kKnownFilterHeight, -1>(
            d, args, out_backprop, filter, rotation, in_backprop, data_format);
  }
}
// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
void LaunchHexRotDepthwiseConvBackpropInputOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
    const T* out_backprop,
    const T* filter,
    const T* rotation,
    T* in_backprop,
    TensorFormat data_format) {

  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  if (args.filter_rows == 3 && args.filter_cols == 3) {
    LaunchHexRotDepthwiseConv2dBackpropInputGPU<T, 3, 3>(
        d, args, out_backprop, filter, rotation, in_backprop, data_format);
  }
  else if (args.filter_rows == 5 && args.filter_cols == 5) {
    LaunchHexRotDepthwiseConv2dBackpropInputGPU<T, 5, 5>(
        d, args, out_backprop, filter, rotation, in_backprop, data_format);
  }
  else {
    LaunchHexRotDepthwiseConv2dBackpropInputGPU<T, -1, -1>(
        d, args, out_backprop, filter, rotation, in_backprop, data_format);
  }
  auto stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream->ok(),
              errors::Internal("Launch of gpu kernel for "
                               "HexRotDepthwiseConv2DBackpropInp"
                               "utGPULaunch failed"));
}
template struct LaunchHexRotDepthwiseConvBackpropInputOp<GPUDevice, Eigen::half>;
template struct LaunchHexRotDepthwiseConvBackpropInputOp<GPUDevice, float>;
template struct LaunchHexRotDepthwiseConvBackpropInputOp<GPUDevice, double>;


// -------------------------------------------------------------------------- //
// Backward Filter conv2d kernel.
// -------------------------------------------------------------------------- //
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
void LaunchHexRotDepthwiseConv2dBackpropFilterGPU(const GpuDevice& d,
                                            const HexRotDepthwiseArgs args,
                                            const T* out_backprop,
                                            const T* input,
                                            const T* rotation,
                                            T* filter_backprop,
                                            TensorFormat data_format) {
  void (*kernel)(const HexRotDepthwiseArgs, const T*, const T*, const T*, T*, int);
  if (data_format == FORMAT_NHWC) {
    kernel = HexRotDepthwiseConv2DBackpropFilterGPUKernelNHWC<
        T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
  } else {
    assert(false && "Incorrect data format not NHWC.");
    return;
  }
  const int num_out_backprop =
      args.batch * args.out_rows * args.out_cols * args.out_depth;
  CudaLaunchConfig config =
      GetCudaLaunchConfig(num_out_backprop, d, kernel, 0, 0);
  kernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      args, out_backprop, input, rotation, filter_backprop, num_out_backprop);
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchHexRotDepthwiseConv2dBackpropFilterGPU(const GpuDevice& d,
                                            const HexRotDepthwiseArgs args,
                                            const T* out_backprop,
                                            const T* input,
                                            const T* rotation,
                                            T* filter_backprop,
                                            TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    LaunchHexRotDepthwiseConv2dBackpropFilterGPU<T, kKnownFilterWidth,
                                           kKnownFilterHeight, 1>(
        d, args, out_backprop, input, rotation, filter_backprop, data_format);
  } else {
    LaunchHexRotDepthwiseConv2dBackpropFilterGPU<T, kKnownFilterWidth,
                                           kKnownFilterHeight, -1>(
        d, args, out_backprop, input, rotation, filter_backprop, data_format);
  }
}
// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
void LaunchHexRotDepthwiseConvBackpropFilterOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
    const T* out_backprop,
    const T* input,
    const T* rotation,
    T* filter_backprop,
    TensorFormat data_format) {

  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  auto stream = ctx->op_device_context()->stream();

  // Initialize the results to 0.
  int num_filter_backprop =
      args.filter_rows * args.filter_cols * args.out_depth;
  perftools::gputools::DeviceMemoryBase filter_bp_ptr(filter_backprop,
                                                      num_filter_backprop);
  stream->ThenMemset32(&filter_bp_ptr, 0, num_filter_backprop * sizeof(T));

  if (args.filter_rows == 3 && args.filter_cols == 3) {
    LaunchHexRotDepthwiseConv2dBackpropFilterGPU<T, 3, 3>(
        d, args, out_backprop, input, rotation, filter_backprop, data_format);
  }
  else if (args.filter_rows == 5 && args.filter_cols == 5) {
    LaunchHexRotDepthwiseConv2dBackpropFilterGPU<T, 5, 5>(
        d, args, out_backprop, input, rotation, filter_backprop, data_format);
  }
  else {
    LaunchHexRotDepthwiseConv2dBackpropFilterGPU<T, -1, -1>(
        d, args, out_backprop, input, rotation, filter_backprop, data_format);
  }
  OP_REQUIRES(ctx, stream->ok(),
              errors::Internal("Launch of gpu kernel for "
                               "HexRotDepthwiseConv2DBackpropFil"
                               "terGPULaunch failed"));
}
template struct LaunchHexRotDepthwiseConvBackpropFilterOp<GPUDevice, Eigen::half>;
template struct LaunchHexRotDepthwiseConvBackpropFilterOp<GPUDevice, float>;
template struct LaunchHexRotDepthwiseConvBackpropFilterOp<GPUDevice, double>;


// -------------------------------------------------------------------------- //
// Backward Rotation conv2d kernel.
// -------------------------------------------------------------------------- //
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
void LaunchHexRotDepthwiseConv2dBackpropRotationGPU(const GpuDevice& d,
                                            const HexRotDepthwiseArgs args,
                                            const T* out_backprop,
                                            const T* input,
                                            const T* filter,
                                            T* rot_backprop,
                                            TensorFormat data_format) {
  void (*kernel)(const HexRotDepthwiseArgs, const T*, const T*, const T*, T*, int);
  if (data_format == FORMAT_NHWC) {
    kernel = HexRotDepthwiseConv2DBackpropRotationGPUKernelNHWC<
        T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
  } else {
    assert(false && "Incorrect data format not NHWC.");
    return;
  }
  const int num_rot_backprop =
      args.batch * args.in_rows * args.in_cols * args.in_depth;
  CudaLaunchConfig config =
      GetCudaLaunchConfig(num_rot_backprop, d, kernel, 0, 0);
  kernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      args, out_backprop, input, filter, rot_backprop, num_rot_backprop);
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchHexRotDepthwiseConv2dBackpropRotationGPU(const GpuDevice& d,
                                            const HexRotDepthwiseArgs args,
                                            const T* out_backprop,
                                            const T* input,
                                            const T* filter,
                                            T* rot_backprop,
                                            TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    LaunchHexRotDepthwiseConv2dBackpropRotationGPU<T, kKnownFilterWidth,
                                           kKnownFilterHeight, 1>(
        d, args, out_backprop, input, filter, rot_backprop, data_format);
  } else {
    LaunchHexRotDepthwiseConv2dBackpropRotationGPU<T, kKnownFilterWidth,
                                           kKnownFilterHeight, -1>(
        d, args, out_backprop, input, filter, rot_backprop, data_format);
  }
}
// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
void LaunchHexRotDepthwiseConvBackpropRotationOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, const HexRotDepthwiseArgs& args,
    const T* out_backprop,
    const T* input,
    const T* filter,
    T* rot_backprop,
    TensorFormat data_format) {

  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  auto stream = ctx->op_device_context()->stream();

  if (args.filter_rows == 3 && args.filter_cols == 3) {
    LaunchHexRotDepthwiseConv2dBackpropRotationGPU<T, 3, 3>(
        d, args, out_backprop, input, filter, rot_backprop, data_format);
  }
  else if (args.filter_rows == 5 && args.filter_cols == 5) {
    LaunchHexRotDepthwiseConv2dBackpropRotationGPU<T, 5, 5>(
        d, args, out_backprop, input, filter, rot_backprop, data_format);
  }
  else {
    LaunchHexRotDepthwiseConv2dBackpropRotationGPU<T, -1, -1>(
        d, args, out_backprop, input, filter, rot_backprop, data_format);
  }
  OP_REQUIRES(ctx, stream->ok(),
              errors::Internal("Launch of gpu kernel for "
                               "HexRotDepthwiseConv2DBackpropRotation"
                               "GPULaunch failed"));
}
template struct LaunchHexRotDepthwiseConvBackpropRotationOp<GPUDevice, Eigen::half>;
template struct LaunchHexRotDepthwiseConvBackpropRotationOp<GPUDevice, float>;
template struct LaunchHexRotDepthwiseConvBackpropRotationOp<GPUDevice, double>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
