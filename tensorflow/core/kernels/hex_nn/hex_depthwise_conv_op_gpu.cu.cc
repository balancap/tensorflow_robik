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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"

#include "hex_depthwise_conv_op.h"

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

// TheHexDepthwiseConv2DGPUKernels perform either forward or backprop input
// convolution depending on a template argument of this enum.
enum HexDepthwiseConv2DDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };


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
    -1, 0, 1, 1, 2, 1, 1, 0, -1, -1, -2, -1
  },
  {
    0,
    0, 1, 1, 1, 0, -1,
    -1, 0, 1, 2, 2, 2, 1, 0, -1, -2, -2, -2
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
    HexDepthwiseConv2DGPUKernelNHWC(const HexDepthwiseArgs args, const T* input,
                                    const T* filter, T* output, int num_outputs) {
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
    HexDepthwiseConv2DBackpropInputGPUKernelNHWC(const HexDepthwiseArgs args,
                                              const T* out_backprop,
                                              const T* filter, T* in_backprop,
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
        const int out_row_sign = (in_r / stride) % 2;
        const int in_col_delta = (in_c % stride);
        if (out_row_sign == in_col_delta) {
          const int out_r = in_r / stride;
          const int out_c = in_c / stride;
          const int out_offset =
                  in_d + in_depth * (out_c + out_cols * (out_r + out_rows * OB));
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
    HexDepthwiseConv2DBackpropFilterGPUKernelNHWC(const HexDepthwiseArgs args,
                                               const T* out_backprop,
                                               const T* input,
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

// -------------------------------------------------------------------------- //
// Old stuff, not implemented in hexagonal net.
// -------------------------------------------------------------------------- //

// Returns whether depthwise convolution forward or backward input pass can be
// performed using the faster ('Small') variant of the kernel.
EIGEN_DEVICE_FUNC bool CanLaunchHexDepthwiseConv2dGPUSmall(
    const HexDepthwiseArgs args) {
  return false && args.depth_multiplier == 1 && args.stride == 1 && args.in_rows <= 32 &&
         args.in_cols <= 32 && args.in_rows == args.out_rows &&
         args.in_cols == args.out_cols && args.pad_rows >= 0 &&
         args.pad_rows < args.filter_rows && args.pad_cols >= 0 &&
         args.pad_cols < args.filter_cols &&
         args.filter_rows * args.filter_cols <=
             (args.in_rows + 1) / 2 * args.in_cols;
}

// Returns whether depthwise convolution backward filter pass can be performed
// using the faster ('Small') variant of the kernel.
EIGEN_DEVICE_FUNC bool CanLaunchHexDepthwiseConv2dBackpropFilterGPUSmall(
    const HexDepthwiseArgs args, const int block_rows) {
  return false && args.depth_multiplier == 1 && args.stride == 1 && args.in_rows <= 32 &&
         args.in_cols <= 32 && args.in_rows == args.out_rows &&
         args.in_cols == args.out_cols && args.pad_rows >= 0 &&
         args.pad_rows < args.filter_rows && args.pad_cols >= 0 &&
         args.pad_cols < args.filter_cols && block_rows <= args.in_rows &&
         args.filter_rows * args.filter_cols <= args.in_cols * block_rows;
}


// CUDA kernel to compute the depthwise convolution forward pass in NHWC format,
// tailored for small images up to 32x32. Stride and depth multiplier must be 1.
// Padding must be 'SAME', which allows to reuse the index computation. Only
// use this kernel if CanLaunchHexDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backprop input direction is the same as forward direction with the filter
// rotated by 180°.
template <typename T, HexDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockSlices,
          bool kKnownEvenRows>
__global__ __launch_bounds__(1024, 2) void HexDepthwiseConv2DGPUKernelNHWCSmall(
    const HexDepthwiseArgs args, const T* input, const T* filter, T* output) {
  assert(CanLaunchHexDepthwiseConv2dGPUSmall(args));
  // Holds block plus halo and filter data for blockDim.x depths.
  extern __shared__ __align__(sizeof(T)) unsigned char shared_memory[];
  T* const shared_data = reinterpret_cast<T*>(shared_memory);

  const int batches = args.batch;
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_cols =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;

  const int block_rows = blockDim.z;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_size = block_rows * in_cols * kBlockSlices;
  const int in_row_size = in_cols * in_depth;
  const int in_size = in_rows * in_row_size;
  const int in_increment = (in_cols - 1) * kBlockSlices;
  const int filter_pixels = filter_rows * filter_cols;
  const int tile_cols = in_cols + filter_cols - 1;
  const int even_rows = kKnownEvenRows || (1 & ~in_rows);
  const int tile_rows = in_rows + filter_rows - even_rows;
  const int tile_row_size = tile_cols * kBlockSlices;
  const int tile_size = tile_rows * tile_row_size;
  const int tile_offset = block_rows * tile_row_size;
  const int pad_offset = pad_rows * tile_cols + pad_cols;
  const int batch_blocks = (in_depth + kBlockSlices - 1) / kBlockSlices;
  const int in_blocks = batch_blocks * batches;
  const int tensor_offset =
      kKnownEvenRows ? in_size / 2 : block_rows * in_row_size;

  const int thread_depth = threadIdx.x;
  const int thread_col = threadIdx.y;
  const int thread_row = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_row * in_cols + thread_col;
  const int thread_idx = thread_pix * kBlockSlices + thread_depth;

  // Initialize tile, in particular the padding.
  for (int i = thread_idx; i < tile_size; i += block_size) {
    shared_data[i] = T(0);
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_pix * in_depth + thread_depth;

  // Position in (padded) shared memory.
  const int data_pix = thread_row * tile_cols + thread_col;
  const int data_idx = data_pix * kBlockSlices + thread_depth;

  // Position in shared memory, offset by pad_rows / pad_cols.
  const int tile_pix = data_pix + pad_offset;
  const int tile_idx = tile_pix * kBlockSlices + thread_depth;

  const int max_depth = in_depth - thread_depth;
  const int filter_write_offset =
      thread_pix < filter_pixels ? tile_size + thread_idx : 0;
  const int filter_read_offset =
      tile_size + thread_depth +
      (kDirection == DIRECTION_FORWARD ? 0 : filter_pixels * kBlockSlices);
  const bool skip_second =
      !kKnownEvenRows && thread_row + (in_rows & 1) == block_rows;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int batch = b / batch_blocks;
    const int stack = b - batch * batch_blocks;

    const int start_depth = stack * kBlockSlices;
    const int filter_offset = tensor_idx + start_depth;
    const int inout_offset = batch * in_size + filter_offset;
    const bool depth_in_range = start_depth < max_depth;

    if (depth_in_range) {
      const T* const in_ptr = inout_offset + input;
      T* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = ldg(in_ptr);
      if (!skip_second) {
        tile_ptr[tile_offset] = ldg(tensor_offset + in_ptr);
      }

      if (filter_write_offset != 0) {
        shared_data[filter_write_offset] = ldg(filter_offset + filter);
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    if (depth_in_range) {
      T sum1 = static_cast<T>(0);
      T sum2 = static_cast<T>(0);
      int shared_offset = data_idx;
      const T* filter_ptr = filter_read_offset + shared_data;
      UNROLL for (int r = 0; r < filter_rows; ++r) {
        UNROLL for (int c = 0; c < filter_cols; ++c) {
          if (kDirection == DIRECTION_BACKWARD) {
            filter_ptr -= kBlockSlices;
          }
          const T filter_value = *filter_ptr;
          const T* const tile_ptr = shared_offset + shared_data;
          sum1 += filter_value * tile_ptr[0];
          sum2 += filter_value * tile_ptr[tile_offset];
          shared_offset += kBlockSlices;
          if (kDirection == DIRECTION_FORWARD) {
            filter_ptr += kBlockSlices;
          }
        }
        shared_offset += in_increment;
      }
      T* const out_ptr = inout_offset + output;
      out_ptr[0] = sum1;
      out_ptr[0] = static_cast<T>(0);
      if (!skip_second) {
        out_ptr[tensor_offset] = sum2;
        out_ptr[tensor_offset] = static_cast<T>(0);
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();
  }
}

// A Cuda kernel to compute the depthwise convolution forward pass
// in NCHW format.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(1024, 2)
    HexDepthwiseConv2DGPUKernelNCHW(const HexDepthwiseArgs args, const T* input,
                                 const T* filter, T* output, int num_outputs) {
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

  CUDA_1D_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int OC = thread_id % out_cols;
    const int OR = (thread_id / out_cols) % out_rows;
    const int OD = (thread_id / out_cols / out_rows) % out_depth;
    const int OB = thread_id / out_cols / out_rows / out_depth;

    // Compute the input depth and the index of depth multiplier
    // based off the output depth index that this thread is
    // computing n.
    const int in_d = OD / depth_multiplier;
    const int multiplier = OD % depth_multiplier;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_depth * in_rows * in_cols values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp = (OB * in_depth + in_d) * (in_rows * in_cols);

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_row_start = OR * stride - pad_rows;
    const int input_col_start = OC * stride - pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;

    T sum = static_cast<T>(0);
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_rows && input_col_end < in_cols) {
      // Loop that doesn't need to check for boundary conditions.
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;

          const int input_offset =
              (input_offset_temp) + (in_r * in_cols) + in_c;
          const int filter_offset =
              multiplier +
              depth_multiplier * (in_d + in_depth * (f_c + filter_offset_temp));
          sum += ldg(input + input_offset) * ldg(filter + filter_offset);
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;
          // TODO(vrv): the in_r check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int in_c = input_col_start + f_c;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_r * in_cols) + in_c;

            const int filter_offset =
                multiplier + depth_multiplier *
                                 (in_d + in_depth * (f_c + filter_offset_temp));
            sum += ldg(input + input_offset) * ldg(filter + filter_offset);
          }
        }
      }
    }

    output[thread_id] = sum;
    // output[thread_id] = static_cast<T>(0);
  }
}

// CUDA kernel to compute the depthwise convolution forward pass in NCHW format,
// tailored for small images up to 32x32. Stride and depth multiplier must be 1.
// Padding must be 'SAME', which allows to reuse the index computation. Only
// use this kernel if CanLaunchHexDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backprop input direction is the same as forward direction with the filter
// rotated by 180°.
template <typename T, HexDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockSlices,
          bool kKnownEvenRows>
__global__ __launch_bounds__(1024, 2) void HexDepthwiseConv2DGPUKernelNCHWSmall(
    const HexDepthwiseArgs args, const T* input, const T* filter, T* output) {
  assert(CanLaunchHexDepthwiseConv2dGPUSmall(args));
  // Holds block plus halo and filter data for blockDim.z depths.
  extern __shared__ __align__(sizeof(T)) unsigned char shared_memory[];
  T* const shared_data = reinterpret_cast<T*>(shared_memory);

  const int batches = args.batch;
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_cols =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;

  // Fixed blockDim.z, tailored for maximum grid size for images of size 16x16.
  const int block_rows = blockDim.y;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_pixels = in_cols * block_rows;
  const int block_size = block_pixels * kBlockSlices;
  const int in_pixels = in_cols * in_rows;
  const int in_increment = in_cols - 1;
  const int filter_pixels = filter_rows * filter_cols;
  const int tile_cols = in_cols + filter_cols - 1;
  const int even_rows = kKnownEvenRows || (1 & ~in_rows);
  const int tile_rows = in_rows + filter_rows - even_rows;
  const int tile_pixels = tile_cols * tile_rows;
  const int tile_size = tile_pixels * kBlockSlices;
  const int tile_offset = block_rows * tile_cols;
  const int pad_offset = pad_rows * tile_cols + pad_cols;
  const int in_slices = in_depth * batches;
  const int in_blocks = (in_slices + kBlockSlices - 1) / kBlockSlices;

  const int thread_col = threadIdx.x;
  const int thread_row = threadIdx.y;
  const int thread_depth = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_row * in_cols + thread_col;
  const int thread_idx = thread_depth * block_pixels + thread_pix;

  // Initialize tile, in particular the padding.
  for (int i = thread_idx; i < tile_size; i += block_size) {
    shared_data[i] = T(0);
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_depth * in_pixels + thread_pix;

  // Position in (padded) shared memory.
  const int data_pix = thread_row * tile_cols + thread_col;
  const int data_idx = thread_depth * tile_pixels + data_pix;

  // Position in shared memory, offset by pad_rows / pad_cols.
  const int tile_idx = data_idx + pad_offset;

  // Filter is always in HWCK format, irrespective of the input/output format.
  const int filter_pix = thread_idx / kBlockSlices;
  const int filter_depth = thread_idx % kBlockSlices;
  const int filter_idx = filter_pix * in_depth;

  const int max_slice = in_slices - thread_depth;
  const int filter_write_offset =
      filter_pix < filter_pixels ? tile_size + thread_idx : 0;
  const int filter_read_offset =
      tile_size + thread_depth +
      (kDirection == DIRECTION_FORWARD ? 0 : filter_pixels * kBlockSlices);
  const bool skip_second =
      !kKnownEvenRows && thread_row + (in_rows & 1) == block_rows;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int slice = b * kBlockSlices;

    const int inout_offset = slice * in_pixels + tensor_idx;
    const bool slice_in_range = slice < max_slice;

    if (slice_in_range) {
      const T* const in_ptr = inout_offset + input;
      T* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = ldg(in_ptr);
      if (!skip_second) {
        tile_ptr[tile_offset] = ldg(block_pixels + in_ptr);
      }
    }

    if (filter_write_offset != 0) {
      const int filter_offset = filter_idx + (slice + filter_depth) % in_depth;
      shared_data[filter_write_offset] = ldg(filter_offset + filter);
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    if (slice_in_range) {
      T sum1 = static_cast<T>(0);
      T sum2 = static_cast<T>(0);
      int shared_offset = data_idx;
      const T* filter_ptr = filter_read_offset + shared_data;
      UNROLL for (int r = 0; r < filter_rows; ++r) {
        UNROLL for (int c = 0; c < filter_cols; ++c) {
          if (kDirection == DIRECTION_BACKWARD) {
            filter_ptr -= kBlockSlices;
          }
          const T filter_value = *filter_ptr;
          const T* const tile_ptr = shared_offset + shared_data;
          sum1 += filter_value * tile_ptr[0];
          sum2 += filter_value * tile_ptr[tile_offset];
          ++shared_offset;
          if (kDirection == DIRECTION_FORWARD) {
            filter_ptr += kBlockSlices;
          }
        }
        shared_offset += in_increment;
      }
      T* const out_ptr = inout_offset + output;
      out_ptr[0] = sum1;
      if (!skip_second) {
        out_ptr[block_pixels] = sum2;
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();
  }
}

template <typename T, HexDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockSlices,
          bool kKnownEvenRows>
void LaunchHexDepthwiseConv2dGPUSmall(const GpuDevice& d, const HexDepthwiseArgs args,
                                   const T* input, const T* filter, T* output,
                                   TensorFormat data_format) {
  const int block_rows = (args.in_rows + 1) / 2;
  dim3 block_dim;
  void (*kernel)(const HexDepthwiseArgs, const T*, const T*, T*);
  if (data_format == FORMAT_NHWC) {
    block_dim = dim3(kBlockSlices, args.in_cols, block_rows);
    kernel = HexDepthwiseConv2DGPUKernelNHWCSmall<T, kDirection, kKnownFilterWidth,
                                               kKnownFilterHeight, kBlockSlices,
                                               kKnownEvenRows>;
  } else if (data_format == FORMAT_NCHW) {
    block_dim = dim3(args.in_cols, block_rows, kBlockSlices);
    kernel = HexDepthwiseConv2DGPUKernelNCHWSmall<T, kDirection, kKnownFilterWidth,
                                               kKnownFilterHeight, kBlockSlices,
                                               kKnownEvenRows>;
  } else {
    assert(false && "Incorrect data format");
    return;
  }
  const int tile_cols = args.in_cols + args.filter_cols - 1;
  const int tile_rows = block_rows * 2 + args.filter_rows - 1;
  const int tile_pixels = tile_rows * tile_cols;
  const int filter_pixels = args.filter_rows * args.filter_cols;
  const int shared_memory_size =
      kBlockSlices * (tile_pixels + filter_pixels) * sizeof(T);
  const int num_outputs =
      args.batch * args.out_rows * args.out_cols * args.out_depth;
  CudaLaunchConfig config =
      GetCudaLaunchConfig(num_outputs, d, kernel, shared_memory_size,
                          block_dim.x * block_dim.y * block_dim.z);
  kernel<<<config.block_count, block_dim, shared_memory_size, d.stream()>>>(
      args, input, filter, output);
}

template <typename T, HexDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockSlices>
void LaunchHexDepthwiseConv2dGPUSmall(const GpuDevice& d, const HexDepthwiseArgs args,
                                   const T* input, const T* filter, T* output,
                                   TensorFormat data_format) {
  if (args.in_rows & 1) {
    LaunchHexDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                  kKnownFilterHeight, kBlockSlices, false>(
        d, args, input, filter, output, data_format);
  } else {
    LaunchHexDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                  kKnownFilterHeight, kBlockSlices, true>(
        d, args, input, filter, output, data_format);
  }
}

template <typename T, HexDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchHexDepthwiseConv2dGPUSmall(const GpuDevice& d, const HexDepthwiseArgs args,
                                   const T* input, const T* filter, T* output,
                                   TensorFormat data_format) {
  // Maximize (power of two) kBlockSlices while keeping a block within 1024
  // threads (2 pixels per thread).
  const int block_pixels = (args.in_rows + 1) / 2 * args.in_cols;
  if (block_pixels > 256) {
    LaunchHexDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                  kKnownFilterHeight, 2>(d, args, input, filter,
                                                         output, data_format);
  } else if (block_pixels > 128) {
    LaunchHexDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                  kKnownFilterHeight, 4>(d, args, input, filter,
                                                         output, data_format);
  } else {
    LaunchHexDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                  kKnownFilterHeight, 8>(d, args, input, filter,
                                                         output, data_format);
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
void LaunchHexDepthwiseConv2dGPU(const GpuDevice& d, const HexDepthwiseArgs args,
                              const T* input, const T* filter, T* output,
                              TensorFormat data_format) {
  void (*kernel)(const HexDepthwiseArgs, const T*, const T*, T*, int);
  if (data_format == FORMAT_NHWC) {
    kernel =
        HexDepthwiseConv2DGPUKernelNHWC<T, kKnownFilterWidth, kKnownFilterHeight,
                                     kKnownDepthMultiplier>;
  } else if (data_format == FORMAT_NCHW) {
    kernel =
        HexDepthwiseConv2DGPUKernelNCHW<T, kKnownFilterWidth, kKnownFilterHeight,
                                     kKnownDepthMultiplier>;
  } else {
    assert(false && "Incorrect data format");
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
           config.thread_per_block, 0, d.stream()>>>(args, input, filter,
                                                     output, num_outputs);
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchHexDepthwiseConv2dGPU(const GpuDevice& d, const HexDepthwiseArgs args,
                              const T* input, const T* filter, T* output,
                              TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    if (CanLaunchHexDepthwiseConv2dGPUSmall(args)) {
      LaunchHexDepthwiseConv2dGPUSmall<T, DIRECTION_FORWARD, kKnownFilterWidth,
                                    kKnownFilterHeight>(d, args, input, filter,
                                                        output, data_format);
      return;
    }

    LaunchHexDepthwiseConv2dGPU<T, kKnownFilterWidth, kKnownFilterHeight, 1>(
        d, args, input, filter, output, data_format);
  } else {
    LaunchHexDepthwiseConv2dGPU<T, kKnownFilterWidth, kKnownFilterHeight, -1>(
        d, args, input, filter, output, data_format);
  }
}

// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
void LaunchHexDepthwiseConvOp<GPUDevice, T>::operator()(OpKernelContext* ctx,
                                                     const HexDepthwiseArgs args,
                                                     const T* input,
                                                     const T* filter, T* output,
                                                     TensorFormat data_format) {
  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  if (args.filter_rows == 3 && args.filter_cols == 3) {
    LaunchHexDepthwiseConv2dGPU<T, 3, 3>(d, args, input, filter, output,
                                      data_format);
  } else {
    LaunchHexDepthwiseConv2dGPU<T, -1, -1>(d, args, input, filter, output,
                                        data_format);
  }
  auto stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream->ok(),
              errors::Internal(
                  "Launch of gpu kernel for HexDepthwiseConv2DGPULaunch failed"));
}

template struct LaunchHexDepthwiseConvOp<GPUDevice, Eigen::half>;
template struct LaunchHexDepthwiseConvOp<GPUDevice, float>;
template struct LaunchHexDepthwiseConvOp<GPUDevice, double>;


template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    HexDepthwiseConv2DBackpropInputGPUKernelNCHW(const HexDepthwiseArgs args,
                                              const T* out_backprop,
                                              const T* filter, T* in_backprop,
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

  // TODO(vrv): Consider assigning threads to output and using
  // atomics for accumulation, similar to the filter case.
  CUDA_1D_KERNEL_LOOP(thread_id, num_in_backprop) {
    // Compute the indexes of this thread in the input.
    const int in_c = thread_id % in_cols;
    const int in_r = (thread_id / in_cols) % in_rows;
    const int in_d = (thread_id / in_cols / in_rows) % in_depth;
    const int b = thread_id / in_depth / in_cols / in_rows;

    T sum = static_cast<T>(0);
    const int out_d_start = in_d * depth_multiplier;
    const int out_d_end = out_d_start + depth_multiplier;

    const int out_r_start =
        tf_max<int>(0, (in_r - filter_rows + pad_rows + stride) / stride);
    const int out_r_end = tf_min(out_rows - 1, (in_r + pad_rows) / stride);
    const int out_c_start =
        tf_max(0, (in_c - filter_cols + pad_cols + stride) / stride);
    const int out_c_end = tf_min(out_cols - 1, (in_c + pad_cols) / stride);

    UNROLL for (int out_d = out_d_start; out_d < out_d_end; ++out_d) {
      UNROLL for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
        const int f_r = in_r + pad_rows - out_r * stride;
        const int filter_dm = out_d - out_d_start;

        const int temp_filter_offset = filter_cols * f_r;
        for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
          const int f_c = in_c + pad_cols - out_c * stride;
          const int filter_offset =
              filter_dm + args.depth_multiplier *
                              (in_d + in_depth * (f_c + temp_filter_offset));

          const int out_backprop_offset =
              (b * out_depth * out_rows * out_cols) +
              (out_d * out_rows * out_cols) + (out_r * out_cols) + (out_c);

          sum += ldg(out_backprop + out_backprop_offset) *
                 ldg(filter + filter_offset);
        }
      }
    }
    const int in_backprop_offset = (b * in_rows * in_cols * in_depth) +
                                   (in_d * in_rows * in_cols) +
                                   (in_r * in_cols) + (in_c);
    in_backprop[in_backprop_offset] = sum;
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
void LaunchHexDepthwiseConv2dBackpropInputGPU(const GpuDevice& d,
                                           const HexDepthwiseArgs args,
                                           const T* out_backprop,
                                           const T* filter, T* in_backprop,
                                           TensorFormat data_format) {
  void (*kernel)(const HexDepthwiseArgs, const T*, const T*, T*, int);
  if (data_format == FORMAT_NHWC) {
    kernel = HexDepthwiseConv2DBackpropInputGPUKernelNHWC<
        T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
  } else if (data_format == FORMAT_NCHW) {
    kernel = HexDepthwiseConv2DBackpropInputGPUKernelNCHW<
        T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
  } else {
    assert(false && "Incorrect data format");
    return;
  }
  const int num_in_backprop =
      args.batch * args.in_rows * args.in_cols * args.in_depth;
  CudaLaunchConfig config =
      GetCudaLaunchConfig(num_in_backprop, d, kernel, 0, 0);
  kernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      args, out_backprop, filter, in_backprop, num_in_backprop);
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchHexDepthwiseConv2dBackpropInputGPU(const GpuDevice& d,
                                           const HexDepthwiseArgs args,
                                           const T* out_backprop,
                                           const T* filter, T* in_backprop,
                                           TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    if (CanLaunchHexDepthwiseConv2dGPUSmall(args)) {
      LaunchHexDepthwiseConv2dGPUSmall<T, DIRECTION_BACKWARD, kKnownFilterWidth,
                                    kKnownFilterHeight>(
          d, args, out_backprop, filter, in_backprop, data_format);
      return;
    }

    LaunchHexDepthwiseConv2dBackpropInputGPU<T, kKnownFilterWidth,
                                          kKnownFilterHeight, 1>(
        d, args, out_backprop, filter, in_backprop, data_format);
  } else {
    LaunchHexDepthwiseConv2dBackpropInputGPU<T, kKnownFilterWidth,
                                          kKnownFilterHeight, -1>(
        d, args, out_backprop, filter, in_backprop, data_format);
  }
}

// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
void LaunchHexDepthwiseConvBackpropInputOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, const HexDepthwiseArgs& args, const T* out_backprop,
    const T* filter, T* in_backprop, TensorFormat data_format) {
  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  if (args.filter_rows == 3 && args.filter_cols == 3) {
    LaunchHexDepthwiseConv2dBackpropInputGPU<T, 3, 3>(
        d, args, out_backprop, filter, in_backprop, data_format);
  } else {
    LaunchHexDepthwiseConv2dBackpropInputGPU<T, -1, -1>(
        d, args, out_backprop, filter, in_backprop, data_format);
  }
  auto stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream->ok(),
              errors::Internal("Launch of gpu kernel for "
                               "HexDepthwiseConv2DBackpropInp"
                               "utGPULaunch failed"));
}

template struct LaunchHexDepthwiseConvBackpropInputOp<GPUDevice, Eigen::half>;
template struct LaunchHexDepthwiseConvBackpropInputOp<GPUDevice, float>;
template struct LaunchHexDepthwiseConvBackpropInputOp<GPUDevice, double>;



// Device function to compute sub-warp sum reduction for a power-of-two group of
// neighboring threads.
template<int kWidth, typename T>
__device__ __forceinline__ T WarpSumReduce(T val) {
  // support only power-of-two widths.
  assert(__popc(kWidth) == 1);
  int sub_warp = cub::LaneId() / kWidth;
  int zeros = sub_warp * kWidth;
  unsigned mask = ((1UL << kWidth) - 1) << zeros;
  for (int delta = kWidth / 2; delta > 0; delta /= 2) {
    val += CudaShuffleXor(mask, val, delta);
  }
  return val;
}

// CUDA kernel to compute the depthwise convolution backward w.r.t. filter in
// NHWC format, tailored for small images up to 32x32. Stride and depth
// multiplier must be 1. Padding must be 'SAME'. Only use this kernel if
// CanLaunchHexDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input tensor are loaded into shared memory before performing the
// convolution. Per iteration and filter element, each thread first performs
// a partial convolution for two elements, one each in the lower and upper half
// of a tile. The intermediate result of all pixels of a warp are then
// accumulated and written to shared memory. Finally, the values in shared
// memory are warp-accumulated (in chunks of kAccumPixels elements) and summed
// up in global memory using atomics.
// Requirements: threads per block must be multiple of 32 and <= launch_bounds,
// kAccumPixels * 64 >= args.in_rows * args.in_cols * kBlockSlices.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kBlockSlices, int kAccumPixels>
__global__
__launch_bounds__(1024, 2) void HexDepthwiseConv2DBackpropFilterGPUKernelNHWCSmall(
    const HexDepthwiseArgs args, const T* output, const T* input, T* filter) {
  assert(CanLaunchHexDepthwiseConv2dBackpropFilterGPUSmall(args, blockDim.z));
  // Holds block plus halo and filter data for blockDim.x depths.
  extern __shared__ __align__(sizeof(T)) unsigned char shared_memory[];
  T* const shared_data = reinterpret_cast<T*>(shared_memory);

  const int batches = args.batch;
  const int in_rows = args.in_rows;
  const int in_cols = blockDim.y;  // slower (see b/62280718): args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_cols =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;

  const int block_rows = blockDim.z;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_size = block_rows * in_cols * kBlockSlices;
  assert((block_size & 31) == 0);
  const int in_row_size = in_cols * in_depth;
  const int in_size = in_rows * in_row_size;
  const int in_increment = (in_cols - 1) * kBlockSlices;
  const int filter_pixels = filter_rows * filter_cols;
  const int tile_cols = in_cols + filter_cols - 1;
  const int tile_rows = 2 * block_rows + filter_rows - 1;
  const int tile_row_size = tile_cols * kBlockSlices;
  const int tile_size = tile_rows * tile_row_size;
  const int tile_offset = block_rows * tile_row_size;
  const int pad_offset = pad_rows * tile_cols + pad_cols;
  const int batch_blocks = (in_depth + kBlockSlices - 1) / kBlockSlices;
  const int in_blocks = batch_blocks * batches;
  const int tensor_offset = block_rows * in_row_size;
  // The accumulator has a fixed number of pixels that can be reduced by one
  // warp. Pixels beyond ceil(in_pixels * kBlockSlices / 64) are never written.
  assert(kAccumPixels * 64 >= in_rows * in_cols * kBlockSlices);
  const int accum_increment = kAccumPixels * kBlockSlices;
  const int accum_size = filter_pixels * accum_increment;

  const int thread_depth = threadIdx.x;
  const int thread_col = threadIdx.y;
  const int thread_row = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_row * in_cols + thread_col;
  const int thread_idx = thread_pix * kBlockSlices + thread_depth;

  // Initialize tile, in particular the padding and accumulator.
  for (int i = thread_idx; i < tile_size + accum_size; i += block_size) {
    shared_data[i] = T(0);
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_pix * in_depth + thread_depth;

  // Position in (padded) shared memory.
  const int data_pix = thread_row * tile_cols + thread_col;
  const int data_idx = data_pix * kBlockSlices + thread_depth;

  // Position in shared memory, offset by pad_rows / pad_cols.
  const int tile_pix = data_pix + pad_offset;
  const int tile_idx = tile_pix * kBlockSlices + thread_depth;

  // Position in accumulator (kBlockSlices per warp, depth major).
  const int accum_pix = thread_pix / (32 / kBlockSlices);
  const int accum_idx = thread_depth * kAccumPixels + accum_pix;

  const int max_depth = in_depth - thread_depth;
  const int accum_offset = tile_size + accum_idx;
  const bool skip_second = block_rows + thread_row >= in_rows;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int batch = b / batch_blocks;
    const int stack = b - batch * batch_blocks;

    const int start_depth = stack * kBlockSlices;
    const int filter_offset = tensor_idx + start_depth;
    const int inout_offset = batch * in_size + filter_offset;
    const bool depth_in_range = start_depth < max_depth;

    if (depth_in_range) {
      const T* const in_ptr = inout_offset + input;
      T* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = ldg(in_ptr);
      if (!skip_second) {
        tile_ptr[tile_offset] = ldg(tensor_offset + in_ptr);
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();
    unsigned active_threads = CudaBallot(CUDA_WARP_ALL, depth_in_range);

    if (depth_in_range) {
      const T* const out_ptr = inout_offset + output;
      const T out1 = ldg(out_ptr);
      const T out2 = skip_second ? T(0) : ldg(tensor_offset + out_ptr);
      int shared_offset = data_idx;
      T* accum_ptr = accum_offset + shared_data;
      UNROLL for (int r = 0; r < filter_rows; ++r) {
        UNROLL for (int c = 0; c < filter_cols; ++c) {
          const T* const tile_ptr = shared_offset + shared_data;
          T val = out1 * tile_ptr[0] + out2 * tile_ptr[tile_offset];
          // Warp-accumulate pixels of the same depth and write to accumulator.
          for (int delta = 16; delta >= kBlockSlices; delta /= 2) {
            val += CudaShuffleDown(active_threads, val, delta);
          }
          if (!(thread_idx & 32 - kBlockSlices) /* lane_idx < kBlockSlices */) {
            *accum_ptr = val;
          }
          shared_offset += kBlockSlices;
          accum_ptr += accum_increment;
        }
        shared_offset += in_increment;
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    const T* const accum_data = tile_size + shared_data;
    for (int i = thread_idx; i < accum_size; i += block_size) {
      const int filter_idx = i / kAccumPixels;
      const int filter_pix = filter_idx / kBlockSlices;
      const int filter_depth = filter_idx % kBlockSlices + start_depth;
      const int filter_offset = filter_pix * in_depth + filter_depth;
      if (filter_depth < in_depth) {
        T val = accum_data[i];
        // Warp-accumulate the pixels of the same depth from the accumulator.
        val = WarpSumReduce<kAccumPixels>(val);
        if (!(thread_idx & kAccumPixels - 1)) {
          CudaAtomicAdd(filter_offset + filter, val);
        }
      }
    }
  }
}

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    HexDepthwiseConv2DBackpropFilterGPUKernelNCHW(const HexDepthwiseArgs args,
                                               const T* out_backprop,
                                               const T* input,
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

  CUDA_1D_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int out_c = thread_id % out_cols;
    const int out_r = (thread_id / out_cols) % out_rows;
    const int out_d = (thread_id / out_cols / out_rows) % out_depth;

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

    const int out_backprop_offset = (b * out_depth * out_rows * out_cols) +
                                    (out_d * out_rows * out_cols) +
                                    (out_r * out_cols) + (out_c);

    const T out_bp = ldg(out_backprop + out_backprop_offset);
    if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
        in_c_end < in_cols) {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = (b * in_depth * in_rows * in_cols) +
                                      (in_d * in_rows * in_cols) +
                                      (in_r * in_cols);

        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;
          const int input_offset = input_offset_temp + in_c;
          T partial_sum = ldg(input + input_offset) * out_bp;
          T* addr = filter_backprop +
                    (dm + depth_multiplier *
                              (in_d + in_depth * (f_c + filter_cols * f_r)));
          CudaAtomicAdd(addr, partial_sum);
        }
      }
    } else {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = (b * in_depth * in_rows * in_cols) +
                                      (in_d * in_rows * in_cols) +
                                      (in_r * in_cols);
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;
          const int addr_temp = filter_cols * f_r;

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int input_offset = input_offset_temp + in_c;
            T partial_sum = ldg(input + input_offset) * out_bp;
            T* addr =
                filter_backprop +
                (dm + depth_multiplier * (in_d + in_depth * (f_c + addr_temp)));
            // Potentially many threads can add to the same address so we have
            // to use atomic add here.
            // TODO(jmchen): If atomic add turns out to be slow, we can:
            // 1. allocate multiple buffers for the gradients (one for each
            // example in a batch, for example). This can reduce the
            // contention on the destination; 2. Have each thread compute one
            // gradient for an element in the filters. This should work well
            // when the input depth is big and filter size is not too small.
            CudaAtomicAdd(addr, partial_sum);
          }
        }
      }
    }
  }
}

// CUDA kernel to compute the depthwise convolution backward w.r.t. filter in
// NCHW format, tailored for small images up to 32x32. Stride and depth
// multiplier must be 1. Padding must be 'SAME'. Only use this kernel if
// CanLaunchHexDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input tensor are loaded into shared memory before performing the
// convolution. Per iteration and filter element, each thread first performs
// a partial convolution for two elements, one each in the lower and upper half
// of a tile. The intermediate result of all pixels of a warp are then
// accumulated and written to shared memory. Finally, the values in shared
// memory are warp-accumulated (in chunks of kAccumPixels elements) and summed
// up in global memory using atomics.
// Requirements: threads per block must be multiple of 32 and <= launch_bounds,
// kAccumPixels * 64 >= args.in_rows * args.in_cols * kBlockSlices.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kBlockSlices, int kAccumPixels>
__global__
__launch_bounds__(1024, 2) void HexDepthwiseConv2DBackpropFilterGPUKernelNCHWSmall(
    const HexDepthwiseArgs args, const T* output, const T* input, T* filter) {
  assert(CanLaunchHexDepthwiseConv2dBackpropFilterGPUSmall(args, blockDim.x));
  // Holds block plus halo and filter data for blockDim.z depths.
  extern __shared__ __align__(sizeof(T)) unsigned char shared_memory[];
  T* const shared_data = reinterpret_cast<T*>(shared_memory);

  const int batches = args.batch;
  const int in_rows = args.in_rows;
  const int in_cols = blockDim.x;  // slower (see b/62280718): args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_cols =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;

  const int block_rows = blockDim.y;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_pixels = in_cols * block_rows;
  const int block_size = block_pixels * kBlockSlices;
  assert((block_size & 31) == 0);
  const int in_pixels = in_cols * in_rows;
  const int in_increment = in_cols - 1;
  const int filter_pixels = filter_rows * filter_cols;
  const int tile_cols = in_cols + filter_cols - 1;
  const int tile_rows = 2 * block_rows + filter_rows - 1;
  const int tile_pixels = tile_cols * tile_rows;
  const int tile_size = tile_pixels * kBlockSlices;
  const int tile_offset = block_rows * tile_cols;
  const int pad_offset = pad_rows * tile_cols + pad_cols;
  const int in_slices = in_depth * batches;
  const int in_blocks = (in_slices + kBlockSlices - 1) / kBlockSlices;
  // The accumulator has a fixed number of pixels that can be reduced by one
  // warp. Pixels beyond ceil(in_pixels * kBlockSlices / 64) are never written.
  assert(kAccumPixels * 64 >= in_rows * in_cols * kBlockSlices);
  const int accum_increment = kAccumPixels * kBlockSlices;
  const int accum_size = filter_pixels * accum_increment;

  const int thread_col = threadIdx.x;
  const int thread_row = threadIdx.y;
  const int thread_depth = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_row * in_cols + thread_col;
  const int thread_idx = thread_depth * block_pixels + thread_pix;

  // Initialize tile, in particular the padding and accumulator.
  for (int i = thread_idx; i < tile_size + accum_size; i += block_size) {
    shared_data[i] = T(0);
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_depth * in_pixels + thread_pix;

  // Position in (padded) shared memory.
  const int data_pix = thread_row * tile_cols + thread_col;
  const int data_idx = thread_depth * tile_pixels + data_pix;

  // Position in shared memory, offset by pad_rows / pad_cols.
  const int tile_idx = data_idx + pad_offset;

  // Position in accumulator (kBlockSlices per warp, depth major).
  const int accum_pix = thread_pix / (32 / kBlockSlices);
  const int accum_idx = thread_depth * kAccumPixels + accum_pix;

  const int max_slice = in_slices - thread_depth;
  const int accum_offset = tile_size + accum_idx;
  const bool skip_second = block_rows + thread_row >= in_rows;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int slice = b * kBlockSlices;

    const int inout_offset = slice * in_pixels + tensor_idx;
    const bool slice_in_range = slice < max_slice;

    if (slice_in_range) {
      const T* const in_ptr = inout_offset + input;
      T* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = ldg(in_ptr);
      if (!skip_second) {
        tile_ptr[tile_offset] = ldg(block_pixels + in_ptr);
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();
    unsigned active_threads = CudaBallot(CUDA_WARP_ALL, slice_in_range);

    if (slice_in_range) {
      const T* const out_ptr = inout_offset + output;
      const T out1 = ldg(out_ptr);
      const T out2 = skip_second ? T(0) : ldg(block_pixels + out_ptr);
      int shared_offset = data_idx;
      T* accum_ptr = accum_offset + shared_data;
      UNROLL for (int r = 0; r < filter_rows; ++r) {
        UNROLL for (int c = 0; c < filter_cols; ++c) {
          const T* const tile_ptr = shared_offset + shared_data;
          T val = out1 * tile_ptr[0] + out2 * tile_ptr[tile_offset];
          // Warp-accumulate pixels of the same depth and write to accumulator.
          for (int delta = 16 / kBlockSlices; delta > 0; delta /= 2) {
            val += CudaShuffleDown(active_threads, val, delta);
          }
          if (!(thread_idx & 32 / kBlockSlices - 1)) {
            *accum_ptr = val;
          }
          ++shared_offset;
          accum_ptr += accum_increment;
        }
        shared_offset += in_increment;
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    const T* const accum_data = tile_size + shared_data;
    for (int i = thread_idx; i < accum_size; i += block_size) {
      const int filter_idx = i / kAccumPixels;
      const int filter_pix = filter_idx / kBlockSlices;
      const int filter_depth = (slice + filter_idx % kBlockSlices) % in_depth;
      const int filter_offset = filter_pix * in_depth + filter_depth;
      if (filter_depth < in_depth) {
        T val = accum_data[i];
        // Warp-accumulate pixels of the same depth from the accumulator.
        val = WarpSumReduce<kAccumPixels>(val);
        if (!(thread_idx & kAccumPixels - 1)) {
          CudaAtomicAdd(filter_offset + filter, val);
        }
      }
    }
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kBlockSlices, int kAccumPixels>
bool TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall(
    const GpuDevice& d, const HexDepthwiseArgs args, const int block_rows,
    const T* out_backprop, const T* input, T* filter_backprop,
    TensorFormat data_format) {
  const int tile_cols = args.in_cols + args.filter_cols - 1;
  const int tile_rows = block_rows * 2 + args.filter_rows - 1;
  const int tile_pixels = tile_rows * tile_cols;
  const int filter_pixels = args.filter_rows * args.filter_cols;
  const int shared_memory_size =
      kBlockSlices * (tile_pixels + filter_pixels * kAccumPixels) * sizeof(T);
  if (shared_memory_size > d.sharedMemPerBlock()) {
    return false;
  }

  dim3 block_dim;
  void (*kernel)(const HexDepthwiseArgs, const T*, const T*, T*);
  if (data_format == FORMAT_NHWC) {
    block_dim = dim3(kBlockSlices, args.in_cols, block_rows);
    kernel = HexDepthwiseConv2DBackpropFilterGPUKernelNHWCSmall<
        T, kKnownFilterWidth, kKnownFilterHeight, kBlockSlices, kAccumPixels>;
  } else if (data_format == FORMAT_NCHW) {
    block_dim = dim3(args.in_cols, block_rows, kBlockSlices);
    kernel = HexDepthwiseConv2DBackpropFilterGPUKernelNCHWSmall<
        T, kKnownFilterWidth, kKnownFilterHeight, kBlockSlices, kAccumPixels>;
  } else {
    assert(false && "Incorrect data format");
    return false;
  }
  const int num_out_backprop =
      args.batch * args.out_rows * args.out_cols * args.out_depth;
  CudaLaunchConfig config =
      GetCudaLaunchConfig(num_out_backprop, d, kernel, shared_memory_size,
                          block_dim.x * block_dim.y * block_dim.z);
  kernel<<<config.block_count, block_dim, shared_memory_size, d.stream()>>>(
      args, out_backprop, input, filter_backprop);
  return true;
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kBlockSlices>
bool TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall(
    const GpuDevice& d, const HexDepthwiseArgs args, const int block_rows,
    const T* out_backprop, const T* input, T* filter_backprop,
    TensorFormat data_format) {
  // Minimize (power of two) kAccumPixels, while satisfying
  // kAccumPixels * 32 >= block_rows * in_cols * kBlockSlices.
  const int block_pixels = block_rows * args.in_cols * kBlockSlices;
  if (block_pixels > 512) {
    return TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall<
        T, kKnownFilterWidth, kKnownFilterHeight, kBlockSlices, 32>(
        d, args, block_rows, out_backprop, input, filter_backprop, data_format);
  } else if (block_pixels > 256) {
    return TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall<
        T, kKnownFilterWidth, kKnownFilterHeight, kBlockSlices, 16>(
        d, args, block_rows, out_backprop, input, filter_backprop, data_format);
  } else {
    return TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall<
        T, kKnownFilterWidth, kKnownFilterHeight, kBlockSlices, 8>(
        d, args, block_rows, out_backprop, input, filter_backprop, data_format);
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
bool TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall(
    const GpuDevice& d, const HexDepthwiseArgs args, const T* out_backprop,
    const T* input, T* filter_backprop, TensorFormat data_format) {
  // Maximize (power of two) kBlockSlices while keeping a block within 1024
  // threads (2 pixels per thread).
  int block_slices = 8;
  int block_rows = (args.in_rows + 1) / 2;
  int round_mask = 1;
  for (; block_slices > 1; block_slices /= 2) {
    // args.in_cols * block_rows * kBlockSlices must be multiple of 32.
    for (; block_rows * args.in_cols * block_slices & 31;
         round_mask = round_mask * 2 + 1) {
      block_rows = block_rows + round_mask & ~round_mask;
    }
    int block_size = block_rows * args.in_cols * block_slices;
    if (block_size <= 1024) {
      break;
    }
  }

  if (!CanLaunchHexDepthwiseConv2dBackpropFilterGPUSmall(args, block_rows)) {
    return false;
  }

  switch (block_slices) {
    case 8:
      return TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall<
          T, kKnownFilterWidth, kKnownFilterHeight, 8>(
          d, args, block_rows, out_backprop, input, filter_backprop,
          data_format);
    case 4:
      return TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall<
          T, kKnownFilterWidth, kKnownFilterHeight, 4>(
          d, args, block_rows, out_backprop, input, filter_backprop,
          data_format);
    case 2:
      return TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall<
          T, kKnownFilterWidth, kKnownFilterHeight, 2>(
          d, args, block_rows, out_backprop, input, filter_backprop,
          data_format);
    default:
      return false;
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
void LaunchHexDepthwiseConv2dBackpropFilterGPU(const GpuDevice& d,
                                            const HexDepthwiseArgs args,
                                            const T* out_backprop,
                                            const T* input, T* filter_backprop,
                                            TensorFormat data_format) {
  void (*kernel)(const HexDepthwiseArgs, const T*, const T*, T*, int);
  if (data_format == FORMAT_NHWC) {
    kernel = HexDepthwiseConv2DBackpropFilterGPUKernelNHWC<
        T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
  } else if (data_format == FORMAT_NCHW) {
    kernel = HexDepthwiseConv2DBackpropFilterGPUKernelNCHW<
        T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
  } else {
    assert(false && "Incorrect data format");
    return;
  }
  const int num_out_backprop =
      args.batch * args.out_rows * args.out_cols * args.out_depth;
  CudaLaunchConfig config =
      GetCudaLaunchConfig(num_out_backprop, d, kernel, 0, 0);
  kernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      args, out_backprop, input, filter_backprop, num_out_backprop);
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchHexDepthwiseConv2dBackpropFilterGPU(const GpuDevice& d,
                                            const HexDepthwiseArgs args,
                                            const T* out_backprop,
                                            const T* input, T* filter_backprop,
                                            TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    if (TryLaunchHexDepthwiseConv2dBackpropFilterGPUSmall<T, kKnownFilterWidth,
                                                       kKnownFilterHeight>(
            d, args, out_backprop, input, filter_backprop, data_format)) {
      return;
    }

    LaunchHexDepthwiseConv2dBackpropFilterGPU<T, kKnownFilterWidth,
                                           kKnownFilterHeight, 1>(
        d, args, out_backprop, input, filter_backprop, data_format);
  } else {
    LaunchHexDepthwiseConv2dBackpropFilterGPU<T, kKnownFilterWidth,
                                           kKnownFilterHeight, -1>(
        d, args, out_backprop, input, filter_backprop, data_format);
  }
}

// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
void LaunchHexDepthwiseConvBackpropFilterOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, const HexDepthwiseArgs& args, const T* out_backprop,
    const T* input, T* filter_backprop, TensorFormat data_format) {
  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  auto stream = ctx->op_device_context()->stream();

  // Initialize the results to 0.
  int num_filter_backprop =
      args.filter_rows * args.filter_cols * args.out_depth;
  perftools::gputools::DeviceMemoryBase filter_bp_ptr(filter_backprop,
                                                      num_filter_backprop);
  stream->ThenMemset32(&filter_bp_ptr, 0, num_filter_backprop * sizeof(T));

  if (args.filter_rows == 3 && args.filter_cols == 3) {
    LaunchHexDepthwiseConv2dBackpropFilterGPU<T, 3, 3>(
        d, args, out_backprop, input, filter_backprop, data_format);
  } else {
    LaunchHexDepthwiseConv2dBackpropFilterGPU<T, -1, -1>(
        d, args, out_backprop, input, filter_backprop, data_format);
  }
  OP_REQUIRES(ctx, stream->ok(),
              errors::Internal("Launch of gpu kernel for "
                               "HexDepthwiseConv2DBackpropFil"
                               "terGPULaunch failed"));
}

template struct LaunchHexDepthwiseConvBackpropFilterOp<GPUDevice, Eigen::half>;
template struct LaunchHexDepthwiseConvBackpropFilterOp<GPUDevice, float>;
template struct LaunchHexDepthwiseConvBackpropFilterOp<GPUDevice, double>;
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
