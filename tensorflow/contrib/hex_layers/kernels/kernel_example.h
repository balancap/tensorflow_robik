// kernel_example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

// #include <iostream>
// #include <Eigen/Dense>
// #include <Eigen/CXX11/Tensor>

// #define EIGEN_USE_GPU

// #include "tensorflow/core/kernels/strided_slice_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// #include "tensorflow/core/framework/tensor_types.h"

// #include "tensorflow/core/framework/register_types.h"

// #include "tensorflow/core/framework/register_types.h"
// #include "tensorflow/core/framework/tensor_types.h"
// #include "tensorflow/core/platform/types.h"


template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

typedef Eigen::GpuDevice GPUDevice;

// #if GOOGLE_CUDA
// // Partially specialize functor for GpuDevice.
// template <Eigen::GpuDevice, typename T>
// struct ExampleFunctor {
//   void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
// };
// #endif
#endif KERNEL_EXAMPLE_H_
