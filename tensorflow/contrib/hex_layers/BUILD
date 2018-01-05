# Description:
#   A Cudnn RNN wrapper.
#   APIs are meant to change over time.
package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_cc_binary",
    "tf_copts",
    "tf_cuda_library",
    "tf_opts_nortti_if_android",
    "tf_kernel_library",
    "tf_mkl_kernel_library",
    "cc_header_only_library",
    "if_not_windows",
)
load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl")

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_tests")
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_kernel_tests_linkstatic",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")


# --------------------------------------------------------------------------
tf_custom_op_library(
    name = "python/ops/_zero_out.so",
    srcs = [
        "kernels/kernel_example.cc",
        "kernels/kernel_example.h",
        "ops/zero_out.cc"
    ],
    gpu_srcs = [
        "kernels/kernel_example.cu.cc",
    ],
    deps = [
        "//tensorflow/core/kernels:bounds_check_lib",
    ],
)

tf_kernel_library(
    name = "hex_layers_kernels",
    srcs = [
        "kernels/kernel_example.cc",
    ],
    hdrs = [
        "kernels/kernel_example.h",
    ],
    gpu_srcs = [
        "kernels/kernel_example.h",
        "kernels/kernel_example.cu.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core/kernels:bounds_check_lib",
        "//third_party/eigen3",
    ],
)

# --------------------------------------------------------------------------
tf_custom_op_library(
    name = "python/ops/_hex_nn_ops.so",
    srcs = [
        "kernels/hex_depthwise_conv_op.cc",
        "kernels/hex_depthwise_conv_op.h",
        "ops/hex_nn_ops.cc",
    ],
    deps = [
        # ":hex_depthwise_convolution",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        # "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "hex_depthwise_convolution",
    srcs = [
        "kernels/hex_depthwise_conv_op.cc",
        "kernels/hex_depthwise_conv_grad_op.cc",
    ],
    hdrs = [
        "kernels/hex_depthwise_conv_op.h",
    ],
    gpu_srcs = [
        "kernels/hex_depthwise_conv_op.h",
        "kernels/hex_depthwise_conv_op_gpu.cu.cc",
    ],
    deps = [
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:conv_2d",
        "//tensorflow/core/kernels:conv_3d",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//third_party/eigen3",
    ] + if_cuda(["@cub_archive//:cub"]),
)

tf_gen_op_libs(
    op_lib_names = ["hex_nn_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)
tf_gen_op_wrapper_py(
    name = "hex_nn_ops",
    deps = [":hex_nn_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "hex_nn_py",
    srcs = [
        "__init__.py",
        "python/ops/hex_nn_ops.py",
        # "python/layers/__init__.py",
        # "python/layers/cudnn_rnn.py",
    ],
    dso = [
        ":python/ops/_hex_nn_ops.so",
    ],
    kernels = [
        ":hex_depthwise_convolution",
        ":hex_nn_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":hex_nn_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers_base",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

# tf_kernel_library(
#     name = "depthwise_conv_op",
#     prefix = "depthwise_conv_op",
#     deps = [
#         ":bounds_check",
#         ":conv_ops",
#         ":ops_util",
#         "//tensorflow/core:core_cpu",
#         "//tensorflow/core:framework",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:nn_ops_op_lib",
#     ] + if_cuda(["@cub_archive//:cub"]),
# )
# tf_kernel_library(
#     name = "depthwise_conv_grad_op",
#     hdrs = [
#         "depthwise_conv_op.h",
#     ],
#     prefix = "depthwise_conv_grad_op",
#     deps = [
#         ":bounds_check",
#         ":ops_util",
#         "//tensorflow/core:core_cpu",
#         "//tensorflow/core:framework",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:nn_ops_op_lib",
#     ],
# )

# --------------------------------------------------------------------------
# cuda_py_test(
#     name = "cudnn_rnn_ops_test",
#     size = "large",
#     srcs = ["python/kernel_tests/cudnn_rnn_ops_test.py"],
#     additional_deps = [
#         ":cudnn_rnn_py",
#         "//tensorflow/core:protos_all_py",
#         "//tensorflow/contrib/rnn:rnn_py",
#         "//tensorflow/python/ops/losses:losses",
#         "//tensorflow/python:array_ops",
#         "//tensorflow/python:client_testlib",
#         "//tensorflow/python:framework",
#         "//tensorflow/python:framework_for_generated_wrappers",
#         "//tensorflow/python:framework_test_lib",
#         "//tensorflow/python:math_ops",
#         "//tensorflow/python:platform_test",
#         "//tensorflow/python:random_ops",
#         "//tensorflow/python:state_ops",
#         "//tensorflow/python:training",
#         "//tensorflow/python:variables",
#     ],
#     shard_count = 6,
#     tags = [
#         "manual",
#         "requires_cudnn5",
#     ],
# )

# cuda_py_test(
#     name = "cudnn_rnn_test",
#     size = "enormous",
#     srcs = ["python/kernel_tests/cudnn_rnn_test.py"],
#     additional_deps = [
#         ":cudnn_rnn_py",
#         "//tensorflow/core:protos_all_py",
#         "//tensorflow/contrib/rnn:rnn_py",
#         "//tensorflow/python/ops/losses:losses",
#         "//tensorflow/python:array_ops",
#         "//tensorflow/python:client_testlib",
#         "//tensorflow/python:framework",
#         "//tensorflow/python:framework_for_generated_wrappers",
#         "//tensorflow/python:framework_test_lib",
#         "//tensorflow/python:math_ops",
#         "//tensorflow/python:platform_test",
#         "//tensorflow/python:random_ops",
#         "//tensorflow/python:state_ops",
#         "//tensorflow/python:training",
#         "//tensorflow/python:variables",
#     ],
#     shard_count = 6,
#     tags = [
#         "manual",
#         "requires_cudnn5",
#     ],
# )

# cuda_py_test(
#     name = "cudnn_rnn_ops_benchmark",
#     size = "small",
#     srcs = ["python/kernel_tests/cudnn_rnn_ops_benchmark.py"],
#     additional_deps = [
#         ":cudnn_rnn_py",
#         "//tensorflow/contrib/rnn:rnn_py",
#         "//tensorflow/python:array_ops",
#         "//tensorflow/python:client",
#         "//tensorflow/python:client_testlib",
#         "//tensorflow/python:control_flow_ops",
#         "//tensorflow/python:framework_for_generated_wrappers",
#         "//tensorflow/python:framework_test_lib",
#         "//tensorflow/python:gradients",
#         "//tensorflow/python:init_ops",
#         "//tensorflow/python:platform",
#         "//tensorflow/python:platform_test",
#         "//tensorflow/python:variables",
#     ],
#     tags = [
#         "noasan",  # http://b/62067814
#         "nomsan",
#         "notsan",
#         "requires_cudnn5",
#     ],
# )

# tf_cc_test(
#     name = "cudnn_rnn_ops_test_cc",
#     size = "small",
#     srcs = [
#         "ops/cudnn_rnn_ops_test.cc",
#     ],
#     deps = [
#         ":cudnn_rnn_ops_op_lib",
#         "//tensorflow/core",
#         "//tensorflow/core:framework",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:test",
#         "//tensorflow/core:test_main",
#         "//tensorflow/core:testlib",
#     ],
# )

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
