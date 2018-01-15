# Description...
package(
    default_visibility = ["//tensorflow:__subpackages__"]
)

licenses(["notice"])  # Apache 2.0
exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "py_test"
)

# --------------------------------------------------------------------------
py_library(
    name = "hex_layers_py",
    srcs = [
        "__init__.py",
        # "python/losses/__init__.py",
        # "python/losses/loss_ops.py",
        # "python/metric_learning/metric_loss_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:summary",
        "//tensorflow/python:util",
    ],
)
# --------------------------------------------------------------------------

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
