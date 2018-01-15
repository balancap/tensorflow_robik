# =========================================================================== #
# [2017] - Robik AI Ltd - Paul Balanca
# All Rights Reserved.

# NOTICE: All information contained herein is, and remains
# the property of Robik AI Ltd, and its suppliers
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Robik AI Ltd
# and its suppliers and may be covered by U.S., European and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Robik AI Ltd.
# =========================================================================== #
"""Collection of hexagonal convolutional layers.
"""
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.ops import variables as tf_variables

# =========================================================================== #
# Depthwise convolution 2d with data format option.
# =========================================================================== #
@add_arg_scope
def hex_depthwise_convolution2d(
        inputs,
        kernel_size,
        depth_multiplier=1,
        stride=1,
        padding='SAME',
        rate=1,
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        data_format='NHWC',
        scope=None):
    """Adds a depthwise 2D convolution with optional batch_norm layer.
    This op performs a depthwise convolution that acts separately on
    channels, creating a variable called `depthwise_weights`. Then,
    if `normalizer_fn` is None,
    it adds bias to the result, creating a variable called 'biases', otherwise,
    the `normalizer_fn` is applied. It finally applies an activation function
    to produce the end result.
    Args:
        inputs: A tensor of size [batch_size, height, width, channels].
        num_outputs: The number of pointwise convolution output filters. If is
          None, then we skip the pointwise convolution stage.
        kernel_size: A list of length 2: [kernel_height, kernel_width] of
          of the filters. Can be an int if both values are the same.
        depth_multiplier: The number of depthwise convolution output channels for
          each input channel. The total number of depthwise convolution output
          channels will be equal to `num_filters_in * depth_multiplier`.
        stride: A list of length 2: [stride_height, stride_width], specifying the
          depthwise convolution stride. Can be an int if both strides are the same.
        padding: One of 'VALID' or 'SAME'.
        rate: A list of length 2: [rate_height, rate_width], specifying the dilation
          rates for atrous convolution. Can be an int if both rates are the same.
          If any value is larger than one, then both stride values need to be one.
        activation_fn: Activation function. The default value is a ReLU function.
          Explicitly set it to None to skip it and maintain a linear activation.
        normalizer_fn: Normalization function to use instead of `biases`. If
          `normalizer_fn` is provided then `biases_initializer` and
          `biases_regularizer` are ignored and `biases` are not created nor added.
          default set to None for no normalizer function
        normalizer_params: Normalization function parameters.
        weights_initializer: An initializer for the weights.
        weights_regularizer: Optional regularizer for the weights.
        biases_initializer: An initializer for the biases. If None skip biases.
        biases_regularizer: Optional regularizer for the biases.
        reuse: Whether or not the layer and its variables should be reused. To be
          able to reuse the layer scope must be given.
        variables_collections: Optional list of collections for all the variables or
          a dictionary containing a different list of collection per variable.
        outputs_collections: Collection to add the outputs.
        trainable: Whether or not the variables should be trainable or not.
        scope: Optional scope for variable_scope.
    Returns:
        A `Tensor` representing the output of the operation.
    """
    with variable_scope.variable_scope(scope, 'HexDepthwiseConv2d', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        # Actually apply depthwise conv instead of separable conv.
        dtype = inputs.dtype.base_dtype
        kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
        stride_h, stride_w = utils.two_element_tuple(stride)
        if data_format == 'NHWC':
            num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
            strides = [1, stride_h, stride_w, 1]
        else:
            # No NCHW for now...
            raise NotImplementedError()
            # num_filters_in = inputs.get_shape().as_list()[1]
            # strides = [1, 1, stride_h, stride_w]

        weights_collections = utils.get_variable_collections(
            variables_collections, 'weights')

        # Depthwise weights variable.
        depthwise_shape = [kernel_h, kernel_w,
                           num_filters_in, depth_multiplier]
        depthwise_weights = variables.model_variable(
            'hex_depthwise_weights',
            shape=depthwise_shape,
            dtype=dtype,
            initializer=weights_initializer,
            regularizer=weights_regularizer,
            trainable=trainable,
            collections=weights_collections)

        outputs = nn.hex_depthwise_conv2d(inputs,
                                          depthwise_weights,
                                          strides,
                                          padding,
                                          rate=utils.two_element_tuple(rate),
                                          data_format=data_format)
        num_outputs = depth_multiplier * num_filters_in

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases_collections = utils.get_variable_collections(
                    variables_collections, 'biases')
                biases = variables.model_variable('biases',
                                                  shape=[num_outputs,],
                                                  dtype=dtype,
                                                  initializer=biases_initializer,
                                                  regularizer=biases_regularizer,
                                                  trainable=trainable,
                                                  collections=biases_collections)
                outputs = nn.bias_add(outputs, biases, data_format=data_format)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)
hex_depthwise_conv2d = hex_depthwise_convolution2d
hex_dw_conv2d = hex_depthwise_convolution2d
