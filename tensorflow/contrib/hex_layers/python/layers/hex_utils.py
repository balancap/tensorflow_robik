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
# Hex. layers / utils
# =========================================================================== #
@add_arg_scope
def hex_downscale2d(
        inputs,
        rate=2,
        outputs_collections=None,
        reuse=None,
        trainable=True,
        data_format='NHWC',
        scope=None):
    """Downscale a hex-tensor by a factor 2.

    Returns:
        A `Tensor` representing the output of the operation.
    """
    with variable_scope.variable_scope(scope, 'HexDownscale2D', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        padding='SAME'
        # Actually apply depthwise conv instead of separable conv.
        dtype = inputs.dtype.base_dtype
        if data_format == 'NHWC':
            num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
            strides = [1, rate, rate, 1]
        else:
            # No NCHW for now...
            raise NotImplementedError()
            # num_filters_in = inputs.get_shape().as_list()[1]
            # strides = [1, 1, stride_h, stride_w]

        # Empty weights. To remove!
        depthwise_shape = [1, 1, num_filters_in, 1]
        zero_weights = tf.zeros(
            shape=depthwise_shape, dtype=dtype, name='zero_weights')
        outputs = nn.hex_depthwise_conv2d(inputs,
                                          zero_weights,
                                          strides,
                                          padding,
                                          rate=utils.two_element_tuple(1),
                                          data_format=data_format)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


@add_arg_scope
def hex_from_cartesian(
        inputs,
        method=tf.image.ResizeMethod.BILINEAR,
        downscale=True,
        extend=True,
        outputs_collections=None,
        reuse=None,
        data_format='NHWC',
        scope=None):
    """Transform a cartesian Tensor to an hexagonal Tensor (NHWC input format).

    The width of the tensor is extended to reflect the different geometry of
    hexagonal tiling.
    """
    with variable_scope.variable_scope(scope, 'HexFromCartesian', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        # New shape: expand width to reflect the different geometry.
        shape = inputs.get_shape().as_list()
        # Output shape?
        out_height = int(shape[1])
        out_width = int(shape[2])
        if downscale:
            # Check width is divisible by two.
            out_width = (out_width // 2) * 2
        else:
            out_height = int(shape[1] * 2)
            out_width = int(shape[2] * 2)
            # Factor sqrt(3) / 2 * a due to hexagonal tiling.
            if extend:
                # out_height = (int(shape[1] * 2 / 1.15470053838) // 2) * 2
                out_width = (int(shape[2] * 2 * 1.15470053838) // 2) * 2
        # Resize input tensor.
        outputs = tf.image.resize_images(
            inputs,
            [out_height, out_width],
            method=method,
            align_corners=False)
        # Un-stack / re-stack
        h_axis = 1
        l_outputs = tf.unstack(outputs, axis=h_axis)
        l_outputs_sub = []
        for i, v in enumerate(l_outputs):
            # sub-sample
            if i % 4 == 0:
                l_outputs_sub.append(v[:, 0::2])
            elif i % 4 == 2:
                l_outputs_sub.append(v[:, 1::2])
        h_axis = 1
        outputs = tf.stack(l_outputs_sub, axis=h_axis)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)
