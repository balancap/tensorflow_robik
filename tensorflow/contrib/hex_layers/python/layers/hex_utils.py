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


