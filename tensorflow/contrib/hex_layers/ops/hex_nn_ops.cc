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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// --------------------------------------------------------------------------
REGISTER_OP("HexDepthwiseConv2dNative")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape)
    .Doc(R"doc(
Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, channel_multiplier]`, containing
`in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
a different filter to each input channel (expanding from 1 channel to
`channel_multiplier` channels for each), then concatenates the results
together. Thus, the output has `in_channels * channel_multiplier` channels.

```
for k in 0..in_channels-1
  for q in 0..channel_multiplier-1
    output[b, i, j, k * channel_multiplier + q] =
      sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                        filter[di, dj, k, q]
```

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
strides: 1-D of length 4.  The stride of the sliding window for each dimension
  of `input`.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, height, width, channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
  `input`. If set to k > 1, there will be k-1 skipped cells between each filter
  element on that dimension. The dimension order is determined by the value of
  `data_format`, see above for details. Dilations in the batch and depth
  dimensions must be 1.
)doc");

REGISTER_OP("HexDepthwiseConv2dNativeBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradients of depthwise convolution with respect to the input.

input_sizes: An integer vector representing the shape of `input`, based
  on `data_format`.  For example, if `data_format` is 'NHWC' then
   `input` is a 4-D `[batch, height, width, channels]` tensor.
filter: 4-D with shape
  `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
out_backprop: 4-D with shape  based on `data_format`.
  For example, if `data_format` is 'NHWC' then
  out_backprop shape is `[batch, out_height, out_width, out_channels]`.
  Gradients w.r.t. the output of the convolution.
strides: The stride of the sliding window for each dimension of the input
  of the convolution.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, height, width, channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
  `input`. If set to k > 1, there will be k-1 skipped cells between each filter
  element on that dimension. The dimension order is determined by the value of
  `data_format`, see above for details. Dilations in the batch and depth
  dimensions must be 1.
output: 4-D with shape according to `data_format`.  For example, if
  `data_format` is 'NHWC', output shape is `[batch, in_height,
  in_width, in_channels]`.  Gradient w.r.t. the input of the
  convolution.
)doc");

REGISTER_OP("HexDepthwiseConv2dNativeBackpropFilter")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradients of depthwise convolution with respect to the filter.

input: 4-D with shape based on `data_format`.  For example, if
  `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
  in_width, in_channels]` tensor.
filter_sizes: An integer vector representing the tensor shape of `filter`,
  where `filter` is a 4-D
  `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
out_backprop: 4-D with shape  based on `data_format`.
  For example, if `data_format` is 'NHWC' then
  out_backprop shape is `[batch, out_height, out_width, out_channels]`.
  Gradients w.r.t. the output of the convolution.
strides: The stride of the sliding window for each dimension of the input
  of the convolution.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, height, width, channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
  `input`. If set to k > 1, there will be k-1 skipped cells between each filter
  element on that dimension. The dimension order is determined by the value of
  `data_format`, see above for details. Dilations in the batch and depth
  dimensions must be 1.
output: 4-D with shape
  `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
  the `filter` input of the convolution.
)doc");

}  // namespace tensorflow
