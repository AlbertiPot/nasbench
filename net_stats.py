"""
Data:2021/08/26
Target: 
计算数据集中全部子网的FLOPs和parameters，并用parameters与官方数据库进行校验

"""

import tensorflow as tf

from nasbench.lib import base_ops
from nasbench.lib import training_time
from nasbench import api
import numpy as np

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


def build_module(spec, inputs, channels, is_training):
    """Build a custom module using a proposed model spec.

  Builds the model using the adjacency matrix and op labels specified. Channels
  controls the module output channel count but the interior channels are
  determined via equally splitting the channel count whenever there is a
  concatenation of Tensors.

  Args:
    spec: ModelSpec object.
    inputs: input Tensors to this module.
    channels: output channel count.
    is_training: bool for whether this model is training.

  Returns:
    output Tensor from built module.

  Raises:
    ValueError: invalid spec
  """
    num_vertices = np.shape(spec.matrix)[0]

    if spec.data_format == 'channels_last':
        channel_axis = 3
    elif spec.data_format == 'channels_first':
        channel_axis = 1
    else:
        raise ValueError('invalid data_format')

    input_channels = inputs.get_shape()[channel_axis].value
    # vertex_channels[i] = number of output channels of vertex i
    vertex_channels = compute_vertex_channels(input_channels, channels,
                                              spec.matrix)

    # Construct tensors from input forward
    tensors = [tf.identity(inputs, name='input')]

    final_concat_in = []
    for t in range(1, num_vertices - 1):
        with tf.variable_scope('vertex_{}'.format(t)):
            # Create interior connections, truncating if necessary
            add_in = [
                truncate(tensors[src], vertex_channels[t], spec.data_format)
                for src in range(1, t) if spec.matrix[src, t]
            ]

            # Create add connection from projected input
            if spec.matrix[0, t]:
                add_in.append(
                    projection(tensors[0], vertex_channels[t], is_training,
                               spec.data_format))

            if len(add_in) == 1:
                vertex_input = add_in[0]
            else:
                vertex_input = tf.add_n(add_in)

            # Perform op at vertex t
            op = base_ops.OP_MAP[spec.ops[t]](is_training=is_training,
                                              data_format=spec.data_format)
            vertex_value = op.build(vertex_input, vertex_channels[t])

        tensors.append(vertex_value)
        if spec.matrix[t, num_vertices - 1]:
            final_concat_in.append(tensors[t])

    # Construct final output tensor by concating all fan-in and adding input.
    if not final_concat_in:
        # No interior vertices, input directly connected to output
        assert spec.matrix[0, num_vertices - 1]
        with tf.variable_scope('output'):
            outputs = projection(tensors[0], channels, is_training,
                                 spec.data_format)

    else:
        if len(final_concat_in) == 1:
            outputs = final_concat_in[0]
        else:
            outputs = tf.concat(final_concat_in, channel_axis)

        if spec.matrix[0, num_vertices - 1]:
            outputs += projection(tensors[0], channels, is_training,
                                  spec.data_format)

    outputs = tf.identity(outputs, name='output')
    return outputs


def projection(inputs, channels, is_training, data_format):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    with tf.variable_scope('projection'):
        net = base_ops.conv_bn_relu(inputs, 1, channels, is_training,
                                    data_format)

    return net


def truncate(inputs, channels, data_format):
    """Slice the inputs to channels if necessary."""
    if data_format == 'channels_last':
        input_channels = inputs.get_shape()[3].value
    else:
        assert data_format == 'channels_first'
        input_channels = inputs.get_shape()[1].value

    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs  # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        if data_format == 'channels_last':
            return tf.slice(inputs, [0, 0, 0, 0], [-1, -1, -1, channels])
        else:
            return tf.slice(inputs, [0, 0, 0, 0], [-1, channels, -1, -1])


def compute_vertex_channels(input_channels, output_channels, matrix):
    """Computes the number of channels at every vertex.

  Given the input channels and output channels, this calculates the number of
  channels at each interior vertex. Interior vertices have the same number of
  channels as the max of the channels of the vertices it feeds into. The output
  channels are divided amongst the vertices that are directly connected to it.
  When the division is not even, some vertices may receive an extra channel to
  compensate.

  Args:
    input_channels: input channel count.
    output_channels: output channel count.
    matrix: adjacency matrix for the module (pruned by model_spec).

  Returns:
    list of channel counts, in order of the vertices.
  """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices -
                                             1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v],
                                             vertex_channels[dst])
        assert vertex_channels[v] > 0

    tf.logging.info('vertex_channels: %s', str(vertex_channels))

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


if __name__ == '__main__':
    spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix=[
            [0, 1, 1, 1, 0, 1, 0],  # input layer
            [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
            [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
            [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
            [0, 0, 0, 0, 0, 0, 0]
        ],  # output layer
        # Operations at the vertices of the module, matches order of matrix
        ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

    config = {}
    # num_train_images = {}

    config['stem_filter_size'] = 128
    config['data_format'] = 'channels_last'
    config['num_stacks'] = 3
    config['num_modules_per_stack'] = 3
    config['num_labels'] = 10
    
    if config['data_format'] == 'channels_last':  # NHWC
        channel_axis = 3
    elif config['data_format'] == 'channels_first':
        # Currently this is not well supported
        channel_axis = 1
    else:
        raise ValueError('invalid data_format')

    assert spec.data_format == config['data_format'],'Wrong ModelSpec data_formar'

    new_graph = tf.Graph()
    with new_graph.as_default():
        # Store auxiliary activations increasing in depth of network. First
        # activation occurs immediately after the stem and the others immediately
        # follow each stack.
        aux_activations = []

        is_training = False

        
        features = tf.placeholder(tf.float32, [1, 32 * 32 * 3], name='features') # 32*32*3
        f_reshape = tf.reshape(features, [1, 32, 32, 3])  # NHWC

        # Initial stem convolution
        with tf.variable_scope('stem'):
            net = base_ops.conv_bn_relu(f_reshape, 3,
                                        config['stem_filter_size'],
                                        is_training, config['data_format'])
            aux_activations.append(net)

        for stack_num in range(config['num_stacks']):
            channels = net.get_shape()[channel_axis].value

            # Downsample at start (except first)
            if stack_num > 0:
                net = tf.layers.max_pooling2d(
                    inputs=net,
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding='same',
                    data_format=config['data_format'])

                # Double output channels each time we downsample
                channels *= 2

            with tf.variable_scope('stack{}'.format(stack_num)):
                for module_num in range(config['num_modules_per_stack']):
                    with tf.variable_scope('module{}'.format(module_num)):
                        net = build_module(spec,
                                           inputs=net,
                                           channels=channels,
                                           is_training=is_training)
                aux_activations.append(net)

        # Global average pool
        if config['data_format'] == 'channels_last':
            net = tf.reduce_mean(net, [1, 2])
        elif config['data_format'] == 'channels_first':
            net = tf.reduce_mean(net, [2, 3])
        else:
            raise ValueError('invalid data_format')

        # Fully-connected layer to labels
        logits = tf.layers.dense(inputs=net, units=config['num_labels'])

    # with tf.Session(graph=tf.get_default_graph()) as sess:
    #     print(sess.run(logits))

    with tf.Session(graph=new_graph) as sess:
        sess.run(tf.global_variables_initializer())
        x =np.random.rand(1,32 * 32 * 3)
        print(sess.run(logits, feed_dict={features: x}))