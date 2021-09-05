"""
Data:2021/09/01
Target: 
在得到模型总可训参数量和FLOPS的基础上，进一步获得模型每个节点的可训练参数量和FLOPS

模型的创建参考了nasbench api
网络参数量的计算参考了https://www.cnblogs.com/o-v-o/p/11042066.html
"""
import os
import json

import operator
import numpy as np
import tensorflow as tf

from nasbench.lib import base_ops
from nasbench.lib import model_spec
from nasbench import api
# from tensorflow.python.framework import graph_util

# os.environ["CUDA_VISIBLE_DEVICES"]='2'

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
        with tf.compat.v1.variable_scope('vertex_{}'.format(t)):
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
    with tf.compat.v1.variable_scope('projection'):
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

    tf.compat.v1.logging.info('vertex_channels: %s', str(vertex_channels))

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

def stats_graph(graph,cmd,is_flops):
    if is_flops:
        flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(),cmd = cmd)
        return flops.total_float_ops
    else:
        params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter(),cmd=cmd)
        return params.total_parameters
    # print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def build_model(spec, config, img_size=32, img_channel=3, is_training = False):

    if config['data_format'] == 'channels_last':  # NHWC
        channel_axis = 3
    elif config['data_format'] == 'channels_first':
        # Currently this is not well supported
        channel_axis = 1
    else:
        raise ValueError('invalid data_format')

    assert spec.data_format == config['data_format'],'Wrong ModelSpec data_format'

    new_graph = tf.Graph()
    with new_graph.as_default():
        features = tf.compat.v1.placeholder(tf.float32, [1, img_size * img_size * img_channel], name='features') # 32*32*3
        f_reshape = tf.reshape(features, [1, img_size, img_size, img_channel])  # NHWC

        # Initial stem convolution
        with tf.compat.v1.variable_scope('stem'):
            net = base_ops.conv_bn_relu(f_reshape, 3,
                                        config['stem_filter_size'],
                                        is_training, config['data_format'])

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

            with tf.compat.v1.variable_scope('stack{}'.format(stack_num)):
                for module_num in range(config['num_modules_per_stack']):
                    with tf.compat.v1.variable_scope('module{}'.format(module_num)):
                        net = build_module(spec,
                                           inputs=net,
                                           channels=channels,
                                           is_training=is_training)

        # Global average pool
        if config['data_format'] == 'channels_last':
            net = tf.reduce_mean(net, [1, 2])
        elif config['data_format'] == 'channels_first':
            net = tf.reduce_mean(net, [2, 3])
        else:
            raise ValueError('invalid data_format')

        # Fully-connected layer to labels
        logits = tf.layers.dense(inputs=net, units=config['num_labels'])

    return new_graph

def vertex_params(spec, config):
    num_vertices = np.shape(spec.matrix)[0]
    cell_params_dict = {}
    for stack_num in range(config['num_stacks']):
            for module_num in range(config['num_modules_per_stack']):
                
                cell_params = [0]*num_vertices
                
                for t in range(1, num_vertices - 1):
                    vertex_params = 0
                    vertex_name ='stack{}/module{}/vertex_{}'.format(stack_num, module_num, t)
                    
                    if tf.compat.v1.trainable_variables(scope = vertex_name) == []:                     # 判断某节点下是否有可训练的参数，如无（返回空列表），continue继续遍历下一个vertex
                        print('{}_params : {}'.format(vertex_name,vertex_params))
                        continue
                    
                    for trainable_variable in tf.compat.v1.trainable_variables(scope = vertex_name):
                        ops_params = 1
                        for dim in trainable_variable.shape:
                            ops_params *= dim.value
                        vertex_params += ops_params
                    cell_params[t] = vertex_params
                    print('{}_params : {}'.format(vertex_name,vertex_params))            
                
                cell_params_dict['stack{}/module{}'.format(stack_num, module_num)] =  cell_params
        
    print(cell_params_dict)
    return cell_params_dict

def compute_params_flops(dataset, nasbench, config):
    
    for i, subnet in enumerate(dataset):
            matrix=[]
            ops = []

            matrix = subnet['module_adjacency']
            ops = subnet['module_operations']
            n_params = subnet['trainable_parameters']
            unique_hash = subnet['unique_hash']
            
            fixed_metrics, _ = nasbench.get_metrics_from_hash(unique_hash)
            assert ((np.array(matrix,dtype=fixed_metrics['module_adjacency'].dtype) if not isinstance(matrix, np.ndarray) else matrix) == fixed_metrics['module_adjacency']).all(), 'Wrong Adjacency'
            assert operator.eq(ops, fixed_metrics['module_operations']),'Wrong ops'
            
            spec = model_spec.ModelSpec(matrix, ops)
            new_graph = build_model(spec, config, img_size=32, img_channel=3, is_training=False)
        
            # print('stats before freezing')
            params = stats_graph(new_graph, cmd = 'scope', is_flops=False)
        
            with tf.Session(graph=new_graph) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
 
                vertex_params_dict = vertex_params(spec, config)
                assert len(vertex_params_dict) == config['num_stacks']*config['num_modules_per_stack'] , 'Wrong cell counts'
                assert len(vertex_params_dict[list(vertex_params_dict.keys())[0]]) == np.shape(spec.matrix)[0] == len(vertex_params_dict[list(vertex_params_dict.keys())[-1]]), 'Wrong cell opts length'
                dataset[i]['vertex_params'] = vertex_params_dict


                # print([n.name for n in tf.get_default_graph().as_graph_def().node])   # find last node in the graph
                output_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, new_graph.as_graph_def(),['dense/BiasAdd']) # freezing parameters

                file_name = 'tmp_output/graph.pb'
                if os.path.exists(file_name):
                    os.remove(file_name)
                
                with tf.gfile.GFile(file_name, "wb") as q:
                    q.write(output_graph.SerializeToString())
                
                # test the ConvNet is run
                # x = np.random.rand(1,32 * 32 * 3)
                # print(sess.run(logits, feed_dict={features: x}))  
                
                # Another way to calculate the params
                # params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
                # print(params)        
            
            graph = load_pb(file_name)
            # print('stats after freezing')
            flops= stats_graph(graph,cmd='scope',is_flops=True)
        
            assert fixed_metrics['trainable_parameters'] == params == dataset[i]['trainable_parameters'] == n_params, 'Wrong calculated parames'
            dataset[i]['flops'] = flops
            assert len(dataset[i]) == 10
            
            del new_graph, graph, output_graph

            # break
    assert len(dataset) == 423624
    return dataset 

def test(config):
    
    dataset = {
        'module_adjacency': [[0, 1, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]], 
      
        'module_operations': ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output'], 
      
        'trainable_parameters': 8555530
    }

    matrix = dataset['module_adjacency']
    ops = dataset['module_operations']
    n_params = dataset['trainable_parameters']

    spec = model_spec.ModelSpec(matrix, ops)

    new_graph = build_model(spec, config, img_size=32, img_channel=3, is_training=False)
    new_graph.as_default()
    # params = stats_graph(new_graph,cmd = 'graph',is_flops=False)
    # assert n_params == params
    
    # import pdb;pdb.set_trace()
    num_vertices = np.shape(spec.matrix)[0]
   
    
    with tf.Session(graph=new_graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        cell_params_dict = {}
        
        for stack_num in range(config['num_stacks']):
            for module_num in range(config['num_modules_per_stack']):
                
                cell_params = [0]*num_vertices
                
                for t in range(1, num_vertices - 1):
                    vertex_params = 0
                    vertex_name ='stack{}/module{}/vertex_{}'.format(stack_num, module_num, t)
                    
                    if tf.compat.v1.trainable_variables(scope = vertex_name) == []:
                        # print('{}_params : {}'.format(vertex_name,vertex_params))
                        continue
                    
                    for trainable_variable in tf.compat.v1.trainable_variables(scope = vertex_name):
                        
                        ops_params = 1
                        for dim in trainable_variable.shape:
                            ops_params *= dim.value
                        vertex_params += ops_params
                    cell_params[t] = vertex_params
                    # print('{}_params : {}'.format(vertex_name,vertex_params))                
                
                cell_params_dict['stack{}/module{}'.format(stack_num, module_num)] =  cell_params
        
        print(cell_params_dict)
    
    
    
    # with tf.Session(graph=new_graph) as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
            
    #     print([n.name for n in tf.get_default_graph().as_graph_def().node])   # find last node in the graph
    #     output_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, new_graph.as_graph_def(),['dense/BiasAdd']) # freezing parameters

    #     file_name = 'tmp_output/graph_test.pb'
    #     if os.path.exists(file_name):
    #         os.remove(file_name)
                
    #     with tf.gfile.GFile(file_name, "wb") as q:
    #         q.write(output_graph.SerializeToString())      
            
    # graph = load_pb(file_name)
    # print('stats after freezing')
    # flops= stats_graph(new_graph,cmd='graph',is_flops=True)
    # print(params, flops)

if __name__ == '__main__':

    config = {}
    config['stem_filter_size'] = 128
    config['data_format'] = 'channels_last'
    config['num_stacks'] = 3
    config['num_modules_per_stack'] = 3
    config['num_labels'] = 10

    # test(config)

    save_file = '/home/ubuntu/workspace/nasbench/tmp_data/423flops.json'
    if os.path.exists(save_file):
        os.remove(save_file)
    
    data_path = '/home/ubuntu/workspace/nasbench/data/nasbench_only108_423.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    f.close()
    # assert len(dataset) == 423624,"the length of the extracted json dataset is not 423624"
    assert len(dataset) == 423,"the length of the extracted json dataset is not 423"
    
    origin_dataset_file = '/home/ubuntu/workspace/nasbench/data/nasbench_only108.tfrecord'
    nasbench = api.NASBench(dataset_file=origin_dataset_file)
    assert len(nasbench.fixed_statistics) == len(nasbench.computed_statistics) == 423624, "Wrong length of the original dataset"

    dataset = compute_params_flops(dataset, nasbench, config)

    # assert len(dataset) == 423624
    assert len(dataset) == 423
    with open(save_file, 'w') as r:
            json.dump(dataset, r)
    r.close()
    
    print('all ok!!!!!!!!!!!!!')


    
    

        