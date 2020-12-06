# credit: this code is derived from https://github.com/snowkylin/ntm
# the major changes made are to make this compatible with the abstract class tf.contrib.rnn.RNNCell
# an LSTM controller is used instead of a RNN controller
# 3 memory inititialization schemes are offered instead of 1
# the outputs of the controller heads are clipped to an absolute value
# we find that our modification result in more reliable training (we never observe gradients going to NaN) and faster convergence

import numpy as np
import tensorflow as tf
import collections
from utils import expand, learned_init, create_linear_initializer

from tensorflow.python.util import nest


NTMControllerState = collections.namedtuple('NTMControllerState', ('controller_state', 'read_vector_list', 'w_list', 'M'))

class NTMCell(tf.keras.layers.Layer):
    def __init__(self,output_dim, controller_layers, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num,
                  shift_range=1,  clip_value=20,
                 init_mode='constant', **kwargs):
        super().__init__(NTMCell, **kwargs)
        self.controller_layers = controller_layers
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.clip_value = clip_value
        self.shift_range = shift_range
        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1 #  [k] + beta + g + [s] + gamma
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num # the latter part represents the Erase and Add vectors
        self.controller =tf.keras.layers.RNN([tf.keras.layers.LSTMCell(controller_units, unit_forget_bias=True) for _ in range(controller_layers)])
        self.output_dim = output_dim
        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(self.controller_units + self.memory_vector_dim * self.read_head_num)
        self.controller_interface = tf.keras.layers.Dense(
                total_parameter_num, activation=None,
                kernel_initializer=self.o2p_initializer,name='controller_to_interface')
        self.final_join = tf.keras.layers.Dense(
                output_dim, activation=None,
                kernel_initializer=self.o2o_initializer)
    
    def call(self, x, prev_state):
        with tf.name_scope("inputs_to_controller"):
            prev_state = nest.pack_sequence_as(self.state_size_nested, prev_state) # prev state is received as a sequence, make a structure of it
            prev_read_vector_list = prev_state.read_vector_list 
            controller_input = tf.concat([x] + prev_read_vector_list, axis=1)  # controler receives the input sequence itself plus the read content from the past state

            controller_output, controller_state = self.controller(controller_input, prev_state.controller_state)

        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1 #  k + beta + g + s + gamma
        num_heads = self.read_head_num + self.write_head_num
        with tf.name_scope("parse_interface"):
            parameters = self.controller_interface(controller_output)
            parameters = tf.clip_by_value(parameters, -self.clip_value, self.clip_value)
        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1)

        prev_w_list = prev_state.w_list
        prev_M = prev_state.M
        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = tf.nn.softplus(head_parameter[:, self.memory_vector_dim])
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            )
            gamma = tf.nn.softplus(head_parameter[:, -1]) + 1
            with tf.name_scope('addressing_head_%d' % i ):
                w = self.addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])
            w_list.append(w)

            # Reading (Sec 3.1)

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            with tf.name_scope('reading_head%d' % i ):
                read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], axis=2) * prev_M, axis=1)
                read_vector_list.append(read_vector)

        # Writing (Sec 3.2)

        write_w_list = w_list[self.read_head_num:]
        for i in range(self.write_head_num):
            with tf.name_scope('writing_head%d' % i ):
                w = tf.expand_dims(write_w_list[i], axis=2)
                erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
                add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
                M = prev_M * (tf.ones([self.memory_size,self.memory_vector_dim]) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)


        with tf.name_scope("join_outputs"):
            if not self.output_dim:
                output_dim = x.get_shape()[1]
            else:
                output_dim = self.output_dim
            NTM_output = self.final_join(tf.concat([controller_output] + read_vector_list, axis=1))
            NTM_output = tf.clip_by_value(NTM_output, -self.clip_value, self.clip_value)

  
        return NTM_output, nest.flatten(NTMControllerState(
            controller_state=controller_state, read_vector_list=read_vector_list, w_list=w_list, M=M))

    @tf.function
    def addressing(self, k, beta, g, s, gamma, prev_M, prev_w):

        # Sec 3.3.1 Focusing by Content

        # Cosine Similarity

        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(prev_M, k)  # multiply previous memory with key to obtain addressed cells (cotient of cosine similarity)
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keepdims=True)) # Vector 2norm
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keepdims=True)) # Vector 2norm
        norm_product = M_norm * k_norm # divisor of cosine similarity
        K = tf.squeeze(inner_product / (norm_product + 1e-8))                   # eq (6) # cosine similarity
        # Calculating w^c
        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keepdims=True)  # eq (5)
        # Sec 3.3.2 Focusing by Location
        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w                                        # eq (7)

        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            axis=1
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      # eq (8)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keepdims=True)        # eq (9)

        return w


    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        if batch_size is None and inputs is not None:
          batch_size = inputs.batch_size
        read_vector_list = [expand(tf.tanh(learned_init(self.memory_vector_dim),name="read_vector"), dim=0, N=batch_size)
            for i in range(self.read_head_num)]
        w_list = [expand(tf.nn.softmax(learned_init(self.memory_size)), dim=0, N=batch_size,name="w")
            for i in range(self.read_head_num + self.write_head_num)]
        controller_init_state = self.controller.get_initial_state(batch_size=batch_size, dtype=dtype)
        M = tf.repeat(tf.expand_dims(tf.fill([ self.memory_size, self.memory_vector_dim], 1e-6),axis=0),batch_size,axis=0,name="M")
        init_state = NTMControllerState(
            controller_state=controller_init_state,
            read_vector_list=read_vector_list,
            w_list=w_list,
            M=M)
        return nest.flatten(init_state)

    @property
    def output_size(self):
        return self.output_dim

    @property
    def state_size_nested(self):
        return NTMControllerState(
            controller_state=self.controller.state_size,
            read_vector_list=[self.memory_vector_dim for _ in range(self.read_head_num)],
            w_list=[self.memory_size for _ in range(self.read_head_num + self.write_head_num)],
            M=tf.TensorShape([self.memory_size,self.memory_vector_dim])
            )

    @property
    def state_size(self):
        return nest.flatten(self.state_size_nested)


    def get_config(self):
        config = {
            'controller': {
                'class_name': self.controller.__class__.__name__,
                'config': self.controller.get_config()
            },
            'memory': {
                'memory_size': self.memory_size,
                'memory_vector_dim': self.memory_vector_dim,
                'read_head_num': self.read_head_num,
                'write_head_num': self.write_head_num,
            },
            'clip_value': self.clip_value,
            'output_dim': self.output_dim,
            'shift_range': self.shift_range,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))