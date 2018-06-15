# This code was inspired by Faur: https://github.com/Faur/CTRNN
# Corresponding to a simple, uni-directional MTRNN with 3 layers

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

import optimizers

# basic cell of the network #
class CTLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, tau, activation=None):
        self._num_units = num_units
        self.tau = tau
        if activation is None:
            self.activation = lambda x: 1.7159 * tf.tanh(2/3*x)
            # from: LeCun et al. 2012: Efficient backprop
        else:
            self.activation = activation
        self.lstmCell = tf.contrib.rnn.BasicLSTMCell(self._num_units, activation=self.activation) 

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            old_u = state[1]    # previous cell state

            new_input = tf.concat(inputs, 1)    # all inputs to the cell, concatenated into a single tf tensor

            with tf.variable_scope('linear'):
                output, states = self.lstmCell(new_input, state)    # get state and output from the LSTM cell

            with tf.variable_scope('applyTau'):
                new_u = (1-1/self.tau)*old_u + 1/self.tau*output    # apply timescale factor

            new_c = self.activation(new_u)  # calculate overall cell activation

        return new_c, (new_c, new_u)


def shape_printer(obj, prefix):
    try:
        print(prefix, obj.shape)
    except AttributeError:
        print(prefix, type(obj))
        for o in obj:
            shape_printer(o, prefix + '\t')

# this function is responsible for the full layer structure #
class MultiLayerHandler():
    def __init__(self, layers):
        """ layers: A list of layers """
        self.layers = layers
        self.num_layers = len(layers)

    def __call__(self, inputs, state, scope=None, reverse = True):

        with tf.variable_scope(scope or type(self).__name__):
            out_state = []
            outputs = [[],[]]
            if reverse:         # When training language
                for i_, l in enumerate(reversed(self.layers)):  # Start with the top level
                    i = self.num_layers - i_ - 1
                    scope = 'CTRNNCell_' + str(i)

                    cur_state = state[i]
                    if i == 0:                      # IO layre for language, last executed
                        cur_input = [inputs[0]] + [state[i+1][0]]
                        outputs1, state_ = l(cur_input, cur_state, scope=scope)
                    elif i == self.num_layers - 1:  # IO layer for actions, first executed
                        cur_input = [inputs[1]] + [state[i-1][0]]
                        outputs2, state_ = l(cur_input, cur_state, scope=scope)
                    else:                           # Inbetween layers
                        cur_input = [state[i-1][0]] + [state[i+1][0]]
                        outputs3, state_ = l(cur_input, cur_state, scope=scope)
                    out_state += [state_]

                out_state = tuple(reversed(out_state))

            else:               # when training actions
                for i_, l in enumerate(self.layers):            # Start with the bottom level
                    i = i_ 
                    scope = 'CTRNNCell_' + str(i)

                    cur_state = state[i]
                    if i == 0:                      # IO layer for language, first executed
                        cur_input = [inputs[0]] + [state[i+1][0]]
                        outputs1, state_ = l(cur_input, cur_state, scope=scope)
                    elif i == self.num_layers - 1:  # IO layer for actions, last executed
                        cur_input = [inputs[1]] + [state[i-1][0]]
                        outputs2, state_ = l(cur_input, cur_state, scope=scope)
                    else:                           # Inbetween layers
                        cur_input = [state[i-1][0]] + [state[i+1][0]]
                        outputs3, state_ = l(cur_input, cur_state, scope=scope)
                    out_state += [state_]

                out_state = tuple(out_state)

            outputs = [outputs2, outputs1] # outputs of the network are only considered for the I/O layers
            shape_printer(out_state, 'MLH')
            return outputs, out_state


class MTLSTMModel(object):
    def __init__(self, num_units, tau, num_steps, lang_dim, motor_dim, learning_rate=1e-4):

        self.num_units = num_units  # vector with number of cells per layer
        self.num_layers = len(self.num_units)   # number of layers
        self.tau = tau              # vector of timescales per layer

        self.lang_dim = lang_dim    # Input/Output dimention for language
        self.motor_dim = motor_dim  # Input/Output dimention for actions

        self.activation = lambda x: 1.7159 * tf.tanh(2/3 * x)
        # activation function to use on the cells (default)


    # Inputs #
        # language matrix #
        # num_sequences x num_timesteps x num_elements_language #
        self.x = tf.placeholder(tf.float32, shape=[None, num_steps, lang_dim], name='inputPlaceholder')
        self.x_reshaped = tf.reshape(tf.transpose(self.x, [1,0,2]), [-1])

        # action matrix #
        # num_sequences x num_timesteps x num_elements_actions #
        self.m = tf.placeholder(tf.float32, shape = [None, num_steps, motor_dim], name = 'sentencePlaceholder')

        # language target #
        # num_sequences x num_timesteps. Each timestep has an integer corresponding to the letter #
        self.y = tf.placeholder(tf.int32, shape=[None, num_steps], name='outputPlaceholder')
        pre_y_reshaped = self.y[:,100:130]      # only the last 30 steps count, corresponding to the language output time #
        self.y_reshaped = tf.reshape(tf.transpose(pre_y_reshaped, [1,0]), [-1])

        # actions target #
        # num_sequences x num_timesteps x num_elements_actions.#
        self.m_o = tf.placeholder(tf.float32, shape=[None, num_steps, motor_dim], name='outputPlaceholder')
        pre_m_o_reshaped = self.m_o[:,30:130,:] # only the last 100 steps count, corresponding to the action output time #
        self.m_o_reshaped = tf.reshape(tf.transpose(pre_m_o_reshaped, [1, 0, 2]), [-1, motor_dim])

        # direction of information flow (True - sentences, False - actions)
        self.direction = tf.placeholder(tf.bool, shape=())


        # initialize states/inputs #
        init_input_lang = tf.placeholder(tf.float32, shape=[None, self.num_units[0]], name='initInputLang')
        init_input_motor = tf.placeholder(tf.float32, shape=[None, self.num_units[6]], name = 'initInputMotor')
        init_input = [init_input_motor, init_input_lang]
        init_state = []
        for i, num_unit in enumerate(self.num_units):
            init_c = tf.placeholder(tf.float32, shape=[None, num_unit], name='initC_' + str(i))
            init_u = tf.placeholder(tf.float32, shape=[None, num_unit], name='initU_' + str(i))
            init_state += [(init_c, init_u)]
        init_state = tuple(init_state)
        self.init_tuple = (init_input, init_state)


        # initialize graph with all cells and layers #
        cells = []
        for i in range(self.num_layers): 
            num_unit = num_units[i]
            tau = self.tau[i]
            cells += [CTLSTMCell(num_unit, tau=tau, activation=self.activation)]
        self.cell = MultiLayerHandler(cells) # First cell (index 0) is IO layer
        

        # main forward propagation loop, conditioned for direction of flow #
        with tf.variable_scope("scan", reuse = tf.AUTO_REUSE):
            self.rnn_outputs, self.final_states = tf.cond(self.direction,
            lambda: tf.scan(lambda state, x: self.cell(x, state[1], reverse = True),
            [tf.transpose(self.x, [1, 0, 2]), tf.transpose(self.m, [1,0,2])],
            initializer=self.init_tuple), 
            lambda: tf.scan(lambda state, x: self.cell(x, state[1], reverse = False), 
            [tf.transpose(self.x, [1, 0, 2]), tf.transpose(self.m, [1,0,2])],
            initializer=self.init_tuple))

        
        # processing the states #
        state_state = []
        for i in range(self.num_layers):
            state_state += [(self.final_states[i][0][-1], self.final_states[i][1][-1])]
        state_state = tuple(state_state)
        self.state_tuple = (self.rnn_outputs[:][-1], state_state)


        # processing the outputs #
        pre_rnn_outputs_lang = self.rnn_outputs[1]
        rnn_outputs_lang = pre_rnn_outputs_lang[100:130, :, :]  # only last 30 steps count
        rnn_outputs_lang = tf.cast(tf.reshape(rnn_outputs_lang, [-1, num_units[0]]), tf.float32)
        rnn_outputs_lang = tf.slice(rnn_outputs_lang, [0, 0], [-1, lang_dim])

        
        pre_rnn_outputs_motor = self.rnn_outputs[0]
        rnn_outputs_motor = pre_rnn_outputs_motor[30:130, :, :] # only last 100 steps count
        rnn_outputs_motor = tf.cast(tf.reshape(rnn_outputs_motor, [-1, num_units[6]]), tf.float32)
        rnn_outputs_motor = tf.slice(rnn_outputs_motor, [0,0], [-1, motor_dim])

################################ Softmax #######################################
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [lang_dim, lang_dim], tf.float32)
            b = tf.get_variable('b', [lang_dim], initializer=tf.constant_initializer(0.0, tf.float32))
            self.logits = tf.matmul(rnn_outputs_lang, W) + b
            self.softmax = tf.nn.softmax(self.logits, dim=-1)


################################ Actions #######################################

        self.logits_motor = rnn_outputs_motor
        
############################# Loss function ####################################

        self.total_loss = tf.cond(self.direction, lambda: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_reshaped)), lambda: tf.reduce_sum(tf.square(tf.subtract(self.m_o_reshaped, self.logits_motor))))
        tf.summary.scalar('training/total_loss', self.total_loss)

############################ Train operation ###################################
        self.train_op = optimizers.AMSGrad(learning_rate).minimize(self.total_loss)

################## If using gradient clipping, uncomment below #################

        #optimizer = optimizers.AMSGrad(learning_rate)
        #gradients, variables = zip(*optimizer.compute_gradients(self.total_loss))
        #gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
        #self.train_op = optimizer.apply_gradients(zip(2*gradients, variables))

################################################################################

        self.TBsummaries = tf.summary.merge_all()

########## uncomment for GPU options (not worth it, more time-consuming) #######

        #config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True)
        #config.gpu_options.per_process_gpu_memory_fraction = 0.10

################################################################################

        config = tf.ConfigProto(device_count = {'CPU': 12,'GPU': 0}, allow_soft_placement = True, log_device_placement = False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.operation_timeout_in_ms = 50000

        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess = tf.Session(config = config)

################################################################################


    def forward_step_test(self):    # Smaller graph for testing network, no train ops #
        self.Inputs_m_t = tf.placeholder(tf.float32, shape = [1, self.motor_dim], name = 'motor_input')
        self.Inputs_sentence_t = tf.placeholder(tf.float32, shape = [1, self.lang_dim], name = 'sentence_input')

        self.direction = tf.placeholder(tf.bool, shape=())

        Inputs_t = [self.Inputs_sentence_t, self.Inputs_m_t]

        with tf.variable_scope("test", reuse = tf.AUTO_REUSE):
            init_state = []
            for i, num_unit in enumerate(self.num_units):
                init_c = tf.placeholder(tf.float32, shape=[None, num_unit], name='initC_' + str(i))
                init_u = tf.placeholder(tf.float32, shape=[None, num_unit], name='initU_' + str(i))
                init_state += [(init_c, init_u)]
            State = tuple(init_state)

        with tf.variable_scope("scan", reuse = tf.AUTO_REUSE):
            self.outputs, self.new_state = tf.cond(self.direction, lambda: self.cell(Inputs_t, State, reverse = True), lambda: self.cell(Inputs_t, State, reverse = False))

        outputs_sentence_sliced = tf.cast(tf.reshape(self.outputs[1], [-1, self.num_units[0]]), tf.float32)
        outputs_sentence_sliced = tf.slice(outputs_sentence_sliced, [0, 0], [-1, self.lang_dim])

        
        with tf.variable_scope("softmax", reuse = tf.AUTO_REUSE):
            W = tf.get_variable('W', [self.lang_dim, self.lang_dim], tf.float32)
            b = tf.get_variable('b', [self.lang_dim], initializer=tf.constant_initializer(0.0, tf.float32))
            logits = tf.matmul(outputs_sentence_sliced, W) + b
            self.softmax = tf.nn.softmax(logits, dim=-1)


# function to obtain the weights of the network, for printing #
    def get_weights(self):
        return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('kernel:0')]
