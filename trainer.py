from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from MTLSTM import MTLSTMModel

import time
import operator
import io
import array
import datetime

import os
import sys

import itertools

def get_sentence(verb, obj):
    verb = float(verb)
    obj = float(obj)
    if verb >= 0.0 and verb < 0.1:
        sentence = "slide left"
    elif verb >= 0.1 and verb < 0.2:
        sentence = "slide right"
    elif verb >= 0.2 and verb < 0.3:
        sentence = "touch"
    elif verb >= 0.3 and verb < 0.4:
        sentence = "reach"
    elif verb >= 0.4 and verb < 0.5:
        sentence = "push"
    elif verb >= 0.5 and verb < 0.6:
        sentence = "pull"
    elif verb >= 0.6 and verb < 0.7:
        sentence = "point"
    elif verb >= 0.7 and verb < 0.8:
        sentence = "grasp"
    else:
        sentence = "lift"
    if obj >= 0.0 and obj < 0.1:
        sentence = sentence + " the " + "tractor"
    elif obj >= 0.1 and obj < 0.2:
        sentence = sentence + " the " + "hammer"
    elif obj >= 0.2 and obj < 0.3:
        sentence = sentence + " the " + "ball"
    elif obj >= 0.3 and obj < 0.4:
        sentence = sentence + " the " + "bus"
    elif obj >= 0.4 and obj < 0.5:
        sentence = sentence + " the " + "modi"
    elif obj >= 0.5 and obj < 0.6:
        sentence = sentence + " the " + "car"
    elif obj >= 0.6 and obj < 0.7:
        sentence = sentence + " the " + "cup"
    elif obj >= 0.7 and obj < 0.8:
        sentence = sentence + " the " + "cubes"
    else:
        sentence = sentence + " the " + "spiky"
    sentence = sentence + "."
    return sentence

######################################################################################
# This function loads data from a file, to train the network
# inputs are sequential (and always same order). 
def loadTrainingData(LangInputNeurons, MotorInputNeurons, Lang_stepEachSeq, Motor_stepEachSeq, numSeq):

    stepEachSeq = Lang_stepEachSeq + Motor_stepEachSeq

    # sequence of letters
    x_train = np.asarray(np.zeros((numSeq , stepEachSeq, LangInputNeurons)),dtype=np.float32)
    y_train = 26 * np.asarray(np.ones((numSeq , stepEachSeq)),dtype=np.int32)

    # motor sequence
    m_train = np.asarray(np.zeros((numSeq, stepEachSeq, MotorInputNeurons)), dtype=np.float32)
    m_gener = np.asarray(np.zeros((numSeq, stepEachSeq, MotorInputNeurons)), dtype=np.float32)

    print("steps: ", stepEachSeq)
    print("number of sequences: ", numSeq)

    dataFile = open("mtrnnTD.txt", 'r')

    totalSeq = 432

    sentence_list = []

    sequences = [k for k in range(0, totalSeq, 1)]

    k = 0 #number of sequences
    t = -1 #number of saved sequences
    while True:
        line = dataFile.readline()
        if line == "":
            break
        if line.find("SEQUENCE") != -1:
            if k in sequences: # In case we want to train particular sequences
                t+=1
                for i in range(0, Motor_stepEachSeq):
                    line = dataFile.readline()
                    line_data = line.split("\t")
                    line_data[-1] = line_data[-1].replace("\r\n",'')
                    if i == 0:
                        sentence = get_sentence(line_data[0], line_data[1])
                        sentence_list += [sentence]
                        p = 0
                        for g in range(Lang_stepEachSeq):
                            if g >= 4 and p < len(sentence):
                                lett = sentence[p]
                                p += 1
                            # during language input, motor input should remain the same
                            m_gener[t, g, 0:MotorInputNeurons] = line_data[1:MotorInputNeurons+1]
                            if g < len(sentence)+4 and g >=4:
                                if lett == ' ':
                                    x_train[t, g,26] = 1
                                    y_train[t, Motor_stepEachSeq + g] = 26
                                elif lett == '.':
                                    x_train[t, g,27] = 1
                                    y_train[t, Motor_stepEachSeq + g] = 27
                                else:
                                    x_train[t, g, ord(lett) - 97] = 1
                                    y_train[t, Motor_stepEachSeq + g] =  ord(lett) - 97
                            else:
                                x_train[t, g,26] = 1
                                y_train[t, Motor_stepEachSeq + g] = 26
                    # we save the values for the encoders at each step
                    m_train[t, i,0:MotorInputNeurons] = line_data[1:MotorInputNeurons+1]
                    m_gener[t, i+Lang_stepEachSeq, 0:MotorInputNeurons] = line_data[1:MotorInputNeurons+1]
                    y_train[t, i] = 26
                    x_train[t, Lang_stepEachSeq + i, 26] = 1

                # now we set the motor output to be constant in the end 
                for i in range(Motor_stepEachSeq, stepEachSeq):
                    m_train[t, i,0:MotorInputNeurons] = line_data[1:MotorInputNeurons+1]
            k = k+1 
        if k == totalSeq:
            break
        
    dataFile.close()
    return x_train, y_train, m_train, m_gener, sentence_list

###########################################


def plot(loss_list, fig, ax):
    ax.semilogy(loss_list, 'b')
    fig.canvas.flush_events()

###########################################

def create_batch(x_train, y_train, m_train, m_gener, m_output, batch_size):
    x_out = np.zeros((batch_size, x_train.shape[1], x_train.shape[2]))
    y_out = np.zeros((batch_size, y_train.shape[1]))
    m_out = np.zeros((batch_size, m_train.shape[1], m_train.shape[2]))
    m_gener_out = np.zeros((batch_size, m_gener.shape[1], m_gener.shape[2]))
    m_output_out = np.zeros((batch_size, m_output.shape[1], m_output.shape[2]))
    for i in range(batch_size):
        seq_index = np.random.randint(0,y_train.shape[0])
        x_out[i, :, :] = x_train[seq_index, :, :]
        y_out[i, :] = y_train[seq_index, :]
        m_out[i, :, :] = m_train[seq_index, :, :]
        m_gener_out[i, :, :] = m_gener[seq_index, :, :]
        m_output_out[i, :, :] = m_output[seq_index, :, :]
    return x_out, y_out, m_out, m_gener_out, m_output_out


my_path= os.getcwd()

########################################## Control Variables ################################
START_FROM_SCRATCH = True  # start model from scratch, or from pre-trained
load_path = my_path + ""    # path to pre-trained file
# Example Path: load_path = my_path + "/mtrnn_387111_loss_0.11538351478520781"

USING_BATCH = True          # using batches or full dataset
batch_size = 32             # size of the batches (in number of sequences)

direction = True            # True - language to actions; False - actions to language
alternate = True            # Alternate direction - False will train only one direction
alpha = 0.5                 # 1 - language loss has more weight, 0 - action loss has more weight

NEPOCH = 1235050            # number of times to train each batch
threshold_lang = 0.001      # early stopping language loss threshold
threshold_motor = 0.5       # early stopping action loss threshold
average_loss = 1000.0       # initial value for the average loss (action+language) - arbitrary

loss_list = []              # list that stores the average loss
lang_loss_list = [2]        # list that stores the language loss
lang_loss = 2               # Save model if language loss below this value
motor_loss_list = [2]       # list that stores the action loss
motor_loss = 2              # Save model if action loss below this value

## formula for calculating the overall best loss of the model ##
best_loss = alpha * lang_loss + (1-alpha) * motor_loss

########################################## Model parameters ################################
lang_input = 28     # size of output/input language
input_layer = 40    # I/O language layer
lang_dim1 = 160     # fast context language layer
lang_dim2 = 35      # slow context language layer
meaning_dim = 25    # meaning layer
motor_dim2 = 35     # slow context action layer
motor_dim1 = 160    # fast context action layer
motor_layer = 140   # I/O action layer
motor_input = 42    # size of output/input action


numSeq = 432            # number of sequences
Lang_stepEachSeq = 30   # timesteps for a sentence
Motor_stepEachSeq = 100 # time steps for a motor action
stepEachSeq = Lang_stepEachSeq + Motor_stepEachSeq  # total number of steps in 1 run

LEARNING_RATE = 5 * 1e-3    # Learning Rate of the network

################################### Network Initialization ###################################
MTRNN = MTLSTMModel([input_layer, lang_dim1, lang_dim2, meaning_dim, motor_dim2, motor_dim1, motor_layer], [2, 5, 60, 100, 60, 5, 2], stepEachSeq, lang_input, motor_input, LEARNING_RATE)


#################################### acquire data ##########################################
x_train, y_train, m_train, m_gener, sentence_list = loadTrainingData(lang_input, motor_input, Lang_stepEachSeq, Motor_stepEachSeq, numSeq)

######### Roll the outputs, so it tries predicting the future ############
# we want the network to output the next position for the robot to go to #
m_output = np.zeros([numSeq, stepEachSeq, motor_input], dtype=np.float32)
m_output[:,:,:] = np.roll(m_gener, -1, axis=1)[:,:,0:motor_input]
m_output[:,-1,:] = m_output[:,-2,:]

# store data in unchanged vectors #
old_x = x_train         
old_y = y_train         
old_m_train = m_train
old_m_gener = m_gener
old_m_output = m_output
old_sentence = sentence_list
old_numSeq = numSeq
###################################

###### Batch creation #######
if USING_BATCH:
    x_train_b, y_train_b, m_train_b, m_gener_b, m_output_b = create_batch(x_train, y_train, m_train, m_gener, m_output, batch_size)
    numSeqmod_b = batch_size
else:
    x_train_b = x_train
    y_train_b = y_train
    m_train_b = m_train
    m_gener_b = m_gener
    m_output_b = m_output
    numSeqmod_b = numSeq
############################
    
print("data loaded")

############################# Initialize States #############################
init_state_IO_l = np.zeros([numSeqmod_b, input_layer], dtype = np.float32)
init_state_fc_l = np.zeros([numSeqmod_b, lang_dim1], dtype = np.float32)
init_state_sc_l = np.zeros([numSeqmod_b, lang_dim2], dtype = np.float32)
init_state_ml = np.zeros([numSeqmod_b, meaning_dim], dtype = np.float32)
init_state_IO_m = np.zeros([numSeqmod_b, motor_layer], dtype = np.float32)
init_state_fc_m = np.zeros([numSeqmod_b, motor_dim1], dtype = np.float32)
init_state_sc_m = np.zeros([numSeqmod_b, motor_dim2], dtype = np.float32)
#############################################################################


############################### training iterations #########################################

MTRNN.sess.run(tf.global_variables_initializer())

flag_save = False           # flag indicating if the network has been saved or not (if it reaches the limit of epochs without having saved yet)

if not START_FROM_SCRATCH:
    MTRNN.saver.restore(MTRNN.sess, load_path)

epoch_idx = 0       # initialize epochs
counter_lang = 0    # initialize counter for number of language training epochs
counter_motor = 0   # initialize counter for number of action training epochs

#complicated logic:
# 1) we train actions and Lang, or;
# 2) we train only Lang, or;
# 3) we train only actions.

while (alternate and (lang_loss_list[-1] > threshold_lang or motor_loss_list[-1] > threshold_motor)) or (not alternate and ((direction and lang_loss_list[-1] > threshold_lang) or (not direction and motor_loss_list[-1] > threshold_motor))): 
    print("Training epoch " + str(epoch_idx))

    motor_inputs = np.zeros([m_train_b.shape[0], m_train_b.shape[1], m_train_b.shape[2]], dtype = np.float32)

    # we recreate random batch at each epoch #
    if USING_BATCH:
        x_train_b, y_train_b, m_train_b, m_gener_b, m_output_b = create_batch(x_train, y_train, m_train, m_gener, m_output, batch_size)
    else:
        x_train_b = x_train
        y_train_b = y_train
        m_train_b = m_train
        m_gener_b = m_gener
        m_output_b = m_output

    if direction:   # if training language
        # There is no language input (all zeros)
        lang_inputs = np.zeros([numSeqmod_b, stepEachSeq, lang_input], dtype = np.float32)

        # complete action input (as in dataset)
        motor_inputs = m_train_b

        # no motor output to compare to (we are generating sentences)
        motor_outputs = np.zeros([numSeqmod_b, stepEachSeq, motor_input], dtype = np.float32)
        counter_lang +=1
    else:           # if training actions
        # full language input (as in dataset)
        lang_inputs = x_train_b

        # we only give initial action input (time step 0)
        motor_inputs[:,0,:] = m_gener_b[:,0,:]

        # full action output (to compare)
        motor_outputs = m_output_b
        counter_motor +=1

    t0 = datetime.datetime.now()
    # run the network with the data we prepared #
    _total_loss, _train_op, _state_tuple = MTRNN.sess.run([MTRNN.total_loss, MTRNN.train_op, MTRNN.state_tuple], feed_dict={MTRNN.x:lang_inputs, MTRNN.y:y_train_b, MTRNN.m:motor_inputs, MTRNN.m_o:motor_outputs, MTRNN.direction:direction, 'initU_0:0':init_state_IO_l, 'initC_0:0':init_state_IO_l, 'initU_1:0':init_state_fc_l, 'initC_1:0':init_state_fc_l, 'initU_2:0':init_state_sc_l, 'initC_2:0':init_state_sc_l, 'initU_3:0':init_state_ml, 'initC_3:0':init_state_ml, 'initU_4:0':init_state_sc_m, 'initC_4:0':init_state_sc_m, 'initU_5:0':init_state_fc_m, 'initC_5:0':init_state_fc_m, 'initU_6:0':init_state_IO_m, 'initC_6:0':init_state_IO_m})
    t1 = datetime.datetime.now()
    print("epoch time: ", (t1-t0).total_seconds())      # check training time #
    if direction:       # if training language, save language loss
        loss = _total_loss
        print("training sentences: ", loss)
        lang_loss = loss
        lang_loss_list.append(lang_loss)
    else:               # if training actions, save actions loss
        loss = _total_loss
        print("training motor: ", loss)
        motor_loss = loss
        motor_loss_list.append(motor_loss)

    average_loss = alpha*lang_loss + (1-alpha)*motor_loss   # calculate average loss
    loss_list.append(average_loss)                          # save average loss
    print("Current best loss: ",best_loss)
    print("#################################")
    print("epoch "+str(epoch_idx)+", loss: "+str(loss))
    if average_loss <= best_loss:           # if average loss lower than best, save model
        model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(average_loss)
        save_path = MTRNN.saver.save(MTRNN.sess, model_path)
        best_loss = average_loss
        flag_save =True
    epoch_idx += 1

    # Below is the implementation that allows the network to focus on
    # a specific training direction, if the other has already crossed
    # the threshold we defined. 
    # Example: if motor_loss is below threshold_motor, then direction
    # will be set to True (always train language).
    # motor loss will be checked again at least every 10 epochs, and
    # updated. If it goes above the threshold, the training defaults
    # back to alternating training
    if alternate:
        direction = not direction
        if motor_loss_list[-1] < threshold_motor:
            direction = True
            if epoch_idx%10 == 0:
                direction = not direction
        elif lang_loss_list[-1] < threshold_lang:
            direction = False
            if epoch_idx%10 == 0:
                direction = not direction

    t2 = datetime.datetime.now()
    print("saving time: ", (t2-t1).total_seconds())
    if epoch_idx > NEPOCH:
        break

print("the network trained language ", counter_lang, " times and motor actions ", counter_motor, " times.")


##################################### Print error graph ####################################
plt.ion()
fig = plt.figure()
ax = plt.subplot(1,1,1)
fig.show()
plot(loss_list, fig, ax)
############################################################################################

#####################  If the network was never saved during the whole training ############
if not flag_save:
    model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(average_loss)
    save_path = MTRNN.saver.save(MTRNN.sess, model_path)
############################################################################################

########################################## TEST ############################################
MTRNN.saver.restore(MTRNN.sess, save_path)
plt.ioff()
plt.show()
print("testing")


PRINT_TABLE = True     # True to print the language output matrix
test_false = True      # True to test action generation
test_true = True        # True to test language generation

jumps = 201               # number of sequences jumped during the test. 1 tests every sequence

init_state_IO_l = np.zeros([1, input_layer], dtype = np.float32)
init_state_fc_l = np.zeros([1, lang_dim1], dtype = np.float32)
init_state_sc_l = np.zeros([1, lang_dim2], dtype = np.float32)
init_state_ml = np.zeros([1, meaning_dim], dtype = np.float32)
init_state_IO_m = np.zeros([1, motor_layer], dtype = np.float32)
init_state_fc_m = np.zeros([1, motor_dim1], dtype = np.float32)
init_state_sc_m = np.zeros([1, motor_dim2], dtype = np.float32)

MTRNN.forward_step_test()   # function that initializes a smaller graph of the network, no training functions
tf.get_default_graph().finalize()

for i in range(0, numSeq, jumps):

    print("sentence: ", sentence_list[i])

    if test_true:
        new_lang_out = np.asarray(np.zeros((1, stepEachSeq)),dtype=np.int32)
        new_motor_in = np.asarray(np.zeros((1, stepEachSeq, motor_input)),dtype=np.float32)
        new_lang_in = np.asarray(np.zeros((1, stepEachSeq, lang_input)), dtype=np.float32)
        new_motor_out = np.asarray(np.zeros((1, stepEachSeq, motor_input)), dtype=np.float32)

        direction = True
        new_motor_in[0, :, :] = m_train[i, :, :]
        softmax_list = np.zeros([stepEachSeq, lang_input], dtype = np.float32)

        input_x = np.zeros([1, motor_input], dtype = np.float32)
        input_sentence = np.zeros([1, lang_input], dtype = np.float32)

        # Define initial state of the network #
        State = ((init_state_IO_l, init_state_IO_l), (init_state_fc_l, init_state_fc_l), (init_state_sc_l, init_state_sc_l), (init_state_ml, init_state_ml), (init_state_sc_m, init_state_sc_m), (init_state_fc_m, init_state_fc_m),(init_state_IO_m, init_state_IO_m))        
        #######################################
        
        for l in range(stepEachSeq):
            input_x[0,:] = new_motor_in[0,l,:]
            input_sentence[0,:] = new_lang_in[0,l,:]
            init_state_00 = State[0][0]
            init_state_01 = State[0][1]
            init_state_10 = State[1][0]
            init_state_11 = State[1][1]
            init_state_20 = State[2][0]
            init_state_21 = State[2][1]
            init_state_30 = State[3][0]
            init_state_31 = State[3][1]
            init_state_40 = State[4][0]
            init_state_41 = State[4][1]
            init_state_50 = State[5][0]
            init_state_51 = State[5][1]
            init_state_60 = State[6][0]
            init_state_61 = State[6][1]

            outputs, new_state, softmax = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state, MTRNN.softmax], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_m_t: input_x, MTRNN.Inputs_sentence_t: input_sentence, 'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20, 'test/initU_3:0':init_state_31, 'test/initC_3:0':init_state_30, 'test/initU_4:0':init_state_41, 'test/initC_4:0':init_state_40, 'test/initU_5:0':init_state_51, 'test/initC_5:0':init_state_50, 'test/initU_6:0':init_state_61, 'test/initC_6:0':init_state_60})

            softmax_list[l, :] = softmax
            State = new_state
            
        sentence = ""
        for t in range(stepEachSeq):
            for g in range(lang_input):
                if softmax_list[t,g] == max(softmax_list[t]): 
                    if g <26:
                        sentence += chr(97 + g)
                    if g == 26:
                        sentence += " "
                    if g == 27:
                        sentence += "."
################################# Print table #####################################
        if PRINT_TABLE:        
            color = 0

            fig, ax = plt.subplots()
            Mat = np.transpose(softmax_list[100:,0:lang_input])
            print(np.shape(Mat))
            cax = ax.matshow(Mat, cmap=plt.cm.binary, vmin = 0, vmax = 1)
            cbar = fig.colorbar(cax, ticks = [0, 1])
            cbar.ax.set_yticklabels(['0', '1'])
            for t in range(lang_input):
                ax.axhline(y=t+0.5, ls='-', color='black')
                if t < 26:
                    plt.text(-2,t+0.5,str(chr(97+t)))
                if t == 26:
                    plt.text(-2,t+0.5," ")
                if t == 27:
                    plt.text(-2,t+0.5,".")
            for t in range(0, 30):
                ax.axvline(x=t+0.5, ls='-', color='black')
            plt.xlabel("timesteps");
            ax.set_yticklabels([])
            plt.show()
 
        print("output: ",sentence)
        print("#######################################")
        sentence = ""
        for g in range(stepEachSeq):
            if y_train[i,g] == 26:
                sentence += " "
            elif y_train[i,g] == 27:
                sentence += "."
            else:
                sentence += chr(97 + y_train[i,g])

        print("target: " ,sentence)
        print("#######################################")

    if test_false:
        new_lang_out = np.asarray(np.zeros((1, stepEachSeq)),dtype=np.int32)
        new_motor_in = np.asarray(np.zeros((1, stepEachSeq, motor_input)),dtype=np.float32)
        new_lang_in = np.asarray(np.zeros((1, stepEachSeq, lang_input)), dtype=np.float32)
        new_motor_out = np.asarray(np.zeros((1, stepEachSeq, motor_input)), dtype=np.float32)

        direction = False
        new_motor_in[0, :, :] = m_gener[i, :, :]
        new_lang_in[0,:,:] = x_train[i,:,:]

        output_list = []

        input_x = np.zeros([1, motor_input], dtype = np.float32)
        input_sentence = np.zeros([1, lang_input], dtype = np.float32)

        # Define initial state of the network #
        State = ((init_state_IO_l, init_state_IO_l), (init_state_fc_l, init_state_fc_l), (init_state_sc_l, init_state_sc_l), (init_state_ml, init_state_ml), (init_state_sc_m, init_state_sc_m), (init_state_fc_m, init_state_fc_m),(init_state_IO_m, init_state_IO_m))
        #######################################
        
        for l in range(stepEachSeq):
            if l == 0:          # only provide input at initial step
                input_x[:,:] = new_motor_in[0,l,:]
                input_sentence[:,:] = new_lang_in[0,l,:]
            else:
                input_x = np.zeros([1, motor_input], dtype = np.float32)
                input_sentence = np.zeros([1, lang_input], dtype = np.float32)
            init_state_00 = State[0][0]
            init_state_01 = State[0][1]
            init_state_10 = State[1][0]
            init_state_11 = State[1][1]
            init_state_20 = State[2][0]
            init_state_21 = State[2][1]
            init_state_30 = State[3][0]
            init_state_31 = State[3][1]
            init_state_40 = State[4][0]
            init_state_41 = State[4][1]
            init_state_50 = State[5][0]
            init_state_51 = State[5][1]
            init_state_60 = State[6][0]
            init_state_61 = State[6][1]

            outputs, new_state = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_m_t: input_x, MTRNN.Inputs_sentence_t: input_sentence, 'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20, 'test/initU_3:0':init_state_31, 'test/initC_3:0':init_state_30, 'test/initU_4:0':init_state_41, 'test/initC_4:0':init_state_40, 'test/initU_5:0':init_state_51, 'test/initC_5:0':init_state_50, 'test/initU_6:0':init_state_61, 'test/initC_6:0':init_state_60})
            output_list += [outputs]

            State = new_state

        output_vec = np.zeros([stepEachSeq, motor_input], dtype = np.float32)
        for t in range(len(output_list)):
            output_vec[t,:] = output_list[t][0][0][0:motor_input]
        for t in range(1, motor_input, 3):
            plt.plot(output_vec[30:,t], 'r')
            plt.plot(m_output[i, 30:, t], 'b')
            plt.show()
        print("\n")

MTRNN.sess.close()

