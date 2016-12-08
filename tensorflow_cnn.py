# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================





from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import input_data
import argparse
import logging
import sys
import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt

log = logging.getLogger()
#command line arguments
parser = argparse.ArgumentParser(description = "path name")	
parser.add_argument("--data_dir", help="the directory where the files are")
parser.add_argument("--train_file", help="training data mnist file")
parser.add_argument("--test_file", help="the test data mnist file")
parser.add_argument("--train_label", help="training labels mnist file")
parser.add_argument("--test_label", help="the test label mnist file")
parser.add_argument("--config_file", help="file with the number of classes")
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no_train', dest='train', action='store_false')
parser.add_argument('--threeconv', dest='threeconv', action='store_true')
parser.add_argument('--no_threeconv', dest='threeconv', action='store_false')
parser.set_defaults(train=True)
parser.set_defaults(no_threeconv=True)
parser.add_argument("--logging_level",type=int)
parser.set_defaults(logging_level = logging.INFO)
args = parser.parse_args()
train = args.train
threeconv = args.threeconv
data_dir = args.data_dir
train_file = args.train_file
test_file = args.test_file
train_label = args.train_label
test_label = args.test_label
conf_f = args.config_file

j = input_data.read_config(conf_f)
no_classes = j['tensorflow']['no_classes']


streamhandler = logging.StreamHandler(sys.stdout)

if args.logging_level==10:
   streamhandler.setLevel(logging.INFO)
   log.setLevel(logging.INFO)
if args.logging_level==20:
   streamhandler.setLevel(logging.DEBUG)
   log.setLevel(logging.DEBUG)

filehandler = logging.FileHandler("logging")
formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

streamhandler.setFormatter(formatter)
log.addHandler(streamhandler)
#read the data sets from the mnist files
data_sets = input_data.read_data_sets(data_dir, no_classes,fake_data=False, one_hot=False,train_only=False, train_file=train_file, test_file=test_file,train_label=train_label,test_label=test_label)


#define the weights, biases, convolution, and max pooling functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#Make the tensorflow interactive session
sess = tf.InteractiveSession()


# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, no_classes]))
b = tf.Variable(tf.zeros([no_classes]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
#Make the first convolution
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#Make the second convolution
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

if threeconv:
    W_conv3 = weight_variable([5,5,64,92])
    b_conv3= bias_variable([92])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
#Make the first fully connected layer
    W_fc1 = weight_variable([4 * 4 * 92, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*92])
else:
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#Set drop out with probability
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
with tf.name_scope('dropout'):
     tf.scalar_summary('dropout_keep_probability', keep_prob)
#Go through the second fully connected layer
W_fc2 = weight_variable([1024, no_classes])
b_fc2 = bias_variable([no_classes])
#Softmax the output vector
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, no_classes])

#Calculate the cross entropy and the AdamOptimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.name_scope('accuracy'):
      tf.scalar_summary('accuracy', accuracy)    
sess.run(tf.initialize_all_variables())

with tf.name_scope('cross_entropy'):
    tf.scalar_summary('cross entropy', cross_entropy)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter("summaries" + '/train',
                                        sess.graph)
test_writer = tf.train.SummaryWriter("summaries" + '/test')

#Create the saver to save the model
saver = tf.train.Saver()

if train:
    steps = []
    accuracys = []
    #Run the cnn with 20000 iterations and batches of 50
    for i in range(20000):
	batch = data_sets.train.next_batch(50)
	if i%100 == 0:
#	    test_batch = data_sets.test.next_batch(1000)
#  	    summary, acc = sess.run([merged, accuracy], feed_dict={
#            x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
#            test_writer.add_summary(summary, i)
	    saver.save(sess, 'temp/checkpoint.chk')
	    #Evaluate training accuracy every 100 steps
	    train_accuracy = accuracy.eval(feed_dict={
	    x:batch[0], y_: batch[1], keep_prob: 1})
	    print("step %d, training accuracy %g"%(i, train_accuracy))
	    steps.append(i)
	    accuracys.append(train_accuracy)
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:.5})
	summary, _ = sess.run([merged, train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5} )
        train_writer.add_summary(summary, i)
    #Evaluate the test accuracy
    print("test accuracy %g"%accuracy.eval(feed_dict={
    x: data_sets.test._images, y_: data_sets.test._labels, keep_prob: 1.0}))
    steps.append(3000)
    accuracys.append(accuracy.eval(feed_dict={
    x: data_sets.test._images, y_: data_sets.test._labels, keep_prob: 1.0}))
    #plt.plot(steps, accuracys)
    #plt.xlabel("Step")
    #plt.ylabel("Accuracy")
    #plt.suptitle("Tensorflow Convolutional Neural Network Accuracys")
    #plt.show()
else:	
    #Restore the saved model
    ckpt = tf.train.get_checkpoint_state('temp')
    if ckpt and ckpt.model_checkpoint_path:
         saver.restore(sess, ckpt.model_checkpoint_path)
         #feed x's and get y
         batch_x = data_sets.train.next_batch(5)
         predictions = sess.run(y_conv, feed_dict={x: batch_x[0], keep_prob:1.0})
	 incl_list = ['97','98','99','120','121','122','61','45','43','47','42','46','48','49','50','51','52','53','54','55','56','57']
        #Find the argmax of the probability distribution vectors and convert to the characters 
	 indices = predictions.argmax(axis=1)
 	 values=[]
	 for i in indices:
		values.append(chr(int(incl_list[i])))
	 print("predictions = ",predictions)
	 print(values)
