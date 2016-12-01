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
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

log = logging.getLogger()
#Command line arguments
parser = argparse.ArgumentParser(description = "path name")
parser.add_argument("--data_dir")
parser.add_argument("--train_file")
parser.add_argument("--test_file")
parser.add_argument("--train_label")
parser.add_argument("--test_label")
parser.add_argument("--config_file")
parser.add_argument('--predict', dest='predict', action='store_true')
parser.add_argument('--no_predict', dest='predict', action='store_false')
parser.set_defaults(train=True)
parser.add_argument("--logging_level",type=int)
parser.set_defaults(logging_level = logging.INFO)
args = parser.parse_args()
data_dir = args.data_dir
train_file = args.train_file
test_file = args.test_file
train_label = args.train_label
test_label = args.test_label
conf_f = args.config_file
predict = args.predict

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
#Read the data sets from the mnist files
data_sets = input_data.read_data_sets(data_dir, no_classes,fake_data=False, one_hot=False,train_only=False, train_file=train_file, test_file=test_file,train_label=train_label,test_label=test_label)
labels = []


test_labels=[]
train_labels=[]
#define function to convert onehot arrays to indices vector
def onehot_to_label(vector):
    """Converts list of onehot arrays to array of indices
    Example: [[0,1,0,0],[1,0,0,0], [0,0,0,1]] --> [1,0,3]
    """
    labels=[]
    for i in vector:
        for j, k in enumerate(i):
            if k == 1:
                labels.append(j)
    return labels
#load the train and test data
test_labels = onehot_to_label(data_sets.test._labels)
train_data = data_sets.train.next_batch(91308)
train_labels= onehot_to_label(train_data[1])

incl_list = ['97','98','99','120','121','122','61','45','43','47','42','46','48','49','50','51','52','53','54','55','56','57']
if not predict:
    #Create the knn benchmark model
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data[0], train_labels)
    #Save the model
    joblib.dump(neigh, 'knn.pkl')
    print(neigh.score(data_sets.test._images, test_labels))
else:
    #Load the saved model
    neigh=joblib.load('knn.pkl')
    thelist=neigh.predict(data_sets.test._images)
    values = []
    for i in thelist:
        values.append(chr(int(incl_list[i])))
    print(values)

index_list = [0] * 22
input_file_list = os.listdir("glyphs_output")
for f in input_file_list:
    line = f.rstrip('\n')
    x = line.split('-')
    y = x[-1].split('.')
    index_list[incl_list.index(y[0])]+=1
#Make bar graph of the distribution of the classes of data set
ind = np.arange(22)
plt.bar(ind, index_list, .5)
plt.ylabel('Number')
plt.title('Number of Characters for Each Class')
plt.xticks(ind, ('a','b','c','x','y','z','=','-','+','/','*','.','0','1','2','3','4','5','6','7','8','9'))
plt.show()

