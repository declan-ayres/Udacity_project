from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn
from sklearn.svm import SVC
import input_data
import argparse
import logging
import sys
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import knn_benchmark

log = logging.getLogger()
#command line arguments
parser = argparse.ArgumentParser(description = "path name")
parser.add_argument("--data_dir", help="the directory where the files are")
parser.add_argument("--train_file", help="the training data mnist file")
parser.add_argument("--test_file", help="the test data mnist file")
parser.add_argument("--train_label", help="the training labels mnist file")
parser.add_argument("--test_label", help="the test label mnist file")
parser.add_argument("--config_file", help="the config file with the number of classes")
parser.add_argument('--predict', dest='predict', action='store_true')
parser.add_argument('--no_predict', dest='predict', action='store_false')
parser.set_defaults(train=True)
parser.add_argument("--logging_level",type=int)
parser.set_defaults(logging_level = logging.INFO)
args = parser.parse_args()
predict = args.predict
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

#the list of the classes in unicode ascii
incl_list = ['97','98','99','120','121','122','61','45','43','47','42','46','48','49','50','51','52','53','54','55','56','57']

ind = np.arange(22)
#convert the onehot arrays to indices array
test_labels = knn_benchmark.onehot_to_label(data_sets.test._labels)
#train the svm
if not predict:
	batch = data_sets.train.next_batch(91308)
	labels = knn_benchmark.onehot_to_label(batch[1])
	clf = SVC(C=100, gamma=.1)
	clf.fit(batch[0], labels)
	joblib.dump(clf, 'svm.pkl')
	print(clf.score(data_sets.test._images, test_labels))
#	plt.bar(ind, clf.n_support_, .5)
#	plt.ylabel('Number')
#	plt.title('Number of Support Vectors for Each Class')
#	plt.xticks(ind, ('a','b','c','x','y','z','=','-','+','/','*','.','0','1','2','3','4','5','6','7','8','9'))
#	plt.show()
else:
	#load the saved model
	clf = joblib.load('svm.pkl')
	thelist=clf.predict(data_sets.test._images)
	values = []
	#convert to the characters
	for i in thelist:
	    values.append(chr(int(incl_list[i])))
        print(values)

