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
#import config
import os
import tensorflow as tf
from sklearn.externals import joblib

log = logging.getLogger()

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
predict = args.predict
data_dir = args.data_dir
train_file = args.train_file
test_file = args.test_file
train_label = args.train_label
test_label = args.test_label
conf_f = args.config_file

j = input_data.read_config(conf_f)
print("it is",j)
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

data_sets = input_data.read_data_sets(data_dir, no_classes,fake_data=False, one_hot=False,train_only=False, train_file=train_file, test_file=test_file,train_label=train_label,test_label=test_label)
labels = []

incl_list = ['97','98','99','120','121','122','61','45','43','47','42','46','48','49','50','51','52','53','54','55','56','57']

test_labels = []

for i in data_sets.test._labels:
    for j, k in enumerate(i):
	if k == 1:
	    test_labels.append(j)
if predict:
	batch = data_sets.train.next_batch(91308)
	for i in batch[1]:
	    for j,k in enumerate(i):
		if k == 1:
		    labels.append(j)
	clf = SVC()
	clf.fit(batch[0], labels)
	joblib.dump(clf, 'svm.pkl')
	print(clf.score(data_sets.test._images, test_labels))
else:
	clf = joblib.load('svm.pkl')
	thelist=clf.predict(data_sets.test._images)
	values = []
	for i in thelist:
	    values.append(incl_list[i])
	print(values)

