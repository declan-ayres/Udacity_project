from __future__ import print_function
import gzip
import os
import urllib
import numpy
import struct
import argparse
import logging
import sys
import json

log = logging.getLogger()

def read_config(filename):
    open_file = open(filename)
    x = open_file.read()
    open_file.close()
    json_file = json.loads(x)
    return json_file

class DataSet(object):
	def __init__(self, images, labels, fake_data=False):
		if fake_data:
			self._num_examples = 10000
		else:
	#		assert images.shape[0] == labels.shape[0] 
#(

#				"images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
			self._num_examples = images.shape[0]
			images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
			images = images.astype(numpy.float32)
			images = numpy.multiply(images, 1.0/255.0)
			self._images = images
			self._labels = labels
			self._epochs_completed = 0
			self._index_in_epoch = 0
			
			@property
			def images(self):
				return self._images
			@property
			def labels(self):
				return self._labels
			@property
			def num_examples(self):
				return self._num_examples
			@property
			def epochs_completed(self):
				return self._epochs_completed

        def next_batch(self, batch_size, fake_data=False):
	      if fake_data:
		      fake_image = [1.0 for _ in xrange(784)]
		      fake_label = 0
		      return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
	      start = self._index_in_epoch
	      self._index_in_epoch += batch_size
	      if self._index_in_epoch > self._num_examples:
		      self._epochs_completed += 1
		      perm = numpy.arange(self._num_examples)
		      print("No examples =", self._num_examples)
		      numpy.random.shuffle(perm)
		      nda = numpy.array(self._images)
		      nda = nda[perm]
		      self._images = list(nda)
		      nda = numpy.array(self._labels)
		      nda = nda[perm]
		      self._labels = list(nda)
		      
		      start = 0
		      self._index_in_epoch = batch_size
		      assert batch_size <= self._num_examples
	      end = self._index_in_epoch
	      return self._images[start:end], self._labels[start:end]

def read_data_sets(data_dir_where_training_data, num_classes,fake_data=False, one_hot=False,train_only=False, train_file=None, test_file=None, train_label=None,test_label=None):
        class DataSets():
            pass
	if fake_data:
    		data_sets.train = DataSet([], [], fake_data=True)
    		data_sets.validation = DataSet([], [], fake_data=True)
    		data_sets.test = DataSet([], [], fake_data=True)
    		data_sets.output = DataSet([], [], fake_data=True)
    		return data_sets
        local_file = maybe_download(train_file,data_dir_where_training_data) # train_file and train_dir were received as two arguments to the 'read_data_sets' function
	train_images = extract_images(local_file,train_only=train_only)
        local_file = maybe_download(train_label,data_dir_where_training_data)
        train_labels = extract_labels(local_file,num_classes,one_hot=True)
	local_file = maybe_download(test_file, data_dir_where_training_data)
	test_images = extract_images(local_file, train_only=train_only)
	local_file = maybe_download(test_label, data_dir_where_training_data)
	test_labels = extract_labels(local_file, num_classes, one_hot=True)
        data_sets = DataSets()
	data_sets.train = DataSet(train_images, train_labels)
	print(test_images.shape[0], test_labels.shape[0])
	data_sets.test = DataSet(test_images, test_labels)
	return data_sets


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'



def maybe_download(train_file, train_dir):
	  if not os.path.exists(train_dir):
    	  	os.mkdir(train_dir)
 	  
	  filepath = os.path.join(train_dir, train_file)
  	  if not os.path.exists(filepath):
    	  	filepath, _ = urllib.urlretrieve(SOURCE_URL + train_file, filepath)
    		statinfo = os.stat(filepath)
    		#print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  	  return filepath

def _read32(bytestream,train_only=True):
    if not train_only:
        return _read_int_4(bytestream) # _read_int_4(bytestream)
    else:
        dt = numpy.dtype(numpy.uint8).newbyteorder('>')
        return numpy.frombuffer(bytestream.read(4), dtype=dt)

def _read_int_4(bytestream):
	x = struct.unpack('i', bytestream.read(4))[0]
	return x

def extract_images(filename,train_only=True):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream,train_only=train_only)
        if magic != 2051:
            raise ValueError(
            'Invalid magic number %d in MNIST image file: %s' %
            (magic, filename))
        num_images = _read32(bytestream,train_only=train_only)
        print ("Num images = ",num_images)
        rows = _read32(bytestream,train_only=train_only)
        cols = _read32(bytestream,train_only=train_only)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
	print(len(data))
        data = data.reshape(num_images, rows, cols, 1)
        print ("Data shape = ",data.shape)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[(index_offset-1) + labels_dense.ravel()] = 1
              
    return labels_one_hot


def extract_labels(filename, num_classes, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream,train_only=False)
        if magic != 2049:
            raise ValueError(
            'Invalid magic number %d in MNIST label file: %s' % (magic, filename))
        num_items = _read32(bytestream,train_only=False)
        print ("Num items = ",num_items)
        labels_buf = []
        for i in range(num_items):
            c = struct.unpack('i',bytestream.read(4))[0]
            labels_buf.append(c) #+=str(c)
        #labels = numpy.frombuffer(labels_buf, dtype=numpy.uint8)
        labels = numpy.asarray(labels_buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels,num_classes=num_classes)
        return labels

if __name__ == '__main__':	
    parser = argparse.ArgumentParser(description = "path name")	
    parser.add_argument("--data_dir")
    parser.add_argument("--train_file")
    parser.add_argument("--test_file")
    parser.add_argument("--train_label")
    parser.add_argument("--test_label")
    parser.add_argument("--config_file")
    parser.add_argument("--logging_level",type=int)
    parser.set_defaults(logging_level = logging.INFO)
    args = parser.parse_args()
    data_dir = args.data_dir
    train_file = args.train_file
    test_file = args.test_file
    train_label = args.train_label
    test_label = args.test_label
    conf_f = args.config_file

    j = read_config(conf_f)
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
    rds = read_data_sets(data_dir, no_classes,False, False, False, train_file=train_file, test_file=test_file,train_label=train_label, test_label=test_label)
    for i in range(5):
        print (rds.train.next_batch(50))

