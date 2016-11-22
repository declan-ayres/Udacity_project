import argparse
import logging
import os
import sys
import gzip
import struct
import glob
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import PIL
import re



log = logging.getLogger()

#define the function to create the mnist files
def create_mnist(input_file, training_data_file_name):
	input_file_list = os.listdir(input_file)
	#Sort the images in numerical order
	input_file_list.sort()
	file_path_list = []
	classifier_list = []
        exclude = ['notelementof','sidewaysC','omega','islash','notA','ia','unknown','triangle','right','dc' ,'elementof','funnychr','om','lessthanequalto','noth','plusminus','ge','forwardslash','derivative','delta','lambda']
        incl_list = ['97','98','99','120','121','122','61','45','43','47','42','46','48','49','50','51','52','53','54','55','56','57']
	#Open the file handle and write the data
	for j in input_file_list:
		file_path_list.append(input_file+"/"+j)
	training_data_file_name_handle = open(training_data_file_name, 'w')
	
	training_data_file_name_handle.write(struct.pack('i',2051))
	
	training_data_file_name_handle.write(struct.pack('i',len(file_path_list)))
	training_data_file_name_handle.write(struct.pack("i",28))
	training_data_file_name_handle.write(struct.pack("i",28))
	
	list_of_train_images=[]
	#Write the numpy arrays to the file
	for i in range(len(file_path_list)):
		print file_path_list[i]
		list_of_train_images.append(np.invert(np.fromfile(file_path_list[i], dtype=np.uint8)))
	composite_array = np.concatenate(list_of_train_images)
	composite_array.tofile(training_data_file_name_handle)
	training_data_file_name_handle.close()
	#Write the file to a gzip file
	file_handle = open(training_data_file_name, 'rb')
	content = file_handle.read()
	file_handle.close()
	file_handle = gzip.open(training_data_file_name, 'wb')
	file_handle.write(content)
	file_handle.close()


	

#Make the command line arguments
if __name__ == '__main__':	
    parser = argparse.ArgumentParser(description = "path name")
    parser.add_argument("--input_file")
    parser.add_argument("--training_data_file_name")
    parser.add_argument("--logging_level",type=int)
    parser.set_defaults(logging_level = logging.INFO)
    args = parser.parse_args()
    input_file = args.input_file
    training_data_file_name = args.training_data_file_name
    
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
    create_mnist(input_file, training_data_file_name)
