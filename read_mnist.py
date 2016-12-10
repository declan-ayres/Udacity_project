import argparse
import logging
import os
import sys
import gzip
import struct
import glob
import json
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import PIL



log = logging.getLogger()

def read_config(filename):
    open_file = open(filename)
    x = open_file.read()
    open_file.close()
    json_file = json.loads(x)
    return json_file



def read_mnist_file(input_file, file_size):
        the_input_file = gzip.open(input_file, 'r')
        magic = struct.unpack("i",the_input_file.read(4))[0]
        if magic != 2049:
            print("This is not an MNIST label file\n")
            exit()
        no_labels = struct.unpack("i",the_input_file.read(4))[0]
	input_file_list = []
	n = 0 
	for i in range(no_labels):
	       n+=1
	 
 	       #input_file_list.append(struct.unpack('i',the_input_file.read(4))[0])#, input_file_list))
 	       c = struct.unpack('i',the_input_file.read(4))[0]#, input_file_list))
	       
               if c <128:
	          print(c,unichr(c).encode('utf-8'))
               else:
                  print(c)
        
        #input_file_list = the_input_file.read(struct.unpack(unpack_read_string, input_file_list))
       
	print n
        the_input_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "path name")
    parser.add_argument("--mnist_label_file")
    parser.add_argument("--logging_level",type=int)
    parser.add_argument("--config_file")
    parser.set_defaults(logging_level = logging.INFO)
    args = parser.parse_args()
    input_file = args.mnist_label_file
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
    read_mnist_file(input_file, no_classes)
