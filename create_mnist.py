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


def create_mnist(input_file, training_data_file_name, training_label_file_name, test_data_file_name, test_label_file_name):
	input_file_list = os.listdir(input_file)
	#input_file_list = the_input_file.readlines()
	#the_input_file.close()
	file_path_list = []
	classifier_list = []
#	dict_handle = open('current_class_count_ascii_only', 'r')
#	dict_list = dict_handle.readlines()
#	dict_with_class_to_ascii_conversion = {}
#	sum1 = 0
#	for j in dict_list:
#                m = re.search('(.*)',str(j))
#                if m:
#                   continue
#		x = j#.replace("(", "").replace(")", "").replace("'", "")
#		x = x.split("-")
#		y = x[2].split(".")
#		print x
#		sum1 += int(x[2].rstrip('\n'))
#		dict_with_class_to_ascii_conversion[x[1]] = str(x[0]).strip()
#	print sum1
        biggest_c = -1
        exclude = ['notelementof','sidewaysC','omega','islash','notA','ia','unknown','triangle','right','dc' ,'elementof','funnychr','om','lessthanequalto','noth','plusminus','ge','forwardslash','derivative','delta','lambda']
        incl_list = ['97','98','99','120','121','122','61','45','43','47','42','46','48','49','50','51','52','53','54','55','56','57']
	for j in input_file_list:
		j = j.rstrip("\n")
		a = j.split('-')
		decimal_code = a[2].split(".")[0]
                c=0
                if decimal_code not in incl_list:
                     continue
		file_path_list.append("glyphs_output/"+j)
                if biggest_c<c:
                    biggest_c = c
		classifier_list.append(int(decimal_code))
	training_data_file_name_handle = open(training_data_file_name, 'w')
	test_data_file_name_handle = open(test_data_file_name, 'w')
        print("Training label file name ",training_label_file_name)
        print("Test label file name ",test_label_file_name)
	training_label_file_name_handle = open(training_label_file_name, 'w')
	test_label_file_name_handle = open(test_label_file_name, 'w')
	
	training_data_file_name_handle.write(struct.pack('i',2051))
	test_data_file_name_handle.write(struct.pack('i',2051))	
	training_label_file_name_handle.write(struct.pack('i',2049))
	test_label_file_name_handle.write(struct.pack('i',2049))

	list_80 = int(.8 * len(file_path_list))
	list_20 = int(.2 * len(file_path_list))
	training_data_file_name_handle.write(struct.pack('i',list_80))
	test_data_file_name_handle.write(struct.pack('i',list_20))
	training_data_file_name_handle.write(struct.pack("i",28))
	training_data_file_name_handle.write(struct.pack("i",28))
	test_data_file_name_handle.write(struct.pack("i",28))	
	test_data_file_name_handle.write(struct.pack("i",28))	
	
	list_of_train_images=[]
	list_of_test_images=[]
	
	for i in range(list_80):
		print file_path_list[i]
		list_of_train_images.append(np.invert(np.fromfile(file_path_list[i], dtype=np.uint8)))
	composite_array = np.concatenate(list_of_train_images)
	composite_array.tofile(training_data_file_name_handle)
	training_data_file_name_handle.close()

	file_handle = open(training_data_file_name, 'rb')
	content = file_handle.read()
	file_handle.close()
	file_handle = gzip.open(training_data_file_name, 'wb')
	file_handle.write(content)
	file_handle.close()

	for i in range(list_80, list_20+list_80):
		list_of_test_images.append(np.invert(np.fromfile(file_path_list[i], dtype = np.uint8)))
	composite_array = np.concatenate(list_of_test_images)
	composite_array.tofile(test_data_file_name_handle)
	test_data_file_name_handle.close()
	
	file_handle = open(test_data_file_name, 'rb')
	content = file_handle.read()
	file_handle.close()
	file_handle = gzip.open(test_data_file_name, 'wb')
	file_handle.write(content)
	file_handle.close()
	
	label80 = int(.8 * len(classifier_list))
	label20 = int(.2 * len(classifier_list))

        new_classifier_list = []
        for j in classifier_list:
            new_classifier_list.append(j)#int(dict_with_class_to_ascii_conversion[j]))
	print len(new_classifier_list)
        print label80
        print label20
        integer_format_string = ""
        training_label_file_name_handle.write(struct.pack("i",label80))
	test_label_file_name_handle.write(struct.pack("i", label20))
	for i in range(label80):
		integer_format_string += "i"

        print ("Length of integer_format_string =",len(integer_format_string))
	training_label_file_name_handle.write(struct.pack(integer_format_string,*new_classifier_list[:label80]))
        training_label_file_name_handle.close()

        file_handle = open(training_label_file_name, 'rb')
        content = file_handle.read()
        file_handle.close()
        file_handle = gzip.open(training_label_file_name, 'wb')
        file_handle.write(content)
        file_handle.close()

	integer_format_string = ""
	for i in range(label20):
		integer_format_string += "i"

	print ("Length of integer_format_string =",len(integer_format_string))
        test_label_file_name_handle.write(struct.pack(integer_format_string,*new_classifier_list[label80:]))
        test_label_file_name_handle.close()

        file_handle = open(test_label_file_name, 'rb')
        content = file_handle.read()
        file_handle.close()
        file_handle = gzip.open(test_label_file_name, 'wb')
        file_handle.write(content)
        file_handle.close()	

        cf = open('config.tensorflow.json','w+')
        cf.write('{\n')
        cf.write('    "tensorflow":{\n')
        cf.write('        "no_classes":'+str(biggest_c)+"\n")
        cf.write('        }'+"\n")
        cf.write('}'+"\n")
        cf.close()



	


if __name__ == '__main__':	
    parser = argparse.ArgumentParser(description = "path name")
    parser.add_argument("--input_file")
    parser.add_argument("--training_data_file_name")
    parser.add_argument("--training_label_file_name")
    parser.add_argument("--test_data_file_name")
    parser.add_argument("--test_label_file_name")
    parser.add_argument("--logging_level",type=int)
    parser.set_defaults(logging_level = logging.INFO)
    args = parser.parse_args()
    input_file = args.input_file
    training_data_file_name = args.training_data_file_name
    training_label_file_name = args.training_label_file_name
    test_data_file_name = args.test_data_file_name
    test_label_file_name = args.test_label_file_name
    
    streamhandler = logging.StreamHandler(sys.stdout)
    
    if args.logging_level==10:
       streamhandler.setLevel(logging.INFO)
       log.setLevel(logging.INFO)
    if args.logging_level==20:
       streamhandler.setLevel(logging.DEBUG)
       log.setLevel(logging.DEBUG)

    filehandler = logging.FileHandler("logging")
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    streamhandler.setFormatter(formatter)
    log.addHandler(streamhandler)
    create_mnist(input_file, training_data_file_name, training_label_file_name, test_data_file_name, test_label_file_name)
