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

# define the function to create the mnist files
def create_mnist(input_file, training_data_file_name, training_label_file_name, test_data_file_name, test_label_file_name):
	input_file_list = os.listdir(input_file)
	file_path_list = []
	classifier_list = []
        exclude = ['notelementof','sidewaysC','omega','islash','notA','ia','unknown','triangle','right','dc' ,'elementof','funnychr','om','lessthanequalto','noth','plusminus','ge','forwardslash','derivative','delta','lambda']
        incl_list = ['97','98','99','120','121','122','61','45','43','47','42','46','48','49','50','51','52','53','54','55','56','57']
	#Extract the unicode ascii values from the file names
	for j in input_file_list:
		j = j.rstrip("\n")
		a = j.split('-')
                print ("A = ",a)
		decimal_code = a[-1].split(".")[0]
                if decimal_code not in incl_list:
                     print("Skipping decimal = ",decimal_code)
                     continue
		file_path_list.append("glyphs_output/"+j)
		classifier_list.append(incl_list.index(decimal_code)+1)
	#Open the file handles
	training_data_file_name_handle = open(training_data_file_name, 'w')
	test_data_file_name_handle = open(test_data_file_name, 'w')
        print("Training label file name ",training_label_file_name)
        print("Test label file name ",test_label_file_name)
	training_label_file_name_handle = open(training_label_file_name, 'w')
	test_label_file_name_handle = open(test_label_file_name, 'w')
	#Write the numbers to the files
	training_data_file_name_handle.write(struct.pack('i',2051))
	test_data_file_name_handle.write(struct.pack('i',2051))	
	training_label_file_name_handle.write(struct.pack('i',2049))
	test_label_file_name_handle.write(struct.pack('i',2049))

	list_80 = int(.8 * len(file_path_list))
	list_20 = len(file_path_list)-list_80
	#Write the lengths to the data files
	training_data_file_name_handle.write(struct.pack('i',list_80))
	test_data_file_name_handle.write(struct.pack('i',list_20))
	training_data_file_name_handle.write(struct.pack("i",28))
	training_data_file_name_handle.write(struct.pack("i",28))
	test_data_file_name_handle.write(struct.pack("i",28))	
	test_data_file_name_handle.write(struct.pack("i",28))	
	
	list_of_train_images=[]
	list_of_test_images=[]
	#Write the numpy arrays to the training data file
	for i in range(list_80):
		print file_path_list[i]
		list_of_train_images.append(np.invert(np.fromfile(file_path_list[i], dtype=np.uint8)))
	composite_array = np.concatenate(list_of_train_images)
	composite_array.tofile(training_data_file_name_handle)
	training_data_file_name_handle.close()
	#Write everything to the gzip training data file
	file_handle = open(training_data_file_name, 'rb')
	content = file_handle.read()
	file_handle.close()
	file_handle = gzip.open(training_data_file_name, 'wb')
	file_handle.write(content)
	file_handle.close()
        #Write the rest of the numpy arrays to the test data file
	for i in range(list_80, list_20+list_80):
		list_of_test_images.append(np.invert(np.fromfile(file_path_list[i], dtype = np.uint8)))
	composite_array = np.concatenate(list_of_test_images)
	composite_array.tofile(test_data_file_name_handle)
	test_data_file_name_handle.close()
	#Write everything to the gzip test data file
	file_handle = open(test_data_file_name, 'rb')
	content = file_handle.read()
	file_handle.close()
	file_handle = gzip.open(test_data_file_name, 'wb')
	file_handle.write(content)
	file_handle.close()
	
	label80 = int(.8 * len(classifier_list))
	label20 = len(classifier_list)-label80
	#Add the classifiers to the list
        new_classifier_list = []
        for j in classifier_list:
            new_classifier_list.append(j)#int(dict_with_class_to_ascii_conversion[j]))
	print len(new_classifier_list)
        print label80
        print label20
	#Write the is to the label files
        integer_format_string = ""
        training_label_file_name_handle.write(struct.pack("i",label80))
	test_label_file_name_handle.write(struct.pack("i", label20))
	for i in range(label80):
		integer_format_string += "i"
	#Write the classifier labels to the training label file
        print ("Length of integer_format_string =",len(integer_format_string))
	training_label_file_name_handle.write(struct.pack(integer_format_string,*new_classifier_list[:label80]))
        training_label_file_name_handle.close()
	#Write it to the gzip file
        file_handle = open(training_label_file_name, 'rb')
        content = file_handle.read()
        file_handle.close()
        file_handle = gzip.open(training_label_file_name, 'wb')
        file_handle.write(content)
        file_handle.close()

	integer_format_string = ""
	for i in range(label20):
		integer_format_string += "i"
	#Write the rest of the classifier labels to the test label file
	print ("Length of integer_format_string =",len(integer_format_string))
        test_label_file_name_handle.write(struct.pack(integer_format_string,*new_classifier_list[label80:]))
        test_label_file_name_handle.close()
	#Write it to the test label gzip file
        file_handle = open(test_label_file_name, 'rb')
        content = file_handle.read()
        file_handle.close()
        file_handle = gzip.open(test_label_file_name, 'wb')
        file_handle.write(content)
        file_handle.close()	
	#Make the config file with the number of classes
        cf = open('config.tensorflow.json','w+')
        cf.write('{\n')
        cf.write('    "tensorflow":{\n')
        cf.write('        "no_classes":'+str(len(incl_list))+"\n")
        cf.write('        }'+"\n")
        cf.write('}'+"\n")
        cf.close()



	

#command line arguments
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
    formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    streamhandler.setFormatter(formatter)
    log.addHandler(streamhandler)
    #Run the function with the command line arguments	
    create_mnist(input_file, training_data_file_name, training_label_file_name, test_data_file_name, test_label_file_name)
