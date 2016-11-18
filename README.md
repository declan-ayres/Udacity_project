# Udacity_project

Use the extract contours program to extract the contours from the image. Then run the create_mnist_train only file on the created directory to create the mnist files. Now you can run either the tensorflow cnn or the svm program to predict.

# Training

To get the font files from the unix server, first run this command: sudo find / -name "*.[ot]tf"  > allfonts-paths-from-server 

This command will put all the font filenames into the allfonts-paths-from-server file. After this run extract_glyphs.py by doing this command: python extract_glyphs.py
 
 This will extract the glyph files and store them in a glyphs_output directory.
 After this you will create the mnist files by running the create_mnist.py file. To run this use the command:
 
 python create_mnist.py --input_file=full path to/glyphs_output --training_data_file_name=a.gz --training_label_file_name=b.gz --test_data_file_name=c.gz --test_label_file_name=d.gz
 
 And this will create the 4 mnist files needed to run the algorithms.
 
 To run the cnn algorithm type this:
 
 python tensorflow_cnn.py --data_dir=. --train_file=a.gz --test_file=c.gz --train_label=b.gz --test_label=d.gz --config_file=config.tensorflow.json --train
 
 To run the svm algorithm: 
 python svm.py --data_dir=. --train_file=a.gz --test_file=c.gz --train_label=b.gz --test_label=d.gz --config_file=config.tensorflow.json --no_predict
 
 If you dont want to do the preprocessing just run the algorithms with the existing a,b,c,d gz files.
 
 
# Predicting

To predict image use my app to take a picture of an equation and send it to the server where the files are. Then run this command:

python extract_and_print_contours --my_path=pathto/image.jpg --root_path=.

This will create an images_output directory inside of which is the image inside of which is the contours directory which contain the numpy arrays images. 

Now to make the one mnist file from this directory run this program command:
python create_mnist_train_only.py --input_file=fullpath to/images_output/image_name/contours --training_data_file_name=f.gz

Now you have the mnist file you can predict with the saved models:
To predict with cnn algorithm:
python tensorflow_cnn.py --data_dir=. --train_file=f.gz --test_file=c.gz --train_label=b.gz --test_label=d.gz --config_file=config.tensorflow.json --no_train

To predict with svm:
python svm.py --data_dir=. --train_file=a.gz --test_file=f.gz --train_label=b.gz --test_label=d.gz --config_file=config.tensorflow.json --predict
