#just for debugging
import sys
import lib_classifier
from lib_classifier import data_read
from lib_classifier import accuracy
from lib_classifier import save_instance
from lib_classifier import load_instance


if sys.argv[1] == '--help':
	print('python tts_prediction_train.py [training_set] [test_set] [save_or_others] [save_name_if_save]')
	sys.exit()

my_in = sys.argv[1] # training set
my_test = sys.argv[2] # test set
mode = sys.argv[3] # if save classifier 

(raw_data, labels) = data_read(my_in) # read training set from file
#print(len(raw_data))
feature_gen = lib_classifier.Feature_Generator() # create feature generator
feature_gen.add_function('gc_content', 10) # usage of GC content feature. NUM means how many bins 
feature_gen.add_function('motif_score', ['meme_probMatrix.txt', '-b']) # usage of motif score feature. 
feature_gen.add_function('motif_score', ['motif1.motif', '-b']) # param is [motif_file_name, mode], where if mode == '-b', it will only return best score
feature_gen.add_function('motif_score', ['motif2.motif', '-b']) 
feature_gen.add_function('motif_score', ['motif3.motif','-b']) 
feature_gen.add_function('motif_score', ['motif4.motif', '-b']) 
feature_gen.add_function('rna_struct', [14, '-up']) # usage of RNA secondary structure feature. NUM means the length of the sequence you want to analyze the structure. more details are in lib_classifier


my_classifier = lib_classifier.My_Classifier(feature_gen, 'tree', [100,1,'rbf_linear_hammington', 100]) # create a classifier 
									# 1: feature generator defined above ; 2: classifier type ; 3. param for classifier
									# now we only have 'svm' and 'logistic' 
#my_classifier.set_up_param() # set param for classifier (sorry, now this is not automatically, so we need to run this line)

my_classifier.train(raw_data, labels) # training

my_classifier.feature_to_csv(raw_data, labels, 'train.csv')

(raw_data_test, labels_test) = data_read(my_test) # read test set from file
my_classifier.feature_to_csv(raw_data, labels_test, 'test.csv')
re = my_classifier.predict(raw_data_test) # predict
#print(raw_data_test)
accuracy(re, labels_test) # generate a report for prediction
#my_classifier.visualize(raw_data, labels) # visualize data as scatter polt (PCA is applied)

if mode == 'save':
	save_name = sys.argv[4]
	save_instance(save_name, my_classifier)

# new_cls = load_instance(save_name)
# new_cls.predict(raw_data_test)
# accuracy(re, labels_test)

