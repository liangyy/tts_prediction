import sys

import lib_classifier
from lib_classifier import data_read
from lib_classifier import accuracy
from lib_classifier import save_instance
from lib_classifier import load_instance

if len(sys.argv) == 1:
	print
	print('Program for discovering TTSs in Maize Genome')
	print('tts_prediction_train is for training the model, please use tts_prediction for predicting the sites')
	print
	print('USAGE:')
	print('python tts_prediction_train.py [training_set] [test_set] [motif_path] [save_or_not] [save_name_if_save]')
	print
	print('if you want to save your classifier, type \'save\' at [save_or_not] and fill in the classifier name to save as')
	print('classifier is saved in three files: XXX.cls, XXX.features and XXX.classifier')
	print
	print('for example:')
	print('[training_set]: training.data')
	print('[test_set]: test.data')
	print('[motif_path]: ./motifs')
	print('[save_or_not]: save')
	print('[save_name_if_save]: XXX')
	print
	sys.exit()


if sys.argv[1] == '--help':
	print
	print('Program for discovering TTSs in Maize Genome')
	print('tts_prediction_train is for training the model, please use tts_prediction for predicting the sites')
	print
	print('USAGE:')
	print('python tts_prediction_train.py [training_set] [test_set] [motif_path] [save_or_not] [save_name_if_save]')
	print
	print('if you want to save your classifier, type \'save\' at [save_or_not] and fill in the classifier name to save as')
	print('classifier is saved in three files: XXX.cls, XXX.features and XXX.classifier')
	print
	print('for example:')
	print('[training_set]: training.data')
	print('[test_set]: test.data')
	print('[motif_path]: ./motifs')
	print('[save_or_not]: save')
	print('[save_name_if_save]: XXX')
	print
	sys.exit()

my_in = sys.argv[1] # training set
my_test = sys.argv[2] # test set
mode = sys.argv[4] # if save classifier 

path = sys.argv[3]

################################# FEATURES ######################################
# Create a Feature Generator Object
feature_gen = lib_classifier.Feature_Generator() # create feature generator

# GC content
feature_gen.add_function('gc_content', 10) # usage of GC content feature. NUM means how many bins 

# Motifs
feature_gen.add_function('motif_score', ['meme_probMatrix.txt', '-b']) # usage of motif score feature. 
feature_gen.add_function('motif_score', ['motif1.motif', '-b']) # param is [motif_file_name, mode], where if mode == '-b', it will only return best score
feature_gen.add_function('motif_score', ['motif2.motif', '-b']) 
feature_gen.add_function('motif_score', ['motif3.motif','-b']) 
feature_gen.add_function('motif_score', ['motif4.motif', '-b']) 

# RNA secondary structure
feature_gen.add_function('rna_struct', [14, '']) # usage of RNA secondary structure feature. NUM means the length of the sequence you want to analyze the structure. more details are in lib_classifier

# Motif Score Summarized by Structure
feature_gen.add_function('motif_struct_pair', ['meme_probMatrix.txt', 20]) # usage of motif score feature. 
feature_gen.add_function('motif_struct_pair', ['motif1.motif', 20])
feature_gen.add_function('motif_struct_pair', ['motif2.motif', 20])
feature_gen.add_function('motif_struct_pair', ['motif4.motif', 20])
feature_gen.add_function('motif_struct_pair', ['motif3.motif', 20])

# RNA Folding Energy
feature_gen.add_function('struct_energy', [14, '']) # usage of RNA secondary structure energy feature. NUM means the length of the sequence you want to analyze the structure. more details are in lib_classifier
#################################################################################

################################ CLASSIFIER #####################################
# Classifier Used
my_classifier = lib_classifier.My_Classifier(feature_gen, 'tree', [100,1,'rbf_linear_structural_motif_sim', 100], path) # create a classifier 
									# 1: feature generator defined above ; 2: classifier type ; 3. param for classifier
#################################################################################									

################################# TRAINING ######################################
# Training
(raw_data, labels) = data_read(my_in) 
my_classifier.train(raw_data, labels) # load training file
my_classifier.feature_to_csv(raw_data, labels, 'train.csv') # output design matrix of training file

# Testing
(raw_data_test, labels_test) = data_read(my_test) # read test set from file
my_classifier.feature_to_csv(raw_data_test, labels_test, 'test.csv') # output design matrix of testing file
re = my_classifier.predict(raw_data_test) # predict

# Report Training Accuracy
accuracy(re, labels_test) # generate a report for prediction
#################################################################################									

################################## SAVE #########################################									
if mode == 'save':
	save_name = sys.argv[5]
	save_instance(save_name, my_classifier)
#################################################################################									

