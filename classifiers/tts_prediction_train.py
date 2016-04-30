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

(raw_data, labels) = data_read(my_in) # read training set from file
#print(len(raw_data))
feature_gen = lib_classifier.Feature_Generator() # create feature generator
feature_gen.add_function('gc_content', 10) # usage of GC content feature. NUM means how many bins 
feature_gen.add_function('motif_score', ['meme_probMatrix.txt', '-b']) # usage of motif score feature. 
feature_gen.add_function('motif_score', ['motif1.motif', '-b']) # param is [motif_file_name, mode], where if mode == '-b', it will only return best score
feature_gen.add_function('motif_score', ['motif2.motif', '-b']) 
feature_gen.add_function('motif_score', ['motif3.motif','-b']) 
feature_gen.add_function('motif_score', ['motif4.motif', '-b']) 





#43 motifs#######
feature_gen.add_function('motif_score', ['dreme 1st AYAYA.txt', '-b'])
feature_gen.add_function('motif_score', ['dreme 2nd AAWWAA.txt', '-b'])
feature_gen.add_function('motif_score', ['dreme 3rd KTTMA.txt', '-b'])
feature_gen.add_function('motif_score', ['dreme 4th AGCAAY.txt', '-b'])
feature_gen.add_function('motif_score', ['dreme 5th GTAATW.txt', '-b'])
feature_gen.add_function('motif_score', ['dreme 6th CATD.txt', '-b'])
feature_gen.add_function('motif_score', ['dreme 7th CAAWCA.txt', '-b'])
feature_gen.add_function('motif_score', ['dreme 8th GCAGY.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 5 1st.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 5 2nd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 5 3rd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 5 4th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 5 5th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 6 1st.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 6 2nd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 6 3rd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 6 4th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 6 5th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 7 1st.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 7 2nd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 7 3rd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 7 4th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 7 5th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 8 1st.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 8 2nd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 8 3rd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 8 4th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 8 5th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 9 1st.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 9 2nd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 9 3rd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 9 4th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 9 5th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 10 1st.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 10 2nd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 10 3rd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 10 4th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w 10 5th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w auto 1st.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w auto 2nd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w auto 3rd.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w auto 4th.txt', '-b'])
feature_gen.add_function('motif_score', ['meme -w auto 5th.txt', '-b'])
###########






feature_gen.add_function('rna_struct', [14, '-up']) # usage of RNA secondary structure feature. NUM means the length of the sequence you want to analyze the structure. more details are in lib_classifier
feature_gen.add_function('motif_struct_pair', ['meme_probMatrix.txt', 20]) # usage of motif score feature. 
feature_gen.add_function('motif_struct_pair', ['motif1.motif', 20])
feature_gen.add_function('motif_struct_pair', ['motif2.motif', 20])
feature_gen.add_function('motif_struct_pair', ['motif4.motif', 20])
feature_gen.add_function('motif_struct_pair', ['motif3.motif', 20])





# feature_gen.add_function('struct_energy', [14, '-up']) # usage of RNA secondary structure energy feature. NUM means the length of the sequence you want to analyze the structure. more details are in lib_classifier



my_classifier = lib_classifier.My_Classifier(feature_gen, 'tree', [100,1,'rbf_linear_structural_motif_sim', 100], path) # create a classifier 
									# 1: feature generator defined above ; 2: classifier type ; 3. param for classifier
									# now we only have 'svm' and 'logistic' 
# my_classifier.set_up_param() # set param for classifier (sorry, now this is not automatically, so we need to run this line)

my_classifier.train(raw_data, labels) # training

# my_classifier.feature_to_csv(raw_data, labels, 'train.csv')

(raw_data_test, labels_test) = data_read(my_test) # read test set from file
# my_classifier.feature_to_csv(raw_data_test, labels_test, 'test.csv')
re = my_classifier.predict(raw_data_test) # predict
# print(raw_data_test)
accuracy(re, labels_test) # generate a report for prediction
# my_classifier.visualize(raw_data, labels) # visualize data as scatter polt (PCA is applied)

if mode == 'save':
	save_name = sys.argv[5]
	save_instance(save_name, my_classifier)

# new_cls = load_instance(save_name)
# new_cls.predict(raw_data_test)
# accuracy(re, labels_test)

