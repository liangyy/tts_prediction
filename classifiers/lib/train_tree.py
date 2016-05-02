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
	print('python train_tree.py [training_set] [motif_path] [save_classifier_as] [output_report]')
	# print
	# print('if you want to save your classifier, type \'save\' at [save_or_not] and fill in the classifier name to save as')
	# print('classifier is saved in three files: XXX.cls, XXX.features and XXX.classifier')
	# print
	print('for example:')
	print('[training_set]: training.data')
	# print('[test_set]: test.data')
	print('[motif_path]: ./motifs')
	# print('[save_or_not]: save')
	# print('[save_name_if_save]: XXX')
	# print
	sys.exit()


raw_data = sys.argv[1] # training set
# my_test = sys.argv[2] # test set
# mode = sys.argv[4] # if save classifier 

path = sys.argv[2]
output = sys.argv[4]
save_name = sys.argv[3]
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

# RNA secondary structure
feature_gen.add_function('rna_struct', [15, '']) # usage of RNA secondary structure feature. NUM means the length of the sequence you want to analyze the structure. more details are in lib_classifier

# Motif Score Summarized by Structure
feature_gen.add_function('motif_struct_pair', ['meme_probMatrix.txt', 20]) # usage of motif score feature. 
feature_gen.add_function('motif_struct_pair', ['motif1.motif', 20])
feature_gen.add_function('motif_struct_pair', ['motif2.motif', 20])
feature_gen.add_function('motif_struct_pair', ['motif4.motif', 20])
feature_gen.add_function('motif_struct_pair', ['motif3.motif', 20])
feature_gen.add_function('motif_struct_pair', ['dreme 1st AYAYA.txt', 20])
feature_gen.add_function('motif_struct_pair', ['dreme 2nd AAWWAA.txt', 20])
feature_gen.add_function('motif_struct_pair', ['dreme 3rd KTTMA.txt', 20])
feature_gen.add_function('motif_struct_pair', ['dreme 4th AGCAAY.txt', 20])
feature_gen.add_function('motif_struct_pair', ['dreme 5th GTAATW.txt', 20])
feature_gen.add_function('motif_struct_pair', ['dreme 6th CATD.txt', 20])
feature_gen.add_function('motif_struct_pair', ['dreme 7th CAAWCA.txt', 20])
feature_gen.add_function('motif_struct_pair', ['dreme 8th GCAGY.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 5 1st.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 5 2nd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 5 3rd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 5 4th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 5 5th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 6 1st.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 6 2nd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 6 3rd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 6 4th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 6 5th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 7 1st.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 7 2nd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 7 3rd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 7 4th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 7 5th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 8 1st.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 8 2nd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 8 3rd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 8 4th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 8 5th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 9 1st.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 9 2nd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 9 3rd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 9 4th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 9 5th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 10 1st.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 10 2nd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 10 3rd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 10 4th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w 10 5th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w auto 1st.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w auto 2nd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w auto 3rd.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w auto 4th.txt', 20])
feature_gen.add_function('motif_struct_pair', ['meme -w auto 5th.txt', 20])
# RNA Folding Energy
feature_gen.add_function('struct_energy', [15, '']) # usage of RNA secondary structure energy feature. NUM means the length of the sequence you want to analyze the structure. more details are in lib_classifier
#################################################################################
(raw_data, labels) = data_read(raw_data) 
output = open(output, 'w')
# output.write('No-param\n')
# gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# c = [1, 5, 10, 20, 50, 100]
# for g in gamma:
# for i in c:
# 	print '------------------------'
# 	print 'N_estimators =', i
my_classifier = lib_classifier.My_Classifier(feature_gen, 'tree', [1000], path) # create a classifier 								
(raw_data, labels) = data_read(raw_data) 
my_classifier.train(raw_data, labels)
re = my_classifier.predict(raw_data)
score = accuracy(re, labels)
	# score = my_classifier.cv(raw_data, labels, fold)
	# print score
print 'accuracy =', score
output.write(' '.join(['training accuracy =', str(score)]))
info = my_classifier.Classifier.get_params()
output.write('### classifer info ###\n')
for i in info.keys():
	output.write('\t'.join([i, str(info[i])]) + '\n')
output.close()

save_instance(save_name, my_classifier)