#just for debugging
import sys
import lib_classifier

def data_read(mf):
	raw_data = []
	labels = []
	f = file(mf).read().splitlines()
	for line in f:
		line = line.split(' ')
		labels.append(line[0])
		raw_data.append(line[1])
	return (raw_data, labels)

def accuracy(re, labels):
	accuracy_list = [ [0, 0] for i in range(2) ]
	for i in range(len(re)):
		a = int(re[i]) - 1
		b = int(labels[i]) - 1
		accuracy_list[a][b] += 1
	print('    1   2   true')
	print('1 '),
	print(accuracy_list[0])
	print('2 '),
	print(accuracy_list[1])
	print('predict')


my_in = sys.argv[1] # training set
my_test = sys.argv[2] # test set

(raw_data, labels) = data_read(my_in) # read training set from file
#print(len(raw_data))
feature_gen = lib_classifier.Feature_Generator() # create feature generator
feature_gen.add_function('gc_content', 10) # usage of GC content feature. NUM means how many bins 
feature_gen.add_function('motif_score', ['meme_probMatrix.txt', '-b']) # usage of motif score feature. 
feature_gen.add_function('motif_score', ['motif1.motif', '-b']) # param is [motif_file_name, mode], where if mode == '-b', it will only return best score
feature_gen.add_function('motif_score', ['motif2.motif', '-b']) 
feature_gen.add_function('motif_score', ['motif3.motif','-b']) 
feature_gen.add_function('motif_score', ['motif4.motif', '-b']) 
feature_gen.add_function('rna_struct', [7, '-up']) # usage of RNA secondary structure feature. NUM means the length of the sequence you want to analyze the structure. more details are in lib_classifier


my_classifier = lib_classifier.My_Classifier(feature_gen, 'svm', [10,1,'rbf_linear_hammington', 1]) # create a classifier 
									# 1: feature generator defined above ; 2: classifier type ; 3. param for classifier
									# now we only have 'svm' and 'logistic' 
#my_classifier.set_up_param() # set param for classifer (sorry, now this is not automatically, so we need to run this line)

my_classifier.train(raw_data, labels) # training
(raw_data_test, labels_test) = data_read(my_test) # read test set from file
re = my_classifier.predict(raw_data_test) # predict
accuracy(re, labels_test) # generate a report for prediction
#my_classifier.visualize(raw_data, labels) # visualize data as scatter polt (PCA is applied)

