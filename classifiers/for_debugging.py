#just for debugging
import sys
import lib_classifier

def gc_content(seq, param):
		tmp = list(seq)
		length = len(seq)
		parts_num = param
		part_size = int(length / param)
		part_starts = [0]
		
		for i in range(1, parts_num):
			part_starts.append(i * part_size)
		
		re = []
		
		for part in part_starts:
			subseq = seq[part : part + part_size]
			tmp_count = 0
			for c in subseq:
				if c == 'G' or c == 'C':
					tmp_count += 1
			re.append(float(tmp_count) / part_size)

		return re

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

feature_gen = lib_classifier.Feature_Generator() # create feature generator
feature_gen.add_function('gc_content', 2) # define first feature generation method (here is trivil -- GC content)
feature_gen.add_function('gc_content', 5) # define another feature generation method (here is trivil -- GC content)

my_svm = lib_classifier.My_Classifier(feature_gen, 'logistic', [10,'l1']) # create a classifier 
									# 1: feature generator defined above ; 2: classifier type ; 3. param for classifier
									# now we only have 'svm' and 'logistic' 
my_svm.set_up_param() # set param for classifer (sorry, now this is not automatically, so we need to run this line)

my_svm.train(raw_data, labels) # training
(raw_data_test, labels_test) = data_read(my_test) # read test set from file
re = my_svm.predict(raw_data_test) # predict
accuracy(re, labels_test) # generate a report for prediction
my_svm.visualize(raw_data, labels) # visualize data as scatter polt (PCA is applied)

