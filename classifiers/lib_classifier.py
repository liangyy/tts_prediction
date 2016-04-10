## library for classifier

## to use svm classifier
## do the following things:
# import lib_classifier
# feature_gen = lib_classifier.Feature_Generator()
# feature_gen.add_function('your_feature_function', 'param_for_this_func')
# feature_gen.add_function('your_feature_function_another', 'param_for_this_func_another')
# my_svm = lib_classifier.My_Classifier(feature_gen, 'svm', [C, gamma])
# my_svm.set_up_param() # this help your modify your param for svm
# my_svm.train(raw_data, lables) # raw_data is literally raw data, here is sequences

import sys
from sklearn import svm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
import matplotlib
class My_Classifier:

	def __init__(self, feature_generator, name, param):
		self.feature_generator = feature_generator
		self.classifier_name = name
		self.classifier_param = param
		self.dict_param = {'svm': 'svm_param', 'logistic': 'log_param'}
		self.dict_classifier = {'svm' : 'svm.SVC()', 'logistic' : 'LogisticRegression()'}
		self.Classifier = eval(self.dict_classifier[name]) # further set up parameters can be done manually or via functions
		
	def set_up_param(self):
		my_cls = Set_Param_For_Classifier()
		
		method = getattr(my_cls, self.dict_param[self.classifier_name])
		method(self.Classifier, self.classifier_param)
	
	def train(self, raw_data, labels): # raw_data is list of seq
		design_matrix = self.convert_data_to_feature(raw_data)
		self.Classifier.fit(design_matrix, labels)

	def convert_data_to_feature(self, raw_data):
		design_matrix = []
		for seq in raw_data:
			seq_feature = self.feature_generator.gen_feature_from_seq(seq)
			design_matrix.append(seq_feature)
		design_matrix = StandardScaler().fit_transform(design_matrix)
		return np.array(design_matrix)

	def predict(self, raw_data):
		design_matrix = self.convert_data_to_feature(raw_data)
		re = list(self.Classifier.predict(design_matrix))
		return re

	def visualize(self, raw_data, labels):
		design_matrix = self.convert_data_to_feature(raw_data)
		colors = ['red','green']
		pca_fit = SparsePCA()
		tmp = pca_fit.fit_transform(design_matrix)
		plt.scatter(tmp[:, 1], tmp[:, 2], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
		plt.show()


class Feature_Generator:

	def __init__(self):
		self.list_of_feature_generation_functions = []

	def gen_feature_from_seq(self, seq):
		features = []
		funcs = self.list_of_feature_generation_functions
		for func in funcs:
			feature = func.run_gen_feature_func(seq)
			features += feature
		return features

	def add_function(self, func_name, func_param):
		func_cls = Call_Feature_Generation_Function(func_name, func_param)
		self.list_of_feature_generation_functions.append(func_cls)


class Call_Feature_Generation_Function:

	def __init__(self, name, param):
		self.func_name = name
		self.func_param = param

	def run_gen_feature_func(self, seq):
		param = self.func_param
		func_name = self.func_name
		my_cls = Feature_Generation_Functions_Lib()
		method = getattr(my_cls, func_name)
		return method(seq, param)


class Feature_Generation_Functions_Lib(object):
	def gc_content(self, seq, param):
		tmp = list(seq)
		length = len(tmp)
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

	#new function here

class Set_Param_For_Classifier(object):
	def svm_param(self, classifier_object, param_list):
		classifier_object.C = param_list[0]
		classifier_object.gamma = param_list[1]
		kernel_dict = {'rbf':1, 'linear':1, 'poly':1, 'sigmoid':1}
		if len(param_list) > 2:
			if param_list[2] in kernel_dict:
				classifier_object.kernel = param_list[2]
			else:
				krn = Kernels()
				method = getattr(krn, param_list[2])
				classifier_object.kernel = method
		#do some set up
	def log_param(self, classifier_object, param_list):
		classifier_object.C = param_list[0]
		classifier_object.penalty = param_list[1]
	## def other_classifiers(self, classifier_object, param_list):

class Kernels(object): #customized kernel
	def test_kernel(self, X, Y):
		return np.dot(X, Y.T)

















