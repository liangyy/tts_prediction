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
import re
import math
import random
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

class My_Classifier:

	def __init__(self, feature_generator, name, param):
		self.feature_generator = feature_generator
		self.classifier_name = name
		self.classifier_param = param
		self.dict_param = {'svm': 'svm_param', 'logistic': 'log_param',\
							'pip_svm' : 'pip_svm_param'}
		self.dict_classifier = {'svm' : 'svm.SVC()', 'logistic' : 'LogisticRegression()',\
								'pip_svm' : 'Pipeline([(\'feature_selection\', LinearSVC(penalty=\"l1\"\
								, dual=False, tol=1e-3)),(\'classification\', LinearSVC())])'}
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
		seq_len = len(raw_data[0])
		for seq in raw_data:
			if len(seq) == 0:
				continue
			delta = seq_len - len(seq)
			if delta != 0:
				#print('warning: sequences have different length')
				#print(seq)
				if delta > 0:
					print(seq)
					print('cannot fix it, exiting')
					sys.exit()
				else:
					seq = seq[:delta]
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

	def motif_score(self, seq, param):
		meme_filename = param[0]
		mode = param[1]
		def pssm(meme_filename):
			alength = 0
			width = 0
			probabilityMatrix = []
			with open(meme_filename, 'r') as memeF:
    				content = memeF.readlines()
				for line in content:
					probabilityMatrix.append(re.findall(r'\S+', line))
			width = len(probabilityMatrix)
			alength = len(probabilityMatrix[0])
			for i in range(0,width):
				for j in range(0,alength):
					probabilityMatrix[i][j] = float(probabilityMatrix[i][j])
				#print probabilityMatrix[i]
			return probabilityMatrix
		def scanSeq(sequence, probabilityMatrix, mode):
			width = len(probabilityMatrix)
			maxScore = -float("inf")
			maxWindow = []
			maxPosition = []
			scores = []
			for position in range(0, len(sequence)-width+1):
				window = ""
				score = 1.0
				for index in range(0, width):
					window = window + sequence[position+index]
					nuc = -1
					if sequence[position+index] == "A":
						nuc = 0
					if sequence[position+index] == "C":
						nuc = 1
					if sequence[position+index] == "G":
						nuc = 2
					if sequence[position+index] == "T":
						nuc = 3
					score = score * probabilityMatrix[index][nuc]
				if score >= maxScore:
					if score > maxScore:
						maxWindow = []
						maxPosition = []
					maxScore = score
					maxWindow.append(window)
					maxPosition.append(position)
				#print position+1, "  ", window, "  ", score
				scores.append(score)
			if mode == '-b':
				return [maxScore]
			else:
				return scores
		probabilityMatrix = pssm(meme_filename)
		scores = scanSeq(seq, probabilityMatrix, mode)
		return scores

## DETAILS of RNA structure predictor
## Suppos here is a sequence (we assume that the very middle one is the TTS) 
## ---------------UP-------MID-------DOWN------------
## rna_struct will return secondary structure of UP------MID, MID------DOWN, and UP----------------DOWN
## where ( is labeled as -1, . is labeled as 0 and ) is labeled as +1
## example: (((.....)))  ==> [-1,-1,-1,0,0,0,0,0,1,1,1]
	def rna_struct(self, seq, param):
		def score(B1, B2, i, j):
			if abs(j-i) <= 4:
				return 0
			if (B1 == "C" and B2 == "G"):
				return 1
			if (B1 == "G" and B2 == "C"):
				return 1
			if (B1 == "A" and B2 == "T"):
				return 1
			if (B1 == "T" and B2 == "A"):
				return 1
			

			return 0
		def fold(sequence):
			length = len(sequence)
			matrix = state = [[[] for _ in range(length)] for _ in range(length)]
			# Initialization
			matrix[0][0] = 0
			for i in range(1, length):
				matrix[i][i] = 0
				matrix[i][i-1] = 0
			# Recursion
			for x in range(1, length):
				for i in range(length):
					j = i + x
					if (i<length and j<length):
						maxOfTheThreeCells = max(matrix[i+1][j],matrix[i][j-1], matrix[i+1][j-1]+score(sequence[i], sequence[j], i, j))
						if j > i+1:
							bifurcation = []
							for k in range(i+1, j):
								bifurcation.append(matrix[i][k]+matrix[k+1][j])
							matrix[i][j] = max(maxOfTheThreeCells, max(bifurcation))
						else:
							matrix[i][j] = maxOfTheThreeCells
			# Traceback
			stack = []
			i = 0
			j = length -1
			stack.append([i,j])
			dot=[]
			basepairs=[]
			for i in xrange(length):
				dot.append(0)
			while not stack==[]:
				pair = stack.pop()
				i = pair[0]
				j = pair[1]
				if (i >= j):
					continue
				elif matrix[i+1][j] == matrix[i][j]:
					stack.append([i+1, j])
				elif matrix[i][j-1] == matrix[i][j]:
					stack.append([i, j-1])
				elif (matrix[i+1][j-1] + score(sequence[i], sequence[j], i, j)) == matrix[i][j]:
					stack.append([i+1, j-1])
					basepairs.append((i,j))	
				else:
					if (j > i+1):
						for k in range(i+1, j):
							if matrix[i][k]+matrix[k+1][j] == matrix[i][j]:
								stack.append([i, k])
								stack.append([k+1, j])
								break
			for i in basepairs:
				k=i[0]
				j=i[1]
				dot[k]=-1
				dot[j]=1
			return dot
		width = param
		mid = len(seq) / 2
		up_seq = seq[mid - width : mid]
		down_seq = seq[mid : mid + width]
		all_seq = seq[mid - width : mid + width]
		up_re = fold(up_seq)
		down_re = fold(down_seq)
		all_re = fold(all_seq)
		# Reference:
		# http://www.tutorialspoint.com/python/list_max.htm
		return up_re + down_re + all_re


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

	def pip_svm_param(self, classifier_object, param_list):
		return
	## def other_classifiers(self, classifier_object, param_list):


class Kernels(object): #customized kernel
	def test_kernel(self, X, Y):
		return np.dot(X, Y.T)

















