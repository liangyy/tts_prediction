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
import matplotlib
import re
import math
import random
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import fold_energy
from sklearn import metrics
from sklearn import cross_validation
class My_Classifier:

	def __init__(self, feature_generator, name, param, path):
		self.feature_generator = feature_generator
		feature_generator.path = path
		self.classifier_name = name
		self.classifier_param = param
		self.dict_param = {'svm': 'svm_param', 'logistic': 'log_param',\
							'pip_svm' : 'pip_svm_param', \
							'centroid' : 'centroid_param', \
							'tree' : 'tree_param', \
							'boost' : 'boost_param'}
		self.dict_classifier = {'svm' : 'svm.SVC()', 'logistic' : 'LogisticRegression()',\
								'pip_svm' : 'Pipeline([(\'feature_selection\', LinearSVC(penalty=\"l1\"\
								, dual=False, tol=1e-3)),(\'classification\', LinearSVC())])'\
								, 'centroid' : 'NearestCentroid()', \
								'tree' : 'RandomForestClassifier()', \
								'boost' : 'AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), \
algorithm=\"SAMME\", n_estimators=200)'}
		self.Classifier = eval(self.dict_classifier[name]) # further set up parameters can be done manually or via functions
		self.window_size = 'unknown'
		self.kernel_mem = ''
		self.path = path
		self.design_matrix = []
		# self.speed = speed

	def set_up_param(self):
		my_cls = Set_Param_For_Classifier()
		
		method = getattr(my_cls, self.dict_param[self.classifier_name])
		self.kernel_mem = method(self.Classifier, self.classifier_param)
	
	def train(self, raw_data, labels): # raw_data is list of seq
		if len(self.design_matrix) == 0:
			[design_matrix, feature_index, feature_name] = self.convert_data_to_feature(raw_data)
			self.design_matrix = [design_matrix, feature_index, feature_name]
		else:
			[design_matrix, feature_index, feature_name] = self.design_matrix
		self.classifier_param.append(feature_name)
		self.classifier_param.append(feature_index)
		self.set_up_param()
		self.Classifier.fit(design_matrix, labels)

	def cv(self, raw_data, labels, fold):
		if len(self.design_matrix) == 0:
			[design_matrix, feature_index, feature_name] = self.convert_data_to_feature(raw_data)
			self.design_matrix = [design_matrix, feature_index, feature_name]
		else:
			[design_matrix, feature_index, feature_name] = self.design_matrix
		self.classifier_param.append(feature_name)
		self.classifier_param.append(feature_index)
		self.set_up_param()
		return cross_validation.cross_val_score(self.Classifier, design_matrix, labels, cv = fold, scoring='accuracy')

	def convert_data_to_feature(self, raw_data):
		design_matrix = []
		seq_len = len(raw_data[0])
		self.window_size = seq_len
		self.feature_generator.path = self.path
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
			[seq_feature, feature_index, feature_name] = self.feature_generator.gen_feature_from_seq(seq)
			design_matrix.append(seq_feature)
		#design_matrix = StandardScaler().fit_transform(design_matrix)
		# [feature_index, feature_name] = self.feature_generator.feature_info(raw_data[0])
		return [np.array(design_matrix), feature_index, feature_name]

	def predict(self, raw_data):
		if len(raw_data) == 0 and 'N' in raw_data[0]:
			return [2]
		[design_matrix, feature_index, feature_name] = self.convert_data_to_feature(raw_data)

		#print(len(design_matrix[0]))
		#print(design_matrix)
		re = list(self.Classifier.predict(design_matrix))
		#print(self.Classifier.predict(design_matrix))
		# print(len(re))
		# print(re)
		# print(['2'])
		return re

	def visualize(self, raw_data, labels):
		[design_matrix, feature_index, feature_name] = self.convert_data_to_feature(raw_data)
		colors = ['red','green']
		pca_fit = SparsePCA()
		tmp = pca_fit.fit_transform(design_matrix)
		plt.scatter(tmp[:, 1], tmp[:, 2], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
		plt.show()
	

	def feature_to_csv(self, raw_data, label, output_name):
		[design_matrix, feature_index, feature_name] = self.convert_data_to_feature(raw_data)
		#print(label)
		label_num = []
		for i in label:
			#print(i)
			label_num.append(int(i))
		label_num = np.matrix(label_num)
		out = np.hstack([design_matrix, label_num.T])
		np.savetxt(output_name, out, delimiter=",")
		print feature_index
		print feature_name

	def __getstate__(self):
		odict = self.__dict__.copy() # copy the dict since we change it
		del odict['feature_generator']              # remove filehandle entry
		del odict['Classifier']
		return odict
	#def __setstate__(self): self.__dict__.update(d)

class Feature_Generator:

	def __init__(self):
		self.list_of_feature_generation_functions = []
		self.list_of_feature_name = []
		self.list_of_feature_index = []
		self.path = ''

	def gen_feature_from_seq(self, seq):
		features = []
		feature_index = []
		feature_name = self.list_of_feature_name
		index = 0
		funcs = self.list_of_feature_generation_functions
		for func in funcs:
			# print(func)
			feature = func.run_gen_feature_func(seq, self.path)
			to = len(feature)
			feature_index.append([index, index + to])
			index = index + to
			features += feature
		self.list_of_feature_index = feature_index
		return [features, feature_index, feature_name]

	def add_function(self, func_name, func_param):
		func_cls = Call_Feature_Generation_Function(func_name, func_param, self.path)
		self.list_of_feature_generation_functions.append(func_cls)
		self.list_of_feature_name.append(func_name)

	# def feature_info(self, seq):
	# 	feature_index = []
	# 	feature_name = self.list_of_feature_name
	# 	funcs = self.list_of_feature_generation_functions
	# 	index = 0
	# 	for func in funcs:
	# 		feature = func.run_gen_feature_func(seq, self.path)
	# 		to = len(feature)
	# 		feature_index.append([index, index + to])
	# 		index = index + to

	# 	self.list_of_feature_index = feature_index
	# 	return [feature_index, feature_name]

class Call_Feature_Generation_Function:

	def __init__(self, name, param, path):
		self.func_name = name
		self.func_param = param
		self.path = path

	def run_gen_feature_func(self, seq, path):
		param = self.func_param
		self.path = path
		func_name = self.func_name
		# print(func_name)
		my_cls = Feature_Generation_Functions_Lib(self.path)
		method = getattr(my_cls, func_name)
		return method(seq, param)


class Feature_Generation_Functions_Lib(object):
	def __init__(self, path):
		self.path = path

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
		meme_filename = self.path + '/' + param[0]
		#print(self.path, param[0], 'sda')
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
		width = param[0]
		mode = param[1]
		if mode == 'local':
			return fold(seq, 1)
		mid = len(seq) / 2
		up_seq = seq[mid - width : mid]
		down_seq = seq[mid : mid + width]
		all_seq = seq[mid - width : mid + width]
		up_re = fold(up_seq, 1)
		down_re = fold(down_seq, 1)
		all_re = fold(all_seq, 1)
		if mode == '-up':
			return up_re
		# Reference:
		# http://www.tutorialspoint.com/python/list_max.htm
		else:
			return up_re + down_re + all_re

	def motif_struct(self, seq, param):
		def motif_length(meme_filename):
			alength = 0
			width = 0
			probabilityMatrix = []
			with open(meme_filename, 'r') as memeF:
    				content = memeF.readlines()
				for line in content:
					probabilityMatrix.append(re.findall(r'\S+', line))
			width = len(probabilityMatrix)
			return width

		motif = param[0]
		motif_len = motif_length(motif)
		mo_up = motif_len / 2
		mo_do = motif_len - mo_up
		width = param[1]
		mid = len(seq) / 2
		up_seq = seq[mid - width : mid]
		down_seq = seq[mid : mid + width]
		up_seq_mo = seq[mid - width - mo_up: mid + mo_do - 1]
		down_seq_mo = seq[mid - mo_up : mid + width + mo_do - 1]
		up_st = np.array(self.rna_struct(up_seq, [-1, 'local']))
		down_st = np.array(self.rna_struct(down_seq, [-1, 'local']))
		up_sc = np.array(self.motif_score(up_seq_mo, [motif, '']))
		down_sc = np.array(self.motif_score(down_seq_mo, [motif, '']))
		nonloop_sum_up = np.dot(abs(up_st), up_sc.T)
		nonloop_sum_down = np.dot(abs(down_st), down_sc.T)
		loop_sum_up = np.dot(up_st == 0, up_sc.T)
		loop_sum_down = np.dot(down_st == 0, down_sc.T)
		return [loop_sum_up, nonloop_sum_up, loop_sum_down, nonloop_sum_down]

	def motif_struct_pair(self, seq, param):
		def motif_length(meme_filename):
			meme_filename = self.path + '/' + meme_filename
			alength = 0
			width = 0
			probabilityMatrix = []
			with open(meme_filename, 'r') as memeF:
    				content = memeF.readlines()
				for line in content:
					probabilityMatrix.append(re.findall(r'\S+', line))
			width = len(probabilityMatrix)
			return width
		#print('path is ', self.path, param[0])
		motif = param[0]
		#print(motif)
		motif_len = motif_length(motif)
		mo_up = motif_len / 2
		mo_do = motif_len - mo_up
		width = param[1]
		mid = len(seq) / 2
		sub_seq = seq[mid - width : mid + width]
		sub_seq_mo = seq[mid - width - mo_up: mid + width + mo_do - 1]
		st = self.rna_struct(sub_seq, [-1, 'local'])
		sc = self.motif_score(sub_seq_mo, [motif, ''])
		return st + sc


		
	def struct_energy(self, seq, param):
		width = param[0]
		mode = param[1]
		mid = len(seq) / 2
		up_seq = re.sub('T', 'U', seq[mid - width : mid])
		down_seq = re.sub('T', 'U', seq[mid : mid + width])
		all_seq = re.sub('T', 'U', seq[mid - width : mid + width])
		# print(fold(up_seq, 2))
		up_dot = ''.join(fold(up_seq, 2))
		down_dot = ''.join(fold(down_seq, 2))
		all_dot = ''.join(fold(all_seq, 2))
		# print(up_dot, up_seq)
		up_re = fold_energy.energy(up_dot, up_seq)
		down_re = fold_energy.energy(down_dot, down_seq)
		all_re = fold_energy.energy(all_dot, all_seq)
		if mode == '-up':
			return [up_re]
		# Reference:
		# http://www.tutorialspoint.com/python/list_max.htm
		else:
			return [up_re, down_re, all_re]
	#new function here

	



class Set_Param_For_Classifier(object):
	def svm_param(self, classifier_object, param_list):
		classifier_object.C = param_list[0]
		classifier_object.gamma = param_list[1]
		kernel_dict = {'rbf':1, 'linear':1, 'poly':1, 'sigmoid':1}
		if len(param_list) > 2:
			if param_list[2] in kernel_dict:
				classifier_object.kernel = param_list[2]
				return ''
			else:
				krn = Kernels(param_list[4], param_list[5], param_list[3])
				method = getattr(krn, param_list[2])
				classifier_object.kernel = method
				return param_list[2]
		#do some set up
	def log_param(self, classifier_object, param_list):
		classifier_object.C = param_list[0]
		classifier_object.penalty = param_list[1]
		return ''
	def pip_svm_param(self, classifier_object, param_list):
		return ''
	def centroid_param(self, classifier_object, param_list):
		return ''
	def tree_param(self, classifier_object, param_list):
		classifier_object.n_estimators = param_list[0]
		return ''
	def boost_param(self, classifier_object, param_list):
		classifier_object.n_estimators = param_list[0]
		return ''
	## def other_classifiers(self, classifier_object, param_list):


class Kernels: #customized kernel

	def __init__(self, entry_name, entry_list, param):
		self.group_index = entry_list
		self.group_name = entry_name
		self.param = param

	def test_kernel(self, X, Y):
		return np.dot(X, Y.T)
	def rbf_linear_hammington(self, X, Y):
		[gc_x, gc_y, motif_x, motif_y, rna_x, rna_y, motifstr_x, motifstr_y\
		, motifstr_x_sc, motifstr_y_sc, motifstr_x_st, motifstr_x_st\
		, energy_x, energy_y] = self.return_feature_by_type(X, Y)
		#print('gc_x = ' + str(gc_x.shape))
		#print('gc_y = ' + str(gc_y.shape))
		gc = np.apply_along_axis(self.rbf, 0, gc_x.T, gc_y).T
		#print('gc = ' + str(gc.shape))
		motif = self.linear(motif_x, motif_y)	
		#print('motif = ' + str(motif.shape))
		rna = np.apply_along_axis(self.hamming, 0, rna_x.T, rna_y).T
		#print('rna = ' + str(rna.shape))
		rna = rna.astype(float)

		re = np.multiply(gc , motif)
		re = np.multiply(re, rna)
		return re

	def rbf_linear_structural_score(self, X, Y):# this one works poorly
		[gc_x, gc_y, motif_x, motif_y, rna_x, rna_y, motifstr_x, motifstr_y\
		, motifstr_x_sc, motifstr_y_sc, motifstr_x_st, motifstr_y_st\
		, energy_x, energy_y] = self.return_feature_by_type(X, Y)
		gc = np.apply_along_axis(self.rbf, 0, gc_x.T, gc_y).T
		motif = self.linear(motif_x, motif_y)
		motifstr = np.apply_along_axis(self.hamming, 0, motifstr_x.T, motifstr_y).T
		#rna = np.apply_along_axis(self.hamming, 0, rna_x.T, rna_y).T
		#rna = rna.astype(float)
		# print(motifstr)
		re = np.multiply(gc , motif)
		#re = np.multiply(re, rna)
		re = np.multiply(re, motifstr)

		# f, axarr = plt.subplots(2, 2)
		# axarr[0, 0].pcolor(gc)
		# axarr[0, 0].set_title('gc')
		# axarr[0, 1].pcolor(motif)
		# axarr[0, 1].set_title('motif')
		# axarr[1, 0].pcolor(motifstr)
		# axarr[1, 0].set_title('motifstr')
		# axarr[1, 1].pcolor(re)
		# axarr[1, 1].set_title('re')
		# plt.show()
		# print('re = ' + str(re.shape))
		return re

	def final_used_kernel(self, X, Y):
		[gc_x, gc_y, motif_x, motif_y, rna_x, rna_y, motifstr_x, motifstr_y\
		, motifstr_x_sc, motifstr_y_sc, motifstr_x_st, motifstr_y_st\
		, energy_x, energy_y] = self.return_feature_by_type(X, Y)
		gc = np.apply_along_axis(self.rbf, 0, gc_x.T, gc_y).T
		motif = self.linear(motif_x, motif_y)
		motifstr = self.compute_similarity_based_on_motif_and_struct(motifstr_x_sc, motifstr_y_sc, motifstr_x_st, motifstr_y_st)
		# motifstr = np.apply_along_axis(self.hamming, 0, motifstr_x.T, motifstr_y).T
		rna = np.apply_along_axis(self.hamming, 0, rna_x.T, rna_y).T
		rna = rna.astype(float)
		# print(motifstr)
		energy = np.apply_along_axis(self.hamming, 0, energy_x.T, energy_y).T
		re = np.add(gc , motif)
		re = np.add(re, rna)
		re = np.add(re, motifstr)
		re = np.add(re, energy)

		# f, axarr = plt.subplots(2, 3)
		# axarr[0, 0].pcolor(gc)
		# axarr[0, 0].set_title('gc')
		# axarr[0, 1].pcolor(motif)
		# axarr[0, 1].set_title('motif')
		# axarr[1, 0].pcolor(motifstr)
		# axarr[1, 0].set_title('motifstr')
		# axarr[1, 1].pcolor(energy)
		# axarr[1, 1].set_title('energy')
		# axarr[0, 2].pcolor(re)
		# axarr[0, 2].set_title('re')
		# plt.show()
		# # print('re = ' + str(re.shape))
		return re
	def compute_similarity_based_on_motif_and_struct(self, xsc, ysc, xst, yst):
		nx = np.shape(xsc)
		# print(nx)
		nx = nx[0]
		ny = np.shape(ysc)
		ny = ny[0]
		re = np.zeros((nx, ny))

		for i in range(nx):
			for j in range(ny):
				tmpxsc = xsc[i, ]
				tmpysc = ysc[j, ]
				tmpxst = xst[i, ]
				tmpyst = yst[j, ]
				tmp = np.multiply((tmpxst == 0), (0 == tmpyst))
				#print(tmp)
				re[i, j] = (np.dot(tmp.T, tmpxsc) + np.dot(tmp.T, tmpysc))
		return re
	def return_feature_by_type(self, X, Y):
		entry_name = self.group_name
		entry_list = self.group_index

		nx = np.shape(X)
		nx = nx[0]
		ny = np.shape(Y)
		ny = ny[0]
		gc_x = np.array([]).reshape(nx, 0)
		gc_y = np.array([]).reshape(ny, 0)
		for f in range(len(entry_name)):
			if entry_name[f] == 'gc_content':
				gc_x = np.hstack([gc_x, X[: , entry_list[f][0] : entry_list[f][1]]])
				gc_y = np.hstack([gc_y, Y[: , entry_list[f][0] : entry_list[f][1]]])

		rna_x = np.array([]).reshape(nx, 0)
		rna_y = np.array([]).reshape(ny, 0)
		for f in range(len(entry_name)):
			if entry_name[f] == 'rna_struct':
				rna_x = np.hstack([rna_x, X[: , entry_list[f][0] : entry_list[f][1]]])
				rna_y = np.hstack([rna_y, Y[: , entry_list[f][0] : entry_list[f][1]]])

		motif_x = np.array([]).reshape(nx, 0)
		motif_y = np.array([]).reshape(ny, 0)
		for f in range(len(entry_name)):
			if entry_name[f] == 'motif_score':
				motif_x = np.hstack([motif_x, X[: , entry_list[f][0] : entry_list[f][1]]])
				motif_y = np.hstack([motif_y, Y[: , entry_list[f][0] : entry_list[f][1]]])

		motifstr_x = np.array([]).reshape(nx, 0)
		motifstr_y = np.array([]).reshape(ny, 0)
		for f in range(len(entry_name)):
			if entry_name[f] == 'motif_struct':
				motifstr_x = np.hstack([motifstr_x, X[: , entry_list[f][0] : entry_list[f][1]]])
				motifstr_y = np.hstack([motifstr_y, Y[: , entry_list[f][0] : entry_list[f][1]]])

		motifstr_x_sc = np.array([]).reshape(nx, 0)
		motifstr_y_sc = np.array([]).reshape(ny, 0)
		motifstr_x_st = np.array([]).reshape(nx, 0)
		motifstr_y_st = np.array([]).reshape(ny, 0)
		for f in range(len(entry_name)):
			if entry_name[f] == 'motif_struct_pair':
				cut = (entry_list[f][1] + entry_list[f][0]) / 2
				# print(entry_list[f][1], cut, entry_list[f][0])
				motifstr_x_st = np.hstack([motifstr_x_st, X[: , entry_list[f][0] : cut]])
				motifstr_y_st = np.hstack([motifstr_y_st, Y[: , entry_list[f][0] : cut]])
				motifstr_x_sc = np.hstack([motifstr_x_sc, X[: , cut : entry_list[f][1]]])
				motifstr_y_sc = np.hstack([motifstr_y_sc, Y[: , cut : entry_list[f][1]]])

		energy_x = np.array([]).reshape(nx, 0)
		energy_y = np.array([]).reshape(ny, 0)
		for f in range(len(entry_name)):
			if entry_name[f] == 'struct_energy':
				energy_x = np.hstack([energy_x, X[: , entry_list[f][0] : entry_list[f][1]]])
				energy_y = np.hstack([energy_y, Y[: , entry_list[f][0] : entry_list[f][1]]])
				
		return [gc_x, gc_y, motif_x, motif_y, rna_x, rna_y\
		, motifstr_x, motifstr_y, motifstr_x_sc, motifstr_y_sc\
		, motifstr_x_st, motifstr_y_st, energy_x, energy_y]

		
	def linear(self, x, y):
		return np.dot(x, y.T)
	def hamming(self, x, y):
		n = np.shape(y)
		n = n[1]
		re = np.apply_along_axis(self._hamming, 1, y, x)
		return re
	def _hamming(self, x, y):
		return math.exp(-sum(abs(x - y)))
	def rbf(self, x, y):
		n = np.shape(y)
		n = n[1]
		re = np.apply_along_axis(self._rbf, 1, y, x)
		return re.T
	def _rbf(self, x, y):
		delta = x - y
		re = sum(np.multiply(delta, delta))
		# print(re, self.param)
		re = math.exp(- re / 2 / self.param)
		return re

def data_read(mf):
	raw_data = []
	labels = []
	f = file(mf).read().splitlines()
	for line in f:
		line = line.split(' ')
		labels.append(int(line[0]))
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
	return (accuracy_list[0][0] + accuracy_list[1][1]) / float(accuracy_list[0][0] + accuracy_list[1][1] + accuracy_list[1][0] + accuracy_list[0][1])

def save_instance(save_name, obj):
	pickle.dump(obj, open(save_name + '.cls', 'wb'))
	pickle.dump(obj.feature_generator, open(save_name + '.features', 'wb'))
	if obj.kernel_mem != '':
		obj.Classifier.kernel = 'rbf'
	pickle.dump(obj.Classifier, open(save_name + '.classifier', 'wb'))

def load_instance(save_name, path):
	f = open(save_name + '.cls', 'rb')
	my_classifier = pickle.load(f)
	f = open(save_name + '.features', 'rb')
	feature_generator = pickle.load(f)
	my_classifier.feature_generator = feature_generator
	f = open(save_name + '.classifier', 'rb')
	classifier = pickle.load(f)
	my_classifier.Classifier = classifier
	if my_classifier.kernel_mem != '':
		my_classifier.set_up_param()
	path = re.sub('/$', '', path)
	my_classifier.path = path
	return my_classifier

def fold(sequence, mode):
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
		if mode == 1:
			dot.append(0)
		elif mode == 2:
			dot.append('.')
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
	if mode == 1:
		for i in basepairs:
			k=i[0]
			j=i[1]
			dot[k]=-1
			dot[j]=1
		return dot
	elif mode == 2:
		for i in basepairs:
			k=i[0]
			j=i[1]
			dot[k]='('
			dot[j]=')'
		return dot

def score(B1, B2, i, j):
	if abs(j-i) <= 4:
		return 0
	if (B1 == "C" and B2 == "G"):
		return 3
	if (B1 == "G" and B2 == "C"):
		return 3
	if (B1 == "A" and B2 == "T"):
		return 2
	if (B1 == "T" and B2 == "A"):
		return 2
	if (B1 == "T" and B2 == "G"):
		return 2
	if (B1 == "G" and B2 == "T"):
		return 2
	return 0
		
def reverse_complementary(reverse_seq):
	seq = list(reverse_seq)
	lib = {'A' : 'T', 'T' : 'A', 'G' : 'C', 'C' : 'G', 'N' : 'N'}
	re = []
	for s in reversed(seq):
		re.append(lib[s])
	return ''.join(re)

def update_reverse_seq(reverse_seq, new_char):
	seq = list(reverse_seq)
	lib = {'A' : 'T', 'T' : 'A', 'G' : 'C', 'C' : 'G', 'N' : 'N'}
	seq.pop()
	seq.insert(0, lib[new_char])
	return ''.join(seq)











