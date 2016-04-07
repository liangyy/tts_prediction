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


from sklearn import svm

class My_Classifier:

	def __init__(self, feature_generator, name, param):
		self.feature_generator = feature_generator
		self.classifier_name = name
		self.classifier_param = param
		self.dict_param = {'svm': 'svm_param'}
		self.dict_classifier = {'svm' : 'svm.SVC()'}
		self.Classifier = eval(name) # further set up parameters can be done manually or via functions
		
	def set_up_param(self):
		print(self.dict_param[self.classifier_name])
		my_cls = Set_Param_For_Classifier()
		
		method = getattr(my_cls, self.dict_param[self.classifier_name])
		method(self.Classifier, self.classifier_param)
	
	def train(self, raw_data, labels): # raw_data is list of seq
		design_matrix = []
		for seq in raw_data:
			seq_feature = self.feature_generator.gen_feature_from_seq(raw_data)
			design_matrix.append(seq_feature)
		self.Classifier.fit(design_matrix, labels)

	def predict(self, raw_data):
		design_matrix = []
		for seq in raw_data:
			seq_feature = self.feature_generator.gen_feature_from_seq(raw_data)
			design_matrix.append(seq_feature)
		re = list(self.Classifier.predict(design_matrix, labels))
		return re

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
		re = []
		re.append(len(seq))
		return re

class Set_Param_For_Classifier(object):
	def svm_param(self, classifier_object, param_list):
		classifier_object.C = param_list[0]
		classifier_object.gamma = param_list[1]
		#do some set up
	
	## def other_classifiers(self, classifier_object, param_list):
