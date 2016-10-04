#Modelling a decision tree to predict

import numpy as np
import itertools


# splits a dataset into two based on the column value
def divide_set(data, column, value):
	split_function = None
	if isinstance(value,int) or isinstance(value, float):
		# split_function = lambda row:row[column] >= value
		set1 = data[data[:,column] >= value]
		set2 = data[data[:,column] < value]
	else:
		# split_function = lambda row:row[column] == value
		set1 = data[data[:,column] == value]
		set2 = data[data[:,column] != value]

	return (set1,set2)



# returns the distinct values and their counts of the column in the given dataset.
# used to find out how mixed a set is
def unique_counts(data, column):
	results = {}
	for row in data:
		r = row[column]
		if r not in results:
			results[r] = 0
		results[r] += 1
	return results


#caluculates the gini impurity for a dataset 
def gini_impurity(data, class_column):
	counts = unique_counts(data,class_column)
	total = np.size(data,0)
	impurity = 0
	for count in counts:
		p = float(counts[count])/total
		p = p**2
		impurity += p

	return (1-impurity)







def age_discretization(value):
	# print type(value)
	if value < 12:
		return "child"
	elif value < 17:
		return "teen"
	elif value < 35:
		return "adult"
	elif value < 59:
		return "middle"
	else:
		return "senior"


def preprocess(data):
	extracted_data = data[:,[2,4,5,1]]
	extracted_data[:,2][extracted_data[:,2] == ''] = '0'
	age = extracted_data[:,2].astype(float)
	discrete_age = [age_discretization(a) for a in age]
	discrete_age = np.array(discrete_age)
	extracted_data[:,2] = discrete_age

	return extracted_data


#split the dataset into two. 
# 1 - attribute in value list
# 2 - attribute not in value list
def split(data, attribute, value_list):
	set1 = data[np.in1d(data[:,attribute], value_list)]
	set2 = data[np.in1d(data[:,attribute], value_list, invert = True)]
	return (set1, set2)




#caluculate the gini impurity (wrt attribute)
def gini_impurity_attr(data, column):
	min_gini = float("inf")
	min_gini_value_list = []
	class_column = data.shape[1]-1 # last column specifies the class
	rows = float(data.shape[0])
	unique_values = np.unique(data[:,column])
	total = np.size(unique_values)
	if total % 2 == 0:
		total = total/2
	else:
		total = (total-1)/2

	total = total + 1
	for i in range(1,total):
		combinations = list(itertools.combinations(unique_values, i))
		combinations = [list(a) for a in combinations]
		for combination in combinations:
			set1,set2 = split(data, column, combination)
			set1_rows = float(set1.shape[0])
			set2_rows = float(set2.shape[1])
			gini_a = (set1_rows/rows)* gini_impurity(set1,class_column) + (set2_rows/rows)*gini_impurity(set2, class_column)
			if gini_a < min_gini:
				min_gini = gini_a
				min_gini_value_list = combination

	return min_gini, min_gini_value_list



def best_split(data):
	#assuming the last column is the class
	# all other columns are in attribute list
	attribute_list = range(0,data.shape[1]-1)
	min_gini_attr_val = float("inf")
	min_gini_attr = -1
	min_gini_selection = []
	for attr in attribute_list:
		gini_a, value_list = gini_impurity_attr(data,attr)
		print gini_a
		if gini_a < min_gini_attr_val:
			min_gini_attr_val = gini_a
			min_gini_attr = attr
			min_gini_selection = value_list
	return min_gini_attr_val, min_gini_attr, value_list

class decisionnode:
	def __init__(self,col=-1, value=None,results= None,tb=None,fb=None):
		self.col = col #column index of the criteria to be tested
		self.value = value #value that the column must match to get a true value
		self.results = results
		self.tb = tb # true node
		self.fb = fb #false node
