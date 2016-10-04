import csv
import numpy as np

csv_obj = csv.reader(open("train.csv", "rb"))
header = csv_obj.next()


data = []
for row in csv_obj:
	data.append(row)

data = np.array(data)


import dt
data = dt.preprocess(data)

attribute_list = range(0,data.shape[1]-1)
tree = dt.decisionnode()
tree.create_tree(data, attribute_list)

tree.print_tree()


csv_test = csv.reader(open("test.csv", "rb"))
header = csv_test.next()

testdata = []
for row in csv_test:
	testdata.append(row)

testdata = np.array(testdata)
testdata = testdata[:,[1,3,4,8,0]]

predictions_file = open("dtmodeltest.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])
for row in testdata:
	print row , tree.predict_value(row[:-1])
	predictions_file_object.writerow([row[-1], tree.predict_value(row[:-1])])
    
# test_file.close()
predictions_file.close()