import csv
import numpy as np

csv_obj = csv.reader(open("train.csv", "rb"))
header = csv_obj.next()


data = []
for row in csv_obj:
	data.append(row)

data = np.array(data)
# print data


#float because we are calculating the proportion
number_passengers = np.size(data[:,1].astype(np.float))
number_survived = np.sum(data[:,1].astype(np.float))
proportion = number_survived/number_passengers


# abelow two values return a list of boolean value for each row in data 
# we use them as a mask to filter

is_female = data[:,4] == "female"
is_male = data[:,4] == "male"

# returns the column 1 values for women and men respectively
women_onboard = data[is_female, 1].astype(float)
men_onboard = data[is_male,1].astype(float)

proportion_men_survived = np.sum(men_onboard)/np.size(men_onboard)
proportion_women_survived = np.sum(women_onboard)/np.size(men_onboard)

print "Proportion of men survied = {0}".format(proportion_men_survived)
print "Proportion of women survied = {0}".format(proportion_women_survived)

# Now that I have my indicator that women were much more likely to survive,
# I am done with the training set.
# Now I will read in the test file and write out my simplistic prediction:
# if female, then model that she survived (1) 
# if male, then model that he did not survive (0)

# First, read in test.csv
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# Also open the a new file so I can write to it. Call it something descriptive
# Finally, loop through each row in the train file, and look in column index [3] (which is 'Sex')
# Write out the PassengerId, and my prediction.

predictions_file = open("gendermodeltest.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])	# write the column headers
for row in test_file_object:									# For each row in test file,
    if row[3] == 'female':										# is it a female, if yes then
        predictions_file_object.writerow([row[0], "1"])			# write the PassengerId, and predict 1
    else:														# or else if male,
        predictions_file_object.writerow([row[0], "0"])			# write the PassengerId, and predict 0.
test_file.close()												# Close out the files.
predictions_file.close()



full_data = data
import dt
data = dt.preprocess(data)
# print data[1:50]
# set1,set2 = dt.split(data,2,["child"])


# val, l = dt.gini_impurity_attr(data,2)
# sol = dt.best_split(data)

attribute_list = range(0,data.shape[1]-1)
tree = dt.decisionnode()
tree.create_tree(data, attribute_list)

print "Yo"
print tree.col
tree.print_tree()



