from sklearn import tree
# CHALLENGE - create 3 more classifiers...
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np

clf = {
	'DesicisionTree':tree.DecisionTreeClassifier(),
	'KNeighborsClassifier':KNeighborsClassifier(n_neighbors=3),
	'GaussianNB':GaussianNB(),
	'MLPClassifier':MLPClassifier(alpha=1)
	}

#accuracy
from sklearn.metrics import accuracy_score

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

i=0
acc_tree = []
# CHALLENGE - ...and train them on our data
#print(prediction)
for items in clf:
	clfs = clf[items].fit(X, Y)
	prediction = clf[items].predict(X)
	acc_tree.append(accuracy_score(Y, prediction) * 100)
	print(items)
	print('Accuracy : {}'.format(acc_tree[i]))
	print(prediction)
	print()
	i+=1


# CHALLENGE compare their reusults and print the best one!
index = np.argmax([acc_tree[0],acc_tree[1],acc_tree[2],acc_tree[3]])
classifiers = {0: 'DesicisionTree', 1: 'KNeighborsClassifier', 2: 'GaussianNB', 3: 'MLPClassifier'}
print('Best gender classifier is {}'.format(classifiers[index]))
