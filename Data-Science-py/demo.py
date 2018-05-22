from sklearn import tree
# CHALLENGE - create 3 more classifiers...
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

clf = [tree.DecisionTreeClassifier(),KNeighborsClassifier(n_neighbors=3),GaussianNB(),MLPClassifier(alpha=1)]

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

x_test = [[190, 70, 43]]

# CHALLENGE - ...and train them on our data
#print(prediction)
for items in clf:
	clfs = items.fit(X, Y)
	prediction = items.predict(x_test)
	print (items)
	print(prediction)


# CHALLENGE compare their reusults and print the best one!
