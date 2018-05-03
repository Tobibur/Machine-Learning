from scipy.spatial import distance

def euc(a,b):
	return distance.euclidean(a,b)

class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
		
	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions
		
	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0
		for i in range(1,len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]


# import a dataset
from sklearn import datasets
iris = datasets.load_iris()
 
x = iris.data
y = iris.target
 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# Classification
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()
# my first classifier
my_classifier = ScrappyKNN()

 
my_classifier.fit(X_train, y_train)
 
predictions = my_classifier.predict(X_test)

#print (predictions)
'''[2 1 0 2 0 0 0 1 2 0 1 0 1 2 0 0 0 2 0 2 2 1 1 1 0 0 0 0 2 2 0 2 2 2 0 1 1
 2 2 0 1 1 2 0 2 0 1 1 2 0 0 0 0 1 1 1 1 2 0 0 1 1 0 0 0 1 1 1 2 2 2 2 0 2
 0]'''


#accuracy
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))

'''0.92'''