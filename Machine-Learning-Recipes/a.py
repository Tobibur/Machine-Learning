from sklearn.datasets import load_iris
iris = load_iris()
print (iris.feature_names)
print (iris.target_names)
for i in range (len(iris.target)):
	print ("Example %d: label %s, features %s" % (i,iris.target[i],iris.data[i]))