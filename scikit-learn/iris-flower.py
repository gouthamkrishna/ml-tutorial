import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pandas as pd

iris = load_iris()

print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]

test_idx = [0,1,2,50,51,52,100,101,102]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

for i in range (9):
    print test_data[i], " - ", test_target[i]

predictions = clf.predict(test_data)

for i in range (9):
    print test_data[i], " - ", predictions[i]