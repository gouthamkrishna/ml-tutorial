from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

predictions = clf.predict([[150, 0], [130, 0], [160, 0], [150, 1], [160, 1], [120, 1]])
print predictions