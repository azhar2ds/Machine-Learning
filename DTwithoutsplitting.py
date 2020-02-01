"""
Import the DecisionTreeClassifier model.
"""

#Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data', names=['animal_name','hair','feathers','eggs','milk',
                                                   'airbone','aquatic','predator','toothed','backbone',
                                                  'breathes','venomous','fins','legs','tail','domestic','catsize','clas'])
print(dataset.shape)
dataset=dataset.drop('animal_name', axis=1)

train_features = dataset.iloc[:80,:-1]
test_features = dataset.iloc[80:,:-1]
train_targets = dataset.iloc[:80,-1]
test_targets = dataset.iloc[80:,-1]

tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)
tree2 = DecisionTreeClassifier(criterion = 'gini').fit(train_features,train_targets)


prediction = tree.predict(test_features)
prediction = tree2.predict(test_features)


print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")
print("The prediction accuracy is: ",tree2.score(test_features,test_targets)*100,"%")
