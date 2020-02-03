import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Importing dataset
balance_data=pd.read_csv('https://raw.githubusercontent.com/azhar2ds/DataSets/master/balance-scale.data', header=None)
print(balance_data.head(3))

#Data Slicing
X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]
print("X Shape:",X.shape)
print("Y Shape:",Y.shape)

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


#Decision Tree Classifier with criterion gini index
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 10,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
#DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
# max_features=None, max_leaf_nodes=None, min_samples_leaf=5,min_samples_split=2, min_weight_fraction_leaf=0.0,
# presort=False, random_state=100, splitter='best')



#Decision Tree Classifier with criterion information gain
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                     max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
#DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
# max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
# min_samples_split=2, min_weight_fraction_leaf=0.0,
# presort=False, random_state=100, splitter='best')


#Predicting the value
a=clf_gini.predict([[4, 4, 3, 3]])
print("Value for the condition [4,4,3,3] is :",a)
y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_entropy.predict(X_test)

#Finding the Model Accuracy
print("Accuracy using Gini Index: ", accuracy_score(y_test,y_pred_gini)*100)
print("Accuracy using Information Gain: ", accuracy_score(y_test,y_pred_entropy)*100)

#azhar2ds@gmail.com