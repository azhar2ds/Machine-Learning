import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#importing the data
df=pd.read_csv('https://raw.githubusercontent.com/azhar2ds/DataSets/master/bb.csv')
#Wrangling the columns
X=df.values[:,2:-1]
Y=df.values[:,-1:]
print(X.shape)
print(Y.shape)

#splitting the dataset in 30%
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=.30, random_state=100)

#Using Decision Tree classifier ALGORITHM
clf=DecisionTreeClassifier(criterion='entropy')#, random_state=100, max_depth=5, min_samples_leaf=3)

#Training the data
clf.fit(X_train,y_train)

#Prediction
prediction=clf.predict(X_test)

#Accuracy Score
print(accuracy_score(prediction,y_test)*100)
