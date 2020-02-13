from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 
iris = datasets.load_iris()
print(iris.target_names)#prints output columns(classes)
print(iris.feature_names) # prints feature(columns names)
#print(iris.data) prints complete dataset
print(iris.data[:5])
print(iris.target) # prints all the target variables()
print(iris.target.shape)


data=pd.DataFrame({'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],'species':iris.target })

#First, we separate the columns into dependent and independent variables
#(or features and labels). Then you split those variables into a training
#and test set.
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # converting in DF
y=data['species'] # converting into pandas series


#splitting the dataset into train & test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# Random Forest Algorigthm object creation
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
a=clf.predict(X_test)
print("Accuracy score is:",metrics.accuracy_score(a,y_test)*100)
print(clf.predict([[3,5,4,2]]))


#extracting the importanct features which has more weight in determining the output
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
print(feature_imp)


# Visualization!!!!Creating a bar plot to measure the weights of the individual features
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()



#If we can remove the "sepal width" feature because it has very 
#low importance, and select the 3 remaining features.
A=data[['petal length', 'petal width', 'sepal length']]
b=data['species']
A_train,A_test,b_train,b_test=train_test_split(A,b,test_size=0.3)
clf2=RandomForestClassifier(n_estimators=100)
clf2.fit(A,b)
pr=clf2.predict(A_test)

#Visualization!!!Creating a bar plot to measure the weights of 3 features
feature_imp = pd.Series(clf2.feature_importances_,index=A.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp,y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
print("Accuracy score with 3 features:",metrics.accuracy_score(pr,b_test)*100)