import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


iris = sns.load_dataset('iris')
#sns.pairplot(iris,hue='species',palette='Dark2')


X=iris.drop('species', axis=1)
y=iris['species']

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)

s=SVC()
s.fit(X_train, y_train)
pre=s.predict(X_test)
print('\n\naccuracy Score before performance_tuning:',accuracy_score(pre,y_test))
print('\nScore before any performance_tuning:',s.score(X_test,y_test))

print("\n\nNow performing cross validation using 6 folds....\n")
n_folds = 6
scores = cross_val_score(SVC(), X, y, cv=n_folds)
print('Each individual fold val:',scores)
cv_error = np.average(cross_val_score(SVC(), X, y, cv=n_folds))
print("\nAverage accuracy Score of folds is: ", cv_error)
print("\nAnother way of finding accuracy of folds is(np.average(list(scores)):", np.average(list(scores)))


print('\n\nConfusion matrix plot....')
cm = confusion_matrix(y_test, pre)
fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.4)
for i in range(cm.shape[0]):
     for j in range(cm.shape[1]):
         ax.text(x=j, y=i,
                s=cm[i, j], 
                va='center', ha='center')
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values')
plt.show()

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
print('Create a GridSearchCV object and fit it to the training data.')

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))

print(classification_report(y_test,grid_predictions))


