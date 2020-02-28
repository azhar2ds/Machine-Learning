import matplotlib.pyplot as plt

#Load libraries for data processing
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.stats import norm

## Supervised learning.
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# visualization
import seaborn as sns 
plt.style.use('fivethirtyeight')
sns.set_style("white")

plt.rcParams['figure.figsize'] = (8,4) 
#plt.rcParams['axes.titlesize'] = 'large'

data = pd.read_csv('https://raw.githubusercontent.com/azhar2ds/DataSets/master/bc_clean-data.csv', index_col=False)
data.drop('Unnamed: 0',axis=1, inplace=True)

X=data.drop('diagnosis', axis=1)
y=data['diagnosis']

l=LabelEncoder()
y=l.fit_transform(y)

scaler=StandardScaler()
Xs=scaler.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(Xs,y,test_size=0.3, random_state=0)

c=SVC(probability=True,random_state=2, gamma='scale')
c.fit(X_train, y_train)

s=SVC()
s.fit(X_train,y_train)
yp=s.predict(X_test)
print("Simple accuracy score all default values for SVC:",s.score(X_train,y_train))
print('accuracy score without any cross validation: ',accuracy_score(yp, y_test))



#cv_error = np.average(cross_val_score(SVC(), Xs, y, cv=3))
#print('The fold cv accuracy score for this classifier is:',3, cv_error)

n_folds = 5
cv_error = np.average(cross_val_score(SVC(), Xs, y, cv=n_folds))
print('\nThe {}-fold cross-validation accuracy score for this classifier is {:.5f}\n'.format(n_folds, cv_error))


#Score can be calculated with METRICS.ACCURACY_SCORE() AND SCORE()
classifier_score = s.score(X_test, y_test)
print('\nThe classifier accuracy score is :', classifier_score)

print('\naccuracy score without any cross validation: ',accuracy_score(yp,y_test))

n_folds = 6
scores = cross_val_score(SVC(), Xs, y, cv=n_folds)
print('\nEach individual fold val:',scores)
cv_error = np.average(cross_val_score(SVC(), Xs, y, cv=n_folds))
print("\nAccuracy Score average of folds is: ", cv_error)
print("\nAnother way of finding average accuracy of folds is(np.average(list(scores)):", np.average(list(scores)))


from sklearn.feature_selection import SelectKBest, f_regression
clf2 = make_pipeline(SelectKBest(f_regression, k=n_folds),SVC(probability=True))

scores = cross_val_score(clf2, Xs, y, cv=n_folds)

# Get average of 3-fold cross-validation score using an SVC estimator.

cv_error = np.average(cross_val_score(SVC(), Xs, y, cv=n_folds))
print('cv_error:', cv_error)


print(scores)
avg = (100*np.mean(scores), 100*np.std(scores)/np.sqrt(scores.shape[0]))
print("Average score and uncertainty: (%.2f +- %.3f)%%"%avg)

# The confusion matrix helps visualize the performance of the algorithm.
y_pred = c.fit(X_train, y_train).predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
#print(cm)


from IPython.display import Image, display

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Reds, alpha=0.3)
for i in range(cm.shape[0]):
     for j in range(cm.shape[1]):
         ax.text(x=j, y=i,
                s=cm[i, j], 
                va='center', ha='center')
plt.xlabel('Predicted Values', )
plt.ylabel('Actual Values')
plt.show()
print(classification_report(y_test, y_pred ))


#ROC Curve

from sklearn.metrics import roc_curve, auc
# Plot the receiver operating characteristic curve (ROC).
plt.figure(figsize=(6,5))
probas_ = c.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.axes().set_aspect(1)




