
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectFromModel

# Importing the dataset
from sklearn.datasets import load_boston
boston = load_boston()
print(type(boston))
#boston is a dictionary.
print(boston.keys())

#get shape of data
print(boston.data.shape)

#get feature names
print(boston.feature_names)

#about the dataset
#print(boston.DESCR)

#create boston dataframe
dataset = pd.DataFrame(boston.data)

#alternatively you can get the dataset using
#https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
#dataset1=pd.read_table("housing.csv")
#assign column name
dataset.columns = boston.feature_names

#add target variable
dataset['PRICE'] = boston.target

#Summary Statistics
print(dataset.describe())

#split x and y
x = dataset.drop('PRICE', axis = 1)
y = dataset['PRICE']

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)



# Fitting Random Forest Regression to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

feature_imp = pd.Series(regressor.feature_importances_,index=x.columns).sort_values(ascending=False)

import seaborn as sns
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# Print the name and gini importance of each feature
#for feature in zip(x.columns, regressor.feature_importances_):
 #   print(feature)
    
sfm = SelectFromModel(regressor, threshold=0.10)
sfm.fit(X_train, y_train)
print("The most important feature are as follows:")    
for a in sfm.get_support(indices=True):
    print(x.columns[a])

X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)


clf_important = RandomForestRegressor(n_estimators=10000, random_state=0)
clf_important.fit(X_important_train, y_train)


print('New Prediction with only two features')
new_pred=clf_important.predict(X_important_test)
print('new_pred')

# Evaluating the Algorithm
from sklearn import metrics
print('MAE with just 2 variables:', metrics.mean_absolute_error(y_test, new_pred))  
print('MSE  with just 2 variables:', metrics.mean_squared_error(y_test, new_pred))  
print('RMSE with just 2 variables:', np.sqrt(metrics.mean_squared_error(y_test, new_pred)))



