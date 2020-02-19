'''
data is comprised of attributes with varying scales, many machine learning algorithms can
benefit from rescaling the attributes to all have the same scale.

Often this is referred to as normalization and attributes are often rescaled into the range
between 0 and 1. 
'''

# Rescale data (between 0 and 1)
import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
url = 'https://raw.githubusercontent.com/azhar2ds/DataSets/master/diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))# all the values will be between 0 and 1
rescaledX = scaler.fit_transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])

print("min", np.min(rescaledX))
print("max", np.max(rescaledX))