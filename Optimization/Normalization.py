'''
Normalization


Normalizing in scikit-learn refers to rescaling each observation (row) to have a length of 1 
(called a unit norm in linear algebra).

This preprocessing can be useful for sparse datasets (lots of zeros) with attributes 
of varying scales when using algorithms that weight input values such as neural networks 
and algorithms that use distance measures such as K-Nearest Neighbors.
'''
from sklearn.preprocessing import Normalizer
import pandas
import numpy
url = "https://raw.githubusercontent.com/azhar2ds/DataSets/master/diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(normalizedX[0:5,:])