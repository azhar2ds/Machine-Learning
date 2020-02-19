#Standardization 

#StandardScaler() will normalize the features (each column of X, INDIVIDUALLY !!!) 
#so that each column/feature/variable will have mean = 0 and standard deviation = 1.

#(X−μ)/σ  WHERE σ is Standard Deviation and μ is mean

#We Standardize the data because if we have feature values measured differntly
# like some values are measured in miles/Kms and some are in feets/yards.

import numpy as np
data = [[6, 2], [4, 2], [6, 4], [8, 2]]
a = np.array(data)

print("standard deviation before the standardization", np.std(a, axis=0))
#Standard deviate is between 0 and 1 of both the features.

print("mean before the standardization",np.mean(a, axis=0))
# mean before standardization

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)

#Xchanged = (X−μ)/σ  WHERE σ is Standard Deviation and μ is mean
z=scaler.transform(data)
print(z)
print('mean after standardization:',np.mean(z))
print('std after standardization:',np.std(z))

# After applying StandardScaler(), each column in X will have mean of 0 and standard deviation of 1
print(z)