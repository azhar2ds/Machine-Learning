# Import dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



data = np.array([[6,2],[4,2],[6,4],[8,2]])

print('mean of complete dataset:', data.mean())
print('STD of complete dataset:', data.std())

print('element wise(row) Mean:',np.mean(data, axis=0))
print('element Wise(colomn) STD', np.std(data, axis=0))

print('Mean before standardization: ',data.mean())
print('Std dev before standardization:',data.std())
scal = StandardScaler()
c=scal.fit_transform(data)
print('scal Mean After tranform: ',c.mean())
print('Scal STD DEV After:',c.std())
print(c)

print('Mean of each feature after Standard Scaler: ',np.mean(c, axis=0))
print('STD of each feature after Standard Scaler:' ,np.std(c, axis=0))

print('Final mean of entire new dataset', c.mean())
print('Final STD of entire new dataset', c.std())
