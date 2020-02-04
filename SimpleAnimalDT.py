import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({"toothed":["True","True","True","False","True","True","True","True","True","False"],
                     "hair":["True","True","False","True","True","True","False","False","True","False"],
                     "breathes":["True","True","True","True","True","True","False","True","True","True"],
                     "legs":["True","True","False","True","True","True","False","False","True","True"],
                     "species":["Mammal","Mammal","Reptile","Mammal","Mammal","Mammal","Reptile","Reptile","Mammal","Reptile"]}, 
                    columns=["toothed","hair","breathes","legs","species"])

features = data[["toothed","hair","breathes","legs"]]
target = data["species"]

#Chaning all boolean values to 'ZERO's & ONE's
le = LabelEncoder() 
data['toothed']= le.fit_transform(data['toothed']) 
data['hair']= le.fit_transform(data['hair']) 
data['legs']= le.fit_transform(data['legs']) 
data['breathes']= le.fit_transform(data['breathes']) 

features = data[["toothed","hair","breathes","legs"]]
target = data["species"]

clf=DecisionTreeClassifier(criterion='gini')
clf.fit(features,target)
predicts=clf.predict(features)
print(accuracy_score(predicts,target)*100)

