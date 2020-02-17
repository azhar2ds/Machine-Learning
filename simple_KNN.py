from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Assigning features and label variables

# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']

# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Label or target varible
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#Changing the catergorical feature into numberical
lencode=preprocessing.LabelEncoder()
weather=lencode.fit_transform(weather)
temp=lencode.fit_transform(temp)
play=lencode.fit_transform(play)

# amalgamate all the features in one list
features=list(zip(temp,play))


# Training a KNN mocel with KNeighborsClassifier
k=KNeighborsClassifier(n_neighbors=3)
k.fit(features,play)
values=k.predict([[0,2]])

#predict the value
print("Predicted value :",values,"(means it can play or not)")


      