from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import pandas as pd

#Playing Gold Dataset

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
temp_encoded=le.fit_transform(temp)
play_encoded=le.fit_transform(play)

features=pd.DataFrame(weather_encoded,temp_encoded)
nb=GaussianNB()
nb.fit(features,play_encoded)
print("0 means cant play, \n1 means play...",nb.predict([[3]]))
print('model accuracy score: ',nb.score(features,play_encoded)*100)

