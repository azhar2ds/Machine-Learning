import pandas as pd

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# 1) Replacing the individual Values
d=pd.DataFrame()
d['weather']=weather
d['temp']=temp

print(d['weather'].value_counts())
cleanup={'weather':{'Sunny':0, 'Overcast':1, 'Rainy':2},
         'temp': {'Hot':0, 'Cool':1,'Mild':2}}
d.replace(cleanup,inplace=True)
print('1st way of replace')
print(d.head())


#2nd ways of encoding - Label Encoder(adding new column)
print("\n\nSecond way using label Encoder")
df=pd.DataFrame({'weather':weather,'temp':temp})
print("Data types of the columns:\n",df.dtypes)

print('\nchanging it to category')
df['weather']=df['weather'].astype('category')
df['temp']=df['temp'].astype('category')
print(df.dtypes)

df['weat_type']=df['weather'].cat.codes
df['temp_type']=df['temp'].cat.codes
df=df.drop(['weather','temp'],axis='columns')
print(df.head())
#VERY IMPORTANT NOTE: we cannot change columns into numberical data
#unless we change columns data type into category as
#it was object datatype earlier


#3rd way of One Hot Encoding
print("\n\n3rd way(get_dummies will create new column of each discrete category)")
dt=pd.DataFrame({'weather':weather,'temp':temp})
dt=pd.get_dummies(dt, prefix=['weat', 'temp'])
#pd.get_dummies(dt, columns=["weather", "temp"], prefix=["weat", "temp"])
print(dt.head())

#4th way
#With the scklearn we have to import preprocessing.LabelEncoder
from sklearn.preprocessing import LabelEncoder
print("\n\n4th way is sklearn.preprocessing.LabelEncoder")
skd=pd.DataFrame({'weather':weather,'temp':temp})
le = LabelEncoder()#Creating object
skd["weather"] = le.fit_transform(skd["weather"])
skd["temp"] = le.fit_transform(skd["temp"])
print(skd)


# 5th way
# There is another way of converting into binary
from sklearn.preprocessing import LabelBinarizer
skb=pd.DataFrame({'weather':weather,'temp':temp})
lb = LabelBinarizer()
lb_results = lb.fit_transform(skb["weather"])
skb=pd.DataFrame(lb_results, columns=lb.classes_)
print("\n\n\nconverting into binary values")
print(skb)

