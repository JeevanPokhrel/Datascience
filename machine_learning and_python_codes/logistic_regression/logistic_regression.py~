import pandas as pd
import numpy as py
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('advertising.csv')

#print(dataset.head())
#print(dataset.info())

#sns.heatmap(dataset.corr())
#plt.show()

#sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#plt.show()

dataset.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1,inplace=True)
#print(dataset.head())

X = dataset[['Daily Time Spent on Site',  'Age',  'Area Income',  'Daily Internet Usage',  'Male']]
y = dataset['Clicked on Ad']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)

from sklearn.linear_model import LogisticRegression

lor = LogisticRegression()

lor.fit(X_train,y_train)

y_predict=lor.predict(X_test)

print (lor.coef_)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))

