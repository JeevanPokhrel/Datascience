import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv('Ecommerce Customers')

#train.head()

X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

y = customers['Yearly Amount Spent']

# To check if the data are missing in the dataset

#print(customers.info())
#print(customers.describe())
#print(customers.head())
#sns.pairplot(customers)
#sns.distplot(customers['Length of Membership'])
#sns.heatmap(customers.corr())
#sns.heatmap(customers.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#sns.countplot(x='',hue='',data=customers,palette='RdBu_r')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
print (lr.coef_)

y_predict=lr.predict(X_test)
plt.scatter(y_test,y_predict)
plt.show()

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

plt.hist((y_test-y_predict),bins=200);
plt.show()

coeff_df = pd.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)



