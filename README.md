# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## DATE:
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
Program to implement univariate Linear Regression to fit a straight line using leas
Developed by: poorna chandra rao
RegisterNumber:2305001012

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("/content/ex1.csv")
df.head

df.head(10)

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')

x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression


lr=LinearRegression()
lr.fit(X_train,Y_train)


plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')

m=lr.coef_
m


b=lr.intercept_
b

pred=lr.predict(X_test)
pred
X_test
Y_test
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test, pred)
print(f'Mean Squared Error (MSE): {mse}')




## Output:
![Screenshot 2024-10-17 104804](https://github.com/user-attachments/assets/99affa21-41ca-4107-9128-029da607a36d)
![image](https://github.com/user-attachments/assets/34dd1934-c537-4e3b-9452-bfd67dd2edb8)
![image](https://github.com/user-attachments/assets/660119fb-0793-421f-a384-134a7110183d)
![image](https://github.com/user-attachments/assets/bcdd14c0-69a7-4ca2-8793-4fb272fdc7ce)
![image](https://github.com/user-attachments/assets/a261506f-2eab-4c42-b1af-336e4b16aaf3)




## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
