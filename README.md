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
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: poorna chandra rao
RegisterNumber:  2305001012
*/
```
```
Program to implement univariate Linear Regression to fit a straight line using leas
 Developed by: PALADUGU VENKATA BALAJI
 RegisterNumber:2305001024
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 df=pd.read_csv('/content/ex1.csv')
 df.head(10)
 plt.scatter(df['X'],df['Y'])
 plt.xlabel('X')
 plt.ylabel('Y')
 x=df.iloc[:,0:1]
 y=df.iloc[:,-1]
 x
 from sklearn.model_selection import train_test_split
 x_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 from sklearn.linear_model import LinearRegression
 lr=LinearRegression()
 lr.fit(x_train,Y_train)
 plt.scatter(df['X'],df['Y'])
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.plot(x_train,lr.predict(x_train),color='red')
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
```
## Output:
![image](https://github.com/user-attachments/assets/761c1d98-690e-42b9-8bdf-f1a4674e829e)
![image](https://github.com/user-attachments/assets/67ab334c-286a-4b5f-a489-15d94a20448c)
![image](https://github.com/user-attachments/assets/1324dff3-0929-4528-922e-dd95635af911)
![image](https://github.com/user-attachments/assets/f4c4f28b-8b47-46c1-a5e3-a228dd458872)
![image](https://github.com/user-attachments/assets/d3cab1cd-1c4c-4c2d-8fde-4d1a40fd9f96)





## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
