# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries       

2.Define Linear Regression Function  

3.Load and Prepare the Dataset       

4.Extract Features and Target Variable 

5.Apply Feature Scaling     

6.Train the Linear Regression Model      

7.Prepare New Data for Prediction      

8.Make Prediction Using Trained Model    

9.Inverse Transform and Output the Prediction 

## Program:
```py
Program to implement the linear regression using gradient descent.
Developed by: Kanagavel A K
RegisterNumber: 212223230096

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        #Update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv(r"C:\Users\admin\Downloads\50_Startups.csv")
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted valeue: {pre}")
```

## Output:
![Screenshot 2024-09-05 114321](https://github.com/user-attachments/assets/128f1438-1da6-4d1c-b327-fc76c96bf8f2)


![image](https://github.com/user-attachments/assets/2a6db511-a47f-4e20-9e2a-f52d5d949d08)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
