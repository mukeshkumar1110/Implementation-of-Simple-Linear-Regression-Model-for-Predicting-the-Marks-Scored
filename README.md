# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MUKESH KUMAR S
RegisterNumber:212223240099 
*/
```
```

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()

df.tail()

#segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting training and test date
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![image](https://github.com/user-attachments/assets/f843d560-0ea6-4cf9-999e-4751f4d0a929)
![image](https://github.com/user-attachments/assets/73535f53-b6a7-4e95-a404-d3deb802d95e)
![image](https://github.com/user-attachments/assets/d70ca034-bceb-418a-8182-a5196dd0e175)
![image](https://github.com/user-attachments/assets/3bbc5a91-0718-4b61-bf5e-98f9ed8542c9)
![image](https://github.com/user-attachments/assets/d1a61c17-41b4-4b55-9610-9f380d69b38d)
![image](https://github.com/user-attachments/assets/e67acbb3-0d84-4cb6-8b4f-41680d087a85)
![image](https://github.com/user-attachments/assets/3666616e-45ce-40e6-9e4c-577da0ec2049)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
