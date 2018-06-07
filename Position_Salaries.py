"""Project - Predict Salary based on Position.
   Rajat Singhal : singhal.rajat97@gmail.com
   B.Tech 4th Semester, Computer Science & Engineering"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#visualising the linear regression result
plt.scatter(X,y,color="blue")
plt.plot(X,lin_reg.predict(X),color="red")
plt.title("LinearRegression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#create dataset of degree of x
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#fittin linear regression to polynomial dataset
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualising the linear regression result
plt.scatter(X,y,color="blue")
plt.plot(X,lin_reg_2.predict(X_poly),color="red")
plt.title("PolynomialRegression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
