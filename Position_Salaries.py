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

#-------------------------------------------------------------------------------------------------

#fitting Linear Regression to the dataset(LinearRegression)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

print(lin_reg.predict(6.5))

#visualising the Linear Regression result
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color="blue")
plt.plot(X_grid,lin_reg.predict(X_grid),color="red")
plt.title("LinearRegression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#-------------------------------------------------------------------------------------------------

#create dataset of degree of x
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#fitting linear regression to Polynomial dataset(PolynomialRegression)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))

#visualising the Polynomial Regression result
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color="blue")
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color="red")
plt.title("PolynomialRegression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#-------------------------------------------------------------------------------------------------

"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_SC = sc_X.fit_transform(X)
sc_y = StandardScaler()
y_SC = sc_y.fit_transform(y.reshape(1, -1))

#fitting Support Vector Regression to the datasset
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_SC,y_SC.reshape(1, -1))

#visualising the Polynomial regression result
plt.scatter(X,y,color="blue")
plt.plot(X,sc_y.inverse_transform(svr_reg.predict(X_SC)),color="red")
plt.title("PolynomialRegression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
"""

#-------------------------------------------------------------------------------------------------

#fitting Decesion Tree Regression to polynomial dataset
from sklearn.tree import DecisionTreeRegressor
dtr_reg = DecisionTreeRegressor(random_state = 0)
dtr_reg.fit(X,y)

print(dtr_reg.predict(6.5))

#visualising the Decesion Tree Regression result
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color="blue")
plt.plot(X_grid,dtr_reg.predict(X_grid),color="red")
plt.title("DecisionTreeRegression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#-------------------------------------------------------------------------------------------------

#fitting Random Forest Regression to polynomial dataset
from sklearn.ensemble import RandomForestRegressor
rfr_reg = RandomForestRegressor(n_estimators = 10,random_state=0)
rfr_reg.fit(X,y)

print(rfr_reg.predict(6.5))

#visualising the Decesion Tree Regression result
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color="blue")
plt.plot(X_grid,rfr_reg.predict(X_grid),color="red")
plt.title("RandomForestRegression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#-------------------------------------------------------------------------------------------------
