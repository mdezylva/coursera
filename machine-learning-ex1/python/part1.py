import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

os.chdir('/home/mitchell/Documents/projects/ng_course/machine-learning-ex1/ex1')
print(os.getcwd())

print('Loading Data...')
data = pd.read_csv('ex1data1.txt', header=None)
print('Done!')

print('Cleaning Data...')
data.columns = ['X', 'Y']
total_entries = data.X.count()+1
x_max = data.X.max()
y_max = data.Y.max()
print("Total number of entries: ", total_entries)
print('Done!')

print('Plotting Data')
plt.scatter(data.X, data.Y)
plt.xlim([0, x_max+10])
plt.xlabel('Population of City in 10,000s')
plt.ylim([0, y_max+10])
plt.ylabel('Profit in $10, 000s')
plt.show()
print('Done!')


x = pd.concat([pd.Series(np.ones(len(data.X))), data.X], axis=1)
theta = np.zeros([1, 2])
theta[0, 1] = 2
theta[0, 0] = 0

iterations = 1500
alpha = 0.01

h_theta = np.multiply(x, theta)
# print(h_theta)

# theta_0 = theta[0]
# theta_1 = theta[1]

X = data.X.values.reshape(-1, 1)
Y = data.Y.values.reshape(-1, 1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.plot(X, Y_pred, color='red')
plt.scatter(data.X, data.Y)
plt.xlim([0, x_max+10])
plt.xlabel('Population of City in 10,000s')
plt.ylim([0, y_max+10])
plt.ylabel('Profit in $10, 000s')
plt.show()

