import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# Read Acidity
try:
    x = np.loadtxt('/Users/chaitanyagupta/Desktop/machine-learning/Linear_Regression/data/linearX.csv', delimiter=",")
except:
    x = np.loadtxt('/home/chaitanya/Desktop/machine-learning/Linear_Regression/data/linearX.csv', delimiter=",")

try:
    m, n = x.shape[0], x.shape[1]
except IndexError:
    m, n = (x.shape[0], 1)

x0 = np.ones(m)

X = np.c_[x0, x]

# Read Density
try:
    y = np.loadtxt('/Users/chaitanyagupta/Desktop/machine-learning/Linear_Regression/data/linearY.csv', delimiter=",")
except:
    y = np.loadtxt('/home/chaitanya/Desktop/machine-learning/Linear_Regression/data/linearY.csv', delimiter=",")



def J(theta, X, y):
    
    # Calculate the hypothesis
    hypothesis = np.dot(X, theta)
    
    # Calculate the loss
    loss = hypothesis - y
 
 	# Return least squared error
    return np.sum( loss ** 2 ) / (2 * m)



# Define variables

l_rate = 0.001
theta = np.zeros(n + 1)

# No. of iterations it took for convergence
num_iter = 0

conv = False

# Cost Function
# Value when theta0 and theta1 are both init with 0
Jn = J(theta, X, y)

print("Learning Rate: ", l_rate)
print("Initial Error: ", Jn)


while (not conv):
    
    theta = theta - (l_rate / m) * np.dot((np.dot(X, theta) - y), X)
    
    Jp = Jn
    
    Jn = J(theta, X, y)
    
#     if (num_iter % 5) == 0:
#         error_plot_points.append((theta, Jn))
    
    if abs(Jp - Jn) < 10 ** -15:
        conv = True
        
    num_iter += 1

print("Final Error: ", Jn)
print("Number of iterations: ", num_iter)
print("Parameters: ", theta)





# Plot input data
# plt.scatter(x, y, marker='x', c='r')
# plt.xlabel('Acidity')
# plt.ylabel('Density')