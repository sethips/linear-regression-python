from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def mse(y_actual, y_pred):
    error = 0
    
    for y, y_prime in zip(y_actual, y_pred):
        error += (y - y_prime) ** 2
    
    return error

def calculate_partial_derivatives(x, y, intercept, slope):
    partial_derivative_slope = 0
    partial_derivative_intercept = 0
    n = len(x)

    for i in range(n):
        
        xi = x[i]
        yi = y[i]

        partial_derivative_intercept += - (2/n) * (yi - ((slope * xi) + intercept))
        partial_derivative_slope += - (2/n) * xi * (yi - ((slope * xi) + intercept))
        
    return partial_derivative_intercept, partial_derivative_slope

def train(x, y, learning_rate, iterations, intercept, slope):

    for i in range(iterations):
        
        partial_derivative_intercept, partial_derivative_slope = calculate_partial_derivatives(x, y, intercept, slope)
            
        intercept = intercept - (learning_rate * partial_derivative_intercept)
        slope = slope - (learning_rate * partial_derivative_slope)
        
    return intercept, slope

x, y = make_regression(n_samples=50, n_features=1, n_informative=1, n_targets=1, noise=5)

learning_rate = 0.01
starting_slope = 0
starting_intercept = float(sum(y)) / len(y)
iterations = 950

plt.scatter(x, y)
plt.plot(x, starting_slope * x + starting_intercept, c='red')
plt.show()

print(mse(y, starting_slope * x + starting_intercept))

intercept, slope = train(x, y, learning_rate, iterations, starting_intercept, starting_slope)

linear_regression_line = [slope * xi + intercept for xi in x]

plt.scatter(x, y)
plt.plot(x, linear_regression_line, c='red')
plt.show()

print(mse(y, linear_regression_line))
