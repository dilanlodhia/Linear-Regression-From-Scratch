# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class LinearRegression():
    def __init__(self, n_features=8):
        '''
        Initializes the linear regression model assigning random values to W and b.
        '''
        self.W = np.random.randn(n_features, 1)
        self.b = np.random.randn(1)
        self.y_pred = None

    def predict(self, X):
        '''
        Computes the dot product of the dataset and vector of weights. This essentially is the equation of the linear regression model.
        It returns a prediction vector of y.
        '''
        y_pred = np.dot(X, self.W) + self.b
        return y_pred

    def update_W_b(self, W, b):
        '''
        Updates W and b once they have been optimized using the minimize_loss method.
        '''
        self.W = W
        self.b = b

    def gd_optimisation(self, X, y):
        '''
        Optimises weights and bias to minimise Mean Squared Error using Gradient Descent Optimisation.
        '''
        m_current = np.random.randn(8)
        b_current = np.random.randn(1)
        n = len(X)
        learning_rate = 0.001
        iterations = 1000
        losses = []
        for i in range(iterations):
            diffs = y - (np.dot(m_current, X.T) + b_current)
            dLdw = -2 * np.sum(X.T * diffs).T/n
            dLdb = -2 * np.sum(diffs) / n
            m_current -= dLdw * learning_rate
            b_current -= dLdb * learning_rate
            self.update_W_b(m_current, b_current)
            losses.append(LinearRegression.mse(self.predict(X), y))
        LinearRegression.plot_losses(losses)
        return m_current, b_current

    @staticmethod
    def mse(y_prediction, y_true):
        '''
        The loss function. Calcluates the difference between each true y value and our predicted y value, squares the value to 
        make sure it is positive and sums each of these together and divides by the number of examples for an average.
        '''
        errors = y_prediction - y_true
        squared_error = errors ** 2
        return np.mean(squared_error)

    @staticmethod
    def plot_losses(loss):
        '''
        Plots loss against iterations.
        '''
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.plot(loss)
        plt.show()

if __name__ == '__main__':
    model = LinearRegression()
    weights, bias = model.gd_optimisation(X_train, y_train)
    y_pred_1 = model.predict(X_train)
    cost = model.mse(y_pred_1, y_train)

    print('y_pred: ', y_pred_1)
    print('Weights: ', weights)
    print('Bias:', bias)
    print('Cost: ', cost)
    
# %%
