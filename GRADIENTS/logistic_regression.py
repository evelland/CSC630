from gradient import *
import numpy

class LogisticRegression():
    def __init__(epochs, learning_rate, self):
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def fit(x, y, self):
        data_point_count = len(x)
        m = Variable(name = 'm') #function inputs
        b = Variable(name = 'b')
        values = [] #fill with random values to give the regression function a starting point, this holds our m and b values throughout the function?
        stored_values = {} # current lowest-cost values for m and b, shifted over time
        stored_values.update({'m': m})
        stored_values.update({'b': b})

        y_hat_container = []
        for i in range(0, data_point_count):
            y_hat_container.append(1/(1 + Variable.exp(-1*(numpy.dot(m, x[i]) + b))))
        cost_function = sum([(-1 * y[i] * Variable.log(y_hat_container[i]) + (y[i] - 1) * Variable.log(1 - y_hat_container[i])) for i in range(0, data_point_count)])

        for epoch in range(self.epochs):
            current_values = {}
            current_values.update({'m': values[epoch]})
            current_values.update({'b': values[epoch]})
            moving_direction = self.cost_function.gradient(current_values)
            values = values - self.learning_rate * moving_direction
            current_values.update({'m': values[epoch]})
            current_values.update({'b': values[epoch]})
            current_cost = cost_function.evaluate(current_values)
            if current_cost < cost_function.evaluate(stored_values):
                stored_values = current_values

        return stored_values # return m and b dictionary

def build_fit(epoch, learning_rate, X, y):
    model = LogisticRegression(epoch, learning_rate)
    optimized_values = model.fit(X, y)
    print(optimized_values) # prints processed and optimized values