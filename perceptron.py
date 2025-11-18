import numpy as np

class Perceptron:
    def __init__(self, input_size:int, learning_rate=0.1):
        self.learning_rate = learning_rate

        array = np.zeros(input_size)
        self.weights = array

        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_probability(self, X):
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return predictions
    
    def predict(self, X):
        predictions = self.predict_probability(X)
        if isinstance(predictions, np.float64):
            if predictions >= 0.5:
                return 1
            else:
                return 0
        else:
            rounded_predictions = []
            for p in predictions:
                if p >= 0.5:
                    rounded_predictions.append(1)
                else:
                    rounded_predictions.append(0)
            return rounded_predictions

        # p = self.predict_probability(X)
        # if p >= 0.5:
        #     rounded_p = 1
        # else:
        #     rounded_p = 0
        # return rounded_p
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            error = predictions - y
            loss = np.mean(error ** 2)
            grad = error * predictions * (1 - predictions)
            self.weights -= self.learning_rate * np.dot(X.T, grad)
            self.bias -= self.learning_rate * np.sum(grad)

# p = Perceptron(4)
#print(p.array)


