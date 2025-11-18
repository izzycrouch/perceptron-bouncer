from data import get_training_data
from perceptron import Perceptron
from sklearn.metrics import accuracy_score

p=Perceptron(input_size=4)

X, y = get_training_data()


p.train(X, y, epochs=10)

pred = p.predict([[1,0,1,0]])
print(pred)

# accuracy = accuracy_score(y, pred)