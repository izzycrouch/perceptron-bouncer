from data import get_training_data
from perceptron import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# p=Perceptron(input_size=4)

# X_train, y_train = get_training_data()

# p.train(X_train, y_train, epochs=10)

# shirt = int(input("Are they wearing a 'I <3 JavaScript' T-shirt? "))
# duck = int(input("Are they carrying a menacing rubber duck? "))
# pennies = int(input("Are they trying to pay with pennies? "))
# queue = int(input("Are they jumping the queue? "))


# input_vector = [shirt, duck, pennies, queue]


# X_test = np.array(test_dataset)
# y_test = np.array(test_targets)

# decision = p.predict_probability(input_vector)
# if  decision > 0.7:
#     allowed_in = "✅ Welcome in!" 
# elif 0.3 < decision <=0.7:
#     allowed_in = "Gray area! Use your best judgement!" 
# else:
#     allowed_in = "🚫 They're on the list. The *bad* list."
# print(decision)
# print(allowed_in)
# accuracy = accuracy_score(y_test, pred)
# print(accuracy)

# train_predictions = p.predict_probability(X_train)

# plt.scatter(train_predictions, y_train, marker='o')
# plt.xlabel('X Train Predictions')
# plt.ylabel('Y Train Results')
# plt.title('Perceptron predictions vs results')
# plt.show()

