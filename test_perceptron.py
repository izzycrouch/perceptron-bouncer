import numpy as np
from perceptron import Perceptron

def test_perceptron_init():
    p = Perceptron(input_size=3)
		
    assert isinstance(p.weights, np.ndarray)
    assert p.weights.shape == (3,)
    assert np.all(p.weights == 0.0)
    
    assert p.learning_rate == 0.1
    
    assert p.bias == 0.0

def test_perceptron_sigmoid():
    p = Perceptron(input_size=3)
    assert np.isclose(p.sigmoid(0), 0.5)
    assert np.isclose(p.sigmoid(100), 1.0, atol=1e-6)
    assert np.isclose(p.sigmoid(-100), 0.0, atol=1e-6)

def test_perceptron_sigmoid_test_whole_array():
    p = Perceptron(input_size=3)
    input_array = np.array([0, 1, -1])
    output = p.sigmoid(input_array)
    expected = 1 / (1 + np.exp(-input_array))
    assert np.allclose(output, expected)

def test_predict_probability_single_input_returns_valid_probability():
	p = Perceptron(input_size=2)
	p.weights = np.array([0.5, -0.5]) 
	p.bias = 0.0 

	result = p.predict_probability(np.array([1,1]))
	assert isinstance(result, float)
	assert 0.0 <= result <= 1.0

def test_predict_probability_batch_input_returns_array_of_probabilities():
	p = Perceptron(input_size=2)
	p.weights = np.array([0.5, -0.5]) 
	p.bias = 0.0 

	result = p.predict_probability(np.array([[1,1], [0,0]]))
	assert isinstance(result, np.ndarray)
	assert result.shape == (2,)
	assert np.all(result >= 0) and np.all(result <=1)

def test_predict_method_returns_1():
    p = Perceptron(input_size=2)
    p.weights = np.array([10, 10]) 
    p.bias = 5.0 

    result = p.predict(np.array([1,0]))
    assert result == 1

def test_predict_method_returns_0():
    p = Perceptron(input_size=2)
    p.weights = np.array([10, 10]) 
    p.bias = -15.0 

    result = p.predict(np.array([1,0]))
    assert result == 0

def test_predict_method_returns_for_multy_array():
    p = Perceptron(input_size=2)
    p.weights = np.array([10, 10]) 
    p.bias = -15.0 

    result = p.predict(np.array([[1,1], [0,0]]))
    assert result == [1,0]

def test_train_changes_weights_and_bias():
    p = Perceptron(input_size=2, learning_rate=0.1)
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])

    initial_weights = p.weights.copy()
    initial_bias = p.bias

    p.train(X, y, epochs=10)

    assert not np.allclose(p.weights, initial_weights)
    assert not np.isclose(p.bias, initial_bias)

def test_train_learns_simple_pattern():
    p = Perceptron(input_size=1, learning_rate=0.5)
        
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    p.train(X, y, epochs=100)

    prob_0 = p.predict_probability(np.array([0]))
    prob_1 = p.predict_probability(np.array([1]))

    assert prob_0 < 0.3
    assert prob_1 > 0.7