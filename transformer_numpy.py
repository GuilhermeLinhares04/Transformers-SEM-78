import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Perceptron:
    def __init__(self, weights=None, bias=-1, activation_threshold=0.5):
        if weights == None:
            self.weights = np.array([1, 1])
        else:
            self.weights = np.array(weights)
        self.bias = bias
        self.activation_threshold = activation_threshold

    def _heaviside(self, x):
        """
        Implementa a função delta de heaviside (famoso degrau)
        Essa é uma função de ativação possível para os nós da rede neural.
        """
        return 1 if x >=  self.activation_threshold else 0

    def _sigmoid(self, x):
        """
        Implementa a função sigmoide
        Essa é uma função de ativação possível para os nós da rede neural.
        """
        return 1/(1 + math.exp(-x))

    def _activation(self, perceptron_output):
        """
        Implementação da função de ativação do perceptron
        Escolha uma das funções de ativação possíveis
        """
        return self._heaviside(perceptron_output)

    def forward_pass(self, data):
        """
        Implementa a etapa de inferência (feedforward) do perceptron.
        """
        weighted_sum = self.bias + np.dot(self.weights, data)
        return self._activation(weighted_sum)

################################################### Implementação com NumPy ###################################################

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.bias_output = np.random.randn(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_pass(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output
    
    def compute_cost(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward_pass(self, X, y_true):
        y_pred = self.forward_pass(X)
        
        output_error = y_true - y_pred
        output_delta = output_error * self.sigmoid_derivative(y_pred)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
        
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate
    
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            self.backward_pass(X, y)
            if epoch % 1000 == 0:
                cost = self.compute_cost(y, self.forward_pass(X))
                print(f"Epoch {epoch}, Custo: {cost}")

# Dados para o problema XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Instanciar e treinar o MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
mlp.train(X, y, epochs=10000)

# Teste
for xi in X:
    print(f'Entrada: {xi}, Saída prevista: {mlp.forward_pass(xi)}')