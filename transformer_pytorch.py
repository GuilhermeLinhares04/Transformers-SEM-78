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

################################################### Implementação com PyTorch ###################################################

import torch
import torch.nn as nn
import torch.optim as optim

class MLP_PyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_PyTorch, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        hidden_output = self.sigmoid(self.hidden(x))
        final_output = self.sigmoid(self.output(hidden_output))
        return final_output

# Dados para o problema XOR
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Instanciar e treinar o MLP
model = MLP_PyTorch(input_size=2, hidden_size=2, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Treinamento
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Custo: {loss.item()}")

# Teste
with torch.no_grad():
    for xi in X:
        output = model(xi)
        print(f'Entrada: {xi.numpy()}, Saída prevista: {output.numpy()}')
