# Implementação de MLP para Porta XOR

Este projeto contém duas implementações de uma Rede Neural Perceptron de Múltiplas Camadas (MLP) para resolver o problema da porta XOR. As implementações são feitas utilizando NumPy e PyTorch.

## Estrutura do Projeto

- `transformer_numpy.py`: Implementação do MLP utilizando NumPy.
- `transformer_pytorch.py`: Implementação do MLP utilizando PyTorch.

## Requisitos

Para executar o projeto, você precisará do Python 3.6 ou superior. As dependências necessárias são:

- NumPy
- PyTorch

## Instalação

### Clonando o Repositório

```bash
git clone https://github.com/GuilhermeLinhares04/Transformers-SEM-78
cd Transformers-SEM-78
```

### Instalando as Dependências

```bash
pip install -r requirements.txt
```

## Executando o Projeto

Para executar o projeto, basta rodar o script `transformer_numpy.py` ou `transformer_pytorch.py`:

```bash
python transformer_numpy.py
python transformer_pytorch.py
```

## Explicação das Implementações
### Numpy <br>
O arquivo `transformer_numpy.py` contém a implementação de um MLP com uma camada escondida utilizando NumPy. A rede é treinada para resolver o problema da porta XOR.

**Estrutura do Código**
- MLP: Classe que define a estrutura do MLP.
    - __init__: Inicializa os pesos e bias.
    - sigmoid: Função de ativação sigmoide.
    - sigmoid_derivative: Derivada da função sigmoide.
    - forward_pass: Realiza a passagem para frente (feedforward).
    - compute_cost: Calcula o custo (erro quadrático médio).
    - backward_pass: Realiza a passagem para trás (backpropagation).
    - train: Treina a rede utilizando os dados de entrada.

### PyTorch
O arquivo `transformer_pytorch.py` contém a implementação de um MLP com uma camada escondida utilizando PyTorch. A rede é treinada para resolver o problema da porta XOR.

**Estrutura do Código**
- MLP_PyTorch: Classe que define a estrutura do MLP.
    - __init__: Define as camadas da rede.
    - forward: Define a passagem para frente (feedforward).
O treinamento é realizado utilizando a classe nn.MSELoss para a função de perda e optim.SGD para o otimizador.

## Vídeo de Demonstração

O vídeo de demonstração do projeto está disponível no YouTube: [https://youtu.be/-Qm6VZQNQzE](https://youtu.be/-Qm6VZQNQzE)