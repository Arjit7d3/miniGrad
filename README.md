# miniGrad
A from scratch implementation of a tiny autograd engine, inspired heavily by karpathy's microGrad.

## Files

### engine.py
This file defines the `Value` class, which is a fundamental component for automatic differentiation. It enables the computation of gradients during the backward pass in the neural network.

### nn.py
This file includes the implementation of the neural network classes: `NeuralNetwork`, `Layer`, and `Neuron`. These classes provide a basic structure for creating, training, and using neural networks.

### example.py
This script demonstrates how to use the neural network classes defined in `nn.py` to train a model on a synthetic dataset. The trained model is then visualized using Matplotlib.

## Usage

1. Import the necessary classes:

    ```python
    from nn import NeuralNetwork, Layer, Neuron
    from engine import Value
    ```

2. Create a neural network:

    ```python
    model = NeuralNetwork([
        Layer(input_size, hidden_size),
        Layer(hidden_size, output_size, nonlin=False)
    ])
    ```

3. Train the model:

    ```python
    # Prepare data (replace with your data)
    x_train = ...
    y_train = ...

    # Train the model
    epochs = 15000
    alpha = 0.0001

    for epochs in range(epochs):
        ypred = list(map(model, x))
        loss = sum((yout - ygt)**2 for ygt, yout in zip(y, ypred)) / len(y)
        if epochs % 100 == 0:
            print(f"loss at {epochs}th: {loss.data}")
    
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
    
        for p in model.parameters():
            p.data += -alpha * p.grad
    ```

4. Visualize the results:

    ```python
    import matplotlib.pyplot as plt

    # Visualize the results
    plt.plot(x_train, list(map(lambda x: x.data, map(model, x_train))), color='orange', zorder=1)
    plt.scatter(x_train, y_train)
    plt.show()
    ```
