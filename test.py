from nn import *
import numpy as np
import matplotlib.pyplot as plt

model = NeuralNetwork(
    [
        Layer(1, 3),
        Layer(3, 1, nonlin=False)
    ]
)

x = np.linspace(0, 10, 20).reshape(-1, 1)
y = x * 2 + 3
y = y.flatten()
x.reshape(-1, 1)
x = x.tolist()
y = y.tolist()

epochs = 10000
alpha = 0.001

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

plt.plot(x, list(map(lambda x: x.data, map(model, x))), color='orange', zorder=1)
plt.scatter(x, y)
plt.show()
