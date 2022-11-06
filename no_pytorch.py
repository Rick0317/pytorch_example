import numpy as np
import torch

data = np.random.randn(200, 1)
label = 3*data + np.random.randn(200, 1)*0.5

x_train = data[:150, :]
y_train = label[:150, :]
x_test = data[:50, :]
y_test = label[:50, :]
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()

def model(x):
    return w*x + b

def criterion(output, y):
    loss = ((output - y)**2).mean()
    return loss

w = torch.tensor(0.0).float()
b = torch.tensor(0.0).float()

lr = 0.01

num_epoch = 1000

train_loss_list = []
for epoch in range(num_epoch):
    output = model(x_train)
    loss = criterion(output, y_train)
    
    grad_w = ((2*x_train)*(output-y_train)).mean()
    grad_b = 2*(output-y_train).mean()
    w -= lr*grad_w
    b -= lr*grad_b
    grad = 0
    if (epoch%5==0):
        train_loss_list.append(loss)
        print(f'【EPOCH {epoch}】 loss : {loss:.5f}')
