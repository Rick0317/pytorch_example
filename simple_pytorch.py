import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

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

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(1, 1, bias=True)
    nn.init.constant_(self.fc.weight, 0.0)
    nn.init.constant_(self.fc.bias, 0.0)
  def forward(self, x):
    out = self.fc(x)
    return out

model = Net()

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epoch = 1000

train_loss_list = []
for epoch in range(num_epoch):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # lossを記録
    if (epoch%5==0):
        train_loss_list.append(loss)
        print(f'【EPOCH {epoch}】 loss : {loss:.5f}')