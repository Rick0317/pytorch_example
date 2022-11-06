# pytorch_example
## Machine Learning without PyTorch: no_pytorch.py
We have to define Model, Loss function, Parameter Updates by ourselves.  
Linear Model:  
y = k * x + bias  

Loss Function:  
loss = ((Output - y_train)**2).mean()

Parameter Updates:  
w = -lr * w_grad  
b = -lr * b_grad  

## Machine Learning with PyTorch:  
### nn.Module class 
variables:  
training : boolean, whether the model is in training or evaluation


### Model
We use nn.Module  


### Loss Function

### Optimizer

## Reference
(website1)[https://dreamer-uma.com/beginner-how-to-use-pytorch/]
