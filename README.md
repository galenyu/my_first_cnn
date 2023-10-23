# My First CNN

This is the simplest implementation of LeNet on MNIST dataset and AlexNet on CiFar10 dataset.

Through these two typical model, you will have a basic understanding in CNN deployment.
### Introduction
![image](https://user-images.githubusercontent.com/25716030/162345646-b13c9af0-bdb5-4ce7-9a62-c0834cba9e5f.png)
![image](https://raw.githubusercontent.com/blurred-machine/Data-Science/master/Deep%20Learning%20SOTA/img/alexnet2.png)
## Requirments
Python3  
PyTorch >= 1.4.0  
torchvision >= 0.1.8
## Usage
```
$git clone https://github.com/galenyu/my_first_cnn.git  
$cd lenet 
```
1. for training:
$python3 train.py
2. for testing:
$python3 test.py 

model will now run on GPU if available

For alexnet, please follow the jupyter notebook.

## Hint
Give your own training process and testing result.
