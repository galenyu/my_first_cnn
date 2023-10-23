from torch.nn import Module
from torch import nn 
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os

class LeNet(Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1=nn.Conv2d(1,6,5,padding=2)
        self.conv2=nn.Conv2d(6,16,5)
        self.linear1=nn.Linear(16*5*5,120)
        self.linear2=nn.Linear(120,84)
        self.linear3=nn.Linear(84,10)
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=F.max_pool2d(x,(2,2))
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,(2,2))
        x = x.view(x.shape[0], -1)
        x=self.linear1(x)
        x=F.relu(x)
        x=self.linear2(x)
        x=F.relu(x)
        x=self.linear3(x)
        return x

if __name__=='__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    lenet=LeNet()
    print(lenet)
    input=torch.randn(1,1,32,32)
    output=lenet(input)
    print(output)
    # toPIL = transforms.ToPILImage()
    # img=toPIL(input)
    # print(img)
    # plt.imshow(img)
    # plt.show()