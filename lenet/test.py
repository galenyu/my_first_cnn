import numpy as np
import os
import torch
from torchvision.datasets.mnist import MNIST
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from LeNet import LeNet
import matplotlib.pyplot as plt
import os
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    test_dataset = MNIST(root='./test', train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    model = LeNet().to(device)
    model=torch.load('./model.pkl')
    all_correct_num = 0
    all_sample_num = 0
    model.eval()
    for idx, (test_x, test_label) in enumerate(test_loader):
        
        test_x = test_x.to(device)
        test_label = test_label.to(device)
        predict_y = model(test_x.float()).detach()
        predict_y =torch.argmax(predict_y, dim=-1)
        if(idx==0):
            fig, axs = plt.subplots(1, batch_size, figsize=(10, 5))
            # test_x = test_x.cpu()
            array = (test_x.to('cpu').numpy() * 255).astype('uint8').transpose((0, 2, 3, 1))
            label = predict_y.to('cpu').numpy()
            for i in range(batch_size):
                axs[i].imshow(array[i])
                axs[i].set_title('Pre:{}'.format(label[i]))
            plt.subplots_adjust(wspace=0.3, hspace=0.1)
            plt.show()
        
        current_correct_num = predict_y == test_label
        # print(current_correct_num.to('cpu').numpy())
        all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]
    acc = all_correct_num / all_sample_num
    print("Acc: {}".format(acc))
    