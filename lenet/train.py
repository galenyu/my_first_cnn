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

def plot(total_acc):
    fig = plt.figure()
    plt.plot(total_acc, label='Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(' Variation')
    plt.show()

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    train_dataset = MNIST(root='./train', train=True, transform=ToTensor(),download = 'True')
    test_dataset = MNIST(root='./test', train=False, transform=ToTensor(),download = 'True')
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = LeNet().to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    loss_fn = CrossEntropyLoss()
    all_epoch = 50
    prev_acc = 0
    print("start training")
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            if(idx%100==99):
                print("Epoch {} - {}/50000".format(current_epoch,(idx+1)*batch_size))
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()
        print("Epoch {} has done, start testing".format(current_epoch))
        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        total_epoch = []
        total_acc = []
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            # print(current_correct_num.to('cpu').numpy()) 
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        total_acc.append(current_epoch)
        total_acc.append(acc)
        print('Epoch {} has done, ACC: {:.3f}'.format(current_epoch,acc), flush=True)
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/mnist_{:.3f}.pth'.format(acc))
        # if np.abs(acc - prev_acc) < 1e-4:
        #     plot(total_acc=total_acc)
        #     break
        prev_acc = acc
        if(current_epoch==50):
            plot(total_acc=total_acc)
