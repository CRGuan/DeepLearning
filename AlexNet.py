import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import matplotlib.pyplot as plt


'''
 !!! 因为fashonMnist数据集图片大小为(1,28,28),无法完成本网路中初始输入图片(1,224,224)
'''


EPOCH = 20
BATCH_SIZE = 100
LR = 0.01
DROPOUT = 0.5
DOWNLOAD_MNIST = False  # 需要下载数据时需要置为Ture
acclist = []
losslist = []


# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="./fashion_data", train=True, transform=trans, download= DOWNLOAD_MNIST)

train_loader = Data.DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True)

mnist_test = torchvision.datasets.FashionMNIST(root="./fashion_data", train=False, transform=trans)

test_x = torch.unsqueeze(mnist_test.data, dim=1).type(torch.FloatTensor)/255.   # shape (10000, 28, 28) -> (10000,1,28,28)
test_y = mnist_test.targets


class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.con1 = nn.Sequential(
            # 这里，我们使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.hidden1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 4096),
            nn.Dropout(DROPOUT),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.out = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.con1(x)
        x = self.hidden1(x)
        output = self.out(x)
        return output


net = Alexnet()
print(net)

net = net.to('cuda')

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.to('cuda')
        b_y = b_y.to('cuda')
        print(b_x, b_x.shape)
        prediction = net(b_x)
        loss = loss_func(prediction, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 :
            test_output = net(test_x.to('cuda'))
            # print(test_output.data.device)
            pre_y_gpu = torch.max(test_output, 1)[1].data
            pre_y_cpu = pre_y_gpu.to('cpu')
            accuracy = 0
            print(len(pre_y_cpu))
            for i in range(len(pre_y_cpu)):
                if pre_y_cpu[i] == test_y.data[i]:
                    accuracy = accuracy + 1
            # print(accuracy, len(test_y.data))
            accuracy = accuracy / len(test_y.data)
            acclist.append(accuracy)
            losslist.append(loss.item())
            print('Epoch:', epoch, '| train loss:%.4f' % loss.item(), '| test accuracy:%.4f' % accuracy)

x = np.arange(len(acclist))
plt.plot(x, acclist)
plt.plot(x, losslist)
plt.xlabel('times')
plt.show()