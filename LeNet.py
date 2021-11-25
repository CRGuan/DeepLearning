import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import matplotlib.pyplot as plt


EPOCH = 20
BATCH_SIZE = 100
LR = 0.01
DROPOUT = 0.1
DOWNLOAD_MNIST = False  # 需要下载数据时需要置为Ture
acclist = []
losslist = []

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="./fashion_data", train=True, transform=trans, download= DOWNLOAD_MNIST)

train_loader = Data.DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True)

mnist_test = torchvision.datasets.FashionMNIST(root="./fashion_data", train=False, transform=trans)

test_x = torch.unsqueeze(mnist_test.data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) -> (2000,1,28,28)
test_y = mnist_test.targets[:2000]

# print(len(mnist_train), len(mnist_test))
# print(mnist_train[0][0].shape)


class Lenet(nn.Module):  # (batch,1,28,28)
    def __init__(self):
        super(Lenet, self).__init__()
        self.con1 = nn.Sequential(  # (1, 28, 28)  conv->(w-k+2P)/s+1
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=6,  # output height
                kernel_size=5,  # filter size
                stride=1,  # filter stride
                padding=2,  # filter pad
            ),  # -> (batch, 6, 28, 28)
            nn.Sigmoid(),
            nn.AvgPool2d(
                kernel_size=2,  # pooling size  -> (batch, 6, 14, 14)
                stride=2,
            )
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # -> (batch, 16, 10, 10)
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2)  # -> (batch, 16, 5, 5)
        )
        self.hidden1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid(),
            # nn.Dropout(DROPOUT),
        )
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.con1(x)  # x.size() -> (batch, 6, 14, 14)
        x = self.con2(x)  # (batch, 16, 5, 5)
        # print(x.shape)
        # print(x.data.device)
        x = self.hidden1(x)
        x = self.hidden2(x)
        output = self.out(x)
        return output


net = Lenet()
print(net)

net = net.to('cuda')

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):  # train dataset 3 times
    for step, (b_x, b_y) in enumerate(train_loader):
        # b_x.shape [50, 1, 28, 28]
        b_x = b_x.to('cuda')
        b_y = b_y.to('cuda')
        prediction = net(b_x)
        loss = loss_func(prediction, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = net(test_x.to('cuda'))
            # print(test_output.data.device)
            pre_y_gpu = torch.max(test_output, 1)[1].data
            pre_y_cpu = pre_y_gpu.to('cpu')
            accuracy = 0
            print(len(pre_y_cpu))
            for i in range(len(pre_y_cpu)):
                if pre_y_cpu[i] == test_y.data[i]:
                    accuracy = accuracy+1
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


