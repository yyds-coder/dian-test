import torch
from torch import nn  # 常用网络
from torch import optim  # 优化工具包
import torchvision  # 视觉数据集
from matplotlib import pyplot as plt

##加载数据
batch_size=512
train_loader = torch.utils.data.DataLoader(#对数据进行batch划分
    torchvision.datasets.MNIST('mnist_data',train=True,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 做一个标准化
                                   ])),
    batch_size=batch_size,shuffle=True)#shuffle打乱数据集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/',train=False,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
    batch_size=batch_size,shuffle=True)
x,y=next(iter(train_loader))#一个迭代器，每一轮返回一个batch的数据
print(x.shape,y.shape,x.min(),x.max())
relu = nn.ReLU()  # 如果使用torch.sigmoid作为激活函数的话正确率只有60%
# 创建网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Conv2d(1,1,(4,4))
        self.fc2 = nn.Linear(625,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self,x):
        x = relu(self.fc1(x))
        xa=x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]/625
        xa=int(xa)
        x=x.view(xa,625)
        # h2 = relu(h1w2+b2)
        x = relu(self.fc2(x))
        # h3 = h2*w3+b3
        x = self.fc3(x)
        return x

def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

## 训练模型
net = Net()
# 返回[w1,b1,w2,b2,w3,b3]  对象，lr是学习过程
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loss = []
mes_loss = nn.CrossEntropyLoss()
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        # x:[b,1,28,28],y:[512]
        # [b,1,28,28]  =>  [b,784]
        x = x.view(x.size(0),1,28,28)
        # =>[b,10]
        out = net(x)
        # [b,10]
        y_onehot = one_hot(y)
        ss=out.shape[0]*out.shape[1]/10
        ss=int(ss)
        out=out.view(ss,10)
        loss = mes_loss(out, y_onehot)
        # 清零梯度
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # w' = w -lr*grad
        # 更新梯度，得到新的[w1,b1,w2,b2,w3,b3]
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

## 准确度测试
total_correct = 0
for x,y in test_loader:
    x = x.view(x.size(0),1,28,28)
    out = net(x)
    # out : [b,10]  =>  pred: [b]
    pred = out.argmax(dim = 1)
    correct = pred.eq(y).sum().float().item()  # .float之后还是tensor类型，要拿到数据需要使用item()
    total_correct += correct
total_num = len(test_loader.dataset)
acc = total_correct/total_num
print('准确率acc:',acc)