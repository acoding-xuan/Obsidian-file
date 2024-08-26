## 手撕代码 part
### 1.线性回归

```python
# 1、算预测值
# 2、算loss
# 3、梯度设为0，并反向传播
# 3、梯度更新
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 构造线性模型,后面都是使用这样的模板
# 至少实现两个函数，__init__构造函数和forward()前馈函数
# backward()会根据我们的计算图自动构建
# 可以继承Functions来构建自己的计算块
class LinerModel(torch.nn.Module):
    def __init__(self):
        # 调用父类的构造
        super(LinerModel, self).__init__()
        # 构造Linear这个对象，对输入数据做线性变换
        # class torch.nn.Linear(in_features, out_features, bias=True)
        # in_features - 每个输入样本的大小
        # out_features - 每个输出样本的大小
        # bias - 若设置为False，这层不会学习偏置。默认值：True
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinerModel()
# 定义MSE(均方差)损失函数，size_average=False不求均值
criterion = torch.nn.MSELoss(size_average=False)
# optim优化模块的SGD，第一个参数就是传递权重，model.parameters()model的所有权重
# 优化器对象
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # loss为一个对象，但会自动调用__str__()所以不会出错
    print(epoch, loss.item())

    # 梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 根据梯度和预先设置的学习率进行更新
    optimizer.step()

# # 打印权重和偏置值,weight是一个值但是一个矩阵
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())
#
# # 测试
# x_test = torch.Tensor([4.0])
# y_test = model(x_test)
# print('y_pred=', y_test.data)

```

### 2.LogisticRegression

```python
# 逻辑斯蒂回归
import torch.nn
import torch.nn.functional as F

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 将sigmoid函数应用到结果中
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()
# 定义MSE(均方差)损失函数，size_average=False不求均值
criterion = torch.nn.BCELoss(size_average=False)
# optim优化模块的SGD，第一个参数就是传递权重，model.parameters()model的所有权重
# 优化器对象
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # loss为一个对象，但会自动调用__str__()所以不会出错
    print(epoch, loss)

    # 梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 根据梯度和预先蛇者的学习率进行更新
    optimizer.step()

# 打印权重和偏置值,weight是一个值但是一个矩阵
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

# 测试
x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_pred=', y_test.data)
```

### 3.处理多维数据特征
```python
import numpy as np
import torch

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
# [-1]加中括号拿出来是矩阵，不加是向量
y_data = torch.from_numpy(xy[:, [-1]])


# // 构建三个线性层， 使最后的结果特征数为1.

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # 这是nn下的Sigmoid是一个模块没有参数，在function调用的Sigmoid是函数
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(size_average=True)  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 优化函数，随机梯度递减

for epoch in range(100):
    # 前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # 反馈
    optimizer.zero_grad()
    loss.backward()

    # 更新
    optimizer.step()
```

### 4.数据集Dataset&DateLoader
#### 常用术语
Epoch：所有的样本都进行了一次前馈计算和反向传播即为一次epoch
Batch-Size：每次训练的时候所使用的样本数量
Iterations：batch数目
注意， 每一个 epoch 都要把所有的数据都使用一次

![[1ac6650dde6748849b1749c2c9b3214a.png]]
DataLoader
核心参数
batch_size,shuffle(洗牌，用于打乱顺序)
核心功能
通过获得DataSet的索引以及数据集大小，来自动得生成小批量训练集
DataLoader先对数据集进行洗牌，再将数据集按照Batch_Size的长度划分为小的Batch，并按照Iterations进行加载，以方便通过循环对每个Batch进行操作


- 本课程的主要任务是通过将原本简单的标量输入，升级为向量输入，构建线性传播模型：
    - 在导入数据阶段就有很大不同：
        - 数据集类里面有三个函数，这三个函数较为固定，分别自己的作用；
    - 继承`Dataset`后我们必须实现三个函数：
        - `__init__()`是初始化函数，之后只要提供数据集路径，就可以进行数据的加载，也就是说，传入`init`的参数，只要有一个文件路径就可以了；
        - `getitem__()`通过索引找到某个样本；
        - `__len__()`返回数据集大小；

```python
import numpy as np
import torch
from torch.utils.data import Dataset  # Dataset是一个抽象类，只能被继承，不能实例化
from torch.utils.data import DataLoader  # 可以直接实例化

'''
四步：准备数据集-设计模型-构建损失函数和优化器-周期训练
'''


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):  # 实例化对象后，该类能支持下标操作，通过index拿出数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv.gz')
# dataset数据集，batch_size小批量的容量，shuffle是否要打乱，num_workers要几个并行进程来读
# DataLoader的实例化对象不能直接使用，因为windows和linux的多线程运行不一样，所以一般要放在函数里运行
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # 这是nn下的Sigmoid是一个模块没有参数，在function调用的Sigmoid是函数
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(size_average=True)  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 优化函数，随机梯度递减

# 变成嵌套循环，实现Mini-Batch
for epoch in range(100):
    # 从数据集0开始迭代
    # 可以简写为for i, (inputs, labels) in enumerate(train_loader, 0):
    for i, data in enumerate(train_loader, 0):
        # 准备数据
        inputs, labels = data
        # 前馈
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 反馈
        optimizer.zero_grad()
        loss.backward()
        # 更新
        optimizer.step()

```

### 5.softmax
![[Pasted image 20231004214528.png]]

注意: pytorch 提供的 CrossEntropyLoss 是上图中三个模块的结合。（最后一层不用激活，直接输给 CrossEntropy() 即可）

![[Pasted image 20231004215525.png]]
```python
import torch
from torchvision import transforms  # 对图像进行处理的工具
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 使用激活函数relu()的包
import torch.optim as optim  # 优化器的包

batch_size = 64
# 对图像进行预处理，将图像转换为
transform = transforms.Compose([
    # 将原始图像PIL变为张量tensor(H*W*C),再将[0,255]区间转换为[0.1,1.0]
    transforms.ToTensor(),
    # 使用均值和标准差对张量图像进行归一化
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 改变形状，相当于numpy的reshape
        # view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变。
        # 一开始tensor 大小 是 batch_size * channel * w * h 要先转化为 二维的。
        x = x.view(-1, 784) 
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 要注意 crossEntropyLoss的输入是不需要经过激活的，直接进行输入即可


model = Net()
# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
# model.parameters()直接使用的模型的所有参数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # momentum动量

def train(epoch):
    running_loss = 0.0
    # 返回了数据下标和数据
    for batch_idx, data in enumerate(train_loader, 0):
        # 送入两个张量，一个张量是64个图像的特征，一个张量图片对应的数字
        inputs, target = data
        # 梯度归零
        optimizer.zero_grad()

        # forward+backward+update
        outputs = model(inputs)
        # 计算损失，用的交叉熵损失函数
        loss = criterion(outputs, target)
        # 反馈
        loss.backward()
        # 随机梯度下降更新
        optimizer.step()

        # 每300次输出一次
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    # 不会计算梯度
    with torch.no_grad():
        for data in test_loader:  # 拿数据
            images, labels = data
            outputs = model(images)  # 预测
            # outputs.data是一个矩阵，每一行10个量，最大值的下标就是预测值
            _, predicted = torch.max(outputs.data, dim=1)  # 沿着第一维度，找最大值的下标，返回最大值和下标
            total += labels.size(0)  # labels.size(0)=64 每个都是64个元素，就可以计算总的元素
            # (predicted == labels).sum()这个是张量，而加了item()变为一个数字，即相等的数量
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))  # 正确的数量除以总数

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

### 6.CNN 基础
##### 卷积原理
###### 单通道卷积

![](<assets/1696428163030.png>)

  
多通道卷积  

![](<assets/1696428163065.png>)

平面的图像转为立体的角度即如下图

![](<assets/1696428163113.png>)

###### 改进多通道

![](<assets/1696428163159.png>)

![[Pasted image 20231005155319.png]]

#### gpu 使用 注意
注意要将数据和模型都放在gpu上，并且训练集和测试集都需要。

```python
import torch

# 输入的通道就是上图的n,输出的通道就是上图的m
in_channels, out_channels = 5, 10
width, height = 100, 100  # 图像的大小
kernel_size = 3  # 卷积盒的大小
batch_size = 1  # 批量大小

# 随机生成了一个小批量=1的5*100*100的张量
input = torch.randn(batch_size, in_channels, width, height)

# Conv2d对由多个输入平面组成的输入信号进行二维卷积
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

output = conv_layer(input)

# print(input)
print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)


结果：
torch.Size([1, 5, 100, 100])
torch.Size([1, 10, 98, 98])
torch.Size([10, 5, 3, 3])
```

```python
import torch

input = [3, 9, 6, 5,
         2, 4, 6, 8,
         1, 6, 2, 1,
         3, 7, 4, 6]

input = torch.Tensor(input).view(1, 1, 4, 4)

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = maxpooling_layer(input)
print(output)

结果：
tensor([[[[9., 8.],
          [7., 6.]]]])

```


![](<assets/1696428163304.png>)

```python
import torch
from torchvision import transforms  # 对图像进行处理的工具
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 使用激活函数relu()的包
import torch.optim as optim  # 优化器的包

batch_size = 64
# 对图像进行预处理，将图像转换为
transform = transforms.Compose([
    # 将原始图像PIL变为张量tensor(H*W*C),再将[0,255]区间转换为[0.1,1.0]
    transforms.ToTensor(),
	    # 使用均值和标准差对张量图像进行归一化
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义两个卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # 定义一个池化层
        self.pooling = torch.nn.MaxPool2d(2)
        # 定义一个全连接的线性层
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        # x.size(0)就是取的n
        batch_size = x.size(0)
        # 用relu做非线性激活
        # 先做卷积再做池化再做relu
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        # 做view把数据变为做全连接网络所需要的输入
        x = x.view(batch_size, -1)
        return self.fc(x)
        # 因为最后一层要做交叉熵损失，所以最后一层不做激活

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # momentum动量


def train(epoch):
    running_loss = 0.0
    # 返回了数据下标和数据
    for batch_idx, data in enumerate(train_loader, 0):
        # 送入两个张量，一个张量是64个图像的特征，一个张量图片对应的数字
        inputs, target = data
        # 把输入输出迁入GPU
        inputs, target = inputs.to(device), target.to(device)
        # 梯度归零
        optimizer.zero_grad()

        # forward+backward+update
        outputs = model(inputs)
        # 计算损失，用的交叉熵损失函数
        loss = criterion(outputs, target)
        # 反馈
        loss.backward()
        # 随机梯度下降更新
        optimizer.step()

        # 每300次输出一次
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    # 不会计算梯度
    with torch.no_grad():
        for data in test_loader:  # 拿数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 预测
            # outputs.data是一个矩阵，每一行10个量，最大值的下标就是预测值
            _, predicted = torch.max(outputs.data, dim=1)  # 沿着第一维度，找最大值的下标，返回最大值和下标
            total += labels.size(0)  # labels.size(0)=64 每个都是64个元素，就可以计算总的元素
            # (predicted == labels).sum()这个是张量，而加了item()变为一个数字，即相等的数量
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))  # 正确的数量除以总数


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

```
### 7.CNN高级

####  `1 * 1` 卷积核的作用

1 X 1 卷积可以加速运算  

![](<assets/1696428163492.png>)

#### GoogLeNet及其实现
GoogLeNet 包括卷积（Convolution），池化（Pooling）、全连接（Softmax）以及连接（Other）四个部分。

而为了减少代码的冗余，将由以上四个模块所组成的相同的部分，封装成一个类 / 函数，在 GoogLeNet 中，这样的部分被称为 Inception Module。

![](<assets/1696428163379.png>)

##### Inception Module

实际上 Inception Module 以及 GoogLeNet 自身只是一种基础的网络结构，他的出现是为了解决构造网络时的部分超参数难以确定的问题。

以卷积核大小 (kernel_size) 为例，虽然无法具体确定某问题中所应使用的卷积核的大小。但是往往可以有几种备选方案，因此在这个过程中，可以利用这样的网络结构，来将所有的备选方案进行计算，并在后续计算过程中增大最佳方案的权重，以此来达到确定超参数以及训练网络的目的。

其中的具体成分可以根据问题进行调整，本文中所详细介绍的 Inception Module 也仅用作参考。  

![](<assets/1696428163425.png>)
####  模块化构建CNN 代码
```python
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 第一个通道，输入通道为in_channels,输出通道为16，卷积盒的大小为1*1的卷积层
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 第二个通道
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # 第三个通道
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # 第四个通道
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 按照channel 这一维进行拼接
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10) # 1408 是最后进过不断地进行变换算出来的。
	
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
```

#### resnet 构建代码

![[Pasted image 20231005201917.png]]
![[Pasted image 20231005201956.png]]


```python

```


### 8.RNN
#### 1.DNN
Dense 网络：稠密网络，有很多线性层对输入数据进行空间上的变换, 又叫 DNN  
输入 x1,x2…xn 是数据样本的不同特征  
Dense 连接就是指全连接

![](https://img-blog.csdnimg.cn/59dd9dbe4d7b49f1ac5fcf1c70e7709d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

比如预测天天气，就需要知道之前几天的数据，每一天的数据都包含若个特征，需要若干天的数据作为输入

假设现在取前 3 天，每一天有 3 个特征

把 x1,x2,x3 拼成有 9 个维度的长向量，然后去训练最后一天是否有雨![](https://img-blog.csdnimg.cn/c1bf4c37f0b24e418b0e9a77c8fa388b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

用全连接稠密网络进行预测，如果输入序列很长，而且每一个序列维度很高的话，对网络训练有很大挑战，因为稠密网络dense参数是比较多的。
对于卷积层：比如输入通道是 128 个，输出通道是 64 个，如果用 5x5 的卷积。权重数就是 25x64x128=204800，卷积层的输入输出只与通道数和卷积核的大小有关，全连接层和变换之后的数据大小有关，比如 3 阶张量经过一系列的卷积变换还剩下 4096 个元素，4096 我们很少直接降成 1 维或者 10 维，而是先降成 1024 维，4096x1024=4194304，所以相比起来，卷积层的权重并不多，而全连接层的权重较多。在网络的全部参数中，全连接层是占大头的。

**为什么卷积神经网络的权重比较少呢？**  
因为使用了权重共享的概念，做卷积时，整个图像的卷积核是共享的，并不是图像上的每一个像素要和下一层的 featureMap 建立连接, 权重数量就少，处理视频的时候，每一帧就少一张图像，我们需要把一组图像做成一个集合，如果用全连接网络的话，使用到的权重的数量就是一个天文数字，难以处理

RNN 专门用来处理带有序列模式的数据，也使用权重共享减少需要训练的权重的数量  
我们把 x1,x2,x3,xn 看成是一个序列，不仅考虑 x1,x2 之间的连接关系，还考虑 x1,x2 的时间上的先后顺序  
x2 依赖于 x1,x3 依赖于 x2, 下一天的天气状况部分依赖于前一天的天气状况，RNN 主要处理这种具有序列连接的  
天气，股市，金融，自然语言处理都是序列数据
#### 2.RNN Cell
RNN Cell 本质是一个线性层（linear），把一个维度映射到另一个维度（比如把输入的 3 维向量 xt 变成输出 5 维向量 ht）
这个线性层与普通的线性层的区别是这个线性层是共享的 
![](https://img-blog.csdnimg.cn/20200908100503831.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

展开就是下图（其中所有的 RNN cell 是同一个线性层，因为是展开的嘛），h0 是先验值，没有就设置成 0 向量![](https://img-blog.csdnimg.cn/1c991f1ae45c4295bb593ca3610a628f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

RNN 过程：h0 和 x1 经过某种运算将他们拼接在一起，即：分别做线性变换，然后求和，生成 h1。然后把 h1, 作为输出送到下一次 RNN cell 计算中，这次输入变成 x2，x2 和 h1 合在一起运算，生成 h2… 

![](https://img-blog.csdnimg.cn/9fb2094beb3a49ed905e0ad378a7867f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)具体的计算过程：
输入 xt 先做线性变换，也是,$x_{t}$的维度是 input_size,$h_{t-1}$ 的维度是 hidden_size, 输出 ht 的维度是 hidden_size，我们需要先把 xt 的维度变成 hidden_size, 所以 Wih 应该是一个 hidden_size $*$ input_size 的矩阵，Wih * xt 得到一个 hidden_size x 1 的矩阵（就是维度为 hidden_size 的向量），bih 是偏置。输入权重矩阵 Whh 是一个 hidden_size* hidden_size 的矩阵。  
whhHt-1+bhh 和 WihXt+bih 都是维度为 hidden_size 的向量，两个向量相加，就把信息融合起来了，融合之后用 tanh 做激活，循环神经网络的激活函数用的是 tanh, 因为 tanh 的取值在 - 1 到 + 1 之间，算出结果得到隐藏层输出 ht
 ![](https://img-blog.csdnimg.cn/33ed2179f33444f1a330535cdf549dd4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

把 RNN Cell 以循环的方式把序列（x1,x2,…）一个一个送进去，然后依次算出隐藏层 (h1,h2…) 的过程, 每一次算出来的 h 会作为下一个 RNN Cell 的输入，这就叫循环神经网络 

#### 3.pytorch 里面构造 RNN 的两种方式
###### ①自己构建 Cell  
需要设定输入的值 input_size，和隐层的值 hidden_size，就能确定权重 W 的维度和偏置 b 的维度

```python
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
#设定参数，输入维度和隐层维度
```

```python
hidden = cell(input, hidden)
#实例化Cell之后，我们需要给定当前的输入input以及当前的hidden，所以需要用循环来处理
```

比如

```
h1=Cell(x1,h0)
```

![](https://img-blog.csdnimg.cn/61ccbc04cd604b6db8479940177d27be.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

 ![](https://img-blog.csdnimg.cn/73e8dd8a8fbb483b8d2624f692b85c76.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_19,color_FFFFFF,t_70,g_se,x_16)![](https://img-blog.csdnimg.cn/20200908145953906.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

batchSize 表示批量  
seqLen=3 表示每一个样本都有 x1,x2,x3 这些特征  
inputSize=4 表示每一个特征都是 4 维的  
hoddenSize=2 表示每一个隐藏层是 2 维
 **代码展示：**
```python
import torch
batch_size=1
seq_len=3
input_size=4
hidden_size=2
Cell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)#初始化，构建RNNCell
dataset=torch.randn(seq_len,batch_size,input_size)#设置dataset的维度
hidden=torch.zeros(batch_size,hidden_size)#隐层的维度：batch_size*hidden_size，先把h0置为0向量
for idx,input in enumerate(dataset):
    print('='*20,idx,'='*20)
    print('Input size:',input.shape)
    hidden=Cell(input,hidden)
    print('Outputs size:',hidden.shape)
    print(hidden)
```
 注释：
1. torch.randn(sizes, out=None) → Tensor      torch.randn() 返回一个包含了从**标准正态分布**中抽取的一组随机数的张量   size：张量的形状，out: 结果张量。
2. 功能 torch.zeros() 返回一个由标量值 0 填充的张量，其形状由变量参数 size 定义。 
**结果:** 
```
==================== 0 ====================
Input size: torch.Size([1, 4])
Outputs size: torch.Size([1, 2])
tensor([[0.8677, 0.8320]], grad_fn=<TanhBackward>)
==================== 1 ====================
Input size: torch.Size([1, 4])
Outputs size: torch.Size([1, 2])
tensor([[-0.9137, -0.5884]], grad_fn=<TanhBackward>)
==================== 2 ====================
Input size: torch.Size([1, 4])
Outputs size: torch.Size([1, 2])
tensor([[0.9840, 0.9235]], grad_fn=<TanhBackward>)
```
##### ②直接使用 RNN
```python
cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
num_layers=num_layers) #num_layers：RNN的层数
```
#num_layers：RNN 的层数，如果 RNN 有多层，每一层都会有输出
```python
out, hidden = cell(inputs, hidden)
```

![](https://img-blog.csdnimg.cn/6b98115bab44469c9d27316ac668d830.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

inputs: 所有的 x,x1,x2,x3,......xn  
用 RNN 不用自己写循环，它自动循环，所以输入的时候要把所有的序列都送进去，然后给定 h0, 然后我们就会得到所有的隐层输出以及最后一层的输出 
![](https://img-blog.csdnimg.cn/20200908153115910.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

当 RNN 有多层，同样颜色的 RNN Cell 是同一个，所以下图是有 3 个线性层（一个 RNNCell 是一个线性层）![](https://img-blog.csdnimg.cn/20200908153346554.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)
```python
import torch
batch_size=1
seq_len=3
input_size=4
hidden_size=2
num_layers=1
cell=torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
#构造RNN时指明输入维度，隐层维度以及RNN的层数
inputs=torch.randn(seq_len,batch_size,input_size)
hidden=torch.zeros(num_layers,batch_size,hidden_size)
out,hidden=cell(inputs,hidden)
print('Output size:',out.shape)
print('Output:',out)
print('Hidden size:',hidden.shape)
print('Hidden',hidden)
```

结果：
```
Output size: torch.Size([3, 1, 2])
Output: tensor([[[-0.9123,  0.9218]],
 
        [[ 0.9394, -0.2471]],
 
        [[-0.9064,  0.5193]]], grad_fn=<StackBackward>)
Hidden size: torch.Size([1, 1, 2])
Hidden tensor([[[-0.9064,  0.5193]]], grad_fn=<StackBackward>)
```

如果初始化 RNN 时，把 batch_first 设置成了 TRUE, 那么 inputs 的参数 batch_size 和 seq_len 需要调换一下位置![](https://img-blog.csdnimg.cn/20200908160110605.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)
####  4.利用 RNN Cell 训练 hello 转换到 ohlol![](https://img-blog.csdnimg.cn/20200908161201637.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

 把字符转成向量![](https://img-blog.csdnimg.cn/20200903201824357.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

inputsize=4, 因为输入有 4 个字符（e h l o）  
这相当于一个多分类问题，输出就是一个 4 维的向量，每一维代表是某一个字符的概率，接交叉熵就能输出概率了![](https://img-blog.csdnimg.cn/bc60b46ab45b48b1b0b899346c1b0004.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16) 代码：
```python
#使用RNNCell
import torch
#参数
input_size = 4
hidden_size = 4
batch_size = 1
#准备数据
idx2char = ['e', 'h', 'l', 'o'] #为了后面可以根据索引把字母取出来
x_data = [1, 0, 2, 3, 3]  # hello中各个字符的下标
y_data = [3, 1, 2, 3, 2]  # ohlol中各个字符的下标
#独热向量
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # (seqLen, inputSize)
#reshape the inputs to (seqlen,batchSize,inputSize)
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
#reshape the labels to (seqlen,1)
labels = torch.LongTensor(y_data).view(-1, 1)
# torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
print(inputs.shape, labels.shape)
 
#设计模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        #对参数进行初始化
        self.batch_size = batch_size #仅构造h0时需要batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
 
    def forward(self, inputs, hidden):
        hidden = self.rnncell(inputs, hidden)  # 输入和隐层转换为下一个隐层 ht = rnncell(xt,ht-1)
        # shape of inputs:(batchSize, inputSize),shape of hidden:(batchSize, hiddenSize),
        return hidden
 
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)  # 提供初始的隐层，生成全0的h0
 
 
net = Model(input_size, hidden_size, batch_size)
#损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)#使用Adam优化器，改进的随机梯度下降优化器进行优化
 
for epoch in range(15):
    loss = 0
    optimizer.zero_grad() #优化器梯度归0
    hidden = net.init_hidden() #每一轮的第一步先初始化hidden,即先计算h0
    print('Predicted string:', end='')
    #shape of inputs:(seqlen序列长度,batchSize,inputSize)  shape of input:(batchSize,inputSize)
    #shape of labeis:(seqsize序列长度,1)  shape of labei:(1)
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        # 注意交叉熵在计算loss的时候维度关系，这里的hidden是([1, 4]), label是 ([1])
        loss += criterion(hidden, label) #不用loss.item,所有的和才是最终的损失
        _, idx = hidden.max(dim=1)  #hidden.max()函数找出hidden里的最大值  _, idx最大值的下标
        print(idx2char[idx.item()], end='')
 
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))
```
####  5.利用 RNN 训练 hello 转换到 ohlol
```python
#使用RNN
import torch
input_size=4
hidden_size=4
num_layers=1
batch_size=1
seq_len=5
# 准备数据
idx2char=['e','h','l','o']
x_data=[1,0,2,2,3] # hello
y_data=[3,1,2,3,2] # ohlol
 
one_hot_lookup=[[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]] #分别对应0,1,2,3项
x_one_hot=[one_hot_lookup[x] for x in x_data] # 组成序列张量
print('x_one_hot:',x_one_hot)
 
# 构造输入序列和标签
inputs=torch.Tensor(x_one_hot).view(seq_len,batch_size,input_size)
labels=torch.LongTensor(y_data)  #要注意在CrossEntropy()中labels维度是: (seqLen * batch_size ，1)
 
# design model
class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers=1):
        super(Model, self).__init__()
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.rnn=torch.nn.RNN(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers)
 
    def forward(self,input):
        hidden=torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        out, _=self.rnn(input,hidden)
        # 为了能和labels做交叉熵，需要reshape一下:(seqlen*batchsize, hidden_size),即二维向量，变成一个矩阵
        return out.view(-1,self.hidden_size)
 
net=Model(input_size,hidden_size,batch_size,num_layers)
 
# loss and optimizer
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(), lr=0.05)
 
# train cycle
for epoch in range(20):
    optimizer.zero_grad()
    #inputs维度是: (seqLen, batch_size, input_size) labels维度是: (seqLen * batch_size * 1)
    #outputs维度是: (seqLen, batch_size, hidden_size)
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()
 
    _, idx=outputs.max(dim=1)
    idx=idx.data.numpy()
    print('Predicted: ',''.join([idx2char[x] for x in idx]),end='')
    print(',Epoch [%d/20] loss=%.3f' % (epoch+1, loss.item()))
 
```
#### 6.embedding（嵌入层）编码方式
--------------------
       独热编码向量维度过高；  
       独热编码向量稀疏，每个向量是一个为 1 其余为 0；  
       独热编码是硬编码，编码情况与数据特征无关；  
       采用一种低维度的、稠密的、可学习数据的编码方式：Embedding。
 Embedding 把一个高维的稀疏的样本映射到一个稠密的低维的空间里面，也就是数据的降维。![](https://img-blog.csdnimg.cn/55f9ce44885f41849f9e46429bfaf8b5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)
输入 4 维，嵌入层 5 维（嵌入层即可以高维也可以低维）
4 维转换为 5 维，构建这样一个矩阵![](https://img-blog.csdnimg.cn/861ea20e06b7418da060b383c4a3e9cb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

 在里边可以做一个查询![](https://img-blog.csdnimg.cn/2b76b3eaefad422faec61f02db009adf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

输入为 2，就表示是第二个字符的索引（索引从 0 开始），找到第 2 行，把这个向量输出，这就叫 embedding
反向传播时导数怎么求？
![](https://img-blog.csdnimg.cn/851ae239bda9471da6816bfc10f77b9b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

输入层必须是长整型张量，输出是（seqlen,4） 
![](https://img-blog.csdnimg.cn/a442815df75449b98bc2f2b146d7ca70.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)embedding 初始化：

`num_bedding`: input 独热向量的维度

`num_bedding() 和 embedding_dim()` 构成矩阵的宽度和高度

输入层必须是长整型张量，输出是（input shape,embedding_shape）

![](https://img-blog.csdnimg.cn/114768fe41334589a9921409a5016fbd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

```python
#Embedding编码方式
import torch
input_size = 4
num_class = 4
hidden_size = 8
embedding_size = 10
batch_size = 1
num_layers = 2
seq_len = 5
 
idx2char_1 = ['e', 'h', 'l', 'o']
idx2char_2 = ['h', 'l', 'o']
 
x_data = [[1, 0, 2, 2, 3]]
y_data = [3, 1, 2, 2, 3]
 
# inputs 维度为（batchsize，seqLen）
inputs = torch.LongTensor(x_data)
# labels 维度为（batchsize*seqLen）
labels = torch.LongTensor(y_data)
 
 
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #告诉input大小和 embedding大小 ，构成input_size * embedding_size 的矩阵
        self.emb = torch.nn.Embedding(input_size, embedding_size)
 
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        # batch_first=True，input of RNN:(batchsize,seqlen,embeddingsize) output of RNN:(batchsize,seqlen,hiddensize)
        self.fc = torch.nn.Linear(hidden_size, num_class) #从hiddensize 到 类别数量的 变换
 
    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size) # 层数 * seqLen *  hidden_size
        x = self.emb(x)  # 进行embedding处理，把输入的长整型张量转变成嵌入层的稠密型张量
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class) #为了使用交叉熵，变成一个矩阵（batchsize * seqlen,numclass）
 
net = Model()
 
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
 
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
 
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
 
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted string: ', ''.join([idx2char_1[x] for x in idx]), end='')
    print(", Epoch [%d/15] loss = %.3f" % (epoch + 1, loss.item()))
```
### 9.RNN高级
用RNN做一个分类器，现在有一个数据集，数据集里有人名和对应的国家，我们需要训练一个模型，输入一个新的名字，模型能预测出是基于哪种语言的（18 种不同的语言，18 分类）![](https://img-blog.csdnimg.cn/a078050f129e47218ba7fa687a6899f5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

 在nlp当中, 通常先把词或字编程一个 one-hot 向量，one-hot 向量维度高，而且过于稀疏，所以一般来说呀先通过嵌入层（Embed）把 one-hot 向量转化成低维的稠密向量，然后经过 RNN，隐层的输出不一定和最终要求的目标一致，所以要用一个线性层把输出映射成和我们的要求一致，

 我们的需求是输出名字所属的语言分类，我们对 01-05 这些输出是没有要求的，即不需要对所有的隐层输出做线性变换，为了解决这个问题，我们可以把网络变得更简单，如下图![](https://img-blog.csdnimg.cn/16e8e04f31454aa4b07e4aa9c8fb41ae.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)

输入向量经过嵌入层之后，输入到 RNN, 输出最终的隐层状态，最终的隐层状态经过一个线性层，我们分成 18 个类别，就可以实现名字分类的任务了  
![](https://img-blog.csdnimg.cn/20201018204743553.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)


 ![](https://img-blog.csdnimg.cn/28c778b8022648699dcecac7ed35bd78.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_20,color_FFFFFF,t_70,g_se,x_16)
#### 主循环  
```python
classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER) 
```

`N_CHARS`：字符数量（输入的是英文字母，每一个字符都要转变成独热向量）  
`HIDDEN_SIZE`：隐层数量（GRU 输出的隐层的维度）  
`N_COUNTRY`：一共有多少个分类  
`N_LAYER`：用来设置所使用的 GRU 层数
```python
if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    #把模型迁移到GPU
    if USE_GPU:
        device = torch.device('cuda:0')
        classifier.to(device)
 
    criterion = torch.nn.CrossEntropyLoss()     #计算损失
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.001)   #更新
 
    start = time.time()
    print("Train for %d epochs..." % N_EPOCHS)
 
    acc_list= []
    for epoch in range(1, N_EPOCHS + 1):
        print('%d / %d:' % (epoch, N_EPOCHS))
        trainModel()
        acc = testModel()
        acc_list.append(acc)
```
在每一个 epoch 做一次训练和测试，把测试的结果添加到 acc_list 列表（可以用来绘图，可以看到训练的损失是如何变化的）
#### 准备数据
拿到的是字符串，先转变成序列，转成列表，列表里面的每一个数就是名字里面的每一个字符![](https://img-blog.csdnimg.cn/20201018210341246.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 接下来做词典，可以用 ASCII 表，ASCII 表是 128 个字符，我们把字典长度设置成 128，求每一个字符对应的 ASCII 值，拼成我们想要的序列![](https://img-blog.csdnimg.cn/20201018210625457.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)
上图中的最右表中每一个数并不是一个数字，而是一个独热向量例如 77。就是一个 128 维的向量，第 77 个数的值为 1，其他的值都是 0.  
`对于 Embed(嵌入层) 来说，只要告诉嵌入层第几个维度是 1 就行了`，所以只需要把 ASCII 值放在这就行了。
注意：这里是不需要输入独热编码的。
##### 序列长短不一怎么解决？ 
![](https://img-blog.csdnimg.cn/20201018211335810.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)
如上图左，每一行是一个序列，我们解决序列长短不一的方法是 padding（因为张量必须保证所有的数据都贴满，不然就不是张量），如右图，就是在做一个 batch 的时候，我们看这一个 batch 里面哪一个字符串的长度最长，然后把其他字符串填充成和它一样的长度，就能保证可以构成一个张量，因为每个维度的数量不一样是没办法构成张量的
##### 分类的处理
我们需要把各个分类（国家）转成一个分类索引，不能直接用字符串作为我们的分类标签![](https://img-blog.csdnimg.cn/20201018211947125.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)
#####  准备数据集代码：
. gzip 和 csv 这两个包可以帮我们读取 gz 文件 (gzip.open() 打开 gzip 文件，然后用 csv.reader()去读里面的数据)  
有很多种不同的方式可以访问数据集，比如有些数据集不是. gz, 而是. pickle, 就可以用 pickle 包，还有 HDFS,HD5 得用 HDFS 的包读取，根据拿到的数据类型不一样，用相应的包把数据读出来。  
我们读到的 rows 是一个元组，形式是（name,language）

```python
# 准备数据集
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'data/names_train.csv.gz' if is_train_set else'data/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader) #一个元组
            
        # 取出名字
        self.names = [row[0] for row in rows] #先把名字都取出来
        self.len = len(self.names) #记录样本数量
        self.countries = [row[1] for row in rows]#把标签language取出来
        self.country_list = list(sorted(set(self.countries)))#set是先把列表变成集合，即去除重复的元素，
        #这样每一个语言就只剩下一个实例，然后用sorted排序 list变成列表
 
        self.country_dict = self.getCountryDict() #根据列表，把列表转变成词典
        self.country_num = len(self.country_list)
 
    def __getitem__(self, index):
        return self.names[index], self.countries_dict[self.countries[index]]
    ##__getitem__根据输入的名字找到对应国家的索引
    #返回两项，一项是输入样本
    #拿到输入样本之后，先把国家取出来，然后根据国家去查找对应的索引
    def getCountriesDict(self):
        countries_dict = dict()  #先做一个空字典
        for index, country_name in enumerate(self.countries_list, 0):
            countries_dict[country_name] = index  #构建键值对
        return countries_dict
 
    def __len__(self): #返回数据集长度
        return self.len
 
    def id2country(self, index): #根据索引返回国家字符串 例：1  Chinese
        return self.countries_list[index]
 
    def getCountriesNum(self):   #返回国家总数量
        return self.countries_num
```

```python
trainset = NameDataset(is_train_set=True) 
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True) 
testset = NameDataset(is_train_set=False) 
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
N_COUNTRY = trainset.getCountriesNum() #总的类别数量，决定模型最终的输出大小
```

N_LAYER = 2    gru用了两层      N_EPOCH = 100   # 将来训练 100 轮
N_CHARS = 128  #128 的字典长度   USE_GPU = False   不用GPU

![](https://img-blog.csdnimg.cn/5bc8616ba69c4464a51c91e6543e9279.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_6,color_FFFFFF,t_70,g_se,x_16)

#### 模型设计
```python
#模型设计
import torch
from torch.nn.utils.rnn import pack_padded_sequence
 
class RNNClassifier(torch.nn.Module):
    # input_size=128, hidden_size=100, output_size=18
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers  # GRU 使用层数
        self.n_directions = 2 if bidirectional else 1  # 是否双向循环神经网络
        self.embedding = torch.nn.Embedding(input_size, hidden_size)  # 输入大小128，输出大小100。
        # 经过Embedding后input的大小是100，hidden_size的大小也是100，所以形参都是hidden_size。
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        # 如果是双向，会输出两个hidden层，要进行拼接，所以线性成的input大小是 hidden_size * self.n_directions，输出是大小是18，是为18个国家的概率。
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)
 
    def _init_hidden(self, batch_size):
        #初始的全0隐层h0
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return hidden
 
    def forward(self, input, seq_lengths):
        # 先对input进行转置，input shape : batch_size*max_seq_lengths -> max_seq_lengths*batch_size 每一列表示姓名
        input = input.t()
        batch_size = input.size(1)  # 总共有多少列，既是batch_size的大小
        hidden = self._init_hidden(batch_size)  # 初始化隐藏层
        embedding = self.embedding(input)  # embedding.shape : max_seq_lengths*batch_size*hidden_size 12*64*100
        # pack_padded_sequence方便批量计算
        gru_input = pack_padded_sequence(embedding, seq_lengths)
        # 进入网络进行计算
        output, hidden = self.gru(gru_input, hidden)
 
        # 如果是双向的，需要进行拼接
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
 
        else:
            hidden_cat = hidden[-1]
 
        # 线性层输出大小为18
        fc_output = self.fc(hidden_cat)
        return fc_output
```

##### **双向循环神经网络**
lstm gru rnn 都有双向的  
下图是单向的 RNN, 其中 RNN Cell 共享权重和偏置，所以 w 和 b 是一样的，Xn-1 的输出只包含它之前的序列的信息，即只考虑过去的信息，实际上在自然语言处理（NLP）我们还需要考虑来自未来的信息![](https://img-blog.csdnimg.cn/20201019093011507.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

![](https://img-blog.csdnimg.cn/20201019093151582.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

正向算完之后，再反向算一下，然后把算出来的隐层输出做拼接，如下图 hN 是 h(0,b) 和 h(N,f) 拼接起来的，h(N-1) 是把 h（1,b）和 h(N-1,f) 拼接起来，这样的循环神经网络叫双向循环神经网络![](https://img-blog.csdnimg.cn/20201019093624390.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 

最终，反向得到一个 h(N,b)![](https://img-blog.csdnimg.cn/20201019095412482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 

每一次调用 GRU 会输出 out 和 hidden 两个项，其中 hidden 包含的项如下![](https://img-blog.csdnimg.cn/20201019095655520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 

 双向循环神经网络的 forward 过程

```python
def forward(self, input, seq_lengths):
 # input shape : B x S - > S x B（S:sequential(序列)，B：batch）
  input = input.t()                    %矩阵转置input shape : B x S - > S x B
  batch_size = input.size(1)           %保存batch_size用来构建最初始的隐层
  hidden = self._init_hidden(batch_size) %创建隐层
  embedding = self.embedding(input)      %把input扔到嵌入层里面，做嵌入
  %嵌入之后，输入的维度就变成了（𝑠𝑒𝑞𝐿𝑒𝑛,𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒,ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒）
  # pack them up 
  gru_input = pack_padded_sequence(embedding, seq_lengths)
  output, hidden = self.gru(gru_input, hidden)  %第二个hidden是初始的隐层，
  %我们想要得到的是第一个hidden的值
  if self.n_directions == 2: 
      hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
      %如果是双向的循环神经网络，会有两个hidden，需要把他们拼接起来
  else: 
      hidden_cat = hidden[-1]  %如果是单向的循环神经网络，就只有1个hidden
  fc_output = self.fc(hidden_cat)             %把最后的隐层输出经过全连接层变换成我们想要的维 
  度做分类
  return fc_output
```

```python
 input = input.t()%矩阵转置input shape : B x S - > S x B
 %功能如下图
```
![](https://img-blog.csdnimg.cn/20201019101425304.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

```python
embedding = self.embedding(input)      %把input扔到嵌入层里面，做嵌入
%嵌入之后，输入的维度就变成了（𝑠𝑒𝑞𝐿𝑒𝑛,𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒,ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒）
```

 ![](https://img-blog.csdnimg.cn/2020101910211229.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

为了提高运行效率，GRU 支持一种提速，尤其是面对序列长短不一的时候，在 pyTorch 中， pack_padded_sequence 的功能如下![](https://img-blog.csdnimg.cn/20201019102718698.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 

先根据长度排序![](https://img-blog.csdnimg.cn/20201019102855256.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 

排好序之后，再经过嵌入层![](https://img-blog.csdnimg.cn/20201019102957514.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 

使用 pack_padded_sequence 做成下面这样的数据，GRU 的运算效率更高哦 (即把没有计算意义的 padding 0 去掉)![](https://img-blog.csdnimg.cn/202010191032206.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 

所以 pack_padded_sequenceh 函数需要输入数据的长度 seq_lengths

```python
gru_input = pack_padded_sequence(embedding, seq_lengths)
```

 GRU 根据上图的 batch_sizes 就决定每一时刻取多少行，GRU 的工作效率就提高了![](https://img-blog.csdnimg.cn/2020101910464277.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

##### 由名字转换成 Tensor 的过程 

 ![](https://img-blog.csdnimg.cn/20201019105100302.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

过程如下：  
1：字符串—> 字符—> 相应的 ASCII 值  
![](https://img-blog.csdnimg.cn/20201019105446962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 然后做 padding 填充![](https://img-blog.csdnimg.cn/20201019105526349.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

填充之后转置![](https://img-blog.csdnimg.cn/20201019105605864.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center) 转置之后排序![](https://img-blog.csdnimg.cn/20201019105631342.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

##### 转换名字为 tensor 
```python
def name2list(name):
    """返回ASCII码表示的姓名列表与列表长度"""
    arr = [ord(c) for c in name]
    return arr, len(arr)

def make_tensors(names, countries):
    # 元组列表，每个元组包含ASCII码表示的姓名列表与列表长度
    sequences_and_lengths = [name2list(name) for name in names]
    # 取出所有的ASCII码表示的姓名列表
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    # 取出所有的列表长度
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    # 将countries转为long型
    countries = countries.long()
 
    # 接下来每个名字序列补零，使之长度一样。
    # 先初始化一个全为零的tensor，大小为 所有姓名的数量*最长姓名的长度
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
 
    # 将姓名序列覆盖到初始化的全零tensor上
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    # 根据序列长度seq_lengths对补零后tensor进行降序怕排列，方便后面加速计算。
    # 返回排序后的seq_lengths与索引变化列表
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    # 根据索引变化列表对ASCII码表示的姓名列表进行排序
    seq_tensor = seq_tensor[perm_idx]
    # 根据索引变化列表对countries进行排序，使姓名与国家还是一一对应关系
    # seq_tensor.shape : batch_size*max_seq_lengths,
    # seq_lengths.shape : batch_size
    # countries.shape : batch_size
    countries = countries[perm_idx]
    return seq_tensor, seq_lengths, countries
```

```python
def name2list(name): 
  arr = [ord(c) for c in name] 
  return arr, len(arr)
  %name2list返回两个，一个是元组，代表列表本身，一个是列表的长度
```

```python
name_sequences = [sl[0] for sl in sequences_and_lengths] 
 %单独拿出列表
```

![](https://img-blog.csdnimg.cn/20201019111423775.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmdtZW5nc2h1eGlhd28=,size_16,color_FFFFFF,t_70#pic_center)

##### 训练过程

```python
# 训练
def time_since(since):
    s = time.time() - since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)
 
def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        # make_tensors函数返回经过降序排列后的 姓名列表，列表长度，国家
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = model(inputs, seq_lengths)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(i)
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss
 
def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for idx, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = model(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total
```
##### 完整代码：
```python
'''
根据名字识别他所在的国家
人名字符长短不一，最长的10个字符，所以处理成10维输入张量，都是英文字母刚好可以映射到ASCII上
Maclean ->  ['M', 'a', 'c', 'l', 'e', 'a', 'n'] ->  [ 77 97 99 108 101 97 110]  ->  [ 77 97 99 108 101 97 110 0 0 0]
共有18个国家，设置索引为0-17
训练集和测试集的表格文件都是第一列人名，第二列国家
'''
import torch
import  time
import csv
import gzip
from  torch.utils.data import DataLoader
import datetime
import matplotlib.pyplot as plt
import numpy as np
 
# Parameters
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
USE_GPU = True
 
class NameDataset():         #处理数据集
    def __init__(self, is_train_set=True):
        filename = 'data/names_train.csv.gz' if is_train_set else 'data/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:    #打开压缩文件并将变量名设为为f
            reader = csv.reader(f)              #读取表格文件
            rows = list(reader)
        self.names = [row[0] for row in rows]   #取出人名
        self.len = len(self.names)              #人名数量
        self.countries = [row[1] for row in rows]#取出国家名
        self.country_list = list(sorted(set(self.countries)))#国家名集合，18个国家名的集合
        #countrys是所有国家名，set(countrys)把所有国家明元素设为集合（去除重复项），sorted（）函数是将集合排序
        #测试了一下，实际list(sorted(set(self.countrys)))==sorted(set(self.countrys))
        self.country_dict = self.getCountryDict()#转变成词典
        self.country_num = len(self.country_list)#得到国家集合的长度18
 
    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]
 
    def __len__(self):
        return self.len
 
    def getCountryDict(self):
	        country_dict = dict()                                       #创建空字典
        for idx, country_name in enumerate(self.country_list,0):    #取出序号和对应国家名
            country_dict[country_name] = idx                        #把对应的国家名和序号存入字典
        return country_dict
 
    def idx2country(self,index):            #返回索引对应国家名
        return self.country_list(index)
 
    def getCountrysNum(self):               #返回国家数量
        return self.country_num
 
trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False)
 
N_COUNTRY = trainset.getCountrysNum()       #模型输出大小
 
def create_tensor(tensor):#判断是否使用GPU 使用的话把tensor搬到GPU上去
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor
 
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size                  #包括下面的n_layers在GRU模型里使用
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
 
        self.embedding = torch.nn.Embedding(input_size, hidden_size)#input.shape=(seqlen,batch) output.shape=(seqlen,batch,hiddensize)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
                                #输入维度       输出维度      层数        说明单向还是双向
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)#双向GRU会输出两个hidden，维度需要✖2，要接一个线性层
 
    def forward(self, input, seq_lengths):
        input = input.t()               #input shaoe :  Batch x Seq -> S x B 用于embedding
        batch_size = input.size(1)
        hidden =self._init_hidden(batch_size)
        embedding = self.embedding(input)
 
        # pack_padded_sequence函数当出入seq_lengths是GPU张量时报错，在这里改成cpu张量就可以，不用GPU直接注释掉下面这一行代码
        seq_lengths = seq_lengths.cpu()#改成cpu张量
        # pack them up
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding, seq_lengths)#让0值不参与运算加快运算速度的方式
        #需要提前把输入按有效值长度降序排列 再对输入做嵌入，然后按每个输入len（seq——lengths）取值做为GRU输入
 
        output, hidden = self.gru(gru_input, hidden)#双向传播的话hidden有两个
        if self.n_directions ==2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output
 
    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return  create_tensor(hidden)
 
#classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
 
#对名字的处理需要先把每个名字按字符都变成ASCII码
def name2list(name):#把每个名字按字符都变成ASCII码
    arr = [ord(c) for c in name]
    return arr, len(arr)
 
def make_tensors(names, countries):     #处理名字ASCII码 重新排序的长度和国家列表
    sequences_and_lengths= [name2list(name) for name in names]                  #把每个名字按字符都变成ASCII码
    name_sequences = [sl[0] for sl in sequences_and_lengths]                    #取出名字列表对应的ACSII码
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])     #取出每个名字对应的长度列表
    countries = countries.long()
 
    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()     #先做一个 名字数量x最长名字长度的全0tensor
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):  #取出序列，ACSII码和长度列表
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)                       #用名字列表的ACSII码填充上面的全0tensor
 
    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)#将seq_lengths按序列长度重新降序排序，返回排序结果和排序序列。
    seq_tensor = seq_tensor[perm_idx]                               #按新序列把ASCII表重新排序
    countries = countries[perm_idx]                                 #按新序列把国家列表重新排序
 
                #返回排序后的 ASCII列表         名字长度降序列表        国家名列表
    return create_tensor(seq_tensor),create_tensor(seq_lengths),create_tensor(countries)
 
def trainModel():
    total_loss = 0
 
    for i, (names, countries) in enumerate(trainloader, 1):
        optimizer.zero_grad()
        inputs, seq_lengths, target = make_tensors(names, countries)#取出排序后的 ASCII列表 名字长度列表 国家名列表
        output = classifier(inputs, seq_lengths)    #把输入和序列放入分类器
        loss = criterion(output, target)            #计算损失
 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
        #打印输出结果
        #if i % 100 == 0:
        #    print(f'Epoch {epoch} ')
        if i == len(trainset) // BATCH_SIZE :
            #print(f'[13374/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
        '''elif i % 10 == 9 :
            print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')'''
    return total_loss
 
def testModel():
    correct = 0
    total = len(testset)
 
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)    #返回处理后的名字ASCII码 重新排序的长度和国家列表
            output = classifier(inputs, seq_lengths)                        #输出
            pred = output.max(dim=1, keepdim=True)[1]                       #预测
            correct += pred.eq(target.view_as(pred)).sum().item()           #计算预测对了多少
 
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total
 
if __name__ == '__main__':
    print("Train for %d epochs..." % N_EPOCHS)
    start = time.time()
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device('cuda:0')
        classifier.to(device)
 
    criterion = torch.nn.CrossEntropyLoss()     #计算损失
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.001)   #更新
 
    acc_list= []
    for epoch in range(1, N_EPOCHS+1):
        #训练
        print('%d / %d:' % (epoch, N_EPOCHS))
        trainModel()
        acc = testModel()
        acc_list.append(acc)
    end = time.time()
    print(datetime.timedelta(seconds=(end - start) // 1))
 
 
    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
 
```

运行：

![](https://img-blog.csdnimg.cn/20210402152708627.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjA0NzY0Mw==,size_16,color_FFFFFF,t_70#pic_center)![](https://img-blog.csdnimg.cn/b57481b2cc314ed8af06d6030d448f62.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Yqq5Yqb5a2m5Lmg55qE5pyx5pyx,size_13,color_FFFFFF,t_70,g_se,x_16)

[参考链接](https://blog.csdn.net/ningmengshuxiawo/article/details/109149735 "参考链接")





## 深度学习的总体的学习思路
![[Pasted image 20231005134137.png]]

