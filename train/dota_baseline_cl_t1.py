from __future__ import print_function, division

import torch
import torch.nn as nn  # nn是神经网络
from torch.autograd import Variable
import numpy as np  # 导入numpy库
import os  # 导入标准库os，对文件和文件夹进行操作
from torch.optim import Adam  # 导入模块
from resnet50 import resnet50,CNN,Bottleneck  # 相当于从一个模块中导入函数
from dataset import DOTA_ONE
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES']='0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU


def requires_grad(model, flag = True):
    for p in model.parameters():
        p.requires_grad = flag


# 加载训练集和测试集
trainset = DOTA_ONE(root='./Images', is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=8, drop_last=False)
testset = DOTA_ONE(root='./Images', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=8, drop_last=False)

# 看电脑的 GPU 能否被 PyTorch 调用
cuda_avail = torch.cuda.is_available()

baselinet1 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
baselinet1.load_state_dict(torch.load('./DOTA_baseline/resnet50model_59', map_location=torch.device('cpu')))

# 测试时加载已经训练的cnn模型
# baselinet1.load_state_dict(torch.load('./save_model/dota_t1wujiandu/baselinet1_98', map_location=torch.device('cpu')))
baselinet1 = nn.DataParallel(baselinet1)


# 优化器及损失函数
optimizer = torch.optim.SGD(baselinet1.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

if cuda_avail:
    baselinet1.cuda()


# 动态更新学习率
def adjust_learning_rate(epoch):
    lr = 0.001
    if epoch > 180:
        lr = lr/1000000
    elif epoch > 150:
        lr = lr/100000
    elif epoch > 120:
        lr = lr/10000
    elif epoch > 90:
        lr = lr/1000
    elif epoch > 60:
        lr = lr/100
    elif epoch > 30:
        lr = lr/10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# 保存模型
def save_models(epoch):
    if not os.path.isdir("./save_model/dota_t1/"):
        os.makedirs("./save_model/dota_t1/")
    savepath = os.path.join('./save_model/dota_t1/', "baselinet1_{}".format(epoch))  # .format和{}有关
    torch.save(baselinet1.state_dict(), savepath)
    print('checkpoint saved')


# 测试每个文件夹的准确度
def test_dir():
    test_acc6 = test_acc9 = test_acc10 = 0.0
    num6 = num9 = num10 = 0
    test_acc_ave = 0.0
    baselinet1.eval()
    road = './save_model/dota_acc1.txt'
    f = open(road, 'a')
    for i, (images, labels) in enumerate(testloader):  # for x in y 循环：x依次表示y中的一个元素，遍历完所有元素循环结束
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs, f1 = baselinet1(images)
        _, prediction = torch.max(outputs.data, 1)
        print(prediction)
        z = int(labels)
        if z == 6:
            num6 += 1
            test_acc6 = test_acc6 + torch.sum(prediction == labels.data)
            cs6 = test_acc6.item() / num6
        if z == 9:
            num9 += 1
            test_acc9 = test_acc9 + torch.sum(prediction == labels.data)
            cs9 = test_acc9.item() / num9
        if z == 10:
            num10 += 1
            test_acc10 = test_acc10 + torch.sum(prediction == labels.data)
            cs10 = test_acc10.item() / num10

        test_acc_ave = test_acc_ave + torch.sum(prediction == labels.data)
    test_acc_ave = test_acc_ave / 18785
    f.write(str(6) + ":" + str(cs6) + '\n' + str(9) + ":" + str(cs9) + '\n' + str(10) + ":" + str(cs10) + '\n'
            + "average_acc" + ":" + str(test_acc_ave) + '\n')
    return test_acc_ave


# 测试总共的准确度
def test():
    test_acc = 0.0
    baselinet1.eval()

    for i, (images, labels) in enumerate(testloader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs, f1 = baselinet1(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc = test_acc + torch.sum(prediction == labels.data)
    test_acc = test_acc/18785

    return test_acc


def train(num_epochs):
    road = './save_model/dota1.txt'
    f = open(road, 'a')

    for epoch in range(num_epochs):
        baselinet1.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            optimizer.zero_grad()  # 优化器梯度置零
            requires_grad(baselinet1, True)
            outputs, f1 = baselinet1(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()  # 进行单次优化

            train_loss += loss.item()*images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data)

        adjust_learning_rate(epoch)

        train_acc = train_acc/71163
        train_loss = train_loss/71163
        test_acc = test()

        save_models(epoch)
        f.write("epoch:"+str(epoch)+","+"train_loss"+str(train_loss)+","+"train_acc"+str(train_acc)+","+"test_acc"+str(test_acc)+'\n')
        print("Epoch:{} ,Train_loss:{},Train_acc:{},test_acc:{}".format(epoch, train_loss, train_acc, test_acc))


if __name__ == "__main__":
    # train(100)
    test_dir()
