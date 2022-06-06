from __future__ import print_function, division

import torch
import torch.nn as nn  # nn是神经网络
from torch.autograd import Variable
import numpy as np  # 导入numpy库
import os  # 导入标准库os，对文件和文件夹进行操作
from resnet50 import resnet50,CNN,Bottleneck  # 相当于从一个模块中导入函数
from dataset import DOTA_THREE
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES']='0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU


def requires_grad(model, flag = True):
    for p in model.parameters():
        p.requires_grad = flag


trainset = DOTA_THREE(root='./Images', is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=8, drop_last=False)

testset = DOTA_THREE(root='./Images', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=8, drop_last=False)

cuda_avail = torch.cuda.is_available()  # 看电脑的 GPU 能否被 PyTorch 调用

t1 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t1.load_state_dict(torch.load('./save_model/dota_t1/baselinet1_98', map_location=torch.device('cpu')))
t2 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t2.load_state_dict(torch.load('./save_model/dota_t2/baselinet2_14', map_location=torch.device('cpu')))
baselinet3 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
baselinet3.load_state_dict(torch.load('./DOTA_baseline/resnet50model_59', map_location=torch.device('cpu')))

# 加载已经训练的cnn模型
baselinet3.load_state_dict(torch.load('./save_model/dota_t3/baselinet3_17', map_location=torch.device('cpu')))
baselinet3 = nn.DataParallel(baselinet3)

optimizer = torch.optim.SGD(baselinet3.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
loss_fn1 = torch.nn.MSELoss(reduce=True, size_average=True)
loss_fn2 = nn.CrossEntropyLoss()


if cuda_avail:
    baselinet3.cuda()


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


def save_models(epoch):
    if not os.path.isdir("./save_model/dota_t3/"):
        os.makedirs("./save_model/dota_t3/")
    savepath = os.path.join('./save_model/dota_t3/', "baselinet3_{}".format(epoch))
    torch.save(baselinet3.state_dict(), savepath)
    print('checkpoint saved')


# 测试每个文件夹的准确度
def test_dir():
    num0 = num5 = num1 = num3 = num11 = num8 = 0
    test_acc3 = test_acc5 = test_acc0 = test_acc1 = test_acc11 = test_acc8 = 0.0

    baselinet3.eval()
    test_acc_ave = 0.0
    road = './save_model/dota_acc3.txt'
    f = open(road, 'a')
    for i, (images, labels) in enumerate(testloader):
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs, f3 = baselinet3(images)
        _, prediction = torch.max(outputs.data, 1)
        print(prediction)
        z = int(labels)

        if z == 3:
            num3 += 1
            test_acc3 = test_acc3 + torch.sum(prediction == labels.data)
            cs3 = test_acc3.item() / num3
        if z == 5:
            num5 += 1
            test_acc5 = test_acc5 + torch.sum(prediction == labels.data)
            cs5 = test_acc5.item() / num5
        if z == 0:
            num0 += 1
            test_acc0 = test_acc0 + torch.sum(prediction == labels.data)
            cs0 = test_acc0.item() / num0
        if z == 1:
            num1 += 1
            test_acc1 = test_acc1 + torch.sum(prediction == labels.data)
            cs1 = test_acc1.item() / num1
        if z == 11:
            num11 += 1
            test_acc11 = test_acc11 + torch.sum(prediction == labels.data)
            cs11 = test_acc11.item() / num11
        if z == 8:
            num8 += 1
            test_acc8 = test_acc8 + torch.sum(prediction == labels.data)
            cs8 = test_acc8.item() / num8

        test_acc_ave = test_acc_ave + torch.sum(prediction == labels.data)
    test_acc_ave = test_acc_ave / 895

    f.write(str(0) + ":" + str(cs0) + '\n' + str(1) + ":" + str(cs1) + '\n' + str(3) + ":" + str(cs3) + '\n'
            + str(5) + ":" + str(cs5) + '\n' + str(8) + ":" + str(cs8) + '\n' + str(11) + ":" + str(cs11) + '\n'
            + "average_acc" + ":" + str(test_acc_ave) + '\n')
    return test_acc_ave


# 测试总共的准确度
def test():
    test_acc = 0.0
    baselinet3.eval()

    for i, (images, labels) in enumerate(testloader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs, f3 = baselinet3(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc = test_acc + torch.sum(prediction == labels.data)
    test_acc = test_acc/895

    return test_acc


def train(num_epochs):
    road = './save_model/dota3.txt'
    f = open(road, 'a')

    for epoch in range(num_epochs):
        baselinet3.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            optimizer.zero_grad()  # 优化器梯度置零
            x1, f1 = t1(images)
            x2, f2 = t2(images)
            requires_grad(t1, False)
            requires_grad(t2, False)
            requires_grad(baselinet3, True)
            outputs, f3 = baselinet3(images)

            loss1 = loss_fn1(f1, f3)
            loss2 = loss_fn1(f3, f2)
            loss3 = loss_fn2(outputs, labels)
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()  # 进行单次优化

            train_loss += loss.item()*images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data)

        adjust_learning_rate(epoch)

        train_acc = train_acc/2610
        train_loss = train_loss/2610

        test_acc = test()
        # 每个epoch训练好后都会被test
        save_models(epoch)
        f.write("epoch:"+str(epoch)+","+"train_loss"+str(train_loss)+","+"train_acc"+str(train_acc)+","+"test_acc"+str(test_acc)+'\n')
        print("Epoch:{} ,Train_loss:{},Train_acc:{},test_acc:{}".format(epoch, train_loss, train_acc, test_acc))


if __name__ == "__main__":
    train(100)
    test_dir()
