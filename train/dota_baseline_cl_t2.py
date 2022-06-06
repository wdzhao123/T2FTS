from __future__ import print_function, division

import torch
import torch.nn as nn  # nn是神经网络
from torch.autograd import Variable
import numpy as np  # 导入numpy库
import os  # 导入标准库os，对文件和文件夹进行操作
from resnet50 import resnet50,CNN,Bottleneck  # 相当于从一个模块中导入函数
from dataset import DOTA_TWO
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES']='0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU

def requires_grad(model, flag = True):
    for p in model.parameters():
        p.requires_grad = flag

# 加载训练集和测试集
trainset = DOTA_TWO(root='./Images', is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=8, drop_last=False)
testset = DOTA_TWO(root='./Images', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=8, drop_last=False)


# 看电脑的 GPU 能否被 PyTorch 调用
cuda_avail = torch.cuda.is_available()


t1 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t1.load_state_dict(torch.load('./save_model/dota_t1/baselinet1_98', map_location=torch.device('cpu')))
baselinet2 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
baselinet2.load_state_dict(torch.load('./DOTA_baseline/resnet50model_59', map_location=torch.device('cpu')))

# 测试时加载已经训练的cnn模型
baselinet2.load_state_dict(torch.load('./save_model/dota_t2/baselinet2_14', map_location=torch.device('cpu')))
baselinet2 = nn.DataParallel(baselinet2)


# 优化器及损失函数
optimizer = torch.optim.SGD(baselinet2.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
loss_fn1 = torch.nn.MSELoss(reduce=True, size_average=True)
loss_fn2 = nn.CrossEntropyLoss()

if cuda_avail:
    baselinet2.cuda()

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
    if not os.path.isdir("./save_model/dota_t2/"):
        os.makedirs("./save_model/dota_t2/")
    savepath = os.path.join('./save_model/dota_t2/', "baselinet2_{}".format(epoch))  # .format和{}有关
    torch.save(baselinet2.state_dict(), savepath)
    print('checkpoint saved')


# 测试每个文件夹的准确度
def test_dir():
    # test_acc = 0.0
    baselinet2.eval()
    num2 = num4 = num7 = num12 = num13 = num14 = 0
    test_acc2 = test_acc4 = test_acc7 = test_acc12 = test_acc13 = test_acc14 = 0.0
    test_acc_ave = 0.0
    road = './save_model/dota_acc2.txt'
    f = open(road, 'a')
    for i, (images, labels) in enumerate(testloader):  # for x in y 循环：x依次表示y中的一个元素，遍历完所有元素循环结束
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs, f2 = baselinet2(images)
        _, prediction = torch.max(outputs.data, 1)
        print(prediction)
        z = int(labels)

        if z == 2:
            num2 += 1
            test_acc2 = test_acc2 + torch.sum(prediction == labels.data)
            cs2 = test_acc2.item() / num2
        if z == 4:
            num4 += 1
            test_acc4 = test_acc4 + torch.sum(prediction == labels.data)
            cs4 = test_acc4.item() / num4
        if z == 7:
            num7 += 1
            test_acc7 = test_acc7 + torch.sum(prediction == labels.data)
            cs7 = test_acc7.item() / num7
        if z == 12:
            num12 += 1
            test_acc12 = test_acc12 + torch.sum(prediction == labels.data)
            cs12 = test_acc12.item() / num12
        if z == 13:
            num13 += 1
            test_acc13 = test_acc13 + torch.sum(prediction == labels.data)
            cs13 = test_acc13.item() / num13
        if z == 14:
            num14 += 1
            test_acc14 = test_acc14 + torch.sum(prediction == labels.data)
            cs14 = test_acc14.item() / num14

        test_acc_ave = test_acc_ave + torch.sum(prediction == labels.data)
    test_acc_ave = test_acc_ave / 9173

    f.write(str(2) + ":" + str(cs2) + '\n' + str(4) + ":" + str(cs4) + '\n' + str(7) + ":" + str(cs7) + '\n'
            + str(12) + ":" + str(cs12) + '\n' + str(13) + ":" + str(cs13) + '\n' + str(14) + ":" + str(cs14) + '\n'
            + "average_acc" + ":" + str(test_acc_ave) + '\n')
    return test_acc_ave


# 测试总共的准确度
def test():
    test_acc = 0.0
    baselinet2.eval()

    for i, (images, labels) in enumerate(testloader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs, f2 = baselinet2(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc = test_acc + torch.sum(prediction == labels.data)
    test_acc = test_acc/9173

    return test_acc


def train(num_epochs):
    road = './save_model/dota2.txt'
    f = open(road, 'a')

    for epoch in range(num_epochs):
        baselinet2.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            optimizer.zero_grad()  # 优化器梯度置零
            x1, f1 = t1(images)
            requires_grad(t1, False)
            requires_grad(baselinet2, True)
            outputs, f2 = baselinet2(images)

            loss1 = loss_fn1(f1, f2)
            loss2 = loss_fn2(outputs, labels)
            loss = loss1+loss2
            loss.backward()
            optimizer.step()  # 进行单次优化

            train_loss += loss.item()*images.size(0)  # .size(0)指batchsize的值
            # item()把字典中每对key和value组成一个元组，并把这些元组放在列表中返回
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data)

        adjust_learning_rate(epoch)

        train_acc = train_acc/25133
        train_loss = train_loss/25133

        test_acc = test()
        # 每个epoch训练好后都会被test
        # if epoch%10==0:
        save_models(epoch)
        f.write("epoch:"+str(epoch)+","+"train_loss"+str(train_loss)+","+"train_acc"+str(train_acc)+","+"test_acc"+str(test_acc)+'\n')
        print("Epoch:{} ,Train_loss:{},Train_acc:{},test_acc:{}".format(epoch, train_loss, train_acc, test_acc))


if __name__ == "__main__":
    # train(100)
    test_dir()
