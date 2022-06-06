from __future__ import print_function, division

import torch
import torch.nn as nn  # nn是神经网络
from torch.autograd import Variable
import numpy as np  # 导入numpy库
import os  # 导入标准库os，对文件和文件夹进行操作
from resnet50 import resnet50,CNN,Bottleneck  # 相当于从一个模块中导入函数
from dataset import FGSC_THREE
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES']='0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


trainset = FGSC_THREE(root='./Images', is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=8, drop_last=False)

testset = FGSC_THREE(root='./Images', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=8, drop_last=False)

cuda_avail = torch.cuda.is_available()  # 看电脑的 GPU 能否被 PyTorch 调用


t1 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t1.load_state_dict(torch.load('./save_model/wujiandut/t1wujiandu/baselinet1_30', map_location=torch.device('cpu')))
t2 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t2.load_state_dict(torch.load('./save_model/classify2/baselinet2_36', map_location=torch.device('cpu')))
baselinet3 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
# 可选择灌不灌baseline最优初始值
# baselinet3.load_state_dict(torch.load('./FGSC_baseline/resnet50model_66', map_location=torch.device('cpu')))
# baselinet3 = nn.DataParallel(baselinet3)
# 加载已经训练的cnn模型
# baselinet3.load_state_dict(torch.load('./save_model/classify3/baselinet3_61', map_location=torch.device('cpu')))


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
    if not os.path.isdir("./save_model/classify3/"):
        os.makedirs("./save_model/classify3/")
    savepath = os.path.join('./save_model/classify3/', "baselinet3_{}".format(epoch))  # .format和{}有关
    # os.path.join()函数用于拼接文件路径，可以传入多个路径，会从第一个以”/”开头的参数开始拼接，之前的参数全部丢弃
    torch.save(baselinet3.state_dict(), savepath)
    print('checkpoint saved')


# 测试每个文件夹的准确度
def test_dir():
    num3 = num5 = num7 = num9 = num11 = num14 = num15 = num16 = num19 = num20 = num21 = num22 = 0
    test_acc3 = test_acc5 = test_acc7 = test_acc9 = test_acc11 = test_acc14 = \
        test_acc15 = test_acc16 = test_acc19 = test_acc20 = test_acc21 = test_acc22 = 0.0

    baselinet3.eval()
    test_acc_ave = 0.0
    road = './save_model/acc3.txt'
    f = open(road, 'a')
    for i, (images, labels) in enumerate(testloader):  # for x in y 循环：x依次表示y中的一个元素，遍历完所有元素循环结束
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
        if z == 7:
            num7 += 1
            test_acc7 = test_acc7 + torch.sum(prediction == labels.data)
            cs7 = test_acc7.item() / num7
        if z == 9:
            num9 += 1
            test_acc9 = test_acc9 + torch.sum(prediction == labels.data)
            cs9 = test_acc9.item() / num9
        if z == 11:
            num11 += 1
            test_acc11 = test_acc11 + torch.sum(prediction == labels.data)
            cs11 = test_acc11.item() / num11
        if z == 14:
            num14 += 1
            test_acc14 = test_acc14 + torch.sum(prediction == labels.data)
            cs14 = test_acc14.item() / num14
        if z == 15:
            num15 += 1
            test_acc15 = test_acc15 + torch.sum(prediction == labels.data)
            cs15 = test_acc15.item() / num15
        if z == 16:
            num16 += 1
            test_acc16 = test_acc16 + torch.sum(prediction == labels.data)
            cs16 = test_acc16.item() / num16
        if z == 19:
            num19 += 1
            test_acc19 = test_acc19 + torch.sum(prediction == labels.data)
            cs19 = test_acc19.item() / num19
        if z == 20:
            num20 += 1
            test_acc20 = test_acc20 + torch.sum(prediction == labels.data)
            cs20 = test_acc20.item() / num20
        if z == 21:
            num21 += 1
            test_acc21 = test_acc21 + torch.sum(prediction == labels.data)
            cs21 = test_acc21.item() / num21
        if z == 22:
            num22 += 1
            test_acc22 = test_acc22 + torch.sum(prediction == labels.data)
            cs22 = test_acc22.item() / num22

        test_acc_ave = test_acc_ave + torch.sum(prediction == labels.data)
    test_acc_ave = test_acc_ave / 213

    f.write(str(3) + ":" + str(cs3) + '\n' + str(5) + ":" + str(cs5) + '\n' + str(7) + ":" + str(cs7) + '\n'
            + str(9) + ":" + str(cs9) + '\n' + str(11) + ":" + str(cs11) + '\n' + str(14) + ":" + str(cs14) + '\n'
            + str(15) + ":" + str(cs15) + '\n' + str(16) + ":" + str(cs16) + '\n' + str(19) + ":" + str(cs19) + '\n'
            + str(20) + ":" + str(cs20) + '\n' + str(21) + ":" + str(cs21) + '\n' + str(22) + ":" + str(cs22) + '\n'
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
    test_acc = test_acc/213

    return test_acc


def train(num_epochs):
    road = './save_model/jieguo3.txt'
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

            train_loss += loss.item()*images.size(0)  # .size(0)指batchsize的值
            # item()把字典中每对key和value组成一个元组，并把这些元组放在列表中返回
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data)

        adjust_learning_rate(epoch)

        train_acc = train_acc/820
        train_loss = train_loss/820

        test_acc = test()
        # 每个epoch训练好后都会被test
        save_models(epoch)
        f.write("epoch:"+str(epoch)+","+"train_loss"+str(train_loss)+","+"train_acc"+str(train_acc)+","+"test_acc"+str(test_acc)+'\n')
        print("Epoch:{} ,Train_loss:{},Train_acc:{},test_acc:{}".format(epoch, train_loss, train_acc, test_acc))


if __name__ == "__main__":
    train(100)
    test_dir()
