from __future__ import print_function, division

import torch
import torch.nn as nn  # nn是神经网络
from torch.autograd import Variable
import numpy as np  # 导入numpy库
import os  # 导入标准库os，对文件和文件夹进行操作
from resnet50 import resnet50, CNN, Bottleneck  # 相当于从一个模块中导入函数
from dataset import SHIP_FG
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES']='0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU


trainset = SHIP_FG(root='./Images', is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=8, drop_last=False)
testset = SHIP_FG(root='./Images', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=8, drop_last=False)

cuda_avail = torch.cuda.is_available()  # 这个指令的作用是看电脑的 GPU 能否被 PyTorch 调用


model = resnet50(pretrained=True)
cnn = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
# CNN为修改之后的resnet50
pretrained_dict = model.state_dict()  # state_dict变量存放训练过程中需要学习的权重和偏执系数，读取参数
cnn_dict = cnn.state_dict()
# 设置网络参数集，pretrained_dict 为预训练网络（已经训练好），cnn_dict 为新定义的网络
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in cnn_dict}
cnn_dict.update(pretrained_dict)  # 使用保留下的参数更新新网络参数集
cnn.load_state_dict(cnn_dict)  # 加载新网络参数集到新网络中

cnn = nn.DataParallel(cnn)  # 进行多GPU训练，并行训练
# 2.加载已经训练的cnn模型，
# cnn.load_state_dict(torch.load('./FGSC_baseline/resnet50model_66', map_location=torch.device('cpu')))


# 加载优化器
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
# torch.optim可以优化神经网络，torch.optim是实现各种优化算法的包

if cuda_avail:
    cnn.cuda()

loss_fn = nn.CrossEntropyLoss()


# 学习率调整函数
def adjust_learning_rate(epoch):
    lr = 0.001
    if epoch>180:
        lr= lr/1000000
    elif epoch>150:
        lr = lr/100000
    elif epoch>120:
        lr = lr/10000
    elif epoch>90:
        lr = lr/1000
    elif epoch>60:
        lr = lr/100
    elif epoch>30:
        lr = lr/10
    # ↓2 动态修改学习率
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# 写出函数的保存模型和评估模型
def save_models(epoch):
    if not os.path.isdir("./save_model/classify/channel_3/"):  # os.path.isdir 判断这个目录是否有“channel_3”文件夹
        os.makedirs("./save_model/classify/channel_3/")  # 通过上面的判断知道这个目录没有该文件夹，所以用 os.makedirs 在这个目录创建“channel_3”文件夹
    savepath = os.path.join('./save_model/classify/channel_3/', "resnet50model_{}".format(epoch))
    torch.save(cnn.state_dict(), savepath)
    print('checkpoint saved')


# 测试每个文件夹的准确度
# enumerate() 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
def test_dir():
    test_acc_ave = 0.0
    cnn.eval()

    num0 = num1 = num2 = num3 = num4 = num5 = num6 = num7 = num8 = num9 = num10 = num11 = num12 = \
        num13 = num14 = num15 = num16 = num17 = num18 = num19 = num20 = num21 = num22 = 0

    test_acc0 = test_acc1 = test_acc2 = test_acc3 = test_acc4 = test_acc5 = test_acc6 = test_acc7 = test_acc8 = \
        test_acc9 = test_acc10 = test_acc11 = test_acc12 = test_acc13 = test_acc14 = test_acc15 = test_acc16 = \
        test_acc17 = test_acc18 = test_acc19 = test_acc20 = test_acc21 = test_acc22 = 0.0

    head_acc_num = 0
    middle_acc_num = 0
    tail_acc_num = 0

    road = './save_model/acc.txt'
    f = open(road, 'a')
    for i, (images, labels) in enumerate(testloader):  # for x in y 循环：x依次表示y中的一个元素，遍历完所有元素循环结束
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = cnn(images)
        _, prediction = torch.max(outputs.data, 1)
        print(prediction)
        z = int(labels)

        if z == 0 or z == 2 or z == 4 or z == 6 or z == 17:
            if z == prediction:
                head_acc_num += 1
        if z == 1 or z == 8 or z == 10 or z == 12 or z == 13 or z == 18:
            if z == prediction:
                middle_acc_num += 1
        if z == 3 or z == 5 or z == 7 or z == 9 or z == 11 or z == 14 or z == 15 or z == 16 or z == 19 or z == 20 or z == 21 or z == 22:
            if z == prediction:
                tail_acc_num += 1

        if z == 0:
            num0 += 1
            test_acc0 = test_acc0 + torch.sum(prediction == labels.data)
            cs0 = test_acc0.item() / num0
        if z == 1:
            num1 += 1
            test_acc1 = test_acc1 + torch.sum(prediction == labels.data)
            cs1 = test_acc1.item() / num1
        if z == 2:
            num2 += 1
            test_acc2 = test_acc2 + torch.sum(prediction == labels.data)
            cs2 = test_acc2.item() / num2
        if z == 3:
            num3 += 1
            test_acc3 = test_acc3 + torch.sum(prediction == labels.data)
            cs3 = test_acc3.item() / num3
        if z == 4:
            num4 += 1
            test_acc4 = test_acc4 + torch.sum(prediction == labels.data)
            cs4 = test_acc4.item() / num4
        if z == 5:
            num5 += 1
            test_acc5 = test_acc5 + torch.sum(prediction == labels.data)
            cs5 = test_acc5.item() / num5
        if z == 6:
            num6 += 1
            test_acc6 = test_acc6 + torch.sum(prediction == labels.data)
            cs6 = test_acc6.item() / num6
        if z == 7:
            num7 += 1
            test_acc7 = test_acc7 + torch.sum(prediction == labels.data)
            cs7 = test_acc7.item() / num7
        if z == 8:
            num8 += 1
            test_acc8 = test_acc8 + torch.sum(prediction == labels.data)
            cs8 = test_acc8.item() / num8
        if z == 9:
            num9 += 1
            test_acc9 = test_acc9 + torch.sum(prediction == labels.data)
            cs9 = test_acc9.item() / num9
        if z == 10:
            num10 += 1
            test_acc10 = test_acc10 + torch.sum(prediction == labels.data)
            cs10 = test_acc10.item() / num10
        if z == 11:
            num11 += 1
            test_acc11 = test_acc11 + torch.sum(prediction == labels.data)
            cs11 = test_acc11.item() / num11
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
        if z == 15:
            num15 += 1
            test_acc15 = test_acc15 + torch.sum(prediction == labels.data)
            cs15 = test_acc15.item() / num15
        if z == 16:
            num16 += 1
            test_acc16 = test_acc16 + torch.sum(prediction == labels.data)
            cs16 = test_acc16.item() / num16
        if z == 17:
            num17 += 1
            test_acc17 = test_acc17 + torch.sum(prediction == labels.data)
            cs17 = test_acc17.item() / num17
        if z == 18:
            num18 += 1
            test_acc18 = test_acc18 + torch.sum(prediction == labels.data)
            cs18 = test_acc18.item() / num18
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

    head_num = num0 + num2 + num4 + num6 + num17
    middle_num = num1 + num8 + num10 + num12 + num13 + num18
    tail_num = num3 + num5 + num7 + num9 + num11 + num14 + num15 + num16 + num19 + num20 + num21 + num22
    test_image_all = head_num + middle_num + tail_num

    head_ave = head_acc_num/head_num
    middle_ave = middle_acc_num/middle_num
    tail_ave = tail_acc_num/tail_num

    test_acc_ave = test_acc_ave / test_image_all

    f.write(str(0) + ":" + str(cs0) + '\n' + str(1) + ":" + str(cs1) + '\n' + str(2) + ":" + str(cs2) + '\n'
            + str(3) + ":" + str(cs3) + '\n' + str(4) + ":" + str(cs4) + '\n' + str(5) + ":" + str(cs5) + '\n'
            + str(6) + ":" + str(cs6) + '\n' + str(7) + ":" + str(cs7) + '\n' + str(8) + ":" + str(cs8) + '\n'
            + str(9) + ":" + str(cs9) + '\n' + str(10) + ":" + str(cs10) + '\n' + str(11) + ":" + str(cs11) + '\n'
            + str(12) + ":" + str(cs12) + '\n' + str(13) + ":" + str(cs13) + '\n' + str(14) + ":" + str(cs14) + '\n'
            + str(15) + ":" + str(cs15) + '\n' + str(16) + ":" + str(cs16) + '\n' + str(17) + ":" + str(cs17) + '\n'
            + str(18) + ":" + str(cs18) + '\n' + str(19) + ":" + str(cs19) + '\n' + str(20) + ":" + str(cs20) + '\n'
            + str(21) + ":" + str(cs21) + '\n' + str(22) + ":" + str(cs22) + '\n' + "average_acc" + ":" + str(test_acc_ave) + '\n'
            + "head_ave:" + str(head_ave) + '\n' + "middle_ave:" + str(middle_ave) + '\n' + "tail_ave:" + str(tail_ave) + '\n')

    return test_acc_ave


# 测试总共的准确度
def test():
    test_acc = 0.0
    cnn.eval()

    for i, (images, labels) in enumerate(testloader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = cnn(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc = test_acc + torch.sum(prediction == labels.data)
    test_acc = test_acc/825

    return test_acc


def train(num_epochs):
    road = './save_model/train_acc.txt'
    f = open(road, 'a')

    for epoch in range(num_epochs):
        cnn.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            optimizer.zero_grad()  # 优化器梯度置零
            outputs = cnn(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()  # 进行单次优化

            train_loss += loss.item()*images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data)

        adjust_learning_rate(epoch)

        train_acc = train_acc/3256
        train_loss = train_loss/3256
        test_acc = test()
        # 每个epoch训练好后都会被test
        save_models(epoch)
        f.write("epoch:"+str(epoch)+","+"train_loss"+str(train_loss)+","+"train_acc"+str(train_acc)+","+"test_acc"+str(test_acc)+'\n')
        print("Epoch:{} ,Train_loss:{},Train_acc:{},test_acc:{}".format(epoch, train_loss, train_acc, test_acc))


if __name__ == "__main__":
    train(100)
    test_dir()

