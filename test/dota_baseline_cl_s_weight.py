from __future__ import print_function, division

import torch
import torch.nn as nn  # nn是神经网络
from torch.autograd import Variable
import numpy as np  # 导入numpy库
import os  # 导入标准库os，对文件和文件夹进行操作
from torch.optim import Adam  # 导入模块
from resnet50 import resnet50, CNN, Bottleneck  # 相当于从一个模块中导入函数
from dataset import DOTA_ALL
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
import warnings
# from ClassAwareSampler import ClassAwareSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import datetime

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


testset = DOTA_ALL(root='./Images', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=8, drop_last=False)

cuda_avail = torch.cuda.is_available()  # 这个指令的作用是看电脑的 GPU 能否被 PyTorch 调用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
baselinestuweight = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
model = resnet50(pretrained=True)
pretrained_dict = model.state_dict()
baselinestuweight_dict = baselinestuweight.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in baselinestuweight_dict}
baselinestuweight_dict.update(pretrained_dict)  # 使用保留下的参数更新新网络参数集
baselinestuweight.load_state_dict(baselinestuweight_dict)  # 加载新网络参数集到新网络中

baselinestuweight = nn.DataParallel(baselinestuweight)
# 加载已经训练的cnn模型
baselinestuweight.load_state_dict(torch.load('./save_model/dota_alpha75/baselinestuweight_alpha_53', map_location="cuda:0"))


optimizer = torch.optim.SGD(baselinestuweight.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

if cuda_avail:
    baselinestuweight.cuda()


# 测试每个文件夹的准确度
# enumerate() 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
def test_dir():

    num0 = num1 = num2 = num3 = num4 = num5 = num6 = num7 = num8 = num9 = num10 = num11 = num12 = num13 = num14 = 0

    test_acc0 = test_acc1 = test_acc2 = test_acc3 = test_acc4 = test_acc5 = test_acc6 = test_acc7 = test_acc8 = \
        test_acc9 = test_acc10 = test_acc11 = test_acc12 = test_acc13 = test_acc14 = 0.0

    test_acc_ave = 0.0
    baselinestuweight.eval()
    road = './save_model/dota_accsweight_alpha.txt'
    f = open(road, 'a')

    for i, (images, labels) in enumerate(testloader):  # for x in y 循环：x依次表示y中的一个元素，遍历完所有元素循环结束
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        z = int(labels)
        outputs, feature = baselinestuweight(images)
        _, prediction = torch.max(outputs.data, 1)
        print(prediction)
        # test_acc = test_acc + torch.sum(prediction==labels.data)
        # a = torch.from_numpy(np.array([m]))
        # torch.from_numpy()把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
        # tensor张量可以轻易地进行卷积，激活，上下采样，微分求导等操作，而numpy数组不行，普通的numpy要先转化为tensor格式才行

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

        test_acc_ave = test_acc_ave + torch.sum(prediction == labels.data)

    test_acc_ave = test_acc_ave / 28853

    f.write(str(0) + ":" + str(cs0) + '\n' + str(1) + ":" + str(cs1) + '\n' + str(2) + ":" + str(cs2) + '\n'
            + str(3) + ":" + str(cs3) + '\n' + str(4) + ":" + str(cs4) + '\n' + str(5) + ":" + str(cs5) + '\n'
            + str(6) + ":" + str(cs6) + '\n' + str(7) + ":" + str(cs7) + '\n' + str(8) + ":" + str(cs8) + '\n'
            + str(9) + ":" + str(cs9) + '\n' + str(10) + ":" + str(cs10) + '\n' + str(11) + ":" + str(cs11) + '\n'
            + str(12) + ":" + str(cs12) + '\n' + str(13) + ":" + str(cs13) + '\n' + str(14) + ":" + str(cs14) + '\n'
            + "average_acc" + ":" + str(test_acc_ave) + '\n')

    return test_acc_ave


if __name__ == "__main__":
    test_dir()
