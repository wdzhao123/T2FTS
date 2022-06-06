from __future__ import print_function, division

import torch
import torch.nn as nn  # nn是神经网络
from torch.autograd import Variable
import numpy as np  # 导入numpy库
import os  # 导入标准库os，对文件和文件夹进行操作
from torch.optim import Adam  # 导入模块
from resnet50 import resnet50, CNN, Bottleneck  # 相当于从一个模块中导入函数
from dataset import FGSC_ALL
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


testset = FGSC_ALL(root='./Images', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=8, drop_last=False)

cuda_avail = torch.cuda.is_available()

baselinestuweight = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
# 加载已经训练的cnn模型
baselinestuweight.load_state_dict(torch.load('./save_model/classifystuweight_alpha75/baselinestuweight_alpha_29', map_location="cuda:0"))
baselinestuweight = nn.DataParallel(baselinestuweight)


optimizer = torch.optim.SGD(baselinestuweight.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)


if cuda_avail:
    baselinestuweight.cuda()


# 测试每个文件夹的准确度
# enumerate() 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
def test_dir():

    num0 = num1 = num2 = num3 = num4 = num5 = num6 = num7 = num8 = num9 = num10 = num11 = num12 = \
        num13 = num14 = num15 = num16 = num17 = num18 = num19 = num20 = num21 = num22 = 0

    test_acc0 = test_acc1 = test_acc2 = test_acc3 = test_acc4 = test_acc5 = test_acc6 = test_acc7 = test_acc8 = \
        test_acc9 = test_acc10 = test_acc11 = test_acc12 = test_acc13 = test_acc14 = test_acc15 = test_acc16 = \
        test_acc17 = test_acc18 = test_acc19 = test_acc20 = test_acc21 = test_acc22 = 0.0

    head_acc_num = 0
    middle_acc_num = 0
    tail_acc_num = 0

    test_acc_ave = 0.0
    baselinestuweight.eval()
    road = './save_model/accsweight_alpha.txt'
    f = open(road, 'a')
    for i, (images, labels) in enumerate(testloader):
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        z = int(labels)
        outputs, feature = baselinestuweight(images)
        _, prediction = torch.max(outputs.data, 1)
        print(prediction)

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

    head_ave = head_acc_num / head_num
    middle_ave = middle_acc_num / middle_num
    tail_ave = tail_acc_num / tail_num

    test_acc_ave = test_acc_ave / 825

    f.write(str(0) + ":" + str(cs0) + '\n' + str(1) + ":" + str(cs1) + '\n' + str(2) + ":" + str(cs2) + '\n'
            + str(3) + ":" + str(cs3) + '\n' + str(4) + ":" + str(cs4) + '\n' + str(5) + ":" + str(cs5) + '\n'
            + str(6) + ":" + str(cs6) + '\n' + str(7) + ":" + str(cs7) + '\n' + str(8) + ":" + str(cs8) + '\n'
            + str(9) + ":" + str(cs9) + '\n' + str(10) + ":" + str(cs10) + '\n' + str(11) + ":" + str(cs11) + '\n'
            + str(12) + ":" + str(cs12) + '\n' + str(13) + ":" + str(cs13) + '\n' + str(14) + ":" + str(cs14) + '\n'
            + str(15) + ":" + str(cs15) + '\n' + str(16) + ":" + str(cs16) + '\n' + str(17) + ":" + str(cs17) + '\n'
            + str(18) + ":" + str(cs18) + '\n' + str(19) + ":" + str(cs19) + '\n' + str(20) + ":" + str(cs20) + '\n'
            + str(21) + ":" + str(cs21) + '\n' + str(22) + ":" + str(cs22) + '\n' + "average_acc" + ":" + str(
        test_acc_ave) + '\n'
            + "head_ave:" + str(head_ave) + '\n' + "middle_ave:" + str(middle_ave) + '\n' + "tail_ave:" + str(tail_ave) + '\n'
            + "test_image_all:" + str(test_image_all) + '\n')

    return test_acc_ave


if __name__ == "__main__":
    test_dir()
