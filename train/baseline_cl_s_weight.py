from __future__ import print_function, division

import torch
import torch.nn as nn  # nn是神经网络
from torch.autograd import Variable
import numpy as np  # 导入numpy库
import os  # 导入标准库os，对文件和文件夹进行操作
from resnet50 import resnet50, CNN, Bottleneck  # 相当于从一个模块中导入函数
from dataset import FGSC_ALL
import warnings
from ClassAwareSampler import ClassAwareSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import WeightedRandomSampler


warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


trainset = FGSC_ALL(root='./Images', is_train=True, data_len=None)
trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False,
                                          sampler=ClassAwareSampler(trainset), num_workers=8, drop_last=False)

testset = FGSC_ALL(root='./Images', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=8, drop_last=False)

cuda_avail = torch.cuda.is_available()  # 这个指令的作用是看电脑的 GPU 能否被 PyTorch 调用

baselinestuweight = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
model = resnet50(pretrained=True)
pretrained_dict = model.state_dict()
baselinestuweight_dict = baselinestuweight.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in baselinestuweight_dict}
baselinestuweight_dict.update(pretrained_dict)  # 使用保留下的参数更新新网络参数集
baselinestuweight.load_state_dict(baselinestuweight_dict)  # 加载新网络参数集到新网络中

t1 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t1.load_state_dict(torch.load('./save_model/classify1/baselinet1_30', map_location=torch.device('cpu')))
t2 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t2.load_state_dict(torch.load('./save_model/classify2/baselinet2_36', map_location=torch.device('cpu')))
t3 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t3.load_state_dict(torch.load('./save_model/classify3/baselinet3_61', map_location=torch.device('cpu')))
baselinestuweight = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
# baselinestuweight.load_state_dict(
#     torch.load('./FGSC_baseline/resnet50model_66', map_location=torch.device('cpu')))   # 初始化参数


optimizer = torch.optim.SGD(baselinestuweight.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

if cuda_avail:
    baselinestuweight.cuda()

loss_fn1 = torch.nn.MSELoss(reduce=True, size_average=True)  # 差值的平方再求平均（分母为所有元素个数）
loss_fn2 = nn.CrossEntropyLoss()


def adjust_learning_rate(epoch):
    lr = 0.001
    if epoch > 200:
        lr = lr / 10000
    elif epoch > 150:
        lr = lr / 1000
    elif epoch > 100:
        lr = lr / 100
    elif epoch > 50:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    if not os.path.isdir("./save_model/classifystuweight_alpha/"):
        os.makedirs("./save_model/classifystuweight_alpha/")
    savepath = os.path.join('./save_model/classifystuweight_alpha/', "baselinestuweight_alpha_{}".format(epoch))  # .format和{}有关
    # os.path.join()函数用于拼接文件路径，可以传入多个路径，会从第一个以”/”开头的参数开始拼接，之前的参数全部丢弃
    torch.save(baselinestuweight.state_dict(), savepath)
    print('checkpoint saved')


# 测试总共的准确度
def test():
    test_acc = 0.0
    baselinestuweight.eval()

    for i, (images, labels) in enumerate(testloader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs, feature = baselinestuweight(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc = test_acc + torch.sum(prediction == labels.data)
    test_acc = test_acc / 825

    return test_acc


def train(num_epochs):
    road = './save_model/trainsweight_alpha.txt'
    f = open(road, 'a')

    for epoch in range(num_epochs):
        baselinestuweight.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainloader1):
            if i == 0:
                if cuda_avail:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                optimizer.zero_grad()  # 优化器梯度置零

                requires_grad(t1, False)
                requires_grad(t2, False)
                requires_grad(t3, False)
                requires_grad(baselinestuweight, True)

                outputs1, feature1 = baselinestuweight(images)
                feature2 = feature1
                feature2 = feature2.cpu().detach().numpy()
                feature2 = feature2.tolist()
                loss_81 = [0, 0, 0, 0, 0, 0, 0, 0]
                loss_231 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                loss2 = loss_fn2(outputs1, labels)
                _, prediction = torch.max(outputs1.data, 1)
                train_acc += torch.sum(prediction == labels.data)
                labels = labels.cpu().detach().numpy()
                labels = labels.tolist()

                for k in range(len(labels)):  # batch_size=8
                    if (labels[k] == 0) or (labels[k] == 2) or (labels[k] == 4) or (labels[k] == 6) or (
                            labels[k] == 17):
                        outputs21, feature21 = t1(images)
                        feature21 = feature21.cpu().detach().numpy()
                        feature21 = feature21.tolist()
                        feature2[k] = feature21[k]
                        b = feature2[k]
                        a = feature1[k]
                        b = torch.Tensor(b).to(device)
                        loss_81[k] = loss_fn1(a, b)
                    elif (labels[k] == 1) or (labels[k] == 8) or (labels[k] == 10) or (labels[k] == 12) \
                            or (labels[k] == 13) or (labels[k] == 18):
                        outputs22, feature22 = t2(images)
                        feature22 = feature22.cpu().detach().numpy()
                        feature22 = feature22.tolist()
                        feature2[k] = feature22[k]
                        b = feature2[k]
                        a = feature1[k]
                        b = torch.Tensor(b).to(device)
                        loss_81[k] = loss_fn1(a, b)
                    elif (labels[k] == 3) or (labels[k] == 5) or (labels[k] == 7) or (labels[k] == 9) or (
                            labels[k] == 11) \
                            or (labels[k] == 14) or (labels[k] == 15) or (labels[k] == 16) or (labels[k] == 19) \
                            or (labels[k] == 20) or (labels[k] == 21) or (labels[k] == 22):
                        outputs23, feature23 = t3(images)
                        feature23 = feature23.cpu().detach().numpy()
                        feature23 = feature23.tolist()
                        feature2[k] = feature23[k]
                        b = feature2[k]
                        a = feature1[k]
                        b = torch.Tensor(b).to(device)
                        loss_81[k] = loss_fn1(a, b)

                feature2 = torch.Tensor(feature2).to(device)

                loss1 = loss_fn1(feature1, feature2)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()  # 进行单次优化

                labels_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # len(labels_num)=23
                for t in range(len(labels)):
                    for tt in range(len(labels_num)):
                        if labels[t] == tt:
                            labels_num[tt] += 1
                            loss_231[tt] += loss_81[t]

                for ttt in range(len(labels_num)):
                    if labels_num[ttt] != 0:
                        loss_231[ttt] = loss_231[ttt] / labels_num[ttt]

                loss_231 = torch.Tensor(loss_231).to(device)
                alpha = 75
                and1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                and1 = torch.Tensor(and1).to(device)
                loss_231_and1 = torch.add(loss_231, and1)
                weight = alpha * torch.log(loss_231_and1)

                weight_num = 0
                for k in range(len(weight)):  # batch_size=8
                    if weight[k] != 0.0:
                        weight_num += 1
                weight_ave = torch.sum(weight) / weight_num
                for k in range(len(weight)):  # batch_size=8
                    if weight[k] == 0.0:
                        weight[k] = weight_ave

                sum_ = 23 * weight.mean()
                weight = weight / sum_
                all_weight = torch.zeros(3256)
                all_weight[0:387] = weight[0] / 387
                all_weight[387:519] = weight[1] / 132
                all_weight[519:953] = weight[2] / 434
                all_weight[953:1039] = weight[3] / 86
                all_weight[1039:1275] = weight[4] / 236
                all_weight[1275:1347] = weight[5] / 72
                all_weight[1347:1581] = weight[6] / 234
                all_weight[1581:1651] = weight[7] / 70
                all_weight[1651:1774] = weight[8] / 123
                all_weight[1774:1845] = weight[9] / 71
                all_weight[1845:2035] = weight[10] / 190
                all_weight[2035:2052] = weight[11] / 17
                all_weight[2052:2166] = weight[12] / 114
                all_weight[2166:2346] = weight[13] / 180
                all_weight[2346:2427] = weight[14] / 81
                all_weight[2427:2485] = weight[15] / 58
                all_weight[2485:2581] = weight[16] / 96
                all_weight[2581:2855] = weight[17] / 274
                all_weight[2855:2987] = weight[18] / 132
                all_weight[2987:3069] = weight[19] / 82
                all_weight[3069:3139] = weight[20] / 70
                all_weight[3139:3213] = weight[21] / 74
                all_weight[3213:3256] = weight[22] / 43

                train_loss += loss.item() * images.size(0)

            else:

                sampler = WeightedRandomSampler(all_weight, num_samples=8, replacement=True)
                trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False,
                                                           sampler=sampler, num_workers=8, drop_last=False)

                for m, (images, labels) in enumerate(trainloader2):
                    if cuda_avail:
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                    optimizer.zero_grad()  # 优化器梯度置零

                    requires_grad(t1, False)
                    requires_grad(t2, False)
                    requires_grad(t3, False)
                    requires_grad(baselinestuweight, True)

                    outputs1, feature1 = baselinestuweight(images)
                    feature2 = feature1
                    feature2 = feature2.cpu().detach().numpy()
                    feature2 = feature2.tolist()
                    loss_81 = [0, 0, 0, 0, 0, 0, 0, 0]
                    loss_231 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                    loss2 = loss_fn2(outputs1, labels)
                    _, prediction = torch.max(outputs1.data, 1)
                    train_acc += torch.sum(prediction == labels.data)
                    labels = labels.cpu().detach().numpy()
                    labels = labels.tolist()

                    for k in range(len(labels)):
                        if (labels[k] == 0) or (labels[k] == 2) or (labels[k] == 4) or (labels[k] == 6) or (
                                labels[k] == 17):
                            outputs21, feature21 = t1(images)
                            feature21 = feature21.cpu().detach().numpy()
                            feature21 = feature21.tolist()
                            feature2[k] = feature21[k]
                            b = feature2[k]
                            a = feature1[k]
                            b = torch.Tensor(b).to(device)
                            loss_81[k] = loss_fn1(a, b)
                        elif (labels[k] == 1) or (labels[k] == 8) or (labels[k] == 10) or (labels[k] == 12) \
                                or (labels[k] == 13) or (labels[k] == 18):
                            outputs22, feature22 = t2(images)
                            feature22 = feature22.cpu().detach().numpy()
                            feature22 = feature22.tolist()
                            feature2[k] = feature22[k]
                            b = feature2[k]
                            a = feature1[k]
                            b = torch.Tensor(b).to(device)
                            loss_81[k] = loss_fn1(a, b)
                        elif (labels[k] == 3) or (labels[k] == 5) or (labels[k] == 7) or (labels[k] == 9) or (
                                labels[k] == 11) \
                                or (labels[k] == 14) or (labels[k] == 15) or (labels[k] == 16) or (labels[k] == 19) \
                                or (labels[k] == 20) or (labels[k] == 21) or (labels[k] == 22):
                            outputs23, feature23 = t3(images)
                            feature23 = feature23.cpu().detach().numpy()
                            feature23 = feature23.tolist()
                            feature2[k] = feature23[k]
                            b = feature2[k]
                            a = feature1[k]
                            b = torch.Tensor(b).to(device)
                            loss_81[k] = loss_fn1(a, b)

                    feature2 = torch.Tensor(feature2).to(device)

                    loss1 = loss_fn1(feature1, feature2)
                    loss = loss1 + loss2
                    loss.backward()
                    optimizer.step()  # 进行单次优化

                    labels_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    for t in range(len(labels)):
                        for tt in range(len(labels_num)):
                            if labels[t] == tt:
                                labels_num[tt] += 1
                                loss_231[tt] += loss_81[t]

                    for ttt in range(len(labels_num)):
                        if labels_num[ttt] != 0:
                            loss_231[ttt] = loss_231[ttt] / labels_num[ttt]

                    loss_231 = torch.Tensor(loss_231).to(device)

                    alpha = 75
                    and1 = torch.ones(23)
                    and1 = torch.Tensor(and1).to(device)
                    loss_231_and1 = torch.add(loss_231, and1)
                    weight = alpha * torch.log(loss_231_and1)

                    weight_num = 0
                    for k in range(len(weight)):  # batch_size=8
                        if weight[k] != 0.0:
                            weight_num += 1
                    weight_ave = torch.sum(weight) / weight_num
                    for k in range(len(weight)):  # batch_size=8
                        if weight[k] == 0.0:
                            weight[k] = weight_ave

                    sum_ = 23 * weight.mean()
                    weight = weight / sum_

                    all_weight = torch.zeros(3256)
                    all_weight[0:387] = weight[0] / 387
                    all_weight[387:519] = weight[1] / 132
                    all_weight[519:953] = weight[2] / 434
                    all_weight[953:1039] = weight[3] / 86
                    all_weight[1039:1275] = weight[4] / 236
                    all_weight[1275:1347] = weight[5] / 72
                    all_weight[1347:1581] = weight[6] / 234
                    all_weight[1581:1651] = weight[7] / 70
                    all_weight[1651:1774] = weight[8] / 123
                    all_weight[1774:1845] = weight[9] / 71
                    all_weight[1845:2035] = weight[10] / 190
                    all_weight[2035:2052] = weight[11] / 17
                    all_weight[2052:2166] = weight[12] / 114
                    all_weight[2166:2346] = weight[13] / 180
                    all_weight[2346:2427] = weight[14] / 81
                    all_weight[2427:2485] = weight[15] / 58
                    all_weight[2485:2581] = weight[16] / 96
                    all_weight[2581:2855] = weight[17] / 274
                    all_weight[2855:2987] = weight[18] / 132
                    all_weight[2987:3069] = weight[19] / 82
                    all_weight[3069:3139] = weight[20] / 70
                    all_weight[3139:3213] = weight[21] / 74
                    all_weight[3213:3256] = weight[22] / 43

                    train_loss += loss.item() * images.size(0)

        adjust_learning_rate(epoch)

        train_acc = train_acc / 3256
        train_loss = train_loss / 3256

        test_acc = test()
        save_models(epoch)
        f.write("epoch:" + str(epoch) + "," + "train_loss" + str(train_loss) + "," + "train_acc" + str(
            train_acc) + "," + "test_acc" + str(test_acc) + '\n')
        print("Epoch:{} ,Train_loss:{},Train_acc:{},test_acc:{}".format(epoch, train_loss, train_acc, test_acc))


if __name__ == "__main__":
    train(100)
