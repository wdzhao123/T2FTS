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
from ClassAwareSampler import ClassAwareSampler
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


trainset = DOTA_ALL(root='./Images', is_train=True, data_len=None)
trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False,
                                          sampler=ClassAwareSampler(trainset), num_workers=8, drop_last=False)

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

t1 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t1.load_state_dict(torch.load('./save_model/dota_t1wujiandu/baselinet1_98', map_location=torch.device('cpu')))
t2 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t2.load_state_dict(torch.load('./save_model/dota_t2/baselinet2_14', map_location=torch.device('cpu')))
t3 = CNN(Bottleneck, [3, 4, 6, 3]).to(device)
t3.load_state_dict(torch.load('./save_model/dota_t3/baselinet3_17', map_location=torch.device('cpu')))

baselinestuweight = nn.DataParallel(baselinestuweight)
# 加载已经训练的cnn模型
# baselinestuweight.load_state_dict(torch.load('./save_model/dota_alpha/baselinestuweight_alpha_53', map_location="cuda:0"))
# baselinestuweight = nn.DataParallel(baselinestuweight)


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
    if not os.path.isdir("./save_model/dota_alpha75/"):
        os.makedirs("./save_model/dota_alpha75/")
    savepath = os.path.join('./save_model/dota_alpha75/', "baselinestuweight_alpha75_{}".format(epoch+53))
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
    test_acc = test_acc / 28853

    return test_acc


def train(num_epochs):
    road = './save_model/dota_jieguosweight_alpha.txt'
    f = open(road, 'a')
    for epoch in range(num_epochs):
        baselinestuweight.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(trainloader1):
            print("i={}".format(i))
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
                loss_151 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                loss2 = loss_fn2(outputs1, labels)
                _, prediction = torch.max(outputs1.data, 1)
                train_acc += torch.sum(prediction == labels.data)
                labels = labels.cpu().detach().numpy()
                labels = labels.tolist()

                for k in range(len(labels)):  # batch_size=8
                    if (labels[k] == 6) or (labels[k] == 9) or (labels[k] == 10):
                        outputs21, feature21 = t1(images)
                        feature21 = feature21.cpu().detach().numpy()
                        feature21 = feature21.tolist()
                        feature2[k] = feature21[k]
                        b = feature2[k]
                        a = feature1[k]
                        b = torch.Tensor(b).to(device)
                        loss_81[k] = loss_fn1(a, b)
                    elif (labels[k] == 2) or (labels[k] == 4) or (labels[k] == 7) or (labels[k] == 12) \
                            or (labels[k] == 13) or (labels[k] == 14):
                        outputs22, feature22 = t2(images)
                        feature22 = feature22.cpu().detach().numpy()
                        feature22 = feature22.tolist()
                        feature2[k] = feature22[k]
                        b = feature2[k]
                        a = feature1[k]
                        b = torch.Tensor(b).to(device)
                        loss_81[k] = loss_fn1(a, b)
                    elif (labels[k] == 3) or (labels[k] == 5) or (labels[k] == 0) or (labels[k] == 1) \
                            or (labels[k] == 11) or (labels[k] == 8):
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

                labels_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # len(labels_num)=23
                for t in range(len(labels)):
                    for tt in range(len(labels_num)):
                        if labels[t] == tt:
                            labels_num[tt] += 1
                            loss_151[tt] += loss_81[t]

                for ttt in range(len(labels_num)):
                    if labels_num[ttt] != 0:
                        loss_151[ttt] = loss_151[ttt] / labels_num[ttt]

                loss_151 = torch.Tensor(loss_151).to(device)
                alpha = 75
                and1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                and1 = torch.Tensor(and1).to(device)
                loss_151_and1 = torch.add(loss_151, and1)
                weight = alpha * torch.log(loss_151_and1)

                weight_num = 0
                for k in range(len(weight)):  # batch_size=8
                    if weight[k] != 0.0:
                        weight_num += 1
                weight_ave = torch.sum(weight) / weight_num
                for k in range(len(weight)):  # batch_size=8
                    if weight[k] == 0.0:
                        weight[k] = weight_ave

                qiuhe = 15 * weight.mean()
                weight = weight / qiuhe
                all_weight = torch.zeros(98906)
                all_weight[0:415] = weight[0] / 415
                all_weight[415:930] = weight[1] / 515
                all_weight[930:2977] = weight[2] / 2047
                all_weight[2977:3302] = weight[3] / 325
                all_weight[3302:9285] = weight[4] / 5983
                all_weight[9285:9915] = weight[5] / 630
                all_weight[9915:26884] = weight[6] / 16969
                all_weight[26884:34855] = weight[7] / 7971
                all_weight[34855:35254] = weight[8] / 399
                all_weight[35254:63322] = weight[9] / 28068
                all_weight[63322:89448] = weight[10] / 26126
                all_weight[89448:89774] = weight[11] / 326
                all_weight[89774:94803] = weight[12] / 5029
                all_weight[94803:96539] = weight[13] / 1736
                all_weight[96539:98906] = weight[14] / 2367

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
                    loss_151 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                    loss2 = loss_fn2(outputs1, labels)
                    _, prediction = torch.max(outputs1.data, 1)
                    train_acc += torch.sum(prediction == labels.data)
                    labels = labels.cpu().detach().numpy()
                    labels = labels.tolist()

                    for k in range(len(labels)):
                        if (labels[k] == 6) or (labels[k] == 9) or (labels[k] == 10):
                            outputs21, feature21 = t1(images)
                            feature21 = feature21.cpu().detach().numpy()
                            feature21 = feature21.tolist()
                            feature2[k] = feature21[k]
                            b = feature2[k]
                            a = feature1[k]
                            b = torch.Tensor(b).to(device)
                            loss_81[k] = loss_fn1(a, b)
                        elif (labels[k] == 2) or (labels[k] == 4) or (labels[k] == 7) or (labels[k] == 12) \
                                or (labels[k] == 13) or (labels[k] == 14):
                            outputs22, feature22 = t2(images)
                            feature22 = feature22.cpu().detach().numpy()
                            feature22 = feature22.tolist()
                            feature2[k] = feature22[k]
                            b = feature2[k]
                            a = feature1[k]
                            b = torch.Tensor(b).to(device)
                            loss_81[k] = loss_fn1(a, b)
                        elif (labels[k] == 3) or (labels[k] == 5) or (labels[k] == 0) or (labels[k] == 1) \
                                or (labels[k] == 11) or (labels[k] == 8):
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

                    labels_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    for t in range(len(labels)):
                        for tt in range(len(labels_num)):
                            if labels[t] == tt:
                                labels_num[tt] += 1
                                loss_151[tt] += loss_81[t]

                    for ttt in range(len(labels_num)):
                        if labels_num[ttt] != 0:
                            loss_151[ttt] = loss_151[ttt] / labels_num[ttt]

                    loss_151 = torch.Tensor(loss_151).to(device)

                    alpha = 75
                    and1 = torch.ones(15)
                    and1 = torch.Tensor(and1).to(device)
                    loss_151_and1 = torch.add(loss_151, and1)
                    weight = alpha * torch.log(loss_151_and1)

                    weight_num = 0
                    for k in range(len(weight)):
                        if weight[k] != 0.0:
                            weight_num += 1
                    weight_ave = torch.sum(weight) / weight_num
                    for k in range(len(weight)):
                        if weight[k] == 0.0:
                            weight[k] = weight_ave

                    qiuhe = 15 * weight.mean()
                    weight = weight / qiuhe

                    all_weight = torch.zeros(98906)
                    all_weight[0:415] = weight[0] / 415
                    all_weight[415:930] = weight[1] / 515
                    all_weight[930:2977] = weight[2] / 2047
                    all_weight[2977:3302] = weight[3] / 325
                    all_weight[3302:9285] = weight[4] / 5983
                    all_weight[9285:9915] = weight[5] / 630
                    all_weight[9915:26884] = weight[6] / 16969
                    all_weight[26884:34855] = weight[7] / 7971
                    all_weight[34855:35254] = weight[8] / 399
                    all_weight[35254:63322] = weight[9] / 28068
                    all_weight[63322:89448] = weight[10] / 26126
                    all_weight[89448:89774] = weight[11] / 326
                    all_weight[89774:94803] = weight[12] / 5029
                    all_weight[94803:96539] = weight[13] / 1736
                    all_weight[96539:98906] = weight[14] / 2367

                    train_loss += loss.item() * images.size(0)

        adjust_learning_rate(epoch)

        train_acc = train_acc / 98906
        train_loss = train_loss / 98906

        test_acc = test()
        save_models(epoch)
        f.write("epoch:" + str(epoch) + "," + "train_loss" + str(train_loss) + "," + "train_acc" + str(
            train_acc) + "," + "test_acc" + str(test_acc) + "chongxin" + '\n')
        print("Epoch:{} ,Train_loss:{},Train_acc:{},test_acc:{}".format(epoch, train_loss, train_acc, test_acc))


if __name__ == "__main__":
    train(100)
    test_dir()
