import imageio
import numpy as np
import scipy.misc
import scipy.io
import os
import torch
import torch.nn as nn

from torchvision import transforms
import PIL.Image as Image


class ONE():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/our_dataset/train', 'train1.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            # line[:-1]去除了这行文本的最后一个字符后剩下的部分
            # split(' ')[0]以‘ ’（空格）为分割符，得到的是第一个空格之前的内容
            train_label_list.append(int(line[:-1].split(' ')[-1]))
        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/our_dataset/test', 'test1.txt'))
        test_file_list = []
        test_label_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'train/1', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'test/1', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((128, 256), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(128,256)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((128, 256), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class TWO():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/our_dataset/train', 'train2.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            # line[:-1]去除了这行文本的最后一个字符后剩下的部分
            # split(' ')[0]以‘ ’（空格）为分割符，得到的是第一个空格之前的内容
            train_label_list.append(int(line[:-1].split(' ')[-1]))
        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/our_dataset/test', 'test2.txt'))
        test_file_list = []
        test_label_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'train/2', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'test/2', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((128,256),Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(128,256)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((128, 256), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class THREE():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/our_dataset/train', 'train3.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            # line[:-1]去除了这行文本的最后一个字符后剩下的部分
            # split(' ')[0]以‘ ’（空格）为分割符，得到的是第一个空格之前的内容
            train_label_list.append(int(line[:-1].split(' ')[-1]))
        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/our_dataset/test', 'test3.txt'))
        test_file_list = []
        test_label_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'train/3', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'test/3', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((128,256),Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(128,256)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((128, 256), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)



class ALL():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/our_dataset/train', 'train.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            # line[:-1]去除了这行文本的最后一个字符后剩下的部分
            # split(' ')[0]以‘ ’（空格）为分割符，得到的是第一个空格之前的内容
            train_label_list.append(int(line[:-1].split(' ')[-1]))
        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/our_dataset/test', 'test.txt'))
        test_file_list = []
        test_label_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'train/4', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'test/4', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((128, 256), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(128,256)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((128, 256), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class FGSC_ONE():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/FGSC', 'FG_train1.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/FGSC', 'FG_test1.txt'))
        test_file_list = []
        test_label_list = []
        test_name_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))
            test_name_list.append(line[:-1].split('.')[0])

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'train/1', train_file)) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_train', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'test/1', test_file)) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_test', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]
            self.test_name = list([os.path.join('./feature', test_file) for test_file in test_name_list[:data_len]])

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(224, 224)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)
            test_name = self.test_name[index]

        return img, target, test_name

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class FGSC_TWO():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/FGSC', 'FG_train2.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/FGSC', 'FG_test2.txt'))
        test_file_list = []
        test_label_list = []
        test_name_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))
            test_name_list.append(line[:-1].split('.')[0])

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'FGSC/train/2', train_file)) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_train', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'FGSC/test/2', test_file)) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_test', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]
            self.test_name = list([os.path.join('./feature', test_file) for test_file in test_name_list[:data_len]])

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(224, 224)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)
            test_name = self.test_name[index]

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class FGSC_THREE():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/FGSC', 'FG_train3.txt'))
        # train_file_txt = open(os.path.join(self.root, 'anno', 'HRSC_train.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/FGSC', 'FG_test3.txt'))
        # test_file_txt = open(os.path.join(self.root, 'anno', 'HRSC_test.txt'))
        test_file_list = []
        test_label_list = []
        test_name_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))
            test_name_list.append(line[:-1].split('.')[0])

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'FGSC/train/3', train_file)) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_train', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'FGSC/test/3', test_file)) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_test', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]
            self.test_name = list([os.path.join('./feature', test_file) for test_file in test_name_list[:data_len]])

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(224, 224)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)
            test_name = self.test_name[index]

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class FGSC_ALL():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/FGSC', 'FG_train.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/FGSC', 'FG_test.txt'))
        test_file_list = []
        test_label_list = []
        test_name_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))
            test_name_list.append(line[:-1].split('.')[0])

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'FGSC/train/4', train_file)) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_train', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'FGSC/test/4', test_file)) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_test', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]
            self.test_name = list([os.path.join('./feature', test_file) for test_file in test_name_list[:data_len]])

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(224, 224)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)
            test_name = self.test_name[index]

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class DOTA_ONE():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/DOTA', 'DOTA_train1.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/DOTA', 'DOTA_test1.txt'))
        test_file_list = []
        test_label_list = []
        test_name_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))
            test_name_list.append(line[:-1].split('.')[0])

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'DOTA_train/1', train_file)) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_train', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'DOTA_test/1', test_file)) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_test', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]
            self.test_name = list([os.path.join('./feature', test_file) for test_file in test_name_list[:data_len]])

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(224, 224)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)
            test_name = self.test_name[index]

        return img, target, test_name

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class DOTA_TWO():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/DOTA', 'DOTA_train2.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/DOTA', 'DOTA_test2.txt'))
        test_file_list = []
        test_label_list = []
        test_name_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))
            test_name_list.append(line[:-1].split('.')[0])

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'DOTA_train/2', train_file)) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_train', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'DOTA_test/2', test_file)) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_test', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]
            self.test_name = list([os.path.join('./feature', test_file) for test_file in test_name_list[:data_len]])

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(224, 224)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)
            test_name = self.test_name[index]

        return img, target, test_name

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class DOTA_THREE():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/DOTA', 'DOTA_train3.txt'))
        # train_file_txt = open(os.path.join(self.root, 'anno', 'HRSC_train.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/DOTA', 'DOTA_test3.txt'))
        # test_file_txt = open(os.path.join(self.root, 'anno', 'HRSC_test.txt'))
        test_file_list = []
        test_label_list = []
        test_name_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))
            test_name_list.append(line[:-1].split('.')[0])

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'DOTA_train/3', train_file)) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_train', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'DOTA_test/3', test_file)) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_test', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]
            self.test_name = list([os.path.join('./feature', test_file) for test_file in test_name_list[:data_len]])

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(224, 224)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)
            test_name = self.test_name[index]

        return img, target, test_name

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class DOTA_ALL():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno/DOTA', 'DOTA_train.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno/DOTA', 'DOTA_test.txt'))
        test_file_list = []
        test_label_list = []
        test_name_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))
            test_name_list.append(line[:-1].split('.')[0])

        #获得图片和标签
        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'DOTA/DOTA_train/4', train_file)) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_train', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'DOTA/DOTA_test/4', test_file)) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'HRSC_test', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]
            self.test_name = list([os.path.join('./feature', test_file) for test_file in test_name_list[:data_len]])

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')  # 实现array到image的转换
            img = transforms.Grayscale(num_output_channels=1)(img)  # 三通道彩色图像转单通道灰度图像
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)  # 将输入PIL图像的大小调整为(224, 224)，Image.BICUBIC（双三次插值）为所需的插值
            img = transforms.RandomHorizontalFlip(p=0.5)(img)  # 以给定的概率p随机水平翻转给定的PIL图像
            img = transforms.ToTensor()(img)  # 转为tensor张量形式，并归一化至[0,1]
            img = transforms.Normalize([.5], [.5])(img)  # 用平均值和标准偏差归一化张量图像，取0.5是归一化，把范围[0,1]的数值调整到[-1,1]，前面均值，后面标准差

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)
            test_name = self.test_name[index]

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)
