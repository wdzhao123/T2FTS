"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import random
import numpy as np
from torch.utils.data.sampler import Sampler

##################################
## Class-aware sampling, partly implemented by frombeijingwithlove
##################################

class RandomCycleIter:
    
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)  # data=23
        self.length = len(self.data_list)  # 23
        self.i = self.length - 1  # 22
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)  # random.shuffle 打乱数据
            
        return self.data_list[self.i]
    
def class_aware_sample_generator (cls_iter, data_iter_list, n, num_samples_cls=1):

    i = 0
    j = 0
    while i < n:
        
#         yield next(data_iter_list[next(cls_iter)])
        
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

class ClassAwareSampler (Sampler):
    
    def __init__(self, data_source, num_samples_cls=1,):
        num_classes = 23
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.train_label):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = 3256     #dota98906 fgsc3256
        self.num_samples_cls = num_samples_cls
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    def __len__ (self):
        return self.num_samples
    
def get_sampler():
    return ClassAwareSampler
