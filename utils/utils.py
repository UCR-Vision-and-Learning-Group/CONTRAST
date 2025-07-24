import os
import torch.nn.functional as F
import numpy as np
import torch
import random
import datetime
import torch.nn as nn
import torch
import math

def get_dataset_mean(training_generator):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in training_generator:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def get_batch_mean(x):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in x:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count
            
class write_logger(object):
    def __init__(self):
        if not os.path.isdir('results'):
            os.mkdir('results')
        now = datetime.datetime.now()
        self.filename_log = 'Results-'+str(now)+'.txt'
    
    def write(self, **kwargs):
        f = open('results/'+self.filename_log, "a")
        for key, value in kwargs.items():
            f.write(str(key) +": " +str(value)+ "\n")
        f.write("\n")
        f.close()

def set_random_seed(seed, deterministic=True):
    """Set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        




    