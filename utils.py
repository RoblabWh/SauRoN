import argparse
import numpy as np
import torch
from PIL import Image
import os

def statesToTensor(list):
    states = np.asarray(list, dtype=object)
    laser = states[:, :, 0].tolist()
    ori = states[:, :, 1].tolist()
    dist = states[:, :, 2].tolist()
    vel = states[:, :, 3].tolist()
    return [torch.tensor(laser, dtype=torch.float32), torch.tensor(ori, dtype=torch.float32),
            torch.tensor(dist, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)]

# TODO maybe use this ???!?!!
def _scan1DTo2D(lidarHits):

    #theta = np.radians(-135)
    #rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
    #                     [np.sin(theta), np.cos(theta)]])
    #data = np.dot(lidarHits, rotMatrix.T)
    data = lidarHits * 5
    data = data.astype(int)
    image = np.zeros((121, 121))

    # comment in to print scans in folder
    image[data[:,0], data[:,1]] = 255

    im = Image.fromarray(image).convert('RGB')
    frmt = "{0:06d}"
    idx_ = len(os.listdir("./scans")) - 1
    idx = frmt.format(idx_)
    name = "./scans/" + idx + "_scan.png"
    im.save(name)
    ######################################

    image[data[:, 0], data[:, 1]] = 1
    return image

def scan1DTo2D(distancesNorm, print=False):
    scanplot = []
    angle_min = 0
    angle_increment = np.radians(0.25)
    for i, point in enumerate(distancesNorm):
        angle = angle_min + (i * angle_increment)
        x = point * np.cos(angle)
        y = point * np.sin(angle)
        scanplot.append([x, y])
    scanplot = np.asarray(scanplot)
    theta = np.radians(-135)
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    data = np.dot(scanplot, rotMatrix.T)
    data = ((data * 20) + 20) * 3
    data = data.astype(int)
    image = np.zeros((121, 121))

    image[data[:, 0], data[:, 1]] = 255

    # prints scans in folder
    if print:
        im = Image.fromarray(image).convert('RGB')
        frmt = "{0:06d}"
        idx_ = len(os.listdir("./scans")) - 1
        idx = frmt.format(idx_)
        name = "./scans/" + idx + "_scan.png"
        im.save(name)

    #image[data[:, 0], data[:, 1]] = 1
    return image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


