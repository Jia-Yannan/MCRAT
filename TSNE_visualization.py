import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from models import *

from loss.loss import MaximalCodingRateReduction

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from dataLoader import data_loader
import torchattacks


import argparse
parser = argparse.ArgumentParser(description="standard training")

parser.add_argument('-fileName', type=str, default='weight', help='Filename for saving weights.')
parser.add_argument('-figName', type=str, default='fig', help='Filename of figure')
args = parser.parse_args()
print(args)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = ResNet18()
checkpoint = torch.load('./checkpoint/'+args.fileName, weights_only=True)
net = torch.nn.DataParallel(net)
net.load_state_dict(checkpoint['net'])
print('Model loaded!')
net = net.to(device)


train_loader, test_loader = data_loader('CIFAR10', 100, 100)

criterion = MaximalCodingRateReduction(eps=0.5)

adversary = torchattacks.PGD(net, eps=0.0314, alpha=0.00784, steps=10, random_start=True)

cudnn.benchmark = True

features = []
labels = []

def test():
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
            
            
        # outputs, outF = net(inputs, -1)
            
        adv = adversary(inputs, targets)
        outputs, outF = net(adv, -1)
            
        labels.append(targets.cpu().detach().numpy())
        features.append(outF.cpu().detach().numpy())
            

def fig():
    global features
    global labels
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(features)
    plt.figure(figsize=(7, 5))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, color in enumerate(colors):
        plt.scatter(features_tsne[labels == i, 0], features_tsne[labels == i, 1], color=color, label=str(i), s=2)


    plt.legend(loc='lower right', fontsize='x-small')    
    # t-SNE visualization of natural samples on standard training、Standard adversarial Training t-SNE、MCRAT Adversarial Training t-SNE
    # plt.title('adversarial samples on MCRAT adversarial training')
    plt.savefig(args.figName + '.jpg')


test()
fig()
