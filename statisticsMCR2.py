import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

import sys

from models import *

from loss.loss import MaximalCodingRateReduction

import torchattacks

import matplotlib.pyplot as plt
from dataLoader import data_loader

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


y_dis1 = []
y_cmp1 = []

y_dis2 = []
y_cmp2 = []


x = []


acc1 = []
acc2 = []

def test():
    net.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
            
        outputs, outF = net(inputs, -1)
        _, dis1, cmp1 = criterion(outputs, targets)

        adv = adversary(inputs, targets)

        outputs, outF = net(adv, -1)
        _, dis2, cmp2 = criterion(outputs, targets)
            
            
        x.append(batch_idx)

             
        y_dis1.append(dis1.cpu().detach().numpy())
        y_cmp1.append(cmp1.cpu().detach().numpy())
        y_dis2.append(dis2.cpu().detach().numpy())
        y_cmp2.append(cmp2.cpu().detach().numpy())
            
            

def fig():
    plt.figure(figsize=(7, 5))
    
    plt.scatter(x, y_dis1, color='#FF0000', marker='o', s=20, label= 'Nat R')
    mean_value = np.mean(y_dis1)
    plt.axhline(mean_value, color='#FF0000', linewidth=1, linestyle=':')
    '''
    plt.text(len(y_dis1)-1, mean_value, f'Mean: {mean_value:.2f}',
         verticalalignment='bottom', horizontalalignment='left', color='#FF0000')
    '''

    plt.scatter(x, y_cmp1, color='#FF0000', marker='^', s=20, label= r'Nat ${R}^{c}$')
    mean_value = np.mean(y_cmp1)
    plt.axhline(mean_value, color='#FF0000', linewidth=1, linestyle=':')
    '''
    plt.text(len(y_cmp1)-1, mean_value, f'Mean: {mean_value:.2f}',
         verticalalignment='bottom', horizontalalignment='left', color='#FF0000')
    '''


    plt.scatter(x, y_dis2, color='#0000FF', marker='o', s=20, label= 'Adv R')
    mean_value = np.mean(y_dis2)
    plt.axhline(mean_value, color='#0000FF', linewidth=1, linestyle=':')
    '''
    plt.text(len(y_dis2)-1, mean_value, f'Mean: {mean_value:.2f}',
         verticalalignment='bottom', horizontalalignment='left', color='#0000FF')
    '''

    plt.scatter(x, y_cmp2, color='#0000FF', marker='^', s=20, label= r'Adv ${R}^{c}$')
    mean_value = np.mean(y_cmp2)
    plt.axhline(mean_value, color='#0000FF', linewidth=1, linestyle=':')
    '''
    plt.text(len(y_cmp2)-1, mean_value, f'Mean: {mean_value:.2f}',
         verticalalignment='bottom', horizontalalignment='left', color='#0000FF')
    '''

    # Randomly Initialize Weights Coding Rate、Standard Training Coding Rate、Standard Adversarial Training Coding Rate、MCRAT Adversarial Training Coding Rate
    plt.title('MCRAT Adversarial Training Coding Rate')
    plt.xlabel('Batch Index')
    plt.ylabel('Coding Rate')
    plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.6))
    # plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.4))
    # plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.265))
    # plt.grid(True)
    plt.savefig(args.figName + '.jpg')

test()
fig()
