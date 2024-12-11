import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt
import torchattacks

from HBaR.source.hbar.utils.dataset import *
from HBaR.source.hbar.model.lenet import *
from HBaR.source.hbar.model.resnet import *

import numpy as np

import argparse
parser = argparse.ArgumentParser(description="eval")


parser.add_argument('-dataSet', type=str, default='CIFAR10', help='Dataset names, including ‘MNIST’, ‘CIFAR10’, and ‘CIFAR100’.')
parser.add_argument('-testBatchSize', type=int, default=128, help='Test batch size.')

parser.add_argument('-fileName', type=str, default='weight', help='Filename for saving weights.')

parser.add_argument('-attack', type=str, default='pgd', help='attack.')
parser.add_argument('-epsAttack', type=float, default=0.03, help='Perturbation constraint radius for training attack.')
parser.add_argument('-stepsAttack', type=int, default=40, help='Iteration count for training attacks.')
parser.add_argument('-alpAttack', type=float, default=0.008, help='Attack step size for training attacks.')

parser.add_argument('-CWC', type=float, default=1, help='Constraints of CW attack.')
parser.add_argument('-CWConf', type=float, default=0, help='Confidence of CW attack.')
parser.add_argument('-CWSteps', type=int, default=50, help='Steps of CW attack.')
parser.add_argument('-CWLr', type=float, default=0.01, help='Learning rate of CW attack.')

args = parser.parse_args()


file_name = args.fileName

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_dict = {'robustness': True}

if args.dataSet == 'MNIST':
    net = LeNet3(**config_dict)
elif args.dataSet == 'CIFAR10':
    net = ResNet18(**config_dict)
else:
    net = ResNet50(**config_dict)

checkpoint = torch.load('./' + args.fileName, weights_only=True)
net.load_state_dict(checkpoint)

net = net.to(device)

_, test_loader = get_dataset_from_code(args.dataSet.lower(), args.testBatchSize)

if args.attack == 'PGD':
    evalAttack = torchattacks.PGD(net, eps=args.epsAttack, alpha=args.alpAttack, steps=args.stepsAttack, random_start=True)
elif args.attack == 'FGSM':
    evalAttack = torchattacks.FGSM(net, eps=args.epsAttack)
elif args.attack == 'DIFGSM':
    evalAttack = torchattacks.DIFGSM(net, eps=args.epsAttack, alpha=args.alpAttack, steps=args.stepsAttack, decay=1.0, resize_rate=0.9, diversity_prob=0.5, random_start=True)
elif args.attack == 'CW':
    evalAttack = torchattacks.CW(net, c=args.CWC, kappa=args.CWConf, steps=args.CWSteps, lr=args.CWLr)
elif args.attack == 'AA':
    if args.dataSet == 'MNIST' or args.dataSet == 'CIFAR10':
        n_classes=10
    elif args.dataSet == 'CIFAR100':
        n_classes=100
    evalAttack = torchattacks.AutoAttack(net, norm='Linf', eps=args.epsAttack, version='standard', n_classes=n_classes, seed=None, verbose=False)
elif args.attack == 'Square':
    if args.dataSet == 'MNIST':
        evalAttack = torchattacks.Square(net, norm='Linf', eps=56/255, n_queries=1000, n_restarts=1, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
    else:
        evalAttack = torchattacks.Square(net, norm='Linf', eps=8/255, n_queries=1000, n_restarts=1, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
elif args.attack == 'Pixle':
    if args.dataSet == 'MNIST':
        evalAttack = torchattacks.Pixle(net, x_dimensions=[1,2], y_dimensions=[1,2], restarts=10, max_iterations=10)
    else:
        evalAttack = torchattacks.Pixle(net, x_dimensions=[0.05, 0.1], y_dimensions=[0.05, 0.1], restarts=10, max_iterations=10)

def get_accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

'''
def test():
    net.eval()
    acc = []
    acc5 = []
    accad = []
    accad5 = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        print(batch_idx)
        outputs  = net(inputs)
        prec1, prec5 = get_accuracy(outputs, targets, topk=(1, 5)) 
        acc.append(prec1.cpu().detach().numpy())
        acc5.append(prec5.cpu().detach().numpy())
        print(float(prec1))
        adv = evalAttack(inputs, targets)
        adv_outputs = net(adv)
        prec1, prec5 = get_accuracy(adv_outputs, targets, topk=(1, 5)) 
        accad.append(prec1.cpu().detach().numpy())
        accad5.append(prec5.cpu().detach().numpy())
        print(float(prec1))
        print('---------------------------------')
    print('\nTotal benign test accuarcy:', np.mean(acc))
    print('Total adversarial test Accuarcy:', np.mean(accad))
'''

def test2():
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))

        adv = evalAttack(inputs, targets)
        adv_outputs = net(adv)


        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)

# test()

test2()

