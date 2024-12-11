import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
from models import *

import matplotlib.pyplot as plt
import torchattacks

from dataLoader import data_loader
from mart import mart_loss


import argparse
parser = argparse.ArgumentParser(description="standard training")

parser.add_argument('-mcrAt', type=int, default=0, help='Whether to use MCRAT regularization, where 1 represents usage and 0 represents original training.')
parser.add_argument('-epsMCR2', type=float, default=0.5, help='Distortion Constraints of MCR2')
parser.add_argument('-dataSet', type=str, default='CIFAR10', help='Dataset names, including ‘MNIST’, ‘CIFAR10’, and ‘CIFAR100’.')
parser.add_argument('--mcrBeta', type=float, default=0.02, help='Beta of MCRAT')


parser.add_argument('-trainBatch', type=int, default=128, help='The batch size of the training set.')
parser.add_argument('-testBatch', type=int, default=128, help='The batch size of the testing set.')
parser.add_argument('-fileName', type=str, default='pgd_adversarial_training', help='Filename for saving weights.')
parser.add_argument('-learningRate', type=float, default=0.01, help='Learning rate for model training.')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('-momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('-weight_decay', type=float, default=0.0005, help='weight_decay')
parser.add_argument('-saveFreq', default=5, type=int, help='save frequency')


parser.add_argument('-epsAttack', type=float, default=0.3, help='Perturbation constraint radius for training attack.')
parser.add_argument('-stepsAttack', type=int, default=40, help='Iteration count for training attacks.')
parser.add_argument('-alpAttack', type=float, default=0.01, help='Attack step size for training attacks.')

parser.add_argument('-martBeta', type=float, default=1.0, help='weight before kl (misclassified examples)')

args = parser.parse_args()


file_name = args.fileName

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, test_loader = data_loader(args.dataSet, args.trainBatch, args.testBatch)

if args.dataSet == 'MNIST':
    net = leNet()
elif args.dataSet == 'CIFAR10':
    net = ResNet18()
elif args.dataSet == 'CIFAR100':
    net = ResNet50()
net = net.to(device)
net = torch.nn.DataParallel(net)

# 预训练
# checkpoint = torch.load('checkpoint/CIFAR10_st_pro_pre')
# net.load_state_dict(checkpoint['net'])

cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.learningRate, momentum=args.momentum, weight_decay=args.weight_decay)

if args.dataSet == 'MNIST':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[55, 75, 90], gamma=0.1)
else:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90], gamma=0.1)

attack = torchattacks.PGD(net, eps=args.epsAttack, alpha=args.alpAttack, steps=args.stepsAttack, random_start=True)

if args.dataSet == 'MNIST' or args.dataSet == 'CIFAR10':
        n_classes=10
elif args.dataSet == 'CIFAR100':
        n_classes=100

criterion2 = nn.CrossEntropyLoss()

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        loss, adv_outputs = mart_loss(mcrAt = args.mcrAt,
                           n_classes=n_classes,
                           epsMCR2=args.epsMCR2,
                           mcrBeta = args.mcrBeta,
                           model=net,
                           x_natural=inputs,
                           y=targets,
                           optimizer=optimizer,
                           step_size=args.alpAttack,
                           epsilon=args.epsAttack,
                           perturb_steps=args.stepsAttack,
                           beta=args.martBeta,
			               distance='l_inf')
        '''
        if torch.isnan(loss).any():
            print("NaN loss detected, skipping this iteration.")
            continue
        '''
        loss.backward()
        if epoch < 10 and (args.dataSet == 'CIFAR10' or args.dataSet == 'CIFAR100'):
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial train loss:', loss.item())

    print('\nTotal adversarial train accuarcy:', 100. * correct / total)
    print('Total adversarial train loss:', train_loss)

def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
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
        loss = criterion2(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign test loss:', loss.item())

        adv = attack(inputs, targets)
        adv_outputs = net(adv)

        loss = criterion2(adv_outputs, targets)
        adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)


    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if epoch % args.saveFreq ==0 or epoch ==args.epochs: 
        torch.save(state, './checkpoint/' + file_name)
        print('Model Saved!')



for epoch in range(1, args.epochs+1):
    train(epoch)
    test(epoch)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}: Current learning rate is {current_lr}")


