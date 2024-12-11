from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models import *
from dataLoader import data_loader
from trades import trades_loss
import torchattacks

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')

parser.add_argument('--mcrAt', type=int, default=0, help='Whether to use MCRAT regularization, where 1 represents usage and 0 represents original training.')
parser.add_argument('--epsMCR2', type=float, default=0.5, help='Distortion Constraints of MCR2')
parser.add_argument('--dataSet', type=str, default='CIFAR10', help='Dataset names, including ‘MNIST’, ‘CIFAR10’, and ‘CIFAR100’.')
parser.add_argument('--mcrBeta', type=float, default=0.02, help='Beta of MCRAT')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=0.03,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', type=int, default=5.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', type=str, default='trades',
                    help='directory of model for saving checkpoint')
parser.add_argument('--fileName', type=str, default='weight', help='Filename for saving weights.')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = 'checkpoint/' + args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader, test_loader = data_loader(args.dataSet, args.batch_size, args.test_batch_size)

criterion2 = nn.CrossEntropyLoss()

print(args)

if args.dataSet == 'MNIST' or args.dataSet == 'CIFAR10':
        n_classes=10
elif args.dataSet == 'CIFAR100':
        n_classes=100

def train(args, model, device, train_loader, optimizer, epoch, attack):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(mcrAt = args.mcrAt,
                           n_classes=n_classes,
                           epsMCR2=args.epsMCR2,
                           mcrBeta=args.mcrBeta,
                           model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        if torch.isnan(loss).any():
            print("NaN loss detected, skipping this iteration.")
            continue
        loss.backward()
        if epoch == 1 and (args.dataSet == 'CIFAR10' or args.dataSet == 'CIFAR100'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader, attack):
    model.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = model(inputs)
        loss = criterion2(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign test loss:', loss.item())

        adv = attack(inputs, targets)
        adv_outputs = model(adv)

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


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if args.dataSet == 'MNIST' and epoch >= 55:
        lr *= 0.1
    if epoch >= 75:
        lr *= 0.1
    if epoch >= 90:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    if args.dataSet == 'MNIST':
        model = leNet().to(device)
    elif args.dataSet == 'CIFAR10':
        model = ResNet18().to(device)
    elif args.dataSet == 'CIFAR100':
        model = ResNet50().to(device)
    
    attack = torchattacks.PGD(model, eps=args.epsilon, alpha=args.step_size, steps=args.num_steps, random_start=True)
    
    # 预训练
    # model = torch.nn.DataParallel(model)
    # checkpoint = torch.load('checkpoint/CIFAR10_st_pro')
    # model.load_state_dict(checkpoint['net'])
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch, attack)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader, attack)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs or epoch > args.epochs - 10:
            torch.save(model.state_dict(), 'checkpoint/'+ args.model_dir + '/' + args.fileName + '_epoch{}.pt'.format(epoch))
            torch.save(optimizer.state_dict(), 'checkpoint/'+ args.model_dir + '/' + args.fileName + '_epoch{}.tar'.format(epoch))


if __name__ == '__main__':
    main()
