import torch
import torchvision
import torchvision.transforms as transforms


def get_transform(img_size=32):

    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform_train, transform_test




def data_loader(dataset='CIFAR10', train_batch=128, test_batch=128):
    
    if dataset == 'CIFAR10':
        transform_train, transform_test = get_transform(32)
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'MNIST':
        transform_train, transform_test = get_transform(28)
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'CIFAR100':
        transform_train, transform_test = get_transform(32)
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=False, num_workers=4)
    
    return train_loader, test_loader
    
    
