
import os
import argparse
import copy
import shutil

import torch
import torch.optim as optim

from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.utils import set_seed, set_torch_deterministic
from advertorch_examples.utils import get_madry_et_al_cifar10_train_transform
from advertorch_examples.utils import get_train_val_loaders, get_test_loader
from advertorch_examples.utils import mkdir

from mma.utils import get_mean_loss_fn, get_sum_loss_fn, get_none_loss_fn
from mma.utils import add_indexes_to_loader
from mma.anpgd import ANPGD, ANPGDForTest
from mma.trainer import Trainer, Evaluator

from models import *


def retrieve_and_overwrite_config(args):
    from mma.config import config
    cfg = copy.deepcopy(config[args.norm + args.dataset.upper()])
    for key, val in vars(args).items():
        if key not in cfg.__dict__ or val is not None:
            setattr(cfg, key, val)
    return cfg


def get_data_loaders(cfg):
    if cfg.dataset.upper() == "MNIST":
        train_transform = None
    elif cfg.dataset.upper() == "CIFAR10":
        train_transform = get_madry_et_al_cifar10_train_transform()
    else:
        raise ValueError(cfg.dataset)

    datasetname = cfg.dataset.upper()
    train_loader, val_loader = get_train_val_loaders(
        datasetname, train_size=cfg.train_size,
        val_size=cfg.val_size, train_batch_size=cfg.training_batch_size,
        val_batch_size=100,
        train_transform=train_transform,
    )
    test_loader = get_test_loader(
        datasetname, test_size=cfg.test_size, batch_size=100)

    return train_loader, val_loader, test_loader


def get_model(cfg):
    
    if cfg.dataset.upper() == "MNIST":
        model = smallNet()
    elif cfg.dataset.upper() == 'CIFAR10':
        model = ResNet18()
    elif cfg.dataset.upper() == 'CIFAR100':
        model = WideResNet_34_10()

    model.to(cfg.device)
    model = torch.nn.DataParallel(model)
    model.train()
    return model


def get_optimizer(cfg, model):
    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), cfg.initial_learning_rate)
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.learning_rate_schedule[0],
            momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(cfg.dataset)
    return optimizer


def get_adversaries(cfg):
    if cfg.norm == "Linf":
        attack_class = LinfPGDAttack
    elif cfg.norm == "L2":
        attack_class = L2PGDAttack
    else:
        raise ValueError("cfg.norm={}".format(cfg.norm))


    train_adv_loss_fn = get_sum_loss_fn(cfg.attack_loss_fn)

    pgdadv = attack_class(
        model, loss_fn=train_adv_loss_fn,
        eps=0.,  # will be set inside ANPGD
        nb_iter=cfg.nb_iter,
        eps_iter=0.,  # will be set inside ANPGD
        rand_init=cfg.rand_init,
        clip_min=cfg.clip_min, clip_max=cfg.clip_max,
    )

    test_adv_loss_fn = get_sum_loss_fn("slm")
    test_pgdadv = attack_class(
        model,
        loss_fn=test_adv_loss_fn,
        eps=cfg.test_eps,
        nb_iter=cfg.nb_iter,
        eps_iter=cfg.test_eps_iter,
        rand_init=cfg.rand_init,
        clip_min=0., clip_max=1.
    )

    cfg.attack_maxeps = cfg.hinge_maxeps * 1.05

    train_adversary = ANPGD(
        pgdadv=pgdadv,
        mineps=cfg.attack_mineps,
        maxeps=cfg.attack_maxeps,
        num_search_steps=cfg.num_search_steps,
        eps_iter_scale=cfg.eps_iter_scale,
        search_loss_fn=get_none_loss_fn(cfg.search_loss_fn),
    )

    test_adversary = ANPGDForTest(
        pgdadv=test_pgdadv,
        maxeps=cfg.attack_maxeps,
        num_search_steps=cfg.num_search_steps,
    )

    return train_adversary, test_adversary



if __name__ == '__main__':
    # see config.py for default values of arguments

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-mcrAt', type=int, default=0, help='Whether to use MCRAT regularization, where 1 represents usage and 0 represents original training.')
    parser.add_argument('-epsMCR2', type=float, default=0.5, help='Distortion Constraints of MCR2')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--deterministic', default=False, action="store_true")
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--norm', required=True, type=str, help="Linf | L2")
    parser.add_argument('--hinge_maxeps', required=True, type=float)

    parser.add_argument('--clean_loss_fn', default=None, type=str,
                        help="xent | slm | lm | cw")
    parser.add_argument('--margin_loss_fn', default=None, type=str,
                        help="xent | slm | lm | cw")
    parser.add_argument('--attack_loss_fn', default=None, type=str,
                        help="xent | slm | lm | cw")
    parser.add_argument('--search_loss_fn', default=None, type=str,
                        help="xent | slm | lm | cw")
    parser.add_argument('--clean_loss_coeff', default=None, type=float)
    
    parser.add_argument('--eps_iter_scale', default=None, type=float)
    parser.add_argument('--num_search_steps', default=None, type=int)
    parser.add_argument('--attack_mineps', default=None, type=float)


    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--train_size', default=None, type=int)
    parser.add_argument('--val_size', default=5000, type=int)
    parser.add_argument('--test_size', default=None, type=int)
    parser.add_argument('--pretrained', default="", type=str)
    parser.add_argument('--disp_interval', default=100, type=int)
    parser.add_argument('--savepath', default="./", type=str)

    args = parser.parse_args()
    cfg = retrieve_and_overwrite_config(args)
    for k, v in cfg.__dict__.items():
        print(k, v)

    if cfg.deterministic:
        print("Set to deterministic behavior")
        set_torch_deterministic()
    set_seed(cfg.seed)
    
    if args.dataset.upper() == 'MNIST' or args.dataset.upper() == 'CIFAR10':
        n_classes=10
    elif args.dataset.upper() == 'CIFAR100':
        n_classes=100
    else:
        raise ValueError("args.dataset={}".format(args.dataset))

    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    add_indexes_to_loader(train_loader)
    add_indexes_to_loader(val_loader)
    add_indexes_to_loader(test_loader)

    model = get_model(cfg)

    train_adversary, test_adversary = get_adversaries(cfg)
    optimizer = get_optimizer(cfg, model)

    clean_loss_fn = get_mean_loss_fn(cfg.clean_loss_fn)
    margin_loss_fn = get_none_loss_fn(cfg.margin_loss_fn)

    trainer = Trainer(
        model, cfg.device, clean_loss_fn, optimizer, train_loader,
        margin_loss_fn,
        hinge_maxeps=cfg.hinge_maxeps,
        clean_loss_coeff=cfg.clean_loss_coeff,
        adversary=train_adversary,
        max_steps=cfg.max_num_training_steps,
        lr_by_epochs=cfg.learning_rate_schedule,
        disp_interval=cfg.disp_interval,
        mcrAt=cfg.mcrAt, 
        epsMCR2=cfg.epsMCR2,
        n_classes=n_classes,
    )

    test_evaluater = Evaluator(
        model, cfg.device, clean_loss_fn, test_loader,
        adversary=test_adversary, dataname="test")
    val_evaluater = Evaluator(
        model, cfg.device, clean_loss_fn, val_loader,
        adversary=test_adversary, dataname="valid")


    # #####################
    # start training
    best_avgeps = 0.
    mkdir(cfg.savepath)
    for i in range(0, cfg.epochs):
        trainer.train_one_epoch()
        val_clnacc, val_advacc, val_avgeps = val_evaluater.test_one_epoch()
        test_clnacc, test_advacc, test_avgeps = test_evaluater.test_one_epoch()

        ckpt = {
            'epoch': trainer.epochs,
            'config': cfg.__dict__,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_dct_eps_record': trainer.dct_eps_record,
        }

        info = {
            'train_dct_eps': trainer.dct_eps,
            'val_dct_eps': val_evaluater.dct_eps,
            'test_dct_eps': test_evaluater.dct_eps,
            'val_clnacc': val_clnacc,
            'val_advacc': val_advacc,
            'val_avgeps': val_avgeps,
            'test_clnacc': test_clnacc,
            'test_advacc': test_advacc,
            'test_avgeps': test_avgeps,
        }

        torch.save(ckpt, os.path.join(
            cfg.savepath, 'ckpt_{}.pt'.format(trainer.epochs)))
        torch.save(info, os.path.join(
            cfg.savepath, 'info_{}.pt'.format(trainer.epochs)))

        if val_avgeps > best_avgeps:
            best_avgeps = val_avgeps
            shutil.copyfile(
                os.path.join(
                    cfg.savepath, 'ckpt_{}.pt'.format(trainer.epochs)),
                os.path.join(cfg.savepath, 'ckpt_best.pt'),
            )
            shutil.copyfile(
                os.path.join(
                    cfg.savepath, 'info_{}.pt'.format(trainer.epochs)),
                os.path.join(cfg.savepath, 'info_best.pt'),
            )
            torch.save({'epoch': trainer.epochs, 'model': model.state_dict()},
                       os.path.join(cfg.savepath, 'model_best.pt'))
