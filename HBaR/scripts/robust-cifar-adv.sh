#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Combining HBaR with adversarial training 

dataset=cifar10
model=resnet18

xw=1
lx=0.0005
ly=0.005
mcrAt=$1
name=$2

run_hbar -cfg config/general-hbar-xentropy-${dataset}.yaml -slmo -xw $xw -lx ${lx} -ly ${ly} \
-adv -ep 100 \
-mf ${dataset}_${model}_xw_${xw}_lx_${lx}_ly_${ly}_adv_$name.pt -mcrAt $mcrAt
