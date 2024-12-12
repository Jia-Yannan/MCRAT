# MCRAT
Code for "Enhancing Adversarial Robustness Based on Maximal Coding Rate Reduction Regularization"


# Contents

* [Requirements](#Requirements)
* [Training](#Training)
* [Evaluation](#Evaluation)
* [StatisticalAnalysis](#StatisticalAnalysis)
* [VisualAnalysis](#VisualAnalysis)
* [Acknowledgement](#Acknowledgement)
* [Contact](#Contact)

## Requirements
* python 3.9.19
* torch 2.4.0
* torchattacks 3.5.1
* CUDA
* numpy

Please run the following command to install the Python dependencies and packages:
```bash
pip install requirements.txt
```

After intalling "torchattacks" package, HBaR needs to modify one place as follows to make sure this framework work. Please go to the installed package directory (`/.../torchattacks/attacks/`), modify `pgd.py` by finding the line `outputs = self.model(adv_images)`, and insert the following code after it:
```bash
if type(outputs) == tuple:
    outputs = outputs[0]
```

## Training
The arguments in the codes are self-explanatory. Detailed commands can be found in the 'cmd' file.

### Standard
- Command Options
```
usage: train_standard.py  [-mcrAt] [-epsMCR2] [-dataSet] [-trainBatch] [-testBatch]
                          [-fileName] [-learningRate] [-epochs] [-momentum] 
                          [-weight_decay] [-saveFreq] [-epsAttack] 
                          [-stepsAttack] [-alpAttack]
```
- example
```python
python train_standard.py -mcrAt 1 -dataSet MNIST -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031 -trainBatch 128 -weight_decay 0.0005 -learningRate 0.01 -epochs 100 -fileName minist_st_pro
```

### TRADES
- Command Options
```
usage: train_trades.py  [--mcrAt] [--epsMCR2] [--dataSet] [--mcrBeta] [--batch-size] [--test-batch-size]
                        [--epochs] [--weight-decay] [--lr] [--momentum] [--no-cuda] [--epsilon]
                        [--num-steps] [--step-size] [--beta] [--seed] [--log-interval] 
                        [--model-dir] [--fileName] [--save-freq]
```
- example
```python
python train_trades.py --mcrAt 1 --mcrBeta 0.02 --dataSet MNIST --beta 5 --epsilon 0.188 --num-steps 20 --step-size 0.031 --batch-size 128 --weight-decay 0.0005 --lr 0.01 --epochs 100 --fileName minist_tds_pro
```


### MART
- Command Options
```
usage: train_mart.py  [-mcrAt] [-epsMCR2] [-dataSet] [-mcrBeta] [-trainBatch] [-testBatch]
                      [-fileName] [-learningRate] [-epochs] [-momentum] [-alpAttack]
                      [-weight_decay] [-saveFreq] [-epsAttack] [-stepsAttack] [-martBeta]                   
```
- example
```python
python train_mart.py -dataSet MNIST -mcrAt 1 --mcrBeta 0.02 -martBeta 5 -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031 -trainBatch 128 -weight_decay 0.001 -learningRate 0.01 -epochs 100 -fileName minist_mart_pro
```


### HBaR
- Command Options
```
usage: scripts/
robust-mnist-adv.sh  [mcrAT] [name]
robust-cifar-adv.sh  [mcrAT] [name]
robust-cifar100-adv.sh  [mcrAT] [name]               
```
- example
```bash
cd HBaR
source env.sh
robust-mnist-adv.sh 1 pro
```


## Evaluation
- Command Options
```
usage: eval.py/eval_HBaR.py [-dataSet] [-testBatchSize] [-fileName] [-attack] [-epsAttack] 
                            [-stepsAttack] [-alpAttack] [-CWC] [-CWConf] [-CWSteps] [-CWLr]
```
- example
```python
python eval.py  -dataSet MNIST -testBatchSize 128 -fileName checkpoint/minist_st_ini -attack PGD -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031
python eval_HBaR.py  -dataSet MNIST -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack PGD -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031
```

## StatisticalAnalysis
- Command Options
```
usage: statisticsMCR2.py [-fileName] [-figName]
```
- example
```python
python statisticsMCR2.py -fileName CIFAR10_st_ini -figName fig1
```

## VisualAnalysis
- Command Options
```
usage: TSNE_visualization.py [-fileName] [-figName]
```
- example
```python
python TSNE_visualization.py -fileName CIFAR10_st_ini -figName fig1
```


## Acknowledgement 
This research was generously supported by Taiyuan City under grant “Double hundred Research action”2024TYJB0127. 
Part of the code is based on the following repo:
- MCR<sup>2</sup>: https://github.com/ryanchankh/mcr2
- TREADES: https://github.com/yaodongyu/TRADES
- MART: https://github.com/YisenWang/MART
- HBaR: https://github.com/neu-spiral/HBaR


## Contact
Please contact jia_yannan@bupt.edu.cn if you have any question on the codes.