
####################################################
# The command to run Standard Adversarial Training #
####################################################
nohup python train_standard.py -dataSet MNIST -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031 -trainBatch 128 -weight_decay 0.0005 -learningRate 0.01 -epochs 100 -fileName minist_st_ini > minist_st_ini.log &

nohup python train_standard.py -mcrAt 1 -dataSet MNIST -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031 -trainBatch 128 -weight_decay 0.0005 -learningRate 0.01 -epochs 100 -fileName minist_st_pro > minist_st_pro.log &

nohup python train_standard.py -dataSet CIFAR10 -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784 -trainBatch 128 -weight_decay 0.0002 -learningRate 0.1 -epochs 100 -fileName CIFAR10_st_ini > CIFAR10_st_ini.log &

nohup python train_standard.py -dataSet CIFAR10 -mcrAt 1 -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784 -trainBatch 128 -weight_decay 0.0002 -learningRate 0.1 -epochs 100 -fileName CIFAR10_st_pro > CIFAR10_st_pro.log &

nohup python train_standard.py -dataSet CIFAR100 -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784 -trainBatch 128 -weight_decay 0.0002 -learningRate 0.1 -epochs 100 -fileName CIFAR100_st_ini > CIFAR100_st_ini.log &

nohup python train_standard.py -dataSet CIFAR100 -mcrAt 1 -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784 -trainBatch 128 -weight_decay 0.0002 -learningRate 0.1 -epochs 100 -fileName CIFAR100_st_pro > CIFAR100_st_pro.log &

nohup python train_standard_beta.py -dataSet CIFAR10 -mcrAt 1 --mcrBeta 0.01 -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784 -trainBatch 128 -weight_decay 0.0002 -learningRate 0.1 -epochs 100 -fileName CIFAR10_st_pro_beta001 > CIFAR10_st_pro_beta001.log &
##############################
# The command to run  TRADES #
##############################
nohup python train_trades.py --dataSet MNIST --beta 5 --epsilon 0.188 --num-steps 20 --step-size 0.031 --batch-size 128 --weight-decay 0.0005 --lr 0.01 --epochs 100 --fileName minist_tds_ini > minist_tds_ini.log &

nohup python train_trades.py --mcrAt 1 --mcrBeta 0.02 --dataSet MNIST --beta 5 --epsilon 0.188 --num-steps 20 --step-size 0.031 --batch-size 128 --weight-decay 0.0005 --lr 0.01 --epochs 100 --fileName minist_tds_pro > minist_tds_pro.og &

nohup python train_trades.py --dataSet CIFAR10 --beta 5 --epsilon 0.0314 --num-steps 10 --step-size 0.00784 --batch-size 128 --weight-decay 0.0002 --lr 0.1 --epochs 100 --fileName CIFAR10_tds_ini > CIFAR10_tds_ini.log &

nohup python train_trades.py --mcrAt 1 --mcrBeta 0.02 --dataSet CIFAR10 --beta 5 --epsilon 0.0314 --num-steps 10 --step-size 0.00784 --batch-size 128 --weight-decay 0.0002 --lr 0.1 --epochs 100 --fileName CIFAR10_tds_pro > CIFAR10_tds_pro.log &

nohup python train_trades.py --dataSet CIFAR100 --beta 5 --epsilon 0.0314 --num-steps 10 --step-size 0.00784 --batch-size 128 --weight-decay 0.0002 --lr 0.1 --epochs 100 --fileName CIFAR100_tds_ini > CIFAR100_tds_ini.log &

nohup python train_trades.py --mcrAt 1 --mcrBeta 0.02 --dataSet CIFAR100 --beta 5 --epsilon 0.0314 --num-steps 10 --step-size 0.00784 --batch-size 128 --weight-decay 0.0002 --lr 0.1 --epochs 100 --fileName CIFAR100_tds_pro > CIFAR100_tds_pro.log &
############################
# The command to run  MART #
############################
nohup python train_mart.py -dataSet MNIST -martBeta 5 -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031 -trainBatch 128 -weight_decay 0.001 -learningRate 0.01 -epochs 100 -fileName minist_mart_ini > minist_mart_ini.log &

nohup python train_mart.py -dataSet MNIST -mcrAt 1 --mcrBeta 0.02 -martBeta 5 -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031 -trainBatch 128 -weight_decay 0.001 -learningRate 0.01 -epochs 100 -fileName minist_mart_pro > minist_mart_pro.log &

nohup python train_mart.py -dataSet CIFAR10 -martBeta 5 -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784 -trainBatch 128 -weight_decay 0.0035 -learningRate 0.01 -epochs 100 -fileName CIFAR10_mart_ini > CIFAR10_mart_ini.log &

nohup python train_mart.py -mcrAt 1 --mcrBeta 0.01 -dataSet CIFAR10 -martBeta 5 -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784 -trainBatch 128 -weight_decay 0.0035 -learningRate 0.01 -epochs 100 -fileName CIFAR10_mart_pro > CIFAR10_mart_pro.log &

nohup python train_mart.py -dataSet CIFAR100 -martBeta 5 -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784 -trainBatch 128 -weight_decay 0.0035 -learningRate 0.01 -epochs 100 -fileName CIFAR100_mart_ini > CIFAR100_mart_ini.log &

nohup python train_mart.py -mcrAt 1 --mcrBeta 0.01 -dataSet CIFAR100 -martBeta 5 -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784 -trainBatch 128 -weight_decay 0.0035 -learningRate 0.01 -epochs 100 -fileName CIFAR100_mart_pro > CIFAR100_mart_pro.log &
############################
# The command to run  HBaR #
############################
cd HBaR
source env.sh
vim config/general-hbar-xentropy-mnist.yaml
vim HBaR/source/hbar/core/train_misc.py #set_optimizer
nohup robust-mnist-adv.sh 0 ini > minist_HBaR_ini.log &
nohup robust-mnist-adv.sh 1 pro > minist_HBaR_pro.log &

nohup robust-cifar-adv.sh 0 ini > CIFAR10_HBaR_ini.log &
nohup robust-cifar-adv.sh 1 pro > CIFAR10_HBaR_pro.log &

nohup robust-cifar100-adv.sh 0 ini > CIFAR100_HBaR_ini.log &
nohup robust-cifar100-adv.sh 1 pro > CIFAR100_HBaR_pro.log &
#######################################
# The command to run model evaluation #
#######################################
python eval.py  -dataSet MNIST -testBatchSize 128 -fileName checkpoint/minist_st_ini -attack FGSM -epsAttack 0.188
python eval.py  -dataSet MNIST -testBatchSize 128 -fileName checkpoint/minist_st_ini -attack PGD -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031
python eval.py  -dataSet MNIST -testBatchSize 128 -fileName checkpoint/minist_st_ini -attack DIFGSM -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031
python eval.py  -dataSet MNIST -testBatchSize 128 -fileName checkpoint/minist_st_ini -attack CW -CWC 10 -CWConf 0 -CWSteps 50 -CWLr 0.01
nohup python eval.py  -dataSet MNIST -testBatchSize 128 -fileName checkpoint/minist_st_ini -attack AA -epsAttack 0.188 > tmp.log &

python eval.py  -dataSet MNIST -testBatchSize 128 -fileName checkpoint/minist_st_ini -attack Square
python eval.py  -dataSet MNIST -testBatchSize 128 -fileName checkpoint/minist_st_ini -attack Pixle

python eval.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName checkpoint/CIFAR10_st_ini -attack FGSM -epsAttack 0.0314
python eval.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName checkpoint/CIFAR10_st_ini -attack PGD -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784
python eval.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName checkpoint/CIFAR10_st_ini -attack DIFGSM -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784
python eval.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName checkpoint/CIFAR10_st_ini -attack CW -CWC 1 -CWConf 0 -CWSteps 20 -CWLr 0.01
nohup python eval.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName checkpoint/CIFAR10_st_ini -attack AA -epsAttack 0.0314 > tmp.log &

python eval.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName checkpoint/CIFAR100_st_ini -attack Square
python eval.py -dataSet CIFAR10/100 -testBatchSize 128 -fileName checkpoint/CIFAR100_st_ini -attack Pixle
........
cd HBaR
python eval_HBaR.py  -dataSet MNIST -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack FGSM -epsAttack 0.188
python eval_HBaR.py  -dataSet MNIST -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack PGD -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031
python eval_HBaR.py  -dataSet MNIST -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack DIFGSM -epsAttack 0.188 -stepsAttack 20 -alpAttack 0.031
python eval_HBaR.py  -dataSet MNIST -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack CW -CWC 10 -CWConf 0 -CWSteps 50 -CWLr 0.01
nohup python eval_HBaR.py  -dataSet MNIST -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack AA -epsAttack 0.188 > tmp.log &
python eval_HBaR.py  -dataSet MNIST -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack Square
python eval_HBaR.py  -dataSet MNIST -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack Pixle
python eval_HBaR.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack FGSM -epsAttack 0.0314
python eval_HBaR.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack PGD -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784
python eval_HBaR.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack DIFGSM -epsAttack 0.0314 -stepsAttack 10 -alpAttack 0.00784
python eval_HBaR.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack CW -CWC 1 -CWConf 0 -CWSteps 20 -CWLr 0.01
nohup python eval_HBaR.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack AA -epsAttack 0.0314 > tmp.log &
python eval_HBaR.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack Square
python eval_HBaR.py  -dataSet CIFAR10/100 -testBatchSize 128 -fileName assets/models/mnist_lenet3_xw_1_lx_0.003_ly_0.001_adv_ini.pt -attack Pixle
###################
# statistics MCR2 #
###################
python statisticsMCR2.py -fileName CIFAR10_st_ini -figName test
######################
# TSNE_visualization #
######################
python TSNE_visualization.py -fileName CIFAR10_st_ini -figName test
=====================================================================
