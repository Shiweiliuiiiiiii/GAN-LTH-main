#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J SNGAN_dense          # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10      # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p gpu                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 3-00:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o SNGAN_dense       # 打印输出的文件
source /public/data2/software/software/a +-----naconda3/bin/activate
conda activate GAN1

#--------- dense training -----------------
python train.py --model sngan_cifar10 --SEMA --exp-name dense_sngan_cifar10_official --init-path initial_weights --random_seed 1
python train.py --model sngan_cifar10 --SEMA --exp-name dense_sngan_cifar10_official --init-path initial_weights --random_seed 2
python train.py --model sngan_cifar10 --SEMA --exp-name dense_sngan_cifar10_official --init-path initial_weights --random_seed 3


conda deactivate

