#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J Test_EMA.out          # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10      # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p p40                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 1-00:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o Test_EMA.out     # 打印输出的文件
source /public/data2/software/software/anaconda3/bin/activate
conda activate GAN1


##--------- unbalanced SNGAN training (static)-----------------
#for densityG in 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.98
#do
# python train.py --model sngan_cifar10 --exp-name sngan_cifar10_unbalanced --init-path initial_weights \
# --sparse --sparse_init ERK --density 0.3 --densityG $densityG --update_frequency 4000 --random_seed 1
#done
#
#for densityG in 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.98
#do
# python train.py --model sngan_cifar10 --exp-name sngan_cifar10_unbalanced --init-path initial_weights \
# --sparse --sparse_init ERK --density 0.3 --densityG $densityG --update_frequency 4000 --random_seed 2
#done
#
#for densityG in 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.98
#do
# python train.py --model sngan_cifar10 --exp-name sngan_cifar10_unbalanced --init-path initial_weights \
# --sparse --sparse_init ERK --density 0.3 --densityG $densityG --update_frequency 4000 --random_seed 3
#done


#--------- unbalanced gan training (DST Ggradient Drandom)-----------------

python train.py --model sngan_cifar10 --exp-name sngan_cifar10_test_EMA --init-path initial_weights \
--sparse --imbalanced --sparse_init ERK --density 0.3 --densityG 0.05 --update_frequency 500 \
--dy_mode GD --G_growth gradient --D_growth random --random_seed 1


#--------- unbalanced gan training (DST Ggradient Dgradient)-----------------
#for densityG in 0.05 0.1 0.2 0.3 0.4 0.5
#do
#  python train.py --model sngan_cifar10 --exp-name sngan_cifar10_unbalanced_DST_Ggradient_Dgradient --init-path initial_weights \
#  --sparse --sparse_init ERK --density 0.3 --densityG $densityG --update_frequency 4000 \
#  --dy_mode --G_growth gradient --D_growth random
#done

conda deactivate

