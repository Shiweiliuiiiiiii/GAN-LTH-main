#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J CGAN_dense_training           # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10      # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p gpu                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 5-00:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o CGAN_dense_training     # 打印输出的文件
source /public/data2/software/software/anaconda3/bin/activate
conda activate GAN1

# training a dense model
for s in 0 1 2
do
  python train_dst.py --dataset horse2zebra --seed $s --rand initial_weights --exp-name dense_training_seed$s
done

# training a balanced sparse model
#for s in 0
#do
#  for density in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#  do
#    python train_dst.py --dataset horse2zebra \
#    --sparse --sparse_init ERK --density $density --update_frequency 10 --rand initial_weights --seed $s \
#    --exp-name sparse_training_balanced_density$density_seed$seed \
#  done
#done
#
#
## training an static imbalanced sparse model with various G-sparsity
#for s in 0
#do
#  for densityG in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
#  do
#    python train_dst.py --dataset horse2zebra \
#    --sparse --imbalanced --sparse_init ERK --update_frequency 100 --rand initial_weights --seed $s \
#    --exp-name sparse_training_unbalanced_static_density$densityG_seed$seed \
#    --densityG $densityG --G_growth gradient --D_growth random
#  done
#done
#

# need to finetue update_frequency: on runing
# training an imbalanced DST sparse GAN with various G-sparsity; D-sparsity is fixed as 0.5
#for s in 0
#do
#  for fre in 100 500 1000 2000 5000 10000 20000 50000 75000
#  do
#    python train_dst.py --dataset horse2zebra \
#    --sparse --imbalanced --sparse_init ERK --update_frequency 100 --rand initial_weights --seed $s \
#    --exp-name sparse_training_unbalanced_Ggradient_Drandom_densityG0.05_fre$fre_seed$seed \
#    --dy_mode GD --densityG 0.05 --G_growth gradient --D_growth random
#  done
#done
#
#for s in 0
#do
#  for fre in 100 500 1000 2000 5000 10000 20000 50000 75000
#  do
#    python train_dst.py --dataset horse2zebra \
#    --sparse --imbalanced --sparse_init ERK --update_frequency 100 --rand initial_weights --seed $s \
#    --exp-name sparse_training_unbalanced_Ggradient_Dgradient_densityG0.05_fre$fre_seed$seed \
#    --dy_mode GD --densityG 0.05 --G_growth gradient --D_growth gradient
#  done
#done
#conda activate GAN1

