#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J CGAN_unbalanced_training           # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10      # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p gpu                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 2-00:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o CGAN_unbalanced_training     # 打印输出的文件
source /public/data2/software/software/anaconda3/bin/activate
conda activate GAN1


# training a unbalanced sparse model
for s in 0
do
  for density in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
  do
    python train_dst.py --dataset horse2zebra \
    --sparse --imbalanced --sparse_init ERK --densityG $density --update_frequency 10 --rand initial_weights --seed $s \
    --exp-name sparse_training_unbalanced_density$density_seed$seed \
  done
done

conda activate GAN1

