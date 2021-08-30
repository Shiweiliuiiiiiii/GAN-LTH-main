#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J CGAN_fre_Ggradient_Dgradient           # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10      # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p gpu                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 7-00:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o CGAN_fre_Ggradient_Dgradient    # 打印输出的文件
source /public/data2/software/software/anaconda3/bin/activate
conda activate GAN1

for s in 0
do
  for fre in 100 500 1000 2000 5000 10000 20000 50000 75000
  do
    python train_dst.py --dataset horse2zebra \
    --sparse --imbalanced --sparse_init ERK --update_frequency $fre --rand initial_weights --seed $s \
    --exp-name sparse_training_unbalanced_Ggradient_Dgradient_densityG0.05_fre$fre_seed$seed \
    --dy_mode GD --densityG 0.05 --G_growth gradient --D_growth gradient
  done
done
#conda activate GAN1

