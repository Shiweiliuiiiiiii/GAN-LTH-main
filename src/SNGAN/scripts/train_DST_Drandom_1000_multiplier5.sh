#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J sngan_DST_Drandom_densityD0.5_multiplier5         # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10      # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p gpu                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 14-00:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o sngan_DST_Drandom_densityD0.5_multiplier5.out       # 打印输出的文件
source /public/data2/software/software/anaconda3/bin/activate
conda activate GAN1
densityD=0.5
#--------- unbalanced gan training (DST Ggradient Drandom)-----------------
for s in 1
do
 for densityG in 0.05 0.1 0.2 0.3
 do
   python train.py --model sngan_cifar10 --exp-name sngan_cifar10_unbalanced_DST_Drandom_fre1000_multiplier_seed$s --init-path initial_weights \
   --sparse --imbalanced --sparse_init ERK --densityD $densityD --densityG $densityG --update_frequency 1000 \
   --dy_mode GD --SEMA --G_growth gradient --D_growth random --random_seed $s --multiplier 5
 done
done

conda deactivate

