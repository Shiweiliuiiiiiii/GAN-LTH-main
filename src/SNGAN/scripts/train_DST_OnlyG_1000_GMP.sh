#!/bin/bash
source /home/TUE/20180170/miniconda3/etc/profile.d/conda.sh
source activate torch151

densityD=1.0
#--------- unbalanced gan training (DST Ggradient Drandom)-----------------
for s in 1
do
 for densityG in 0.05 0.1 0.2
 do
   python train.py --model sngan_cifar10 --exp-name sngan_cifar10_unbalanced_GMP_onlyG_fre5000_seed1 --init-path initial_weights \
   --sparse --imbalanced --sparse_init dense --densityD 1.0 --densityG 0.1 --update_frequency 10 \
   --dy_mode G --SEMA --G_growth gradient --D_growth random --random_seed 1 --sparse_mode GMP --initial_prune_time 0.0
 done
done

conda deactivate

