#!/bin/bash

# attack='pgd', attack_iters=10, batch_size=128, cutout=False, cutout_len=None, data_dir='../cifar-data', epochs=200, epsilon=8, eval=False, fgsm_alpha=1, fgsm_init='random', fname='experiments/cifar10_validation/preactresnet18', half=False, l1=0, l2=0, lr_max=0.1, lr_schedule='piecewise', mixup=False, mixup_alpha=None, model='PreActResNet18', norm='l_inf', pgd_alpha=2, restarts=1, resume=0, seed=0, val=True, width_factor=10

# parser.add_argument('--model', default='PreActResNet18')
# parser.add_argument('--l2', default=0, type=float)
# parser.add_argument('--l1', default=0, type=float)
# parser.add_argument('--batch-size', default=128, type=int)
# parser.add_argument('--data-dir', default='../cifar-data', type=str)
# parser.add_argument('--epochs', default=200, type=int)
# parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
# parser.add_argument('--lr-max', default=0.1, type=float)
# parser.add_argument('--lr-one-drop', default=0.01, type=float)
# parser.add_argument('--lr-drop-epoch', default=100, type=int)
# parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
# parser.add_argument('--epsilon', default=8, type=int)
# parser.add_argument('--attack-iters', default=10, type=int)
# parser.add_argument('--restarts', default=1, type=int)
# parser.add_argument('--pgd-alpha', default=2, type=float)
# parser.add_argument('--fgsm-alpha', default=1.25, type=float)
# parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
# parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
# parser.add_argument('--fname', default='cifar_model', type=str)
# parser.add_argument('--seed', default=0, type=int)
# parser.add_argument('--half', action='store_true')
# parser.add_argument('--width-factor', default=10, type=int)
# parser.add_argument('--resume', default=0, type=int)
# parser.add_argument('--cutout', action='store_true')
# parser.add_argument('--cutout-len', type=int)
# parser.add_argument('--mixup', action='store_true')
# parser.add_argument('--mixup-alpha', type=float)
# parser.add_argument('--eval', action='store_true')
# parser.add_argument('--val', action='store_true')
# parser.add_argument('--use-probes', action='store_true')
# parser.add_argument('--chkpt-iters', default=10, type=int)

srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name cifar10-probe-adv --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/robust_overfitting/train_cifar_probes.py --data-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
            --attack='pgd' --attack-iters=10 --batch-size=128 --epochs=200 --epsilon=8 --fgsm-alpha=1 --fgsm-init='random' \
            --fname='experiments/cifar10_probe/preactresnet18' --lr-schedule='piecewise' --model='PreActResNet18' --norm='l_inf' \
            --pgd-alpha=2 --restarts=1 --resume=0 --seed=0 --width-factor=10 --use-probes # --val=True
            # > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar10_ce_stop_train_noise_${noise_level}_baseline_dynamic_thresh_3rd_ep.log 2>&1 &
