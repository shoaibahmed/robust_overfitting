#!/bin/bash

# python3 plot_results.py experiments/cifar10_validation/preactresnet18/output.log results_val.png

python3 plot_results.py experiments/cifar10_probe/preactresnet18/output.log results.png
python3 plot_results.py experiments/cifar10_probe/preactresnet18_val/output.log results_val.png
python3 plot_results.py experiments/cifar10_probe/preactresnet18_probe/output.log results_probe.png
