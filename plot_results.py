#!/bin/python

import os
import sys
import simplejson

import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <Log file> <Output file>")
    exit()

file_name = sys.argv[1]
output_file_name = sys.argv[2]
assert output_file_name.endswith(".png"), output_file_name

parsed_columns = []
assert os.path.exists(file_name), file_name
header = "Epoch \t Train Time \t Test Time \t LR \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc"
header = header.split(" \t ")
assert len(header) == 12, header
output_dict = {h: [] for h in header}
idx2name_map = {i: h for i, h in enumerate(header)}

with open(file_name, "r") as f:
    for i, l in enumerate(f):
        try:
            l = l.strip().split(" - ")[1]  # Ignore the timestamp
            values = l.split(" \t ")
            if len(values) != len(header):
                continue
            assert len(values) == len(header), values
            
            values = [float(x.replace(" \t ", "")) for x in values]
            values[0] = int(values[0])  # Convert epoch to int
            parsed_columns.append(values)
            
            for j in range(len(values)):
                name = idx2name_map[j]
                output_dict[name].append(values[j])
        
        except ValueError:
            continue  # Irrelevant line

print(simplejson.dumps(output_dict))

# Plot the final results
linewidth = 4
plt.plot(output_dict["Epoch"], [1. - x for x in output_dict["Train Acc"]], color='r', label="Train standard", linewidth=linewidth)
plt.plot(output_dict["Epoch"], [1. - x for x in output_dict["Test Acc"]], color='g', label="Test standard", linewidth=linewidth)

plt.plot(output_dict["Epoch"], [1. - x for x in output_dict["Train Robust Acc"]], color='orange', label="Train robust", linewidth=linewidth)
plt.plot(output_dict["Epoch"], [1. - x for x in output_dict["Test Robust Acc"]], color='b', label="Test robust", linewidth=linewidth)

plt.xlabel("Epochs")
plt.ylabel("Error")

# Set tick interval
plt.xticks([x for x in output_dict["Epoch"] if x % 50 == 0] + [output_dict["Epoch"][-1] + 1])
plt.yticks(np.arange(0.0, 1.0, 0.2))

plt.legend()
plt.tight_layout()
plt.savefig(output_file_name, dpi=300)
