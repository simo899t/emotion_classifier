from datasets import load_dataset
import numpy as np


ds = load_dataset("dair-ai/emotion", "split")

## sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)

inputs = ds["train"][:]["text"]
labels = ds["train"][:]["label"]

freq_bin = np.bincount(np.array(labels))

# print(freq_bin)
# [4666 5362 1304 2159 1937  572]

total = np.sum(freq_bin)
acc_list = []
for acc in freq_bin:
    acc_list.append(round(float(acc/total * 100),1)) # pct

# >>> print(acc_list)
# [29.2, 33.5, 8.2, 13.5, 12.1, 3.6] 

# avg accuracy
# >>> print(round(sum(acc_list)/(len(acc_list)),1))
# 16.7