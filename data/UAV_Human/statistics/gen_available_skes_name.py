import numpy as np
import os
from tqdm import tqdm


miss_names = './samples_with_missing_skeletons.txt'
skes_train_path = '../../../../Skeletons/train'
skes_test_path = '../../../../Skeletons/test'
availabel_train_skes_name = './skes_train_available_name.txt'
availabel_test_skes_name = './skes_test_available_name.txt'

miss_names = np.loadtxt(miss_names, dtype=str)
skes_train_names = os.listdir(skes_train_path)
skes_test_names = os.listdir(skes_test_path)

skes_train_names.sort()
skes_train_names.sort()

available_train_skes = []
for name in tqdm(skes_train_names):
    name = name.split('.')[0]
    if name not in miss_names:
        available_train_skes.append(name)

with open(availabel_train_skes_name, 'w') as fw:
    for name in available_train_skes:
        name = name + '\n'
        fw.write(name)

available_test_skes = []
for name in tqdm(skes_test_names):
    name = name.split('.')[0]
    if name not in miss_names:
        available_test_skes.append(name)

with open(availabel_test_skes_name, 'w') as fw:
    for name in available_test_skes:
        name = name + '\n'
        fw.write(name)

print('Finishing~')
