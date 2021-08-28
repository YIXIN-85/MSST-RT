import argparse
import re
import fit

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(

    network_joint_10='UAV_joint_10_lr0005_e50',
    network_bone_10='UAV_bone_10_lr0005',
    network_joint_20='UAV_joint_20_lr0005_e50',
    network_bone_20='UAV_bone_20_lr0005',

    dataset='UAV',
    alpha=1,
    belta=1,
    )
args = parser.parse_args()

dataset = args.dataset

network_joint_10 = args.network_joint_10
network_bone_10 = args.network_bone_10
network_joint_20 = args.network_joint_20
network_bone_20 = args.network_bone_20


l_joint_10 = open('./results/'+dataset+'/'+network_joint_10+'/0_lable.txt', 'r')
l_joint_10 = l_joint_10.readlines()

f_joint_10 = open('./results/'+dataset+'/'+network_joint_10+'/0_pred.txt', 'r')
r_joint_10 = f_joint_10.readlines()
f_bone_10 = open('./results/'+dataset+'/'+network_bone_10+'/0_pred.txt', 'r')
r_bone_10 = f_bone_10.readlines()
f_joint_20 = open('./results/'+dataset+'/'+network_joint_20+'/0_pred.txt', 'r')
r_joint_20 = f_joint_20.readlines()
f_bone_20 = open('./results/'+dataset+'/'+network_bone_20+'/0_pred.txt', 'r')
r_bone_20 = f_bone_20.readlines()


right_num = total_num = 0
for i in tqdm(range(len(l_joint_10))):

    l = l_joint_10[i]
    r1 = re.split(r"[ ]", r_joint_10[i])
    r2 = re.split(r"[ ]", r_bone_10[i])
    r3 = re.split(r"[ ]", r_joint_20[i])
    r4 = re.split(r"[ ]", r_bone_20[i])


    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    r3 = np.array(r3, dtype=float)
    r4 = np.array(r4, dtype=float)

    r = np.array(r1+r2+r3+r4)

    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
print("acc:", acc)