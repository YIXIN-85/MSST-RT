import numpy as np


train_label_file = 'train_label.txt'
test_label_file = 'test_label.txt'
skes_train_name_file = 'skes_train_available_name.txt'
skes_test_name_file = 'skes_test_available_name.txt'

skes_train_names = np.loadtxt(skes_train_name_file, dtype=np.string_)
skes_test_names = np.loadtxt(skes_test_name_file, dtype=np.string_)


train_label = []
test_label=[]

for name in skes_train_names:
    train_label_num = int(name[-14:-11])
    train_label.append(train_label_num)

for name in skes_test_names:
    test_label_num = int(name[-14:-11])
    test_label.append(test_label_num)

train_label = np.asarray(train_label, dtype=np.int)
test_label = np.asarray(test_label, dtype=np.int)
np.savetxt(train_label_file, train_label, fmt='%d')
np.savetxt(test_label_file, test_label, fmt='%d')


