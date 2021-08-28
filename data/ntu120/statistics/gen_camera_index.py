import numpy as np


camera_file = 'camera.txt'
skes_name_file = 'skes_available_name.txt'
skes_names = np.loadtxt(skes_name_file, dtype=np.string_)

cameras = []
for name in skes_names:
    camera_num = int(name[5:8])
    cameras.append(camera_num)

cameras = np.asarray(cameras, dtype=np.int)
np.savetxt(camera_file, cameras, fmt='%d')



