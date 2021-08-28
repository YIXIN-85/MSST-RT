# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import h5py

missing_count = 0
noise_len_thres = 11
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754
noise_mot_thres_lo = 0.089925
noise_mot_thres_hi = 2

def get_raw_bodies_data(ske_path, ske_name):
    """
    Get raw bodies data from a skeleton sequence.

    Each body's data is a dict that contains the following keys:
      - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
      - interval: a list which stores the frame indices of this body.
      - motion: motion amount (only for the sequence with 2 or more bodyIDs).

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """
    ske_file = osp.join(ske_path, ske_name + '.skeleton')
    assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
    # Read all data from .skeleton file into a list (in string format)
    print('Reading data from %s' % ske_file[-29:])
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()

    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1  # 0-based index
    current_line = 1

    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1

        if num_bodies == 0:  # no data in this frame, drop it
            frames_drop.append(f)  # 0-based index
            continue

        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)

        for b in range(num_bodies):
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
            current_line += 1

            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                current_line += 1

            if bodyID not in bodies_data:  # Add a new body's data
                body_data = dict()
                body_data['joints'] = joints[b]  # ndarray: (25, 3)
                body_data['interval'] = [valid_frames]  # the index of the first frame
            else:  # Update an already existed body's data
                body_data = bodies_data[bodyID]
                # Stack each body's data of each frame along the frame order
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index

            bodies_data[bodyID] = body_data  # Update bodies_data

    num_frames_drop = len(frames_drop)
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)


    # Calculate motion (only for the sequence with 2 or more bodyIDs)
    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data():

    raw_data = get_raw_bodies_data(ske_path, ske_name)
    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_data, fw, pickle.HIGHEST_PROTOCOL)

    print('Saved raw bodies data into %s' % save_data_pkl)
    return raw_data

def denoising_by_length(ske_name, bodies_data):
    """
    Denoising data based on the frame length for each bodyID.
    Filter out the bodyID which length is less or equal than the predefined threshold.

    """
    new_bodies_data = bodies_data.copy()
    for (bodyID, body_data) in new_bodies_data.items():
        length = len(body_data['interval'])
        if length <= noise_len_thres:
            del bodies_data[bodyID]

    return bodies_data

def get_valid_frames_by_spread(points):
    """
    Find the valid (or reasonable) frames (index) based on the spread of X and Y.

    :param points: joints or colors
    """
    num_frames = points.shape[0]
    valid_frames = []
    for i in range(num_frames):
        x = points[i, :, 0]
        y = points[i, :, 1]
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):  # 0.8
            valid_frames.append(i)
    return valid_frames

def denoising_by_spread(ske_name, bodies_data):
    """
    Denoising data based on the spread of Y value and X value.
    Filter out the bodyID which the ratio of noisy frames is higher than the predefined
    threshold.

    bodies_data: contains at least 2 bodyIDs
    """
    denoised_by_spr = False  # mark if this sequence has been processed by spread.

    new_bodies_data = bodies_data.copy()
    # for (bodyID, body_data) in bodies_data.items():
    for (bodyID, body_data) in new_bodies_data.items():
        if len(bodies_data) == 1:
            break
        valid_frames = get_valid_frames_by_spread(body_data['joints'].reshape(-1, 25, 3))
        num_frames = len(body_data['interval'])
        num_noise = num_frames - len(valid_frames)
        if num_noise == 0:
            continue

        ratio = num_noise / float(num_frames)
        motion = body_data['motion']
        if ratio >= noise_spr_thres2:  # 0.69754
            del bodies_data[bodyID]
            denoised_by_spr = True
        else:  # Update motion
            joints = body_data['joints'].reshape(-1, 25, 3)[valid_frames]
            body_data['motion'] = min(motion, np.sum(np.var(joints.reshape(-1, 3), axis=0)))
            # TODO: Consider removing noisy frames for each bodyID


    return bodies_data, denoised_by_spr

def denoising_bodies_data(bodies_data):
    """
    Denoising data based on some heuristic methods, not necessarily correct for all samples.

    Return:
      denoised_bodies_data (list): tuple: (bodyID, body_data).
    """
    bodies_data = bodies_data['data']

    # Step 1: Denoising based on frame length.
    bodies_data, noise_info_len = denoising_by_length(ske_name, bodies_data)

    if len(bodies_data) == 1:  # only has one bodyID left after step 1
        return bodies_data.items(), noise_info_len

    # Step 2: Denoising based on spread.
    bodies_data, denoised_by_spr = denoising_by_spread(ske_name, bodies_data)

    if len(bodies_data) == 1:
        return bodies_data.items()

    bodies_motion = dict()  # get body motion
    for (bodyID, body_data) in bodies_data.items():
        bodies_motion[bodyID] = body_data['motion']
    # Sort bodies based on the motion
    # bodies_motion = sorted(bodies_motion.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
    bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
    denoised_bodies_data = list()
    for (bodyID, _) in bodies_motion:
        denoised_bodies_data.append((bodyID, bodies_data[bodyID]))

    return denoised_bodies_data


def get_one_actor_points(body_data, num_frames):
    """
    Get joints for only one actor.
    For joints, each frame contains 75 X-Y-Z coordinates.
    """
    joints = np.zeros((num_frames, 75), dtype=np.float32)
    start, end = body_data['interval'][0], body_data['interval'][-1]
    joints[start:end + 1] = body_data['joints'].reshape(-1, 75)

    return joints


def remove_missing_frames(ske_name, joints):
    """
    Cut off missing frames which all joints positions are 0s

    For the sequence with 2 actors' data, also record the number of missing frames for
    actor1 and actor2, respectively (for debug).
    """

    # Find valid frame indices that the data is not missing or lost
    # For two-subjects action, this means both data of actor1 and actor2 is missing.
    valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # 0-based index
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]
    num_missing = len(missing_indices)

    if num_missing > 0:  # Update joints
        joints = joints[valid_indices]

    return joints

def get_two_actors_points(bodies_data):
    """
    Get the first and second actor's joints positions locations.

    # Arguments:
        bodies_data (dict): 3 key-value pairs: 'name', 'data', 'num_frames'.
        bodies_data['data'] is also a dict, while the key is bodyID, the value is
        the corresponding body_data which is also a dict with 4 keys:
          - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
          - interval: a list which records the frame indices.
          - motion: motion amount

    # Return:
        joints.
    """
    num_frames = bodies_data['num_frames']

    bodies_data = denoising_bodies_data(bodies_data)  # Denoising data

    bodies_data = list(bodies_data)
    if len(bodies_data) == 1:  # Only left one actor after denoising

        bodyID, body_data = bodies_data[0]
        joints = get_one_actor_points(body_data, num_frames)
    else:


        joints = np.zeros((num_frames, 150), dtype=np.float32)

        bodyID, actor1 = bodies_data[0]  # the 1st actor with largest motion
        start1, end1 = actor1['interval'][0], actor1['interval'][-1]
        joints[start1:end1 + 1, :75] = actor1['joints'].reshape(-1, 75)
        del bodies_data[0]

        start2, end2 = [0, 0]  # initial interval for actor2 (virtual)

        while len(bodies_data) > 0:
            bodyID, actor = bodies_data[0]
            start, end = actor['interval'][0], actor['interval'][-1]
            if min(end1, end) - max(start1, start) <= 0:  # no overlap with actor1
                joints[start:end + 1, :75] = actor['joints'].reshape(-1, 75)
                # Update the interval of actor1
                start1 = min(start, start1)
                end1 = max(end, end1)
            elif min(end2, end) - max(start2, start) <= 0:  # no overlap with actor2
                joints[start:end + 1, 75:] = actor['joints'].reshape(-1, 75)
                # Update the interval of actor2
                start2 = min(start, start2)
                end2 = max(end, end2)
            del bodies_data[0]

    return joints

def get_raw_denoised_data(raw_data):
    num_bodies = len(raw_data['data'])
    if num_bodies == 1:  # only 1 actor
        num_frames = raw_data['num_frames']
        body_data = list(raw_data['data'].values())[0]
        joints= get_one_actor_points(body_data, num_frames)
    else:  # more than 1 actor, select two main actors
        joints= get_two_actors_points(raw_data)
        # Remove missing frames
        joints= remove_missing_frames(ske_name, joints)
    return joints

def seq_translation(ske_joints):
    num_frames = ske_joints.shape[0]
    num_bodies = 1 if ske_joints.shape[1] == 75 else 2
    if num_bodies == 2:
        missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
        missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
        cnt1 = len(missing_frames_1)
        cnt2 = len(missing_frames_2)

    i = 0  # get the "real" first frame of actor1
    while i < num_frames:
        if np.any(ske_joints[i, :75] != 0):
            break
        i += 1

    origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

    for f in range(num_frames):
        if num_bodies == 1:
            ske_joints[f] -= np.tile(origin, 25)
        else:  # for 2 actors
            ske_joints[f] -= np.tile(origin, 50)

    if (num_bodies == 2) and (cnt1 > 0):
        ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

    if (num_bodies == 2) and (cnt2 > 0):
        ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

    return ske_joints

def one_hot_vector(label):
    label_vector = np.zeros((1, 60))
    label_vector[0, int(label)-1] = 1
    print(label_vector)

    return label_vector

if __name__ == '__main__':
    save_path = './'
    ske_path = '../../../Raw_Skeleton_data/NTU-RGB-D60'
    # ske_name = 'S013C003P017R002A038'
    # ske_name = 'S017C001P008R001A010'
    ske_name = 'S013C003P028R001A024'

    visual_label = ske_name[-3:]

    if not osp.exists('./visual_data'):
        os.makedirs('./visual_data')

    save_data_pkl = osp.join(save_path, 'visual_data', 'visual_skes_data.pkl')

    raw_data = get_raw_skes_data()
    joints = get_raw_denoised_data(raw_data)
    ske_joints = seq_translation(joints)
    ske=[]
    ske.append(ske_joints)
    h5file = h5py.File(osp.join(save_path, 'NTU_visual_24.h5'), 'w')
    # Visual set
    h5file.create_dataset('visual_x', data=ske)
    visual_one_hot_labels = one_hot_vector(visual_label)
    h5file.create_dataset('visual_y', data=visual_one_hot_labels)

    h5file.close()