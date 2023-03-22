import logging
import os
import ujson as json
import os
import math
import random
import pdb
from glob import glob
import re
import sys
from pickle import UnpicklingError


list_name = './datasets/train.txt'
json_path = '/media/lixiaofan/xf/bevdata/anpdataset_mini/label/'

with open(list_name, 'rt') as f:
    infos = f.readlines()
    infos = [d.split()[0] for d in infos]

taskids = infos  # all taskids in curr dataset

global_ego_server_cnt = 0
for info_dir in infos:
    image_list = sorted(glob(os.path.join(json_path, info_dir, '*/images.list')))


    for name in image_list:
        with open(name, 'rt') as f:
            lines = f.readlines()
            # lines = lines[0]
        if not lines:
            continue

        rec_name = os.path.basename(os.path.dirname(name))

        ts_list = []
        for i, line in enumerate(lines):
            obs_cam_name = line.strip().split('\t')[1]
            cam_id = line.strip().split('/')[0]
            ts = int(os.path.basename(obs_cam_name).split('_')[-1][:-4])
            ts_list.append((ts, i))
        ts_list = sorted(ts_list, key=lambda t: t[0])

        for ts, idx in ts_list:
            line = lines[idx]

            # labels
            label_name = (os.path.join(info_dir, rec_name), idx)
            if 0: # lukou
                path, index = label_name
                cur_cls_mat_path = os.path.join(self.json_path, '%s/road_seg/' % path,
                                                '%s_%s.npz' % (idx, 'SEG_STOPLINE'))
                cur_cls_mat = np.load(cur_cls_mat_path)['arr_0']
                if not np.any(cur_cls_mat == 1):
                    continue
            self.labels.append(label_name)
            self.ts.append(ts)

            # images
            img_scene_names = line.strip().split('\t')
            img_names = img_scene_names[:3]
            img_names = [os.path.join(info_dir, 'image_data_resized', rec_name, img_name) for img_name in img_names]
            self.images.append(img_names)

            self.idx_to_pose_server.append(global_ego_server_cnt)

