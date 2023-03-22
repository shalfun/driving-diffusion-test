import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
from glob import glob
from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)
import data.transformer as transformer
import cv2
import data.proj_utils as pu

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


class GenerateDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, data_path='', image_size=[224, 224], loader=pil_loader,
                 local_mode=True):
        self.data_root = data_root

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

        ## anp
        self.labels = []
        self.images = []

        print('local_mode:', local_mode)
        if local_mode:
            list_name = data_flist  # './datasets/train.txt'
            self.json_path = '/media/lixiaofan/xf/bevdata/anpdataset_mini/label/'
            self.image_dir = data_path  # '/media/lixiaofan/xf/bevdata/anpdataset_mini/data/'
            params_dir = self.image_dir
        else:
            list_name = data_flist  # '/root/paddlejob/workspace/diff/datasets/train.txt'
            self.json_path = '/root/local_bev_data/anp/label_0113_fix_raw/'
            # self.image_dir = '/root/0000_mount/bev-data/road-structure/data/'
            self.image_dir = data_path  # '/root/local_storage_0001/bev-data/road-structure/data/'

            params_dir = self.image_dir

        with open(list_name, 'rt') as f:
            infos = f.readlines()
            infos = [d.split()[0] for d in infos]

        self.taskids = infos  # all taskids in curr dataset

        global_ego_server_cnt = 0
        for info_dir in infos:
            image_list = sorted(glob(os.path.join(self.json_path, info_dir, '*/images.list')))

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
                    if 0:  # lukou
                        path, index = label_name
                        cur_cls_mat_path = os.path.join(self.json_path, '%s/road_seg/' % path,
                                                        '%s_%s.npz' % (idx, 'SEG_STOPLINE'))
                        cur_cls_mat = np.load(cur_cls_mat_path)['arr_0']
                        if not np.any(cur_cls_mat == 1):
                            continue
                    self.labels.append(label_name)
                    # self.ts.append(ts)

                    # images
                    img_scene_names = line.strip().split('\t')
                    img_names = img_scene_names[:3]
                    if 1:  # cfg.DATASETS.UNDISTORTED:
                        for i in range(len(img_names)):
                            img_names[i] = img_names[i][:-4] + '_undistorted.jpg'
                    img_names = [os.path.join(info_dir, 'image_data_resized', rec_name, img_name) for img_name in
                                 img_names]
                    self.images.append(img_names)

        self.transform_server_dict = {taskid: transformer.TransformServer(
            os.path.join(params_dir, taskid, 'params'), False) for taskid in self.taskids}
        # print(self.transform_server_dict)

    def get_instance(self, index):
        def _to_rfu_segment(points_s, points_e, h, w, mat, img_front, scale, color):
            """ pass
            """
            '''start'''
            points_s = points_s.copy()
            points_s[:, 0] = points_s[:, 0] - w / 2
            points_s[:, 1] = (h - 1) - points_s[:, 1]
            points_s = points_s * 100 / h  # h=800 h_r=100
            z = np.ones((points_s.shape[0], 2), dtype=points_s.dtype) * (-0.364)  # 0.509  0.364
            z[:, 1] = 1
            points_s = np.concatenate((points_s, z), 1)
            # points_s = np.dot(mat, points_s.T).T  # [n, 3]
            '''end'''
            points_e = points_e.copy()
            points_e[:, 0] = points_e[:, 0] - w / 2
            points_e[:, 1] = (h - 1) - points_e[:, 1]
            points_e = points_e * 100 / h  # h=800 h_r=100
            z = np.ones((points_e.shape[0], 2), dtype=points_e.dtype) * (-0.364)  # 0.509  0.364
            z[:, 1] = 1
            points_e = np.concatenate((points_e, z), 1)
            # points_e = np.dot(mat, points_e.T).T  # [n, 3]
            '''segment'''
            for s, e in zip(points_s, points_e):
                homo_point1 = mat.dot(e)
                homo_point2 = mat.dot(s)
                thred = 0
                if homo_point1[-1] < thred and homo_point2[-1] < thred:
                    continue
                if homo_point1[-1] > 100 and homo_point2[-1] > 100:
                    continue
                if homo_point1[-1] > thred and homo_point2[-1] > thred:
                    image_point1 = homo_point1 / homo_point1[-1]
                    image_point2 = homo_point2 / homo_point2[-1]
                    direction = image_point2[0: 2] - image_point1[0: 2]
                    cv2.line(img_front,
                             (int(image_point1[0] * scale), int(image_point1[1] * scale)),
                             (int(image_point2[0] * scale), int(image_point2[1] * scale)), color, 1)
                else:
                    # find the farther point
                    if homo_point1[-1] > homo_point2[-1]:
                        anchor_point = e
                        cos_vector = pu.compute_directional_vector(e, s)
                    else:
                        anchor_point = s
                        cos_vector = pu.compute_directional_vector(s, e)
                    jacobian = pu.compute_J(mat, anchor_point)
                    line_direction = jacobian.dot(cos_vector)
                    line_direction = line_direction / np.linalg.norm(line_direction, 2)
                    projected_anchor_point = mat.dot(anchor_point)
                    projected_anchor_point = projected_anchor_point / projected_anchor_point[-1]
                    mock_point = projected_anchor_point[:2] + line_direction * 10000
                    # print(projected_anchor_point, line_direction)
                    cv2.line(img_front,
                             (int(projected_anchor_point[0] * scale), int(projected_anchor_point[1] * scale)),
                             (int(mock_point[0] * scale), int(mock_point[1] * scale)), color, 1)

            # res = []
            # for s, e in zip(points_s, points_e):
            #     r = pu.project_line3D_by_two_points(s, e, mat)
            # points = points / (points[:, 2:3] + 1e-6)
            # return res

        def _to_rfu(points, h, w, mat):
            """pass
            """
            points = points.copy()
            points[0] = points[0] - w / 2
            points[1] = (h - 1) - points[1]
            points = points * 100 / h  # h=800 h_r=100
            # z = np.ones((1, 2), dtype=points.dtype) * (-0.364)  # 0.509  0.364
            # z[:, 1] = 1
            z = [-0.509, 1.0]
            points = np.concatenate((points, z)),
            points = np.dot(mat, np.array(points).T).T  # [n, 3]
            points = np.array(points)[0]
            points = points / (points[2] + 1e-6)
            return points

        ret = {}

        # print(index)
        if 0:  # origin list
            file_name = str(self.flist[index]).zfill(5)
            img = self.tfs(self.loader('{}/{}'.format(self.data_root, file_name)))
        else:  # anp dataset
            img_path = os.path.join(self.image_dir, self.images[index][1])
            # im = cv2.imread(img_path)
            img = Image.open(img_path).convert('RGB')  # 0: fl  1:f  2:fr
            img = self.tfs(img)

        cond_image = torch.randn_like(img)
        # black_image = np.zeros((1860,2880))
        s_h = self.image_size[0] / 1860
        s_w = self.image_size[1] / 2800

        # import cv2
        # cv2.imwrite('./debug.jpg', cond_image)

        # params
        w = 224

        taskid = self.images[index][1].split('/')[0]
        trans_server = self.transform_server_dict.get(taskid, None)
        # cams = ["spherical_left_forward", "onsemi_obstacle", "spherical_right_forward"]
        extrinsics = np.array(np.asarray(trans_server.get_static_transform('vehicle', 'onsemi_obstacle')[1][:3]),
                              dtype=np.float32)
        intrin_param = trans_server.intrinsics_dict['onsemi_obstacle'][0]  # disgarding D & image_shape

        # flag, novatel2vehicle = trans_server.get_static_transform('novatel', 'vehicle')
        # assert flag, 'can not find novatel2vehicle, please check your params dir extrinsics'

        ## proj
        # label
        item = self.labels[index]
        path, idx = item
        rv = {}

        seg_dict = {}
        reg_dict = {}

        name_list = ['seg', 'curb', 'stopline', 'crosswalk']
        seg_list = ['SEG', 'SEG_CURB', 'SEG_STOPLINE', 'SEG_CROSSWALK']
        reg_list = ['road_reg', 'road_reg_curb', 'road_reg_stopline', 'road_reg_crosswalk']
        # prompt_value_list = [-0.5,-1, 0.5,1]
        prompt_value_list = [0.25, 0.5, 0.75 , 1]
        prompt_value_list = [1,1,1,1]


        black_image = np.zeros((4, self.image_size[0], self.image_size[1]))  # 4 means element number

        for index, name in enumerate(name_list):  # road_seg/SEG_xxx

            # if index in [1,2,3]: continue
            cur_cls_mat_path = os.path.join(self.json_path, '%s/road_seg/' % path, '%s_%s.npz' % (idx, seg_list[index]))
            seg_dict[name] = np.load(cur_cls_mat_path)['arr_0']
            reg_dict[name] = np.load(os.path.join(self.json_path, '{}/{}'.format(path, reg_list[index]),
                                                  '%s_REG.npz' % idx))['arr_0']

            h, w = seg_dict[name].shape[-2:]
            row = torch.arange(0, h)
            col = torch.arange(0, w)
            ii, jj = torch.meshgrid(row, col)
            grid = torch.stack((ii, jj), 2)  # .repeat(1, 1, 1, 1)
            gt_offset_mask = (seg_dict[name] > 0)
            gt_reg = (np.array(grid) + np.transpose(reg_dict[name], [1, 2, 0]))[gt_offset_mask]

            # f_img = cv2.imread('front_007.jpg')

            mat = np.dot(intrin_param, extrinsics)  # current batch, front image

            for pt in gt_reg:
                pt = pt[::-1]  # w=160 h=400
                if pt[1] < 100: continue
                if pt[0] > 120 or pt[0] < 40: continue

                pt2d = _to_rfu(pt, 400, 160, mat)  # on 2880*1860
                try:
                    h = int(pt2d[1] * s_h)
                    w = int(pt2d[0] * s_w)
                    if h > 0 and w > 0:
                        black_image[index, int(pt2d[1] * s_h), int(pt2d[0] * s_w)] = prompt_value_list[index]
                except:
                    pass
                # cv2.circle(im, (int(pt2d[0]), int(pt2d[1])), 5, (0, 255, 0))

        # cv2.imwrite('front_007_atmp.jpg', black_image * 255)

        # cv2.imwrite('front_007_tmp.jpg', im)
        # _to_rfu_segment(gt_s, gt_e, h, w, mat, black_image, 1, (30, 0, 190))

        #######################################3
        ret['gt_image'] = img
        ret['cond_image'] = torch.from_numpy(black_image.astype(np.float32))  # cond_image
        ret['path'] = img_path  # file_name
        return ret

    def __getitem__(self, index):
        try:
            return self.get_instance(index)
        except:
            print('ret 0', index)
            return self.get_instance(0)

    def __len__(self):
        return len(self.images)
        # return len(self.flist)


class GenerateDatasetDeprecated(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, data_path='', image_size=[224, 224], loader=pil_loader,
                 local_mode=True):
        self.data_root = data_root

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

        ## anp
        self.labels = []
        self.images = []

        print('local_mode:', local_mode)
        if local_mode:
            list_name = data_flist  # './datasets/train.txt'
            self.json_path = '/media/lixiaofan/xf/bevdata/anpdataset_mini/label/'
            self.image_dir = data_path  # '/media/lixiaofan/xf/bevdata/anpdataset_mini/data/'
            params_dir = self.image_dir
        else:
            list_name = data_flist  # '/root/paddlejob/workspace/diff/datasets/train.txt'
            self.json_path = '/root/local_bev_data/anp/label_0113_fix_raw/'
            # self.image_dir = '/root/0000_mount/bev-data/road-structure/data/'
            self.image_dir = data_path  # '/root/local_storage_0001/bev-data/road-structure/data/'

            params_dir = self.image_dir

        with open(list_name, 'rt') as f:
            infos = f.readlines()
            infos = [d.split()[0] for d in infos]

        self.taskids = infos  # all taskids in curr dataset

        global_ego_server_cnt = 0
        for info_dir in infos:
            image_list = sorted(glob(os.path.join(self.json_path, info_dir, '*/images.list')))

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
                    if 0:  # lukou
                        path, index = label_name
                        cur_cls_mat_path = os.path.join(self.json_path, '%s/road_seg/' % path,
                                                        '%s_%s.npz' % (idx, 'SEG_STOPLINE'))
                        cur_cls_mat = np.load(cur_cls_mat_path)['arr_0']
                        if not np.any(cur_cls_mat == 1):
                            continue
                    self.labels.append(label_name)
                    # self.ts.append(ts)

                    # images
                    img_scene_names = line.strip().split('\t')
                    img_names = img_scene_names[:3]
                    if 1:  # cfg.DATASETS.UNDISTORTED:
                        for i in range(len(img_names)):
                            img_names[i] = img_names[i][:-4] + '_undistorted.jpg'
                    img_names = [os.path.join(info_dir, 'image_data_resized', rec_name, img_name) for img_name in
                                 img_names]
                    self.images.append(img_names)

        self.transform_server_dict = {taskid: transformer.TransformServer(
            os.path.join(params_dir, taskid, 'params'), False) for taskid in self.taskids}
        # print(self.transform_server_dict)

    def get_instance(self, index):
        def _to_rfu_segment(points_s, points_e, h, w, mat, img_front, scale, color):
            """ pass
            """
            '''start'''
            points_s = points_s.copy()
            points_s[:, 0] = points_s[:, 0] - w / 2
            points_s[:, 1] = (h - 1) - points_s[:, 1]
            points_s = points_s * 100 / h  # h=800 h_r=100
            z = np.ones((points_s.shape[0], 2), dtype=points_s.dtype) * (-0.364)  # 0.509  0.364
            z[:, 1] = 1
            points_s = np.concatenate((points_s, z), 1)
            # points_s = np.dot(mat, points_s.T).T  # [n, 3]
            '''end'''
            points_e = points_e.copy()
            points_e[:, 0] = points_e[:, 0] - w / 2
            points_e[:, 1] = (h - 1) - points_e[:, 1]
            points_e = points_e * 100 / h  # h=800 h_r=100
            z = np.ones((points_e.shape[0], 2), dtype=points_e.dtype) * (-0.364)  # 0.509  0.364
            z[:, 1] = 1
            points_e = np.concatenate((points_e, z), 1)
            # points_e = np.dot(mat, points_e.T).T  # [n, 3]
            '''segment'''
            for s, e in zip(points_s, points_e):
                homo_point1 = mat.dot(e)
                homo_point2 = mat.dot(s)
                thred = 0
                if homo_point1[-1] < thred and homo_point2[-1] < thred:
                    continue
                if homo_point1[-1] > 100 and homo_point2[-1] > 100:
                    continue
                if homo_point1[-1] > thred and homo_point2[-1] > thred:
                    image_point1 = homo_point1 / homo_point1[-1]
                    image_point2 = homo_point2 / homo_point2[-1]
                    direction = image_point2[0: 2] - image_point1[0: 2]
                    cv2.line(img_front,
                             (int(image_point1[0] * scale), int(image_point1[1] * scale)),
                             (int(image_point2[0] * scale), int(image_point2[1] * scale)), color, 1)
                else:
                    # find the farther point
                    if homo_point1[-1] > homo_point2[-1]:
                        anchor_point = e
                        cos_vector = pu.compute_directional_vector(e, s)
                    else:
                        anchor_point = s
                        cos_vector = pu.compute_directional_vector(s, e)
                    jacobian = pu.compute_J(mat, anchor_point)
                    line_direction = jacobian.dot(cos_vector)
                    line_direction = line_direction / np.linalg.norm(line_direction, 2)
                    projected_anchor_point = mat.dot(anchor_point)
                    projected_anchor_point = projected_anchor_point / projected_anchor_point[-1]
                    mock_point = projected_anchor_point[:2] + line_direction * 10000
                    # print(projected_anchor_point, line_direction)
                    cv2.line(img_front,
                             (int(projected_anchor_point[0] * scale), int(projected_anchor_point[1] * scale)),
                             (int(mock_point[0] * scale), int(mock_point[1] * scale)), color, 1)

            # res = []
            # for s, e in zip(points_s, points_e):
            #     r = pu.project_line3D_by_two_points(s, e, mat)
            # points = points / (points[:, 2:3] + 1e-6)
            # return res

        def _to_rfu(points, h, w, mat):
            """pass
            """
            points = points.copy()
            points[0] = points[0] - w / 2
            points[1] = (h - 1) - points[1]
            points = points * 100 / h  # h=800 h_r=100
            # z = np.ones((1, 2), dtype=points.dtype) * (-0.364)  # 0.509  0.364
            # z[:, 1] = 1
            z = [-0.509, 1.0]
            points = np.concatenate((points, z)),
            points = np.dot(mat, np.array(points).T).T  # [n, 3]
            points = np.array(points)[0]
            points = points / (points[2] + 1e-6)
            return points

        ret = {}

        # print(index)
        if 0:  # origin list
            file_name = str(self.flist[index]).zfill(5)
            img = self.tfs(self.loader('{}/{}'.format(self.data_root, file_name)))
        else:  # anp dataset
            img_path = os.path.join(self.image_dir, self.images[index][1])
            # im = cv2.imread(img_path)
            img = Image.open(img_path).convert('RGB')  # 0: fl  1:f  2:fr
            img = self.tfs(img)

        cond_image = torch.randn_like(img)
        # black_image = np.zeros((1860,2880))
        black_image = np.zeros((self.image_size[0], self.image_size[1]))
        s_h = self.image_size[0] / 1860
        s_w = self.image_size[1] / 2800

        # import cv2
        # cv2.imwrite('./debug.jpg', cond_image)

        # params
        w = 224

        taskid = self.images[index][1].split('/')[0]
        trans_server = self.transform_server_dict.get(taskid, None)
        # cams = ["spherical_left_forward", "onsemi_obstacle", "spherical_right_forward"]
        extrinsics = np.array(np.asarray(trans_server.get_static_transform('vehicle', 'onsemi_obstacle')[1][:3]),
                              dtype=np.float32)
        intrin_param = trans_server.intrinsics_dict['onsemi_obstacle'][0]  # disgarding D & image_shape

        # flag, novatel2vehicle = trans_server.get_static_transform('novatel', 'vehicle')
        # assert flag, 'can not find novatel2vehicle, please check your params dir extrinsics'

        ## proj
        # label
        item = self.labels[index]
        path, idx = item
        rv = {}

        seg_dict = {}
        reg_dict = {}

        name_list = ['seg', 'curb', 'stopline', 'crosswalk']
        seg_list = ['SEG', 'SEG_CURB', 'SEG_STOPLINE', 'SEG_CROSSWALK']
        reg_list = ['road_reg', 'road_reg_curb', 'road_reg_stopline', 'road_reg_crosswalk']
        # prompt_value_list = [-0.5,-1, 0.5,1]
        prompt_value_list = [0.25, 0.5, 0.75 , 1]

        for index, name in enumerate(name_list):  # road_seg/SEG_xxx
            # if index in [1,2,3]: continue
            cur_cls_mat_path = os.path.join(self.json_path, '%s/road_seg/' % path, '%s_%s.npz' % (idx, seg_list[index]))
            seg_dict[name] = np.load(cur_cls_mat_path)['arr_0']
            reg_dict[name] = np.load(os.path.join(self.json_path, '{}/{}'.format(path, reg_list[index]),
                                                  '%s_REG.npz' % idx))['arr_0']

            h, w = seg_dict[name].shape[-2:]
            row = torch.arange(0, h)
            col = torch.arange(0, w)
            ii, jj = torch.meshgrid(row, col)
            grid = torch.stack((ii, jj), 2)  # .repeat(1, 1, 1, 1)
            gt_offset_mask = (seg_dict[name] > 0)
            gt_reg = (np.array(grid) + np.transpose(reg_dict[name], [1, 2, 0]))[gt_offset_mask]

            # f_img = cv2.imread('front_007.jpg')

            mat = np.dot(intrin_param, extrinsics)  # current batch, front image

            for pt in gt_reg:
                pt = pt[::-1]  # w=160 h=400
                if pt[1] < 100: continue
                if pt[0] > 120 or pt[0] < 40: continue

                pt2d = _to_rfu(pt, 400, 160, mat)  # on 2880*1860
                try:
                    h = int(pt2d[1] * s_h)
                    w = int(pt2d[0] * s_w)
                    if h > 0 and w > 0:
                        black_image[int(pt2d[1] * s_h), int(pt2d[0] * s_w)] = prompt_value_list[index]
                except:
                    pass
                # cv2.circle(im, (int(pt2d[0]), int(pt2d[1])), 5, (0, 255, 0))

        # cv2.imwrite('front_007_atmp.jpg', black_image * 255)

        # cv2.imwrite('front_007_tmp.jpg', im)
        # _to_rfu_segment(gt_s, gt_e, h, w, mat, black_image, 1, (30, 0, 190))

        #######################################3
        ret['gt_image'] = img
        ret['cond_image'] = torch.from_numpy(
            np.tile(black_image[np.newaxis, :, :], (3, 1, 1)).astype(np.float32))  # cond_image
        ret['path'] = img_path  # file_name
        return ret

    def __getitem__(self, index):
        try:
            return self.get_instance(index)
        except:
            print('ret 0', index)
            return self.get_instance(0)

    def __len__(self):
        return len(self.images)
        # return len(self.flist)
