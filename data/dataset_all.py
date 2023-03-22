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


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[64, 64], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0, 2) < 1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class GenerateDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, data_path='', image_size=[224, 224], loader=pil_loader, local_mode=True):
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
            list_name = data_flist  #'./datasets/train.txt'
            self.json_path = '/media/lixiaofan/xf/bevdata/anpdataset_mini/label/'
            self.image_dir = data_path #'/media/lixiaofan/xf/bevdata/anpdataset_mini/data/'
            params_dir = self.image_dir
        else:
            list_name = data_flist# '/root/paddlejob/workspace/diff/datasets/train.txt'
            self.json_path = '/root/local_bev_data/anp/label_1207/'
            # self.image_dir = '/root/0000_mount/bev-data/road-structure/data/'
            self.image_dir = data_path # '/root/local_storage_0001/bev-data/road-structure/data/'


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
                    if 1:#cfg.DATASETS.UNDISTORTED:
                        for i in range(len(img_names)):
                            img_names[i] = img_names[i][:-4] + '_undistorted.jpg'
                    img_names = [os.path.join(info_dir, 'image_data_resized', rec_name, img_name) for img_name in
                                 img_names]
                    self.images.append(img_names)

        self.transform_server_dict = {taskid: transformer.TransformServer(
            os.path.join(params_dir, taskid, 'params'), False) for taskid in self.taskids}
        # print(self.transform_server_dict)

    def __getitem__(self, index):
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
        cur_cls_mat_path = os.path.join(self.json_path, '%s/road_seg/' % path, '%s_%s.npz' % (idx, 'SEG'))
        gt_seg = np.load(cur_cls_mat_path)['arr_0']

        rv['road_reg'] = np.load(os.path.join(self.json_path, '%s/road_reg' % path,
                                              '%s_REG.npz' % idx))['arr_0']

        # gt_lines = rv['road_reg'][gt_batch_idx.long(), gt_line_idx.long()].detach().cpu().numpy() * scale
        h, w = gt_seg.shape[-2:]
        cv2.imwrite('front_s.jpg', gt_seg * 255)
        row = torch.arange(0, h)
        col = torch.arange(0, w)
        ii, jj = torch.meshgrid(row, col)
        grid = torch.stack((ii, jj), 2)  # .repeat(1, 1, 1, 1)
        gt_offset_mask = (gt_seg > 0)
        gt_reg = (np.array(grid) + np.transpose(rv['road_reg'], [1, 2, 0]))[gt_offset_mask]

        # f_img = cv2.imread('front_007.jpg')

        mat = np.dot(intrin_param, extrinsics)  # current batch, front image

        for pt in gt_reg:
            pt = pt[::-1]  # w=160 h=400
            if pt[1] < 100: continue
            if pt[0] > 120 or pt[0] < 40: continue

            pt2d = _to_rfu(pt, 400, 160, mat) # on 2880*1860
            try:
                h = int(pt2d[1]*s_h)
                w = int(pt2d[0]*s_w)
                if h > 0 and w > 0:
                    black_image[int(pt2d[1]*s_h), int(pt2d[0]*s_w)] = 1

            except:
                pass
            # cv2.circle(im, (int(pt2d[0]), int(pt2d[1])), 5, (0, 255, 0))

        # cv2.imwrite('front_007_atmp.jpg', black_image * 255)

        # cv2.imwrite('front_007_tmp.jpg', im)
        # _to_rfu_segment(gt_s, gt_e, h, w, mat, black_image, 1, (30, 0, 190))

        #######################################3
        ret['gt_image'] = img
        ret['cond_image'] = torch.from_numpy(np.tile(black_image[np.newaxis,:,:],(3,1,1)).astype(np.float32)) #cond_image
        ret['path'] = img_path #file_name
        return ret

    def __len__(self):
        return len(self.images)
        # return len(self.flist)
