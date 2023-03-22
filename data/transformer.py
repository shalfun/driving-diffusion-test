"""
borow from https://console.cloud.baidu-int.com/devops/icode/repos/baidu/idt/dueye/blob/andes_lane/dueye/common/transformer.py
not support tf-related
"""
import os
import logging

import pyquaternion
import yaml
import numpy as np

from .object import Transform


class GlobalEgoPoseServer(object):
    """for query global ego_pose from tf.txt"""
    def __init__(self, tf_path):
        assert(os.path.exists(tf_path))
        self.tf = []
        with open(tf_path) as f:
            for line in f.readlines():
                tf = Transform()
                data = line.strip().split()
                tf.filename = int(data[0])
                tf.ts = float(data[0]) * 1e-9
                tf.tx = float(data[1])
                tf.ty = float(data[2])
                tf.tz = float(data[3])
                tf.qx = float(data[4])
                tf.qy = float(data[5])
                tf.qz = float(data[6])
                tf.qw = float(data[7])
                self.tf.append(tf)

    def get_novatel_2_world_trans(self, ts):
        """novatel to world transform"""
        for i in range(len(self.tf) - 1):
            if self.tf[i].ts <= ts < self.tf[i + 1].ts:
                transform = self.tf[i].covert_quaternion2matrix()
                return True, transform
        # for border, give 1 second
        if self.tf[0].ts - 1 < ts < self.tf[0].ts:
            transform = self.tf[0].covert_quaternion2matrix()
            return True, transform
        num = len(self.tf)
        if self.tf[num - 1].ts + 1 > ts >= self.tf[num - 1].ts - 0.01:
            transform = self.tf[num - 1].covert_quaternion2matrix()
            return True, transform
        print('tf not found')
        return False, np.zeros((4, 4), dtype=np.float32)


class TransformServer(object):
    """pass"""
    def __init__(self, params_dir, strict_mode=False):
        """pass"""
        self.vertices = []
        self.edges = []
        self.intrinsics_dict = {}
        self.enable = os.path.exists(params_dir)
        self.strict_mode = strict_mode
        self.hesai_height = None
        self.origin_height = None
        if self.enable:
            self.load_params(params_dir)
            self.init_z()
        else:
            self.load_fake_params()

    def init_z(self):
        """init ground z @ vehicle_coordinate"""
        # NOTES here we use vehicle coords and use the rough height
        if "hesai40" in self.vertices:
            status, trans2vehicle = self.get_static_transform('hesai40', 'vehicle')
            status, trans2novatel = self.get_static_transform('hesai40', 'novatel')
        elif "hesai90" in self.vertices:
            status, trans2vehicle = self.get_static_transform('hesai90', 'vehicle')
            status, trans2novatel = self.get_static_transform('hesai90', 'novatel')
        else:
            raise Exception("no hesai40/hesai90 found in params, init height failed")

        if self.hesai_height is not None:
            # origin_height means ground_z in vehicle_coords
            # novatel_height means ground_z in novatel_coords
            self.origin_height = trans2vehicle.astype(np.float32)[2, 3] - self.hesai_height
            self.novatel_height = trans2novatel.astype(np.float32)[2, 3] - self.hesai_height

    def read_extrinsics(self, filename):
        """pass"""
        if not self.enable:
            return '', '', np.eye(4)

        ex = yaml.load(open(filename), Loader=yaml.SafeLoader)
        r = ex['transform']['rotation']
        t = ex['transform']['translation']
        q = pyquaternion.Quaternion(r['w'], r['x'], r['y'], r['z'])
        m = q.rotation_matrix
        m = np.matrix(m).reshape((3, 3))
        t = np.matrix([t['x'], t['y'], t['z'], 1]).T
        trans = np.matrix(np.zeros((4, 4)))
        trans[:3, :3] = m
        trans[:, 3] = t
        child_frame_id, frame_id = ex['child_frame_id'], ex["header"]['frame_id']
        # NOTES *For L4 dataset* : correct frame_id: imu-->novatel, otherwise can not get vehicle2hesai
        if 'vehicle_imu_extrinsics' in filename and frame_id == 'imu':
            frame_id = 'novatel'
        return child_frame_id, frame_id, trans

    def read_intrinsics(self, filename):
        """pass"""
        if not self.enable:
            return np.eye(3), np.ones(12), (1860, 2880)

        intrinsic = yaml.load(open(filename), Loader=yaml.SafeLoader)
        k = np.matrix(intrinsic['K'])
        k = k.reshape((3, 3))
        d = np.array(intrinsic['D'])
        width = intrinsic['width']
        height = intrinsic['height']
        shape = (height, width)
        return k, d, shape

    def load_params(self, params_dir):
        """pass"""
        for fname in os.listdir(params_dir):
            full_path = os.path.join(params_dir, fname)
            # if 'extrinsics' in fname:
            if full_path.endswith('extrinsics.yaml'):
                try:
                    child_frame_id, frame_id, trans = self.read_extrinsics(full_path)
                    self.add_transform(child_frame_id, frame_id, trans)
                except:
                    continue
            elif 'intrinsics' in fname:
                try:
                    intri_param = self.read_intrinsics(full_path)
                    cam_name = fname[:-16]
                    self.intrinsics_dict[cam_name] = intri_param
                except:
                    continue
            elif 'hesai40_height' in fname or 'hesai90_height' in fname:
                with open(full_path, 'rt') as f:
                    hesai_param = yaml.load(f, Loader=yaml.SafeLoader)
                self.hesai_height = hesai_param['vehicle']['parameters']['height']
            else:
                continue

    def load_fake_params(self):
        """pass"""
        pass

    def add_transform(self, child_frame_id, frame_id, transform):
        """pass"""
        self.vertices.append(child_frame_id)
        self.vertices.append(frame_id)
        self.edges.append((child_frame_id, frame_id, transform))
        self.edges.append((frame_id, child_frame_id, np.linalg.inv(transform)))

    def find_transform(self, child_frame_id, frame_id, visited):
        """pass"""
        visited[child_frame_id] = True
        for e in self.edges:
            if e[0] != child_frame_id or visited[e[1]]:
                continue
            # print 'from ', e[0], ' to ', e[1]

            if e[1] == frame_id:
                return True, e[2]

            bfound, transform = self.find_transform(e[1], frame_id, visited)
            if bfound:
                loc_transform = transform * e[2]
                return True, loc_transform

        return False, np.identity(4)

    def get_static_transform(self, child_frame_id, frame_id):
        """pass"""
        visit = {}
        for v in self.vertices:
            visit[v] = False
        if child_frame_id not in self.vertices:
            logging.warning('No vert {}'.format(child_frame_id))
            raise Exception('No vert {}'.format(child_frame_id))
        if frame_id not in self.vertices:
            logging.warning('No vert {}'.format(frame_id))
            raise Exception('No vert {}'.format(frame_id))
        bfound, transform = self.find_transform(child_frame_id, frame_id, visit)
        if not bfound:
            logging.error('Failed to find transform from %s to %s' % (child_frame_id, frame_id))
            if self.strict_mode:
                raise(ValueError('please check your input params'))
        return bfound, transform

def trans_point(pts, trans, height):
    """transform point given transform matrix
    pt: np.array, shape: [n, 2], only consider x, y
    trans: np.array, shape: [4, 4]
    """
    n, _ = pts.shape
    z = np.ones((n, 2), dtype=pts.dtype) * (-height)
    z[:, 1] = 1
    pts = np.concatenate((pts, z), 1)
    pts = np.dot(trans[:3], pts.T).T  # [n, 3]
    return pts[:, :2]

def trans_point_torch(pts, trans):
    """transform point given transform matrix
    pt: np.array, shape: [n, 2], only consider x, y
    trans: np.array, shape: [4, 4]
    """
    import torch
    n, _ = pts.shape
    z = torch.ones((n, 2), dtype=pts.dtype) * (-0.509)
    z = z.to(pts.device)
    z[:, 1] = 1
    pts = torch.cat((pts, z), 1)
    pts = torch.matmul(trans[:3], pts.T).T # [n, 3]
    return pts[:, :2]