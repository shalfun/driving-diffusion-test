"""
borrow from dueye
"""

import pyquaternion
import numpy as np

class BBox:
    """BBox"""
    def __init__(self, x1, y1, x2, y2):
        """init"""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def upper_left(self):
        """pass"""
        return self.x1, self.y1

    def lower_right(self):
        """pass"""
        return self.x2, self.y2

    def center(self):
        """pass"""
        return self.x1 + self.w() / 2.0, self.y1 + self.h() / 2.0

    def w(self):
        """pass"""
        return self.x2 - self.x1

    def h(self):
        """pass"""
        return self.y2 - self.y1

    def shape(self):
        """pass"""
        return self.w(), self.h()

    def join(self, box):
        """
        join two box, get max area
        """
        self.x1 = min(self.x1, box.x1)
        self.x2 = max(self.x2, box.x2)
        self.y1 = min(self.y1, box.y1)
        self.y2 = max(self.y2, box.y2)

    def __str__(self):
        """pass"""
        return '{} {} {} {}'.format(self.x1, self.y1, self.x2, self.y2)


class TData:
    """
        Utility class to load data.
    """

    def __init__(self, frame=-1, obj_type="unset", truncation=-1, occlusion=-1, \
                 obs_angle=-10, x1=-1, y1=-1, x2=-1, y2=-1, w=-1, h=-1, l=-1, \
                 X=-1000, Y=-1000, Z=-1000, yaw=-10, score=-1000, track_id=-1):
        """
            Constructor, initializes the object given the parameters.
        """

        # init object data
        self.frame = frame
        self.track_id = track_id
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.yaw = yaw
        self.score = score
        self.ignored = False
        self.valid = False
        self.tracker = -1
        self.tracker_local_track_id = -1
        self.camera_name = ""
        self.group_id = -1
        self.local_track_id = -1
        self.local_frame_id = -1
        self.vx = -1
        self.vy = -1
        self.vz = -1
        self.local_X = 0.0
        self.local_Y = 0.0
        self.local_Z = 0.0
        self.ids = False
        self.track_confidence = -1000.0
        self.distance_to_maincar = -1000.0
        self.euler_x = 0.0
        self.euler_y = 0.0
        self.euler_z = 0.0
        self.local_euler_x = 0.0
        self.local_euler_y = 0.0
        self.local_euler_z = 0.0
        self.image_name = ""
        self.image_timestamp = 0.0
        self.is_on_road = -1 # -1 means not, 1 means on road, -2 means unknown
        self.brake_light = -1 # -1 means unset, 0 means unknown, 1 means on, 2 means off, 3 means label-unknown
        self.left_turn_light = -1 # -1 means unset, 0 means unknown, 1 means on, 2 means off
        self.right_turn_light = -1 # -1 means unset, 0 means unknown, 1 means on, 2 means off
        self.brake_light_on_prob = -1 
        self.left_turn_light_on_prob = -1 
        self.right_turn_light_on_prob = -1 
        self.road_state = -1 #corresponding offline
        self.seq = -1

    def __str__(self):
        """
            Print read data.
        """

        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())

    def get_line(self):
        """To be completed"""
        line = "%d %d %d %f %f %f %f %d " % (self.frame, self.track_id, self.local_track_id, self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.tracker)
        return line

    def get_simulation_format_line(self):
        """ return the line by the simulation format"""
        line = \
            "%d %d %s %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s %f %f %f %f %f %f %f %f %f %f" % (
                self.frame,
                self.track_id,
                self.obj_type,
                self.truncation,
                self.occlusion,
                self.obs_angle,
                self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.h,
                self.w,
                self.l,
                self.X,
                self.Y,
                self.Z,
                self.yaw,
                self.score,
                self.vx,
                self.vy,
                self.vz,
                self.camera_name,
                self.local_X,
                self.local_Y,
                self.local_Z,
                self.euler_x,
                self.euler_y,
                self.euler_z,
                self.local_euler_x,
                self.local_euler_y,
                self.local_euler_z,
                self.image_timestamp)
        return line

    def get_label_format_line(self):
        """ return the line by the offline format"""
        line = \
            "%d %d %s %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s %f %f %f %d %d %d %f %d %d %d" % (
                self.frame,
                self.track_id,
                self.obj_type,
                self.truncation,
                self.occlusion,
                self.obs_angle,
                self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.h,
                self.w,
                self.l,
                self.X,
                self.Y,
                self.Z,
                self.yaw,
                self.score,
                self.vx,
                self.vy,
                self.vz,
                self.camera_name,
                self.local_X,
                self.local_Y,
                self.local_Z,
                self.local_track_id,
                self.local_frame_id,
                self.road_state,
                self.track_confidence,
                self.brake_light,
                self.left_turn_light,
                self.right_turn_light)
        return line

class Transform:
    """pass"""
    def __init__(self, ts=-1, tx=-1, ty=-1, tz=-1, qx=-1, qy=-1, qz=-1, qw=-1):
        """pass"""
        self.ts = ts
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw

    def from_matrix44(self, trans, ts):
        """pass"""
        self.tx, self.ty, self.tz = trans[0, 3], trans[1, 3], trans[2, 3]
        quat = pyquaternion.Quaternion(matrix=trans)
        self.qw, self.qx, self.qy, self.qz = quat.q
        self.ts = ts
        return self

    def covert_quaternion2matrix(self):
        """pass"""
        transform = np.zeros((4, 4))
        q = pyquaternion.Quaternion(self.qw,
                                    self.qx,
                                    self.qy,
                                    self.qz)

        transform[:3, :3] = q.rotation_matrix
        transform[0, 3] = self.tx
        transform[1, 3] = self.ty
        transform[2, 3] = self.tz
        transform[3, 3] = 1
        return transform

    def slerp(self, trans_rhs, ratio, ts):
        """pass"""
        qs = pyquaternion.Quaternion(self.qw, self.qx, self.qy, self.qz)
        qe = pyquaternion.Quaternion(trans_rhs.qw, trans_rhs.qx, trans_rhs.qy, trans_rhs.qz)
        slerp_q = pyquaternion.Quaternion.slerp(qs, qe, ratio)
        x = self.tx * (1.0 - ratio) + trans_rhs.tx * ratio
        y = self.ty * (1.0 - ratio) + trans_rhs.ty * ratio
        z = self.tz * (1.0 - ratio) + trans_rhs.tz * ratio
        qw, qx, qy, qz = slerp_q.q
        return Transform(ts, x, y, z, qx, qy, qz, qw)
    
    def covert_quaternion2position(self):
        """pass"""
        return self.tx, self.ty, self.tz
    
    def covert_quaternion2yaw_pitch_roll(self):
        """pass"""
        position = np.zeros((3))
        q = pyquaternion.Quaternion(self.qw,
                                    self.qx,
                                    self.qy,
                                    self.qz)
        return q.yaw_pitch_roll

    def __str__(self):
        """
            Print read data.
        """

        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())

class VisualFrame:
    """frame info for dubug visual"""

    def __init__(self, img=None, frame_info=None, objects_2d=None, objects_3d=None,
                 novatel2world=None):
        """pass"""
        self.img = img
        self.frame_info = frame_info
        self.objects_2d = objects_2d
        self.objects_3d = objects_3d
        self.novatel2world = novatel2world


class FrameInfo:
    """info to description one frame in a seq."""

    def __init__(self, global_frame_id=-1, local_frame_id=-1,
                 camera_name="", image_name="", group=-1, seq="0000", timestamp_second=0.0):
        """pass"""
        self.global_frame_id = global_frame_id
        self.local_frame_id = local_frame_id
        self.camera_name = camera_name
        self.image_name = image_name
        self.group = group
        self.timestamp_second = timestamp_second
        self.seq = seq

    def __str__(self):
        """
            Print read data.
        """

        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())


class SeqListInfo:
    """pass"""
    def __init__(self, seq_list_path, main_camera_name="onsemi_obstacle"):
        """pass"""
        self.seq_list_path = seq_list_path
        self.seq_name = self.seq_list_path.split("/")[-1].split(".txt")[0]
        self.main_camera_name = main_camera_name
        self.read_list(seq_list_path)

    def read_list(self, seq_list_path):
        """pass"""
        self.frame_num = 0
        self.global_frame_id_info_map = {}
        self.camera_name_local_frame_id_map = {}
        self.info_list = []
        group = 0
        with open(seq_list_path) as infile:
            for line in infile:
                fields = line.strip().split("/")
                camera_name = str(fields[0])
                image_name = str(fields[1])
                if camera_name not in self.camera_name_local_frame_id_map:
                    self.camera_name_local_frame_id_map[camera_name] = 0
                else:
                    self.camera_name_local_frame_id_map[camera_name] += 1
                self.frame_num += 1
                frame_info = FrameInfo(
                    self.frame_num - 1, self.camera_name_local_frame_id_map[camera_name],
                    camera_name, image_name, group, self.seq_name, float(image_name)*1e-9)
                self.global_frame_id_info_map[self.frame_num - 1] = frame_info
                self.info_list.append(frame_info)
                if camera_name == self.main_camera_name:
                    group += 1

    def get_info_by_camera(self, camera_name):
        """ get list info by camera_name.

        Args:
            camera_name: str

        Returns:
            info_list: each info is FrameInfo
        """
        info_list = []
        for i, info in enumerate(self.info_list):
            if info.camera_name == camera_name:
                info_list.append(info)
        return info_list

    def get_info_by_global_frame_id(self, global_frame_id):
        """pass"""
        if global_frame_id not in self.global_frame_id_info_map:
            print("seq does not have frame_id:", global_frame_id)
            assert (False)
        return self.global_frame_id_info_map[global_frame_id]

    def get_info_list(self):
        """pass"""
        return self.info_list


class Trajectory(object):
    """pass"""
    def __init__(self, track_id):
        """pass"""
        self.track_id = track_id
        self.data_list = []
        self.associate_data_list = []

    def add_data(self, data):
        """pass"""
        self.data_list.append(data)
        self.associate_data_list.append(None)

    def add_association_data(self, global_frame_id, association_data):
        """pass"""
        tracked_frame = [data.frame for data in self.data_list]
        set_tracked_frame = set(tracked_frame)
        assert (len(tracked_frame) == len(set_tracked_frame))
        index = tracked_frame.index(data.frame)
        assert (index < len(self.associate_data_list))
        self.data_list[index].tracker = association_data.track_id
        self.associate_data_list[index] = association_data

    def add_association_pair(self, data, association_data):
        """pass"""
        data.tracker = association_data.track_id
        association_data.track = data.track_id
        self.data_list.append(data)
        self.associate_data_list.append(association_data)

subtype_map = {
    0: "UNKNOWN",
    1: "UNKNOWN_MOVABLE",
    2: "UNKNOWN_UNMOVABLE", 
    3: "CAR",
    4: "VAN",
    5: "TRUCK",
    6: "BUS",
    7: "CYCLIST",
    8: "MOTORCYCLIST",
    9: "TRICYCLIST",
    10: "PEDESTRIAN",
    11: "TRAFFICCONE",
    12: "CAT",
    13: "DOG",
    14: "TRAFFICCONE_DOWN",
    15: "SAFETY_TRIANGLE",
    16: "SECURITY_ARM_BARRIER",
    17: "CONFUSED"
}

subtype2id_map = {
    'UNKNOWN': 0,
    'UNKNOWN_MOVABLE': 1,
    'UNKNOWN_UNMOVABLE': 2,
    'CAR': 3,
    "VAN": 4,
    "TRUCK": 5,
    "BUS": 6,
    "CYCLIST": 7,
    "MOTORCYCLIST": 8,
    "TRICYCLIST": 9,
    "PEDESTRIAN": 10,
    "TRAFFICCONE": 11,
    "CAT": 12,
    "DOG": 13,
    "TRAFFICCONE_DOWN": 14,
    "SAFETY_TRIANGLE": 15,
    "SECURITY_ARM_BARRIER": 16,
    "CONFUSED": 17
}



type_map = {
  0: "UNKNOWN",
  1: "UNKNOWN_MOVABLE",
  2: "UNKNOWN_UNMOVABLE",
  3: "PEDESTRIAN",
  4: "BICYCLE",
  5: "VEHICLE",
  6: "MAX_OBJECT_TYPE"
}
