#!/usr/bin/env python
# coding: utf-8
"""
   File Name: pyape/data/obstacle/io.py
      Author: Li Xiaofan
      E-mail: lixiaofan03@baidu.com
  Created on: 2022-01-20 11:12:27 +0800
 Description:
"""

import cv2
from math import cos, sin, pi, atan2, asin
import numpy as np


def generate_Pmat(K, x, y, z, T):
    """
        inputs: roll, pitch, yaw (x, y, z)
        return: K[R|T]
    """
    T = np.array(T).reshape(1, 3)
    r11, r12, r13 = cos(z) * cos(y), cos(z) * sin(y) * sin(x) - sin(z) * cos(x), cos(z) * sin(y) * cos(x) + sin(
        z) * sin(x)
    r21, r22, r23 = sin(z) * cos(y), sin(z) * sin(y) * sin(x) + cos(z) * cos(x), sin(z) * sin(y) * cos(x) - cos(
        z) * sin(x)
    r31, r32, r33 = -sin(y), cos(y) * sin(x), cos(y) * cos(x)
    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    return K.dot(np.hstack([R, T.T]))


def project_points(Pmat, points_3D):
    """
        inputs:
            Pmat: projection mat 3 X 4
            points_3D: 3D points N X 3
        return:
            projected 2D points
    """
    nb_points = points_3D.shape[0]
    homo_points = np.hstack([points_3D, np.ones([nb_points, 1])])
    projected_points = Pmat.dot(homo_points.T)
    image_points = projected_points / (projected_points[-1, :] + 1E-20)
    return image_points.T


def compute_J(Pmat, point_3d):
    """
    compute_J:
        compute jacobian matrix of Pmat
        this func only works for dz = 0
    """
    assert Pmat.shape == (3, 4)
    assert point_3d.shape[0] == 4
    vx, vy, vz = Pmat.dot(point_3d)
    j_11 = Pmat[0, 0] / vz - Pmat[2, 0] * vx / (vz ** 2)
    j_12 = Pmat[0, 1] / vz - Pmat[2, 1] * vx / (vz ** 2)
    j_21 = Pmat[1, 0] / vz - Pmat[2, 0] * vy / (vz ** 2)
    j_22 = Pmat[1, 1] / vz - Pmat[2, 1] * vy / (vz ** 2)
    ret = np.zeros([2, 3])
    ret[0, 0] = j_11
    ret[0, 1] = j_12
    ret[1, 0] = j_21
    ret[1, 1] = j_22
    return ret


def compute_J_single(Pmat, point_3d):
    """
    compute_J:
        compute jacobian matrix of Pmat
        this func only works for dz = 0
    """
    assert Pmat.shape == (3, 3)

    vx, vy, vz = Pmat.dot(point_3d)
    j_11 = Pmat[0, 0] / vz - Pmat[2, 0] * vx / (vz ** 2)
    j_12 = Pmat[0, 1] / vz - Pmat[2, 1] * vx / (vz ** 2)
    j_13 = Pmat[0, 2] / vz - Pmat[2, 2] * vx / (vz ** 2)

    j_21 = Pmat[1, 0] / vz - Pmat[2, 0] * vy / (vz ** 2)
    j_22 = Pmat[1, 1] / vz - Pmat[2, 1] * vy / (vz ** 2)
    j_23 = Pmat[1, 2] / vz - Pmat[2, 2] * vy / (vz ** 2)

    ret = np.zeros([2, 3])
    ret[0, 0] = j_11
    ret[0, 1] = j_12
    ret[0, 2] = j_13
    ret[1, 0] = j_21
    ret[1, 1] = j_22
    ret[1, 2] = j_23
    return ret


def compute_directional_vector(point1, point2):
    """
    compute_directional_vector: from point1 to point2
        line: alpha * x + beta * y + gamma * z
        where alpha ** 2 + beta ** 2 + gamma ** 2 = 1
    return:
        [alpha, beta, gamma]
    """
    point1_3D = point1[0: 3]
    point2_3D = point2[0: 3]
    delta = point2_3D - point1_3D
    delta = delta / (np.linalg.norm(delta) + 1E-20)
    return delta


def project_line3D_by_two_points(point1, point2, Pmat, thred=0):
    """
    project_line3D_by_two_points:
        point1, point2: 4 X 1
        Pmat: 3 X 4
        thred: minium distance from the principle plane
    return:
        1. None - if projected two points are all < thred
        2. anchor_point, direction ([del_u, del_v]) - if one point > thred
        3. image_point1, image_point2, direction - if both points > thred
    """
    homo_point1 = Pmat.dot(point1)
    homo_point2 = Pmat.dot(point2)
    # print(homo_point2)  # [-375999.9999973  -372002.93856138    -949.99999999]
    if homo_point1[-1] < thred and homo_point2[-1] < thred:
        return None
    if homo_point1[-1] > 100 and homo_point2[-1] > 100:
        return None
    if homo_point1[-1] > thred and homo_point2[-1] > thred:
        # two points on the image
        image_point1 = homo_point1 / homo_point1[-1]
        image_point2 = homo_point2 / homo_point2[-1]
        direction = image_point2[0: 2] - image_point1[0: 2]
        return image_point1, image_point2, direction / np.linalg.norm(direction, 2)
    else:
        # find the farther point
        if homo_point1[-1] > homo_point2[-1]:
            anchor_point = point1
            cos_vector = compute_directional_vector(point1, point2)
        else:
            anchor_point = point2
            cos_vector = compute_directional_vector(point2, point1)
        jacobian = compute_J(Pmat, anchor_point)
        line_direction = jacobian.dot(cos_vector)
        line_direction = line_direction / np.linalg.norm(line_direction, 2)
        projected_anchor_point = Pmat.dot(anchor_point)
        projected_anchor_point = projected_anchor_point / projected_anchor_point[-1]
        return projected_anchor_point, line_direction


if __name__ == '__main__':
    img_front = cv2.imread('data/onsemi_obstacle.jpg')
    novatel2world = np.array([[-8.01528139e-01, - 5.97813694e-01, - 1.30930841e-02, 2.26729011e+05],
                              [5.97758205e-01, - 8.01634136e-01, 8.23656449e-03, 3.37541579e+06],
                              [-1.54197942e-02, - 1.22466026e-03, 9.99880358e-01, 1.55169511e+01],
                              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    extrinsics_front = np.matrix([[0.00940942, -0.01788846, 0.99979571, 0.46919522],
                                  [-0.99988938, -0.01168621, 0.00920121, -0.03360787],
                                  [0.01151923, -0.99977169, -0.01799644, -0.3943116],
                                  [0., 0., 0., 1.]])
    extrinsics_hesai40_novatel = np.matrix([[0.01189847, -0.99992661, -0.00227904, -0.09138513],
                                            [0.99990137, 0.01191511, -0.00743454, 2.01929912],
                                            [0.00746115, -0.00219035, 0.99996977, 1.387],
                                            [0., 0., 0., 1.]])
    intrinsics_front = np.matrix([[2.85361776e+03, 0.00000000e+00, 1.44734730e+03],
                                  [0.00000000e+00, 2.85361776e+03, 9.21366341e+02],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    world2novatel = np.linalg.inv(novatel2world)

    road_info_cache_utm_path = 'data/hdmap_data_draw_temp.npy'
    hdmap_data_draw = np.load(road_info_cache_utm_path)

    hdmap_data_draw[:, 2] = 15.516951058269894 - 0.5089999999999999
    hdmap_data_draw[:, 6] = 15.516951058269894 - 0.5089999999999999
    hdmap_data_draw[:, 0:4] = [np.dot(world2novatel, w) for w in hdmap_data_draw[:, 0:4]]  # novatel
    hdmap_data_draw[:, 4:8] = [np.dot(world2novatel, w) for w in hdmap_data_draw[:, 4:8]]

    camera_points_end = [np.linalg.multi_dot(
        [np.linalg.inv(extrinsics_front), np.linalg.inv(extrinsics_hesai40_novatel), w]) for w
        in hdmap_data_draw[:, 0:4]]
    camera_points_start = [np.linalg.multi_dot(
        [np.linalg.inv(extrinsics_front), np.linalg.inv(extrinsics_hesai40_novatel), w]) for w
        in hdmap_data_draw[:, 4:8]]

    camera_points_end_np = np.array(
        [[w.item(0), w.item(1), w.item(2)] for w in
         camera_points_end])
    camera_points_start_np = np.array(
        [[w.item(0), w.item(1), w.item(2)] for w in
         camera_points_start])

    # project
    for ind, (ep_camcoor, sp_camcoor) in enumerate(zip(camera_points_end_np, camera_points_start_np)):
        Pmat = np.array(intrinsics_front)
        homo_point1 = Pmat.dot(ep_camcoor)
        homo_point2 = Pmat.dot(sp_camcoor)
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
                     (int(image_point1[0]), int(image_point1[1])),
                     (int(image_point2[0]), int(image_point2[1])), (70, 200, 0), 12)
        else:
            # find the farther point
            if homo_point1[-1] > homo_point2[-1]:
                anchor_point = ep_camcoor
                cos_vector = compute_directional_vector(ep_camcoor, sp_camcoor)
            else:
                anchor_point = sp_camcoor
                cos_vector = compute_directional_vector(sp_camcoor, ep_camcoor)
            jacobian = compute_J_single(Pmat, anchor_point)
            line_direction = jacobian.dot(cos_vector)
            line_direction = line_direction / np.linalg.norm(line_direction, 2)
            projected_anchor_point = Pmat.dot(anchor_point)
            projected_anchor_point = projected_anchor_point / projected_anchor_point[-1]
            mock_point = projected_anchor_point[:2] + line_direction * 10000
            # print(projected_anchor_point, line_direction)
            cv2.line(img_front,
                     (int(projected_anchor_point[0]), int(projected_anchor_point[1])),
                     (int(mock_point[0]), int(mock_point[1])), (0, 0, 0), 12)
            cv2.imwrite('data/res.jpg', img_front)