# Copyright (c) Baidu, Inc. and its affiliates. All Rights Reserved
"""
bevmap.data.utils.scene.py
Read scene-label from cyberecord_xxx_info.txt
"""
"""
SCENE_TAG: int, scene_label parsed from xxx_info.txt, un-exlclusive
SCENE_NAME: str, scene name for human-read
SCENE_CLS: int, scene class-id, exlclusive
"""
import os
import ast

import numpy as np


SCENE_NAMES2CLS = dict(others=0,
                       fork=1,
                       head_change=2,
                       empty=4, # temp keep
                       merge=3,
                       ramp=5,
                       tunnel=6,
                       crossing=7, # by xiexiaolu
                       curve=8,
                       big_curve=9,
                       y_ramp=10,
                       roundabout=11,
                       )

# NOTES : move SCENE_CLS2REP_FACTORS to config.yaml

SCENE_CLS2NAMES = {v: k for k, v in SCENE_NAMES2CLS.items()}

SCENE_TAG2CLS = {
    -100: 4, # temp not used
    112: 10, # y_ramp
    109: 11, # roundabout
    108: 6, # tunnel
    103: 3, # merge
    403: 2, # lane_change   # TODO  ruda ??
    105: 5, # in-ramp
    106: 5, # out-ramp
    110: 5, # on-ramp
    104: 1, # fork
    # 101: 10 # big wandao, radius < 500, not use
}

HIGH_SCENE_CLS = {10, 11} # TODO


class RecordSceneLabel:
    """scenelabel parser for a record
    """
    def __init__(self, info_path):
        self.ts2label = {}
        self.ts2radius = {}
        if not os.path.exists(info_path):
            return
        with open(info_path, 'rt') as f:
            scene_lines = f.readlines()
        if len(scene_lines) > 0 and '\t' in scene_lines[0]:
            delmiter = '\t'
        else:
            delmiter = '['
        for line in scene_lines[1:]: # skip header
            splits = line.split(delmiter)
            if delmiter == '\t':
                scene_str = splits[-2]
                radius = splits[-1][:-1]
                ts = splits[8]
            else:
                scene_str = '[' + splits[-1]
                radius = '10000'
                ts = line.split(' ')[8]
            scene_tag_set = set(ast.literal_eval(scene_str)) # {int}
            self.ts2label[ts] = scene_tag_set # (onsemi_ts, scene_tag_set)
            self.ts2radius[ts] = int(radius) if radius.isnumeric() else 10000.

    def get_label(self, ts):
        """ get label
        Args:
            ts: onsemi_obstacle timestamp, str
        Return:
            label: scene class label, int
            radius: sample ego_lane radius, float
        """
        label = 0
        scene_set = self.ts2label.get(ts, [])
        for scene_tag, cls in SCENE_TAG2CLS.items():
            if scene_tag in scene_set:
                label = cls
                break
        radius = self.ts2radius.get(ts, 10000)
        if label not in HIGH_SCENE_CLS:
            if 100 < radius < 500:
                label = 8
            elif 50 < radius <= 100:
                label = 9
        return label, radius

    @staticmethod
    def static_radius_distribution(radius):
        """static radius"""
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        num = 100
        path = 'radius-histogram_{}.png'.format(num)
        if type(radius) is dict:
            radius = list(radius.values())
        elif type(radius) is list:
            pass
        radius = np.array(radius)
        radius = np.sort(radius)
        plt.hist(radius, bins=num, range=(0, 10000), facecolor='g')
        plt.xlim(0, 10000)
        # plt.ylim(0, 1000)
        plt.grid(True)
        plt.title("radius-histogram")
        # plt.savefig(path)
        plt.show()
        print("Save to {}".format(path))
        import pandas as pd

        path = 'radius_{}.xlsx'.format(num)
        hist, bins = np.histogram(radius, bins=num, range=(0, 10000))
        data = np.vstack([hist, bins[1:]]).transpose()
        data_df = pd.DataFrame(data)
        writer = pd.ExcelWriter(path)
        data_df.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        print("Save to {}".format(path))
