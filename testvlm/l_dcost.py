#这两个可以直接一起算
#然后针对v1，不需要别的一些处理，直接对着原始数据去算
import argparse
import copy
import datetime
import json
import math
import os
import random
import shutil
import sys
import traceback
import ast
import cv2
import mmengine
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from allknowgai1 import Theallknow
###注意了，yes和naive的pkl都是一样的。
###360的是单独弄得。
scene_info_path = "/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/hm3rscandata/xuanran/ceshioutputscan/3rscan_instance_data/scenes_train_val_info_w_images.pkl"
scene_info = mmengine.load(scene_info_path)
###很好这个用来计算v360应该也性
to = '/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/hm3rscandata/xuanran/ceshioutputscan/query_analysisto/today.csv'

df_a = pd.read_csv(to)
yes = '/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/hm3rscandata/xuanran/ceshioutputscan/query_analysisyes/yesterday.csv'

df_b = pd.read_csv(yes)
scene_idkechongfu = []
#+"*"+df_b.loc[i, 'scan_id']
###注意了，这个是给yesterday用的
for i, row in df_a.iterrows():
    scene_idkechongfu.append(df_a.loc[i, 'scan_id'])
print("总长度，", len(scene_idkechongfu))
def angle_between_rotations(R1, R2):
    # 计算相对旋转矩阵 R_rel = R1^T * R2
    #print(R1)
    #print(R2)
    R_rel = np.dot(R1.T, R2)

    # 计算旋转角度（处理数值误差）
    trace = np.trace(R_rel)
    # 确保值在 [-1, 1] 范围内，避免arccos数值错误
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    #theta_deg = np.degrees(theta)
    return theta
def jisuancost(pre_po, now_po):
    pre_l1 = pre_po[:3, 3]
    pre_l2 = now_po[:3, 3]
    center_distance = np.linalg.norm(pre_l2 - pre_l1)
    pre_r1 = pre_po[:3, :3]
    pre_r2 = now_po[:3, :3]
    """rotation_quaternion1 = R.from_matrix(pre_r1).as_quat()
    rotation_quaternion2 = R.from_matrix(pre_r2).as_quat()

    # 计算四元数之间的夹角（弧度制）
    quaternion_dot_product = np.dot(rotation_quaternion1, rotation_quaternion2)
    quaternion_angle = 2 * np.arccos(np.clip(quaternion_dot_product, -1, 1))"""

    quaternion_angle = angle_between_rotations(pre_r2, pre_r1)

    # 将夹角转换为度
    #quaternion_angle_degrees = np.degrees(quaternion_angle)
    quaternion_angle_degrees = quaternion_angle

    center_distance = center_distance / 0.5

    return center_distance, quaternion_angle_degrees

def patscost(chushipose, ensem_id_pose):
    pre_pose = chushipose
    l_cost = 0
    degree_cost = 0
    for pose in ensem_id_pose:
        l, d =jisuancost(pre_pose, pose)
        l_cost+=l
        degree_cost+=d
        pre_pose=pose
    return l_cost, degree_cost

#前面都是对的，关键是计算初始的pose了。

def easyjisuancost(scene_id):
    actioncost = 0



    actioncost += len(scene_info[scene_id]["images_info"])

    center, id, zuizhongpose, xuhao = Theallknow(scene_info, scene_id).zhaodaozhongxin()

    result = list(range(id, len(xuhao))) + list(range(0, id))

    assert len(xuhao) == len(scene_info[scene_id]["images_info"])
    #我们要把这个中心序号改成，或者不改动，但是注意因为模糊，有跳脚。
    poses = []
    #print(result)
    for id in result:
        #print(id)
        id = xuhao[id]
        #print("改制后",id)
        #id = 'frame-' + f"{id:06}"
        poses.append(zuizhongpose[id])
    #w = np.eye(4)
    w = center

    ###稍等，这里用了单位阵。说明她直接给对其了。
    l, d = patscost(w, poses)
    return actioncost, l, d
"""
action_cost = 0
L = 0
D = 0
t = 0
for item in scene_idkechongfu:
    t += 1
    a, l, d= easyjisuancost(item)
    action_cost += a
    L += l
    D += d
    #print(t)
print("action ", action_cost)
print("线性",L)
print("角度",D)"""
