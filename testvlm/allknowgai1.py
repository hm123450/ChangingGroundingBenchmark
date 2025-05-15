# {
#   scene_id:
#   {
# 	  num_posed_images: n
# 	  intrinsic_matrix: 4 * 4 array
# 	  images_info:
# 	  {
# 		  {image_id:05d}:
# 		  {
# 			  'image_path': 'data/scannet/posed_images/{scene_id}/{image_id:05d}'
# 			  'depth_image_path': ...png
# 			  'extrinsic_matrix': 4 * 4 array
# 		  }
# 	  }
#      object_id: * N (0-indexed)
#      {
#        "aligned_bbox": numpy.array of (7, )
#        "unaligned_bbox": numpy.array of (7, )
#        "raw_category": str
#      }
#      "axis_aligned_matrix": numpy.array of (4, 4)
#      "num_objects": int
#   }
# }

import os

import cv2
import mmengine
import numpy as np

import torch
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R


def chulialign(view3):
    view3[:, 2] = -view3[:, 2]
    view3[:, [1, 2]] = view3[:, [2, 1]]
    view3[2, 3] += 1.40
    #view3[:, [0, 1]] = view3[:, [1, 0]]
    #view3[:, 1] = -view3[:, 1]
    return view3
#是给昨天的id去比较今天的id哪个好。
#所以昨天的id这个东西才是变量
class Theallknow:
    def __init__(self, scene_info, scanto_id):
        self.scene_info = scene_info
        self.to_scan_id = scanto_id

    def zhaodaozhongxin(self):
        center1 = self.scene_info[self.to_scan_id]["axis_align_matrix"]
        center1 = np.linalg.inv(center1)
        center = chulialign(center1)
        self.center = center
        min_key, zuizhongpose, xuhao = self.get_toimage_id_top()
        #frame-000067
        #id = int(min_key.split("-")[1])
        id = xuhao.index(min_key)
        return center, id, zuizhongpose, xuhao
    def zhaodaozhongxinid(self):
        center1 = self.scene_info[self.to_scan_id]["axis_align_matrix"]
        center1 = np.linalg.inv(center1)
        center = chulialign(center1)
        self.center = center
        min_key, zuizhongpose, xuhao = self.get_toimage_id_top()
        # frame-000067
        id = int(min_key.split("-")[1])
        #id = xuhao.index(min_key)
        ###这个id是为了找pose
        return id


    def calculate_line_similarity(self, pose_matrix1, pose_matrix2):
        """
        计算两个相机位姿矩阵的相似性

        参数:
        pose_matrix1 (numpy.ndarray): 第一个相机的位姿矩阵，形状为 (4, 4)
        pose_matrix2 (numpy.ndarray): 第二个相机的位姿矩阵，形状为 (4, 4)

        返回:
        similarity (float): 相似性度量值，值越小表示相似性越高
        """

        # 提取位置向量
        position1 = pose_matrix1[:3, 3]
        position2 = pose_matrix2[:3, 3]
        # print(position1)
        # print(position2)

        # 计算位置向量的欧氏距离
        position_distance = np.linalg.norm(position1 - position2)
        # print(position_distance)

        similarity = position_distance

        return similarity

    def calculate_degree_similarity(self, pose_matrix1, pose_matrix2):
        """
        计算两个相机位姿矩阵的相似性

        参数:
        pose_matrix1 (numpy.ndarray): 第一个相机的位姿矩阵，形状为 (4, 4)
        pose_matrix2 (numpy.ndarray): 第二个相机的位姿矩阵，形状为 (4, 4)

        返回:
        similarity (float): 相似性度量值，值越小表示相似性越高
        """

        # 提取旋转矩阵
        rotation_matrix1 = pose_matrix1[:3, :3]
        rotation_matrix2 = pose_matrix2[:3, :3]

        # 将旋转矩阵转换为四元数
        rotation_quaternion1 = R.from_matrix(rotation_matrix1).as_quat()
        rotation_quaternion2 = R.from_matrix(rotation_matrix2).as_quat()

        # 计算四元数之间的夹角（弧度制）
        quaternion_dot_product = np.dot(rotation_quaternion1, rotation_quaternion2)
        quaternion_angle = 2 * np.arccos(np.clip(quaternion_dot_product, -1, 1))

        # 将夹角转换为度
        quaternion_angle_degrees = np.degrees(quaternion_angle)

        # 计算相似性度量值，这里简单地将位置距离和旋转角度加权求和
        # 你可以根据实际需求调整权重
        similarity = quaternion_angle_degrees

        return similarity

    def calculate_cosine_similarity(self, pose_matrix1, pose_matrix2):
        """
        计算两个矩阵的余弦相似度

        """
        matrix1 = pose_matrix1[:3, :3]
        matrix2 = pose_matrix2[:3, :3]
        dot_product = np.trace(matrix1.T @ matrix2)
        norm1 = np.linalg.norm(matrix1)
        norm2 = np.linalg.norm(matrix2)
        cosine_similarity = dot_product / (norm1 * norm2)
        return cosine_similarity

    def juzhenchajugai(self, pose_matrix1, pose_matrix2):
        position1 = pose_matrix1[:3, 3]
        position2 = pose_matrix2[:3, 3]
        # print(position1)
        # print(position2)

        # 计算位置向量的欧氏距离
        position_distance = np.linalg.norm(position1 - position2)
        # (position_distance)
        # print("位置距离：", position_distance)

        # 提取旋转矩阵
        rotation_matrix1 = pose_matrix1[:3, :3]
        rotation_matrix2 = pose_matrix2[:3, :3]

        # 将旋转矩阵转换为四元数
        rotation_quaternion1 = R.from_matrix(rotation_matrix1).as_quat()
        rotation_quaternion2 = R.from_matrix(rotation_matrix2).as_quat()

        # 计算四元数之间的夹角（弧度制）
        quaternion_dot_product = np.dot(rotation_quaternion1, rotation_quaternion2)
        quaternion_angle = 2 * np.arccos(np.clip(quaternion_dot_product, -1, 1))

        # 将夹角转换为度
        quaternion_angle_degrees = np.degrees(quaternion_angle)
        # print("度数差距：", quaternion_angle_degrees)

        # 计算相似性度量值，这里简单地将位置距离和旋转角度加权求和
        # 你可以根据实际需求调整权重
        weight_position = 0.5
        weight_rotation = 0.5
        similarity = weight_position * position_distance + weight_rotation * quaternion_angle_degrees

        return similarity

    def angle_between_rotations(self, RR1, RR2):
        # 计算相对旋转矩阵 R_rel = R1^T * R2
        # print(R1)
        # print(R2)
        R1= RR1[:3,:3]
        R2=RR2[:3, :3]
        R_rel = np.dot(R1.T, R2)

        # 计算旋转角度（处理数值误差）
        trace = np.trace(R_rel)
        # 确保值在 [-1, 1] 范围内，避免arccos数值错误
        cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        # theta_deg = np.degrees(theta)
        return theta

    def convert_imageid(self, image_id):
        # try to convert image_id as int
        if type(image_id) != str:
            print("不是字符串")
            print(image_id)
            image_id = 'frame-' + f"{image_id:06}"

        elif image_id[0] == 'f':
            print("不是数字")
            image_id = image_id.split('.')[0]
            print(image_id)
        else:
            print("是数字字符串")
            print(image_id)
            print(type(image_id))

            image_id = 'frame-' + "0" + image_id
            print(image_id)
        return image_id

    def get_yesimage_align_pose(self):
        align_pose = np.eye(4)
        return align_pose

    def get_toimage_id_top(self):


        chajuline = {}
        chajudegree = {}
        zuizhongpose = {}
        xuhao = []
        for key, value in self.scene_info[self.to_scan_id]["images_info"].items():
            rescan_yuanshi_pose = value["extrinsic_matrix"]
            zuizhongpose[key] = rescan_yuanshi_pose
            xuhao.append(key)

            chajuline[key] = self.calculate_line_similarity(rescan_yuanshi_pose, self.center)
            chajudegree[key] = self.angle_between_rotations(rescan_yuanshi_pose, self.center)
        # 直接从前面开始删除
        sorted_keys = sorted(chajuline, key=lambda k: chajuline[k])
        min_10_keys = sorted_keys[:10]
        ceshiline = {}
        for jing in min_10_keys:
            ceshiline[jing] = chajuline[jing]
        #print(ceshiline)
        #print(min_10_keys)
        zuizhong = {}
        for item in min_10_keys:
            zuizhong[item] = chajudegree[item]
        #print(zuizhong)
        sorted_zuizhong = sorted(zuizhong, key=lambda k: zuizhong[k])

        min_key = min(zuizhong, key=lambda k: zuizhong[k])
        #print(f"现在键值对中值最小的键是：{min_key}，对应的值是：{zuizhong[min_key]}")

        return min_key, zuizhongpose, xuhao


if __name__ == "__main__":
    scene_info_path = "../hm3rscandata/scannet/ceshioutputz/3rscan_instance_data/scenes_train_val_info_w_images.pkl"
    yesscan_id = "0cac7536-8d6f-2d13-8dc2-2f9d7aa62dc4"
    yesimage_id = 44
    toscan_id = "dc42b36c-8d5c-2d2a-86aa-19a8929361fd"
    ceshi = []
    god = Theallknow(scene_info_path, yesimage_id, yesscan_id, toscan_id)
    xin_id, chaju = god.get_toimage_id_top(ceshi)
    # chaju = god.chaju
    print(xin_id)
    # print(chaju)
