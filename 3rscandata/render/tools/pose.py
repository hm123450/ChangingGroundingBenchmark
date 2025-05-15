# Modified from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py # noqa
import os
import re
import argparse
current_directory = os.getcwd()
print(current_directory)
# print(os.path.isdir('../../../../../dongtaiscan/data/.ipynb_checkpoints'))
# print(os.path.isdir('scans/scene0000_00'))
# if os.path.isdir('../../../../../dongtaiscan/data/.ipynb_checkpoints'):
#    print("cunzai sahnchu")
#    os.removedirs('../../../../../dongtaiscan/data/.ipynb_checkpoints')

import struct
import time
import zlib
from argparse import ArgumentParser
from functools import partial
from PIL import Image
import os
import zipfile
import imageio.v2 as imageio  # * to surpress warning
import mmengine
import numpy as np


# 算了，人家用pgm存的好好的，你大不了再用IMgae打开不就好了


root_dir = '../ceshioutputscan'  # 替换为你的根目录路径
data_dir = "../../../../../3rscandata/data"
posed_images_dir = os.path.join(root_dir, 'posed_imageschushi')

# 确保posed_images目录存在
if not os.path.exists(posed_images_dir):
    os.makedirs(posed_images_dir)

scan_id = []
aa = 0
# 遍历data目录中的所有子文件夹
"""for subdir, dirs, files in os.walk(data_dir):
    for dir in dirs:
        if dir == ".ipynb_checkpoints":
            continue
        else:
            scan_id.append(dir)"""
with open("valid.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    # 去除每行末尾的换行符
    result = [line.strip() for line in lines]
    scan_id = result
print(len(scan_id))


def chuli(dir):
    # 构建子文件夹的完整路径

    subdir_path = os.path.join(data_dir, dir)
    # 构建ZIP文件的路径
    zip_file_path = os.path.join(subdir_path, 'sequence.zip')

    # 检查ZIP文件是否存在
    if os.path.exists(zip_file_path):
        # 构建解压目标文件夹的路径
        extract_to = os.path.join(posed_images_dir, dir)

        # 确保目标文件夹存在
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)

        # 解压ZIP文件
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"已解压 {zip_file_path} 到 {extract_to}")
        # batch_convert_images(extract_to, extract_to)
        # 假设文本文件名为'data.txt'
        infofile = os.path.join(extract_to, "_info.txt")

        # 读取文件内容
        with open(infofile, 'r') as file:
            content = file.readlines()

        for line in content:
            if line.startswith('m_calibrationColorIntrinsic = '):
                # 分割字符串并提取数字
                print(line)
                line = line[30:]
                numbers = list(map(float, line.split()))
                print(numbers)

        # 将列表转换为NumPy矩阵
        calibration_matrix = np.array(numbers).reshape((4, 4))  # 假设这是一个3x3矩阵
        print("Calibration Matrix:")
        print(calibration_matrix)
        insfile = os.path.join(extract_to, "intrinsic.txt")
        with open(insfile, 'w') as file:
            # 遍历矩阵的每一行
            for row in calibration_matrix:
                # 将每一行的元素格式化为字符串，并用空格分隔
                row_str = ' '.join(f"{num:.6f}" for num in row)
                # 写入文件，并在每一行后添加换行符
                file.write(row_str + '\n')
###非常号。就是把zip展开以及替换出来instrinsic这个文件

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_range_begin', type=int, default=0)
    parser.add_argument('--scene_range_end', type=int,
                            default=-1)  # * -1 means to the end

    args = parser.parse_args()
    #排下序，这样不会重复
    scene_range_begin = args.scene_range_begin
    if args.scene_range_end == -1:
        scene_range_end = len(scan_id)
    else:
        scene_range_end = args.scene_range_end
    scan_id.sort()
    scan_id = scan_id[scene_range_begin:scene_range_end]
    for item in scan_id:
        extract_to = os.path.join(posed_images_dir, item)
        print(extract_to)
        # 确保目标文件夹存在
        if os.path.exists(extract_to):
            print("进来了么")
            aa += 1
            print("pass already process")
            continue
        chuli(item)
    print(aa)
    # process_directory(scan_id)
