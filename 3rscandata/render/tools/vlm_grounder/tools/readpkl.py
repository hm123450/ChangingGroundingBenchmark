#查看pkl里面数据类型
import json
import pickle
import mmengine
import os

# 设置文件夹路径
folder_path1 = "xuezhangpatsceshi.pkl"


zuihzong = {}

with open(folder_path1, 'rb') as file:
    # 加载文件内容
    dict1 = pickle.load(file)
    merged_dict = dict1.copy()  # 首先复制dict1，避免直接修改原字典
    zuihzong.update(merged_dict)