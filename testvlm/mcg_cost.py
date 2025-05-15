import os

import numpy as np
import pandas as pd
from l_dcost import *
from allknowgai1 import *
to = '/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/hm3rscandata/xuanran/ceshioutputscan/query_analysisto/today.csv'
df_a = pd.read_csv(to)
scene_info_path = "/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/hm3rscandata/xuanran/ceshioutputscan/3rscan_instance_data/scenes_train_val_info_w_images.pkl"
scene_info = mmengine.load(scene_info_path)
scene_idkechongfu = []
for i, row in df_a.iterrows():
    scene_idkechongfu.append(df_a.loc[i, 'scan_id'])
scene_idkechongfu = list(set(scene_idkechongfu))
def find_posels_json_files(folder_path):
    posels_json_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'posels.json':
                posels_json_paths.append(os.path.join(root, file))
    return posels_json_paths

#outputs/visual_groundingxuanran0324/2025-04-23_v2_ceshi_bufen_ceshi_1_2/intermediate_results/0ad2d3a3-79e2-2212-9a51-9094be707ec2

def chuliposels(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    posels = []
    for key, value in data.items():
        value = np.array(value)
        posels.append(value)
    return posels

if __name__ == "__main__":
    zongl = 0
    zongaction = 0
    zongd = 0
    zong = []
    zongchangdu = 0
    for i in range(1,2):
        duiyingdict = {}
        
        #2025-04-23_v2_ceshi_posels_qubaodiposexiuzheng_5fen_1
        j = str(i)
        #root_path = f'outputs/visual_groundingxuanran0324/2025-04-23_v2_ceshi_posels_xinpose_3/intermediate_results/'  # 这里可以替换为你要查找的文件夹路径
        root_path = f'outputs/visual_groundingxuanran0324/2025-04-23_v2_ceshi_posels_qumem_high_{j}/intermediate_results/'  # 这里可以替换为你要查找的文件夹路径
        #root_path = f'outputs/visual_groundingxuanran0324/2025-04-23_v2_ceshi_posels_qubaodiposexiuzheng_5fen_{j}/intermediate_results/'  # 这里可以替换为你要查找的文件夹路径

        for scanid in scene_idkechongfu:
            ###这里应该没有就跳过就可以了
            folder_path = os.path.join(root_path, scanid)
            if not os.path.exists(folder_path):
                continue
            result = find_posels_json_files(folder_path)
            duiyingdict[scanid] = result
            print(result)
            zongchangdu += len(result)

        

        ###存好了正式开始计算了
        action_cost = 0
        l_cost = 0
        d_cost = 0

        for key, values in duiyingdict.items():
            ###一个一个scan去算
            ###起点都是align
            ###取出center
            center1 = scene_info[key]["axis_align_matrix"]
            center1 = np.linalg.inv(center1)
            center = chulialign(center1)
            for value in values:
                dangqianposels = chuliposels(value)
                l, d = patscost(center, dangqianposels)
                l_cost+=l
                d_cost+=d
                action_cost+=len(dangqianposels)
        zongaction+=action_cost
        zongl += l_cost
        zongd += d_cost
    print("修改后的有效posels的长度", zongchangdu)
    print("action ", zongaction)
    print("线性", zongl)
    print("角度", zongd)
    print("时间", zongl+zongd)

