import json
import numpy as np
import os

def cunchuliebiao(matrix_list, filedir):
    if matrix_list is None:
        return
    filename = os.path.join(filedir, "posels.json")
    matrix_dict = {}
    for i, matrix in enumerate(matrix_list):
        if matrix is None:
            continue
        matrix_dict[i] = matrix.tolist()

    with open(filename, 'w') as f:
        json.dump(matrix_dict, f, indent=4)


# 示例矩阵列表
matrix_list = [np.array([[1,2,3,4], [5,6,7,8]])]

# 保存到 JSON 文件
cunchuliebiao(matrix_list, 'ceshi')