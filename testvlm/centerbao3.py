# 这是专门为了替代pats的函数
# description由主函数提供，其它的一些东西也由主函数提供
# 注意，就在这吧，主函数已经太长了
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
import supervision as sv
import torch
from mmengine.utils.dl_utils import TimeCounter
from PIL import Image
from supervision import Position
from supervision.draw.color import Color
from tqdm import tqdm
from ultralytics import SAM
from ultralytics.engine.results import Results
from sklearn.cluster import DBSCAN
from vlm_grounder.grounder.prompts.visual_grounder_prompts import *
from vlm_grounder.utils import *
from vlm_grounder.utils import (
    DetInfoHandler,
    MatchingInfoHandler,
    OpenAIGPT,
    SceneInfoHandler,
    UltralyticsSAMHuge,
    calculate_iou_3d,
)
from scipy.spatial.transform import Rotation as R

DEFAULT_OPENAIGPT_CONFIG = {"temperature": 1, "top_p": 1, "max_tokens": 4095}

# from fuzhu import huodemaodiandet_paths
from promptpaixu import paixu_prompt2
from promptpaixu import *

import numpy as np

from zhiwei.nvdiffrast_tool.godmokuai import *

custom_color = sv.Color(r=255, g=0, b=0)  # 红色边框
default_bbox_annotator = sv.BoundingBoxAnnotator(color=custom_color, thickness=4)
default_label_annotator = sv.LabelAnnotator()
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

import cv2
import mmengine
import pandas as pd
import supervision as sv
import torch
from mmengine.utils.dl_utils import TimeCounter
from PIL import Image
from supervision import Position
from supervision.draw.color import Color
from tqdm import tqdm
from ultralytics import SAM
from ultralytics.engine.results import Results

from vlm_grounder.grounder.prompts.visual_grounder_prompts import *
from vlm_grounder.utils import *
from vlm_grounder.utils import (
    DetInfoHandler,
    MatchingInfoHandler,
    OpenAIGPT,
    SceneInfoHandler,
    UltralyticsSAMHuge,
    calculate_iou_3d,
)

DEFAULT_OPENAIGPT_CONFIG = {"temperature": 1, "top_p": 1, "max_tokens": 768}
from zhiwei.nvdiffrast_tool.godmokuai import *
from scipy.spatial.transform import Rotation as R


DEFAULT_OPENAIGPT_CONFIG = {"temperature": 1, "top_p": 1, "max_tokens": 4095}

# from fuzhu import huodemaodiandet_paths
from promptpaixu import paixu_prompt2
from promptpaixu import *
import numpy as np

from zhiwei.nvdiffrast_tool.godmokuai import *

DEFAULT_OPENAIGPT_CONFIG = {"temperature": 0.1, "top_p": 0.3, "max_tokens": 4095}
openaigpt_config = {"model": "gemini-2.0-flash", **DEFAULT_OPENAIGPT_CONFIG}
openaigpt = OpenAIGPT(**openaigpt_config)
kernel_size = 7
post_process_component = True
post_process_dilation = False
post_process_erosion = True
post_process_component_num = 2
apibbox_promptceshi = """Great! Here is the detailed version of the image you've selected. You will be provided with a description of the target object, which includes details such as the color of the target object, its position in the picture, and other relevant features.
There are {num_candidate_bboxes} candidate objects shown in the image. I have annotated each object at the center with an object ID in white text on a black background. Please find out which marked object in the picture exactly matches the details in the description.
You need to think through three steps. First, the candidate objects provided to you are not necessarily all of the target object category. You must first identify which of the candidate objects belong to the target object category. Second, cross-reference the identified candidate objects of the target category with the content in the description. Finally, after the analysis of the first two steps, select the object ID among the found candidate IDs of the target object category that is most likely to satisfy the description.
Here is the description: {description}
Please reply in JSON format with two keys, "reasoning" and "object_id", in the following format:
{{
"reasoning": "Your reasoning process", // Explain the justification for your selection of the object ID. Describe your three-step thinking process, including how you identified the candidate objects of the target object category, how you cross-referenced them with the description, and how you made the final selection among them.
"object_id": 0 // The object ID you've selected. Always provide one object ID from the image that you are most confident about, even if you think the correct object is not present in the image.
}}
"""
def uniform_downsampling(arr, sample_rate):
    """
    按照一定的采样率进行均匀间隔采样

    参数:
    arr (numpy.ndarray): 输入的形状为 (n, 5) 的数组
    sample_rate (float): 采样率，取值范围在 0 到 1 之间

    返回:
    numpy.ndarray: 降采样后的数组
    """
    num_samples = int(arr.shape[0] * sample_rate)
    indices = np.linspace(0, arr.shape[0] - 1, num_samples, dtype=int)
    downsampled_arr = arr[indices]
    return downsampled_arr


def find_local_minima_horizontal(arr1):
    """
    Find local minima in a 2D NumPy array, where each point is smaller than its horizontal neighbors.

    Parameters:
    - arr (ndarray): The input 2D array.

    Returns:
    - list of tuples: A list of coordinates (i, j) where the value is a local minimum horizontally.
    """
    minima_coords = []
    t = 0
    arr = arr1.copy()
    arr = arr.astype(np.int32)
    for i in range(arr.shape[0]):
        for j in range(1, arr.shape[1] - 1):
            #if arr[i, j] < arr[i, j-1] and arr[i, j] < arr[i, j+1]:
            #    minima_coords.append((i, j))

            if arr[i, j] <= 250:
                minima_coords.append((i, j))

            if arr[i, j-1] == 0 and arr[i, j+1] == 0:
                minima_coords.append((i, j))

            #if arr[i, j-1] == 0 and arr[i, j] != 0 and arr[i, j] < (arr[i, j+1]-150):
            #    minima_coords.append((i, j))
            #if arr[i, j+1] == 0 and arr[i, j] != 0 and arr[i, j] < (arr[i, j-1]-150):
            #    minima_coords.append((i, j))

            ###最多只会损耗一个正常值，所以没关系，绝对不能迭代

            if abs(arr[i, j-1] - arr[i, j])>10 or abs(arr[i, j+1] - arr[i, j])>10:
                minima_coords.append((i, j))
                t+=1

            ###其实可能没有关系，像那种中间点如果是正常的只会有一个，我们赋值为0自然就弄走了，只有不正常的才会连续的变化，
            ###而且注意这个不能像我直接说的那样来回搞，不然会迭代消失
    print("有多少个点这样", t)

    return minima_coords
def replace_local_minima_with_previous(arr):
    minima_coords = find_local_minima_horizontal(arr)
    print(type(arr))
    for i, j in minima_coords:
        arr[i, j] = 0
    #another = find_points_with_zero_neighbors(arr)
    #for i, j in another:
    #    arr[i, j]=0
    return arr


def chulialign(view3):
    view3[:, 2] = -view3[:, 2]
    view3[:, [1, 2]] = view3[:, [2, 1]]
    view3[2, 3] += 1.40
    # view3[:, [0, 1]] = view3[:, [1, 0]]
    # view3[:, 1] = -view3[:, 1]
    return view3


def rotate_local_axis(pose, axis='z', angle_deg=0):
    """绕局部坐标系某轴旋转（右乘旋转矩阵）"""
    angle_rad = np.deg2rad(angle_deg)
    rot = R.from_euler(axis, angle_rad).as_matrix()

    # 分解当前位姿的旋转和平移
    current_rot = pose[:3, :3]
    current_trans = pose[:3, 3]

    # 更新旋转矩阵（右乘）
    new_rot = current_rot @ rot
    new_pose = np.eye(4)
    new_pose[:3, :3] = new_rot
    new_pose[:3, 3] = current_trans
    return new_pose


def move_along_local_axis(pose, delta_x=0, delta_y=0, delta_z=0):
    """沿局部坐标系平移（修改平移部分）"""
    new_pose = pose.copy()
    new_pose[:3, 3] += np.array([delta_x, delta_y, delta_z])
    return new_pose


def cunxuanran(img, depth, patspose, imgfile, depthfile, invimgfile, posefile):
    # 是这样的，只要我们的inv存的是一样的图像，之后不需要mask求得时候再转。
    data1 = depth * 1000
    data1 = data1.astype(np.uint32)
    downsample_factor = 4

    # 进行间隔采样
    data1 = data1[::downsample_factor, ::downsample_factor]
    data1 = Image.fromarray(data1)
    data1.save(depthfile)
    # np.save(t_file_pgm, depth)

    # img = Image.fromarray(img)

    img.save(imgfile)
    imginv = img
    imginv.save(invimgfile)
    # 不需要旋转了，只需要正常存储三个就可以了

    np.savetxt(posefile, patspose)


def cunquxuanrantupianyiqi(xianzaiid, intfiledir, godxuanran, yespose):
    # 先暂时存today的pose
    # 因为这一步之前的pose我们都处理好了，不需要换坐标轴什么的
    # 所以要用yiqi
    fanhuiimg, fanhuidepth, fanhuipose, patspose = godxuanran.toposeyiqi(yespose)
    nowpose = fanhuipose  # nowpose是用来算距离的，yespose才是用来旋转，这个是求过逆的
    # xuanrantupianintdir = os.path.join(self.dangqianoutdir, toscan_id, "xuanrantupianintdir")
    nowimage = "frame-" + f"{int(xianzaiid):06d}" + ".jpg"
    nowdepth = "frame-" + f"{int(xianzaiid):06d}" + ".pgm"
    nowimageinv = "frame-" + f"{int(xianzaiid):06d}inv" + ".jpg"
    nowposetxt = "frame-" + f"{int(xianzaiid):06d}" + ".txt"
    xuanranimagefile = os.path.join(intfiledir, nowimage)
    xuanrandepthfile = os.path.join(intfiledir, nowdepth)
    xuanraninvimagefile = os.path.join(intfiledir, nowimageinv)
    xuanranposefile = os.path.join(intfiledir, nowposetxt)
    cunxuanran(fanhuiimg, fanhuidepth, patspose, xuanranimagefile, xuanrandepthfile, xuanraninvimagefile,
               xuanranposefile)


def apixuanranbbox(
        imageid, imagefilepath, apibbox_prompts, xinintermediate_output_dir, classtarget1,
        jiancemodel, des
):
    # 去选出来新的image里面合适的id
    # ceshiid是为了存文件用的
    # 这个是需要路径的
    print(imagefilepath)
    xinimage = Image.open(imagefilepath)
    detections = jiancemodel.detect(imagefilepath, classtarget1)
    detections = detections[detections.confidence > 0.20]
    num_candidate = len(detections)
    print("是零么", num_candidate)
    if num_candidate == 0:
        print("进入返回了么")
        return 0, None

    messages = []
    bbox_index_gpt_select_output_dir = os.path.join(
        xinintermediate_output_dir, "bbox_index_gpt_select_prompts", f"index_{imageid}"
    )
    mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)

    labels = [f"ID:{ID}" for ID in range(len(detections))]
    label_annotator = sv.LabelAnnotator(
        text_position=Position.CENTER,
        text_thickness=2,
        text_scale=1,
        color=Color.BLACK,
    )
    annotated_image = label_annotator.annotate(
        scene=xinimage, detections=detections, labels=labels
    )
    # save the annotated_image
    annotated_image.save(
        f"{bbox_index_gpt_select_output_dir}/{imageid}_annotated.jpg"
    )

    if num_candidate == 1:
        return 0, detections[0]
    if num_candidate == 0:
        return -1, None

    # convert the annotated image to base64
    annotated_image_base64 = encode_PIL_image_to_base64(
        resize_image(annotated_image)
    )
    # 这里我们需要处理的只有一个图像
    # 欧，我想起来了，一个是框，一个是id。
    # 我们可以用format去搞

    multi_prompt = apibbox_prompts.format(
        num_candidate_bboxes=num_candidate,
        description=des,
    )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": multi_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{annotated_image_base64}",
                        "detail": "high",
                    },
                },
            ],
        }
    )

    # call VLM to get the result
    cost = 0
    retry = 1
    while retry <= 2:
        bbox_index = -1
        gpt_message = None

        gpt_response = openaigpt.safe_chat_complete(
            messages, response_format={"type": "json_object"}, content_only=True
        )

        cost += gpt_response["cost"]
        gpt_content = gpt_response["content"]
        if (gpt_content[0] == '`'):
            gpt_content = gpt_content.strip()[8:-4]
        print(gpt_content)
        gpt_content_json = json.loads(gpt_content)
        print(gpt_content_json)
        gpt_message = {
            "role": "assistant",
            "content": [{"text": gpt_response["content"], "type": "text"}],
        }
        messages.append(gpt_message)
        # save gpt_content_json

        bbox_index = int(gpt_content_json.get("object_id", -1))
        if bbox_index < 0 or bbox_index >= num_candidate:
            # invalid bbox index
            retry += 1
            bbox_invalid_prompt = f"""Your selected bounding box ID: {bbox_index} is invalid, it should start from 0 and there are only {num_candidate_bboxes} candidate objects in the image. Now try again to select one. Remember to reply using JSON format with the required keys."""

            messages.append({"role": "user", "content": bbox_invalid_prompt})

            continue
        else:
            break

    if isinstance(messages[-1], dict) and messages[-1]["role"] == "user":
        # means the process is not complete, whether VLM produces no response or VLM provides invalid bbox id
        # the outer module should start from the begining
        message_his = None
    else:
        # means, VLM give a valid bbox_index
        message_his = messages
    if bbox_index == -1:
        return -1, None
    else:
        return bbox_index, detections[bbox_index]


def project_mask_to_3dceshi(
        depth_image,
        intrinsic_matrix,
        extrinsic_matrix,
        mask=None,
        world_to_axis_align_matrix=None,
        color_image=None,
):
    """
    Projects a mask to 3D space using the provided depth map and camera parameters.
    Optionally appends RGB values from a color image to the 3D points. (RGB order with 0-255 range)

    Parameters:
    - depth_image (str or ndarray): Path to the depth image or a numpy array of depth values. h, w
    - intrinsic_matrix (ndarray): The camera's intrinsic matrix. 4 * 4
    - extrinsic_matrix (ndarray): The camera's extrinsic matrix. 4 * 4
    - mask (ndarray): A binary mask (zero, non-zero array) where True values indicate pixels to project, which has the same shape with color_image. H, W. Could be None, where all pixels are projected.
    - world_to_axis_align_matrix (ndarray, optional): Matrix to align the world coordinates. 4 * 4
    - color_image (str or ndarray, optional): Path to the color image or a numpy array of color values. H, W, 3

    Returns:
    - ndarray: Array of 3D coordinates, optionally with RGB values appended. All False mask will give `array([], shape=(0, C), dtype=float64)`
    """
    # 交换第一行和第二行
    """new_matrix = intrinsic_matrix.copy()
    new_matrix[[0, 1], :] = new_matrix[[1, 0], :]

    # 交换第一列和第二列
    new_matrix[:, [0, 1]] = new_matrix[:, [1, 0]]
    new_matrix[1,2] = -new_matrix[1,2]
    intrinsic_matrix = new_matrix"""
    # mask = np.rot90(mask)  # ok，也就是说我已经把旋转过的图片应该有的mask搞出来，之后返回去就可以直接利用之前的depth_imagele

    # Load depth image from path if it's a string
    if isinstance(depth_image, str):
        depth_image = cv2.imread(depth_image, -1)
    depth_image = replace_local_minima_with_previous(depth_image)

    # Load color image from path if it's a string
    if isinstance(color_image, str):
        color_image = cv2.imread(color_image)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    if mask is None:
        mask = np.ones(color_image.shape[:2], dtype=bool)

    # Calculate scaling factors
    scale_y = depth_image.shape[0] / mask.shape[0]
    scale_x = depth_image.shape[1] / mask.shape[1]

    # Get coordinates of True values in mask
    mask_indices = np.where(mask)
    mask_y = mask_indices[0]
    mask_x = mask_indices[1]

    # Scale coordinates to match the depth image size
    # depth_y = (mask_y * scale_y).astype(int)
    # depth_x = (mask_x * scale_x).astype(int)

    # # scale and use round
    depth_y = np.round(mask_y * scale_y).astype(int)
    depth_x = np.round(mask_x * scale_x).astype(int)

    # Clip scaled coordinates to ensure they are within the image boundary
    depth_y = np.clip(depth_y, 0, depth_image.shape[0] - 1)
    depth_x = np.clip(depth_x, 0, depth_image.shape[1] - 1)

    # Extract depth values
    depth_values = (
            depth_image[depth_y, depth_x] * 0.001
    )  # Assume depth is in millimeters

    # Filter out zero depth values
    valid = depth_values > 0
    depth_values = depth_values[valid]
    mask_x = mask_x[valid]
    mask_y = mask_y[valid]

    # Construct normalized pixel coordinates
    normalized_pixels = np.vstack(
        (
            mask_x * depth_values,
            mask_y * depth_values,
            depth_values,
            np.ones_like(depth_values),
        )
    )

    # Compute points in camera coordinate system
    cam_coords = np.dot(np.linalg.inv(intrinsic_matrix), normalized_pixels)

    # Transform to world coordinates
    world_coords = np.dot(extrinsic_matrix, cam_coords)

    # Apply world-to-axis alignment if provided
    if world_to_axis_align_matrix is not None:
        world_coords = np.dot(world_to_axis_align_matrix, world_coords)

    # Append color information if color image is provided
    if color_image is not None:
        # Scale mask coordinates for the color image
        rgb_values = color_image[mask_y, mask_x]
        return np.hstack((world_coords[:3].T, rgb_values))

    return world_coords[:3].T


def project_image_to_3d(
        posefile, depthfile, colorfile, scene_id, image_id, scene_info, mask=None, with_color=False
):
    # 传进来的file都是没有转的
    # 只有
    # 第一个image不会被用到
    intrinsic_matrix = scene_info[scene_id]["intrinsic_matrix"]
    extrinsic_matrix = np.loadtxt(posefile)

    world_to_axis_align_matrix = scene_info[scene_id]["axis_align_matrix"]
    # world_to_axis_align_matrix = np.linalg.inv(world_to_axis_align_matrix)
    ###我在想如果不做align呢

    # 看好了，我传进去的都是未转
    depth_image_path = depthfile
    if with_color:
        color_image = colorfile
    else:
        color_image = None
    points_3d = project_mask_to_3dceshi(
        depth_image_path,
        intrinsic_matrix,
        extrinsic_matrix,
        mask,
        world_to_axis_align_matrix,
        color_image=color_image,
    )
    return points_3d


def ensemble_pred_points(
        scene_info,
        xuanranwenjianjia,
        posefile1,
        depthfile1,
        colorfile1,
        scene_id,
        image_id,
        pred_target_class,
        sam_mask,
        sam_mask_output_dir,
        intermediate_output_dir,
        detections,
):
    """
    Ensemble the predicted points from different images.

    Args:
        scene_id (str): The scene ID.
        image_id (str or int): The image ID.
        pred_target_class (str): The predicted target class.
        sam_mask (np.ndarray): The predicted segmentation mask.
    Returns:
        ensemble_points (np.ndarray): The ensemble points.

    """

    # 文件夹没有作用的所以不用管
    ensemble_image_ids = [image_id]
    ensemble_masks = [sam_mask]
    ensemble_points = []
    # projections for all the ids
    for current_image_id, current_mask in zip(ensemble_image_ids, ensemble_masks):
        current_aligned_points_3d = project_image_to_3d(posefile1, depthfile1, colorfile1, scene_id, image_id,
                                                        scene_info, sam_mask,
                                                        with_color=True)

        ensemble_points.append(current_aligned_points_3d)

    aligned_points_3d = np.concatenate(ensemble_points, axis=0)
    aligned_points_3d = uniform_downsampling(aligned_points_3d, 0.1)

    return aligned_points_3d


def post_process_mask(mask):
    """
    Process a binary mask to smooth and optionally remove small components.

    Args:
        mask (np.array): A 2D numpy array where the mask is boolean (True/False) or is `ultralytics.engine.results.Results`.
        opening (bool): If True, perform morphological opening to remove noise.
        remove_small_component (bool): If True, remove all but the largest connected component.

    Returns:
        np.array: The processed mask as a boolean numpy array.
    """
    is_ultralytics = False
    if isinstance(mask, Results):
        is_ultralytics = True
        ultra_result = mask.new()
        # get the masks
        mask = mask.masks.data.cpu().numpy()[0]

    # Convert boolean mask to uint8
    img = np.uint8(mask) * 255

    # Define the kernel for morphological operations
    kernel = np.ones((kernel_size * 2 + 1, kernel_size * 2 + 1), np.uint8)

    # Apply morphological erosion if requested
    if post_process_erosion:
        img = cv2.erode(img, kernel, iterations=1)

    # Apply morphological dilation if requested
    if post_process_dilation:
        img = cv2.dilate(img, kernel, iterations=1)

    # Find all connected components
    num_labels, labels_im = cv2.connectedComponents(
        img
    )  # label 0 is background, so start from 1
    if post_process_component and num_labels > 1:
        # Calculate the area of each component and sort them, keeping the largest k
        component_areas = [
            (label, np.sum(labels_im == label)) for label in range(1, num_labels)
        ]
        component_areas.sort(key=lambda x: x[1], reverse=True)
        largest_components = [
            x[0] for x in component_areas[: post_process_component_num]
        ]
        img = np.isin(labels_im, largest_components).astype(np.uint8)

    # Return the processed image as a boolean mask
    new_mask = img.astype(bool)

    if is_ultralytics:
        new_mask = torch.from_numpy(new_mask[None, :])  # should be tensor
        ultra_result.update(masks=new_mask)
        return ultra_result

    return new_mask


# 针对2d的我们可以这样，就是如果dz==0的话，或者哪一个是0的话
# 我们稍微给一点量0.05那样。
def huodezuijindedian(scene_info, pose, bboxpred1, scene_id):
    # 注意，传入的pose必须已经是对齐的了
    position = pose[:3, 3]
    center = bboxpred1[:3]
    # center = np.array(center)
    x, y, z, dx, dy, dz = bboxpred1
    radius = 2.0 * (math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)) / 2
    radius = max(1.5, radius)
    # 感觉可能没什么问题，不管是原始相机位姿在里面还是外面
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.sin(v) * np.cos(u)
    y = center[1] + radius * np.sin(v) * np.sin(u)
    z = center[2] + radius * np.cos(v)

    # 上扬角度
    theta = np.deg2rad(30)  # 30度转换为弧度

    # 计算圆的半径和中心
    r_circle = radius * np.cos(theta)  # 圆的半径
    z_circle = center[2] + radius * np.sin(theta)  # 圆的中心在z轴上的位置

    # 生成圆周上的点
    phi = np.linspace(0, 2 * np.pi, 16)  # 参数角度
    x_circle = center[0] + r_circle * np.cos(phi)
    y_circle = center[1] + r_circle * np.sin(phi)
    z_circle = np.full_like(phi, z_circle)  # z坐标是常数

    # 这是咱们的圆周
    # 下面来找离他最近的一个点
    # 计算相机位姿到圆周上所有点的距离
    # distances = np.sqrt((x_circle - position[0]) ** 2 + (y_circle - position[1]) ** 2 + (z_circle - position[2]) ** 2)

    # 找到距离最短的点
    # min_distance_index = np.argmin(distances)
    # closest_point = (x_circle[min_distance_index], y_circle[min_distance_index], z_circle[min_distance_index])
    poses = []
    for i in range(16):
        closest_point = (x_circle[i], y_circle[i], z_circle[i])
        # 做出这个点的位姿，x取投影的垂直分量
        # 1. 位置向量
        A = np.array(closest_point)
        C = np.array(center)

        # 2. 计算 z 轴方向（从 A 指向 C 的向量）
        z_axis = C - A
        z_axis = z_axis / np.linalg.norm(z_axis)  # 归一化

        # 3. 计算 x 轴方向（(0, 0, -1) 在 z 轴方向上的投影的垂直分量）
        up_vector = np.array([0, 0, -1])
        x_axis = up_vector - np.dot(up_vector, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)  # 归一化

        # 4. 计算 y 轴方向（通过右手定则，y 轴方向为 z 轴和 x 轴的叉积）
        y_axis = np.cross(z_axis, x_axis)

        ###之后要对x,y轴进行交换

        # 组合成旋转矩阵
        R = np.vstack((x_axis, y_axis, z_axis)).T
        R[:, 1] = -R[:, 1]
        R[:, [0, 1]] = R[:, [1, 0]]

        # view1[:, [0, 1]] = view1[:, [1, 0]]
        # view1[:, 1] = -view1[:, 1]

        # 组合成齐次变换矩阵
        posechushi = np.eye(4)
        posechushi[:3, :3] = R
        posechushi[:3, 3] = A
        np.save("chushipose.txt", posechushi)

        alignjuzhen = scene_info[scene_id]["axis_align_matrix"]
        alignnv = np.linalg.inv(alignjuzhen)
        posechushiqualign = np.dot(alignnv, posechushi)
        poses.append(posechushiqualign)

    return poses


def duiyinghuodezuijindedian(scene_info, pose, bboxpred1, scene_id):
    # 注意，传入的pose必须已经是对齐的了
    position = pose[:3, 3]
    center = bboxpred1[:3]
    # center = np.array(center)
    x, y, z, dx, dy, dz = bboxpred1
    radius = 4.0 * (math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)) / 2
    radius = max(1.3, radius)
    # 感觉可能没什么问题，不管是原始相机位姿在里面还是外面
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.sin(v) * np.cos(u)
    y = center[1] + radius * np.sin(v) * np.sin(u)
    z = center[2] + radius * np.cos(v)

    # 上扬角度
    theta = np.deg2rad(30)  # 30度转换为弧度
    # 20，低一点，这样能看到下面一些
    # 省的被挡住

    # 从下面看感觉不太好，感觉拉开距离会不会好一点

    # 计算圆的半径和中心
    r_circle = radius * np.cos(theta)  # 圆的半径
    z_circle = center[2] + radius * np.sin(theta)  # 圆的中心在z轴上的位置

    # 生成圆周上的点
    phi = np.linspace(0, 2 * np.pi, 16)  # 参数角度
    x_circle = center[0] + r_circle * np.cos(phi)
    y_circle = center[1] + r_circle * np.sin(phi)
    z_circle = np.full_like(phi, z_circle)  # z坐标是常数

    # 这是咱们的圆周
    # 下面来找离他最近的一个点
    # 计算相机位姿到圆周上所有点的距离
    # distances = np.sqrt((x_circle - position[0]) ** 2 + (y_circle - position[1]) ** 2 + (z_circle - position[2]) ** 2)

    # 找到距离最短的点
    # min_distance_index = np.argmin(distances)
    # closest_point = (x_circle[min_distance_index], y_circle[min_distance_index], z_circle[min_distance_index])
    poses = []
    for i in range(16):
        closest_point = (x_circle[i], y_circle[i], z_circle[i])
        # 做出这个点的位姿，x取投影的垂直分量
        # 1. 位置向量
        A = np.array(closest_point)
        C = np.array(center)

        # 2. 计算 z 轴方向（从 A 指向 C 的向量）
        z_axis = C - A
        z_axis = z_axis / np.linalg.norm(z_axis)  # 归一化

        # 3. 计算 x 轴方向（(0, 0, -1) 在 z 轴方向上的投影的垂直分量）
        up_vector = np.array([0, 0, -1])
        x_axis = up_vector - np.dot(up_vector, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)  # 归一化

        # 4. 计算 y 轴方向（通过右手定则，y 轴方向为 z 轴和 x 轴的叉积）
        y_axis = np.cross(z_axis, x_axis)

        # 组合成旋转矩阵
        R = np.vstack((x_axis, y_axis, z_axis)).T

        # 组合成齐次变换矩阵
        posechushi = np.eye(4)
        posechushi[:3, :3] = R
        posechushi[:3, 3] = A
        np.save("chushipose.txt", posechushi)

        alignjuzhen = scene_info[scene_id]["axis_align_matrix"]
        alignnv = np.linalg.inv(alignjuzhen)
        posechushiqualign = np.dot(alignnv, posechushi)
        poses.append(posechushiqualign)

    return poses


def get_det_bbox3d_from_image_idsingle(
        sam_predictor,
        scene_info,
        xuanranwenjianjia,
        scene_id,
        image_id,
        pred_target_class,
        pred_detection,
        desc,
        intermediate_output_dir,
        bbox_index=None,
        pred_sam_mask=None,
):
    # 这个也得要除了聚类，你这个单独的也需要一起用的。
    """
    Get the 3D bbox from the image_id.

    Args:
        scene_id (str): The scene ID.
        image_id (str): The image ID
        pred_target_class (str): The predicted target class
        pred_detection (Detection): The predicted detection
        gt_object_id (int): The ground truth object ID
        query (str): The query
        matched_image_ids (list): The matched image IDs
        intermediate_output_dir (str): The intermediate output directory
        bbox_index (int): The bbox index, used while ensembling points
        pred_sam_mask (np.ndarray): The predicted segmentation mask

    Raises:
        NotImplementedError: If the point filter type is not implemented.

    Returns:
        tuple: The predicted 3D bbox.
    """
    # ok，image_path弄出来之后，后面就只剩下pred_points那里的事情了

    # image_id是int所以不用变了
    if type(image_id) == str:
        return None
    zhuanhuan = "frame-" + f"{image_id:06d}" + "inv.jpg"
    color1 = "frame-" + f"{image_id:06d}" + ".jpg"
    depth1 = "frame-" + f"{image_id:06d}" + ".pgm"
    pose1 = "frame-" + f"{image_id:06d}" + ".txt"

    img_path = os.path.join(xuanranwenjianjia, zhuanhuan)
    colorfile1 = os.path.join(xuanranwenjianjia, color1)
    depthfile1 = os.path.join(xuanranwenjianjia, depth1)
    posefile1 = os.path.join(xuanranwenjianjia, pose1)

    sam_mask_output_dir = os.path.join(
        intermediate_output_dir,
        f"baseyesv2sam_mask_{image_id}_{pred_target_class.replace('/', 'or')}_target_{desc[:20]}",
    )

    if pred_sam_mask is not None:
        # pred seg
        result = pred_sam_mask
    else:
        # from scratch

        bbox_2d = pred_detection.xyxy

        mmengine.mkdir_or_exist(sam_mask_output_dir)

        # Get sam mask from 2d bbox
        # sam_predictor does not support batched inference at the moment, so len(results) is always 1
        # bbox_2d can be n * 4 or 1-d array of 4. If 1-d array, masks.data.shape would still be 1, H, W

        for xx1, yy1, xx2, yy2 in bbox_2d:
            x1 = xx1
            y1 = yy1
            x2 = xx2
            y2 = yy2
        center_x = (x1 + x2) / 2
        # 计算中心点的 y 坐标
        center_y = (y1 + y2) / 2

        labelss = np.array([1])
        pointss = np.array([[center_x, center_y]])

        result = sam_predictor(img_path, bboxes=bbox_2d, points=pointss, labels=labelss, verbose=False)[0]
        # save the mask
        result.save(f"{sam_mask_output_dir}/anchor_sam_raw.jpg")

        mmengine.dump(result, f"{sam_mask_output_dir}/anchor_sam_raw.pkl")
        mmengine.dump(pred_detection, f"{sam_mask_output_dir}/pred_detection.pkl")
        if True:
            result = post_process_mask(result)
            result.save(f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg")

        mmengine.dump(
            result, f"{sam_mask_output_dir}/anchor_sam_ks{kernel_size}.pkl"
        )

        # 暂时不用process了，后面再说

        # annotate the bbox on the ori image
        print(img_path)
        ori_image = Image.open(img_path)
        annotated_image = default_bbox_annotator.annotate(
            scene=ori_image, detections=pred_detection
        )
        annotated_image = default_label_annotator.annotate(
            scene=annotated_image, detections=pred_detection
        )
        # save the image
        annotated_image.save(f"{sam_mask_output_dir}/anchor_bbox.jpg")

        # save the original image for reference
        shutil.copy(img_path, f"{sam_mask_output_dir}/ancho_ori.jpg")
    # ok，获得好sam之后
    # 前面其实获得单张图片的sam也是没有问题的，所以问题关键再ensemble_pred_points这里

    # results[0].masks.data.shape would be torch.Size([n, H, W]), n is the number of the input bbox_2d
    sam_mask = result.masks.data.cpu().numpy()[0]
    if pred_detection is None:
        pred_detection = mmengine.load(f"{sam_mask_output_dir}/pred_detection.pkl")
    # ensemble adjacent images
    with TimeCounter(tag="EnsemblePredPoints", log_interval=60):
        aligned_points_3d = ensemble_pred_points(
            scene_info,
            xuanranwenjianjia,
            posefile1,
            depthfile1,
            colorfile1,
            scene_id,
            image_id,
            pred_target_class,
            sam_mask=sam_mask,
            sam_mask_output_dir=sam_mask_output_dir,
            intermediate_output_dir=intermediate_output_dir,
            detections=pred_detection
        )
    # 这一下的都不用改，我们需要改的就是上面怎么获得3dbbox的
    # 而且是align的
    with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
        if True:
            aligned_points_3d_filtered = remove_statistical_outliers(
                aligned_points_3d,
                nb_neighbors=5,
                std_ratio=1.0
            )

    # save the aligned points 3d
    aligned_points_3d_output_dir = os.path.join(
        intermediate_output_dir, "projected_points"
    )
    mmengine.mkdir_or_exist(aligned_points_3d_output_dir)
    np.save(
        f"{aligned_points_3d_output_dir}/ensemble{1}_matching.npy",
        aligned_points_3d,
    )
    np.save(
        f"{aligned_points_3d_output_dir}/ensemble{1}__matching_filtered.npy",
        aligned_points_3d_filtered,
    )

    # if nan in points, return None
    if (
            aligned_points_3d.shape[0] == 0
            or aligned_points_3d_filtered.shape[0] == 0
            or np.isnan(aligned_points_3d).any()
            or np.isnan(aligned_points_3d_filtered).any()
    ):
        return None, None, None, None

    pred_bbox = calculate_aabb(aligned_points_3d_filtered)
    poseceshi = np.loadtxt(posefile1)
    alignjuzhen = scene_info[scene_id]["axis_align_matrix"]
    poseceshialign = np.dot(alignjuzhen, poseceshi)

    return pred_bbox, poseceshialign, alignjuzhen, aligned_points_3d_filtered


def zhaodaozuijin(zhongxinls, indexls, cankaozhongxin):
    min_distance = float('inf')
    # 初始化最近点的索引
    closest_index = -1

    # 遍历所有点
    for i, point in enumerate(zhongxinls):
        # 计算当前点与参考点之间的欧几里得距离
        distance = math.sqrt((point[0] - cankaozhongxin[0]) ** 2 +
                             (point[1] - cankaozhongxin[1]) ** 2 +
                             (point[2] - cankaozhongxin[2]) ** 2)
        # 如果当前距离小于最小距离，更新最小距离和最近点的索引
        if distance < min_distance:
            min_distance = distance
            closest_index = i
    print("closest_index是多少为甚么会报错", "closest_index是多少", closest_index, "index_ls是多少", indexls)
    # 返回最近点的对应标识id
    return indexls[closest_index], min_distance


def get_det_bbox3d_center(
        sam_predictor,
        scene_info,
        xuanranwenjianjia,
        scene_id,
        image_id,
        pred_target_class,
        pred_detection,
        desc,
        intermediate_output_dir,
        bbox_index=None,
        pred_sam_mask=None,
):
    # 这个也得要除了聚类，你这个单独的也需要一起用的。
    """
    Get the 3D bbox from the image_id.

    Args:
        scene_id (str): The scene ID.
        image_id (str): The image ID
        pred_target_class (str): The predicted target class
        pred_detection (Detection): The predicted detection
        gt_object_id (int): The ground truth object ID
        query (str): The query
        matched_image_ids (list): The matched image IDs
        intermediate_output_dir (str): The intermediate output directory
        bbox_index (int): The bbox index, used while ensembling points
        pred_sam_mask (np.ndarray): The predicted segmentation mask

    Raises:
        NotImplementedError: If the point filter type is not implemented.

    Returns:
        tuple: The predicted 3D bbox.
    """
    # ok，image_path弄出来之后，后面就只剩下pred_points那里的事情了

    # image_id是int所以不用变了
    if type(image_id) == str:
        return None, None
    zhuanhuan = "frame-" + f"{image_id:06d}" + "inv.jpg"
    color1 = "frame-" + f"{image_id:06d}" + ".jpg"
    depth1 = "frame-" + f"{image_id:06d}" + ".pgm"
    pose1 = "frame-" + f"{image_id:06d}" + ".txt"

    img_path = os.path.join(xuanranwenjianjia, zhuanhuan)
    colorfile1 = os.path.join(xuanranwenjianjia, color1)
    depthfile1 = os.path.join(xuanranwenjianjia, depth1)
    posefile1 = os.path.join(xuanranwenjianjia, pose1)

    sam_mask_output_dir = os.path.join(
        intermediate_output_dir,
        f"{image_id}zhongxin",
    )

    if pred_sam_mask is not None:
        # pred seg
        result = pred_sam_mask
    else:
        # from scratch

        bbox_2d = pred_detection.xyxy

        # Get sam mask from 2d bbox
        # sam_predictor does not support batched inference at the moment, so len(results) is always 1
        # bbox_2d can be n * 4 or 1-d array of 4. If 1-d array, masks.data.shape would still be 1, H, W
        result = sam_predictor(img_path, bboxes=bbox_2d, verbose=False)[0]
        if True:
            result = post_process_mask(result)

        # 暂时不用process了，后面再说

        # annotate the bbox on the ori image

    # ok，获得好sam之后
    # 前面其实获得单张图片的sam也是没有问题的，所以问题关键再ensemble_pred_points这里

    # results[0].masks.data.shape would be torch.Size([n, H, W]), n is the number of the input bbox_2d
    sam_mask = result.masks.data.cpu().numpy()[0]
    # ensemble adjacent images
    with TimeCounter(tag="EnsemblePredPoints", log_interval=60):
        aligned_points_3d = ensemble_pred_points(
            scene_info,
            xuanranwenjianjia,
            posefile1,
            depthfile1,
            colorfile1,
            scene_id,
            image_id,
            pred_target_class,
            sam_mask=sam_mask,
            sam_mask_output_dir=sam_mask_output_dir,
            intermediate_output_dir=intermediate_output_dir,
            detections=pred_detection
        )
    # 这一下的都不用改，我们需要改的就是上面怎么获得3dbbox的
    # 而且是align的
    with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
        if True:
            aligned_points_3d_filtered = remove_statistical_outliers(
                aligned_points_3d,
                nb_neighbors=5,
                std_ratio=1.0
            )

    # if nan in points, return None
    if (
            aligned_points_3d.shape[0] == 0
            or aligned_points_3d_filtered.shape[0] == 0
            or np.isnan(aligned_points_3d).any()
            or np.isnan(aligned_points_3d_filtered).any()
    ):
        return None, None

    pred_bbox = calculate_aabb(aligned_points_3d_filtered)
    x, y, z, xx, yy, zz = pred_bbox
    return (x, y, z), result


def find_largest_cluster(clustered_data):
    """
    从聚类结果中找出包含点数最多的簇。

    参数:
    clustered_data (list): 存储聚类结果的列表，每个元素是一个 numpy 数组。

    返回:
    numpy.ndarray: 包含点数最多的簇的数据。
    int: 包含点数最多的簇的编号。
    """
    max_size = 0
    largest_cluster_index = None
    largest_cluster_data = None
    for i, cluster in enumerate(clustered_data):
        if len(cluster) > max_size:
            max_size = len(cluster)
            largest_cluster_index = i
            largest_cluster_data = cluster
    return largest_cluster_data


def cluster_numpy_array(arr):
    # 提取前三维坐标用于聚类
    coordinates = arr[:, :3]

    # 初始化 DBSCAN 聚类器，可根据需要调整参数
    dbscan = DBSCAN(eps=0.1, min_samples=50)
    # 进行聚类
    labels = dbscan.fit_predict(coordinates)

    # 获取聚类的数量
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters: {num_clusters}")

    # 存储聚类结果
    clustered_data = []
    for cluster_id in range(num_clusters):
        # 找到属于当前聚类的索引
        indices = np.where(labels == cluster_id)[0]
        # 将属于当前聚类的所有原始数据添加到结果中
        clustered_data.append(arr[indices])

    return clustered_data


def get_det_bbox3d_from_image_idjulei(
        sam_predictor,
        scene_info,
        xuanranwenjianjia,
        scene_id,
        image_id,
        pred_target_class,
        pred_detection,
        desc,
        intermediate_output_dir,
        bbox_index=None,
        pred_sam_mask=None,
):
    """
    Get the 3D bbox from the image_id.

    Args:
        scene_id (str): The scene ID.
        image_id (str): The image ID
        pred_target_class (str): The predicted target class
        pred_detection (Detection): The predicted detection
        gt_object_id (int): The ground truth object ID
        query (str): The query
        matched_image_ids (list): The matched image IDs
        intermediate_output_dir (str): The intermediate output directory
        bbox_index (int): The bbox index, used while ensembling points
        pred_sam_mask (np.ndarray): The predicted segmentation mask

    Raises:
        NotImplementedError: If the point filter type is not implemented.

    Returns:
        tuple: The predicted 3D bbox.
    """
    # ok，image_path弄出来之后，后面就只剩下pred_points那里的事情了

    # image_id是int所以不用变了
    # 允许初始的图片放在ceshi虾米那
    if type(image_id) == str:
        return None
    zhuanhuan = "frame-" + f"{image_id:06d}" + "inv.jpg"
    color1 = "frame-" + f"{image_id:06d}" + ".jpg"
    depth1 = "frame-" + f"{image_id:06d}" + ".pgm"
    pose1 = "frame-" + f"{image_id:06d}" + ".txt"

    img_path = os.path.join(xuanranwenjianjia, zhuanhuan)
    colorfile1 = os.path.join(xuanranwenjianjia, color1)
    depthfile1 = os.path.join(xuanranwenjianjia, depth1)
    posefile1 = os.path.join(xuanranwenjianjia, pose1)

    sam_mask_output_dir = os.path.join(
        xuanranwenjianjia,
        "juleimasktemp",
    )

    if pred_sam_mask is not None:
        # pred seg
        result = pred_sam_mask
    else:
        # from scratch

        bbox_2d = pred_detection.xyxy

        mmengine.mkdir_or_exist(sam_mask_output_dir)

        # Get sam mask from 2d bbox
        # sam_predictor does not support batched inference at the moment, so len(results) is always 1
        # bbox_2d can be n * 4 or 1-d array of 4. If 1-d array, masks.data.shape would still be 1, H, W
        result = sam_predictor(img_path, bboxes=bbox_2d, verbose=False)[0]
        # save the mask
        result.save(f"{sam_mask_output_dir}/anchor_sam_raw.jpg")

        mmengine.dump(result, f"{sam_mask_output_dir}/anchor_sam_raw.pkl")
        mmengine.dump(pred_detection, f"{sam_mask_output_dir}/pred_detection.pkl")
        if True:
            result = post_process_mask(result)
            result.save(f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg")

        mmengine.dump(
            result, f"{sam_mask_output_dir}/anchor_sam_ks{kernel_size}.pkl"
        )

        # 暂时不用process了，后面再说

        # annotate the bbox on the ori image
        print(img_path)
        ori_image = Image.open(img_path)
        annotated_image = default_bbox_annotator.annotate(
            scene=ori_image, detections=pred_detection
        )
        annotated_image = default_label_annotator.annotate(
            scene=annotated_image, detections=pred_detection
        )
        # save the image
        annotated_image.save(f"{sam_mask_output_dir}/anchor_bbox.jpg")

        # save the original image for reference
        shutil.copy(img_path, f"{sam_mask_output_dir}/ancho_ori.jpg")
    # ok，获得好sam之后
    # 前面其实获得单张图片的sam也是没有问题的，所以问题关键再ensemble_pred_points这里

    # results[0].masks.data.shape would be torch.Size([n, H, W]), n is the number of the input bbox_2d
    sam_mask = result.masks.data.cpu().numpy()[0]
    if pred_detection is None:
        pred_detection = mmengine.load(f"{sam_mask_output_dir}/pred_detection.pkl")
    # ensemble adjacent images
    with TimeCounter(tag="EnsemblePredPoints", log_interval=60):
        aligned_points_3d = ensemble_pred_points(
            scene_info,
            xuanranwenjianjia,
            posefile1,
            depthfile1,
            colorfile1,
            scene_id,
            image_id,
            pred_target_class,
            sam_mask=sam_mask,
            sam_mask_output_dir=sam_mask_output_dir,
            intermediate_output_dir=intermediate_output_dir,
            detections=pred_detection
        )
    # 这一下的都不用改，我们需要改的就是上面怎么获得3dbbox的
    # 而且是align的
    ###这里我们不考虑聚类了
    #ttt = cluster_numpy_array(aligned_points_3d)
    tt = 0
    w = []
    t = []
    t.append(aligned_points_3d)
    #t += ttt
    # 把最早的这个也放进去,并且放在前面
    for aligned_points_3d in t:
        tt += 1
        print("ceshijulei ", type(aligned_points_3d))

        with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
            if True:
                aligned_points_3d_filtered = remove_statistical_outliers(
                    aligned_points_3d,
                    nb_neighbors=5,
                    std_ratio=1.0
                )
        # save the aligned points 3d
        aligned_points_3d_output_dir = os.path.join(
            sam_mask_output_dir, "projected_points"
        )
        mmengine.mkdir_or_exist(aligned_points_3d_output_dir)
        np.save(
            f"{aligned_points_3d_output_dir}/ensemble_index_{tt}_matching.npy",
            aligned_points_3d,
        )
        np.save(
            f"{aligned_points_3d_output_dir}/ensemble_index_{tt}_matching_filtered.npy",
            aligned_points_3d_filtered,
        )

        # if nan in points, return None
        if (
                aligned_points_3d.shape[0] == 0
                or aligned_points_3d_filtered.shape[0] == 0
                or np.isnan(aligned_points_3d).any()
                or np.isnan(aligned_points_3d_filtered).any()
        ):
            return None, None, None

        pred_bbox = calculate_aabb(aligned_points_3d_filtered)
        poseceshi = np.loadtxt(posefile1)
        alignjuzhen = scene_info[scene_id]["axis_align_matrix"]
        poseceshialign = np.dot(alignjuzhen, poseceshi)
        w.append(pred_bbox)

###注意这里不会引起歧义，因为pose和align都是固有的而w得话我们只会取第一个
    return w, poseceshialign, alignjuzhen


def get_det_3dpoints_from_image_idcircle(sam_predictor, scene_info, xuanranwenjianjia, to_scanid, image_id,
                                         pred_target_class, pred_detection, desc, pred_sam_mask=None):
    # 讓ceshidir和xuanran一樣算了
    """
        Get the 3D bbox from the image_id.

        Args:
            scene_id (str): The scene ID.
            image_id (str): The image ID
            pred_target_class (str): The predicted target class
            pred_detection (Detection): The predicted detection
            gt_object_id (int): The ground truth object ID
            query (str): The query
            matched_image_ids (list): The matched image IDs
            intermediate_output_dir (str): The intermediate output directory
            bbox_index (int): The bbox index, used while ensembling points
            pred_sam_mask (np.ndarray): The predicted segmentation mask

        Raises:
            NotImplementedError: If the point filter type is not implemented.

        Returns:
            tuple: The predicted 3D bbox.
        """
    # ok，image_path弄出来之后，后面就只剩下pred_points那里的事情了

    # image_id是int所以不用变了
    if type(image_id) == str:
        return None
    zhuanhuan = "frame-" + f"{image_id:06d}" + "inv.jpg"
    color1 = "frame-" + f"{image_id:06d}" + ".jpg"
    depth1 = "frame-" + f"{image_id:06d}" + ".pgm"
    pose1 = "frame-" + f"{image_id:06d}" + ".txt"

    img_path = os.path.join(xuanranwenjianjia, zhuanhuan)
    colorfile1 = os.path.join(xuanranwenjianjia, color1)
    depthfile1 = os.path.join(xuanranwenjianjia, depth1)
    posefile1 = os.path.join(xuanranwenjianjia, pose1)

    sam_mask_output_dir = os.path.join(
        xuanranwenjianjia,
        "objectcentermask",
        f"sam_mask_{image_id}",
    )

    if pred_sam_mask is not None:
        # pred seg
        result = pred_sam_mask
    else:
        # from scratch

        bbox_2d = pred_detection.xyxy

        mmengine.mkdir_or_exist(sam_mask_output_dir)

        # Get sam mask from 2d bbox
        # sam_predictor does not support batched inference at the moment, so len(results) is always 1
        # bbox_2d can be n * 4 or 1-d array of 4. If 1-d array, masks.data.shape would still be 1, H, W
        result = sam_predictor(img_path, bboxes=bbox_2d, verbose=False)[0]
        # save the mask
        result.save(f"{sam_mask_output_dir}/anchor_sam_raw.jpg")

        mmengine.dump(result, f"{sam_mask_output_dir}/anchor_sam_raw.pkl")
        mmengine.dump(pred_detection, f"{sam_mask_output_dir}/pred_detection.pkl")
        if True:
            result = post_process_mask(result)
            result.save(f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg")

        mmengine.dump(
            result, f"{sam_mask_output_dir}/anchor_sam_ks{kernel_size}.pkl"
        )

        # 暂时不用process了，后面再说

        # annotate the bbox on the ori image
        print(img_path)
        ori_image = Image.open(img_path)
        annotated_image = default_bbox_annotator.annotate(
            scene=ori_image, detections=pred_detection
        )
        annotated_image = default_label_annotator.annotate(
            scene=annotated_image, detections=pred_detection
        )
        # save the image
        annotated_image.save(f"{sam_mask_output_dir}/anchor_bbox.jpg")

        # save the original image for reference
        shutil.copy(img_path, f"{sam_mask_output_dir}/ancho_ori.jpg")
    # ok，获得好sam之后
    # 前面其实获得单张图片的sam也是没有问题的，所以问题关键再ensemble_pred_points这里

    # results[0].masks.data.shape would be torch.Size([n, H, W]), n is the number of the input bbox_2d
    sam_mask = result.masks.data.cpu().numpy()[0]
    if pred_detection is None:
        pred_detection = mmengine.load(f"{sam_mask_output_dir}/pred_detection.pkl")
    # ensemble adjacent images
    with TimeCounter(tag="EnsemblePredPoints", log_interval=60):
        aligned_points_3d = ensemble_pred_points(
            scene_info,
            xuanranwenjianjia,
            posefile1,
            depthfile1,
            colorfile1,
            to_scanid,
            image_id,
            pred_target_class,
            sam_mask=sam_mask,
            sam_mask_output_dir=sam_mask_output_dir,
            intermediate_output_dir=xuanranwenjianjia,
            detections=pred_detection
        )
    # 这一下的都不用改，我们需要改的就是上面怎么获得3dbbox的
    # 而且是align的

    with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
        if True:
            aligned_points_3d_filtered = remove_statistical_outliers(
                aligned_points_3d,
                nb_neighbors=5,
                std_ratio=1.0
            )

    # if nan in points, return None
    if (
            aligned_points_3d.shape[0] == 0
            or aligned_points_3d_filtered.shape[0] == 0
            or np.isnan(aligned_points_3d).any()
            or np.isnan(aligned_points_3d_filtered).any()
    ):
        return None

    return aligned_points_3d_filtered


def stitch_and_encode_images_to_base64(
        images, images_id, intermediate_output_dir=None
):
    """
    Stitch and encode a list of images into a grid and convert them to base64 format.

    Args:
        images (list): A list of PIL images to be stitched.
        images_id (list): A list of image IDs corresponding to the images.
        intermediate_output_dir (str, optional): The directory to save the grid images. Defaults to None.

    Returns:
        list: A list of base64 encoded images.

    """
    # determine layouts

    # square stitching
    num_views_pre_selection = len(images)
    grid_nums = math.ceil(num_views_pre_selection / 6)
    column = math.ceil(grid_nums ** 0.5)
    row = math.ceil(grid_nums / column)
    grid_layout = (row, column)

    # stitch images

    grid_images = dynamic_stitch_images_fix_v2(
        images,
        fix_num=6,
        ID_array=images_id,
        ID_color="red",
        annotate_id=True,
    )

    # save all the grid images
    if intermediate_output_dir is not None:
        grid_images_output_dir = os.path.join(
            intermediate_output_dir, "grid_images"
        )
        mmengine.mkdir_or_exist(grid_images_output_dir)

        # save one by one
        for index, grid_image in enumerate(grid_images):
            grid_image.save(f"{grid_images_output_dir}/grid{index}.jpg")

    # encode to base64
    base64Frames = [
        encode_PIL_image_to_base64(resize_image(image)) for image in grid_images
    ]
    return base64Frames


# okk，额外的函数写好了
def apixuanranweitiaoleibbox(
        imageid, imagefilepath, apibbox_prompts, xinintermediate_output_dir, classtarget1,
        jiancemodel, des, cankaotuxiangbase, chushidaxiao
):
    # 去选出来新的image里面合适的id
    # ceshiid是为了存文件用的
    # 这个是需要路径的
    xinimage = Image.open(imagefilepath)
    detections = jiancemodel.detect(imagefilepath, classtarget1)
    detections = detections[detections.confidence > 0.25]
    num_candidate = len(detections)
    if num_candidate == 0:
        return -1, None
    messages = []
    bbox_index_gpt_select_output_dir = os.path.join(
        xinintermediate_output_dir, "findfbboxselect", f"index_{imageid}"
    )
    mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)

    jiheids = []
    jiheanoimages = []
    for detgeshu in range(num_candidate):
        www = detections[detgeshu]
        for xx1, yy1, xx2, yy2 in www.xyxy:
            x1 = xx1
            y1 = yy1
            x2 = xx2
            y2 = yy2
        width = x2 - x1  # xmax - xmin
        height = y2 - y1  # ymax - ymin
        bili = float(width * height / 1254432)
        print("现在的detection比例多少", bili)
        if bili >= 0.75:
            # 一般来说不会这么大
            print("是那种恶心的框全图的情况")
            continue

        aili = float(width * height / chushidaxiao)
        print("现在的detection与初始比例多少", aili)
        if aili <= 0.49:
            # 一般来说不会这么大
            print("是那种太小的框的情况")
            continue

        jiheids.append(detgeshu)

        ori_image = xinimage
        annotated_image = default_bbox_annotator.annotate(
            scene=ori_image, detections=detections[detgeshu]
        )
        annotated_image = default_label_annotator.annotate(
            scene=annotated_image, detections=detections[detgeshu]
        )
        jiheanoimages.append(annotated_image)
    print("现在的jiheids是多少，会不会有问题", jiheids)
    base64Frames = stitch_and_encode_images_to_base64(
        jiheanoimages, jiheids, intermediate_output_dir=bbox_index_gpt_select_output_dir
    )

    # 这里我们需要处理的只有一个图像
    # 欧，我想起来了，一个是框，一个是id。
    # 我们可以用format去搞

    multi_prompt = apibbox_prompts.format(
        target=classtarget1,
        num_candidate_bboxes=num_candidate,
    )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": multi_prompt},
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                            "detail": "high",
                        },
                    },
                    base64Frames,
                ),
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{cankaotuxiangbase}",
                        "detail": "high",
                    },
                },

            ],
        }
    )

    # call VLM to get the result
    cost = 0
    retry = 1
    while retry <= 2:
        bbox_index = -1
        gpt_message = None

        gpt_response = openaigpt.safe_chat_complete(
            messages, response_format={"type": "json_object"}, content_only=True
        )

        cost += gpt_response["cost"]
        gpt_content = gpt_response["content"]
        if (gpt_content[0] == '`'):
            gpt_content = gpt_content.strip()[8:-4]
        print(gpt_content)
        gpt_content_json = json.loads(gpt_content)
        gpt_message = {
            "role": "assistant",
            "content": [{"text": gpt_response["content"], "type": "text"}],
        }
        messages.append(gpt_message)
        # save gpt_content_json

        bbox_index = int(gpt_content_json.get("object_id", -1))
        if bbox_index > (num_candidate - 1):
            print("重新试一下")
            retry += 1
            continue
        else:
            # 防止一直循环
            break
    # 我们看看它怎么选框的
    print("它每次是怎么选具体是哪个框的，", imageid, " ", gpt_content_json)
    if bbox_index == -1 or bbox_index > (num_candidate - 1):
        return -1, None
    else:
        return bbox_index, detections[bbox_index]


def apixuanranweitiaoleigaibbox(
        imageid, imagefilepath, apibbox_prompts, xinintermediate_output_dir, classtarget1,
        jiancemodel, des, cankaotuxiangbase, chuzhongxindian, sam_predictor, scene_info, toscanid, bigstuff
):
    # 去选出来新的image里面合适的id
    # ceshiid是为了存文件用的
    # 这个是需要路径的
    xinimage = Image.open(imagefilepath)
    detections = jiancemodel.detect(imagefilepath, classtarget1)
    detections = detections[detections.confidence > 0.25]
    num_candidate = len(detections)
    if num_candidate == 0:
        return -1, None, None
    messages = []
    bbox_index_gpt_select_output_dir = os.path.join(
        xinintermediate_output_dir, "findfbboxselect", f"index_{imageid}"
    )
    sam_dir = os.path.join(
        xinintermediate_output_dir, "sammask", f"index_{imageid}"
    )

    mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)

    jiheids = []
    jiheanoimages = []
    jihedet = []
    for detgeshu in range(num_candidate):
        www = detections[detgeshu]
        for xx1, yy1, xx2, yy2 in www.xyxy:
            x1 = xx1
            y1 = yy1
            x2 = xx2
            y2 = yy2
        width = x2 - x1  # xmax - xmin
        height = y2 - y1  # ymax - ymin
        bili = float(width * height / 1254432)
        print("现在的detection比例多少", bili)
        if bili >= 0.8:
            # 一般来说不会这么大
            print("是那种恶心的框全图的情况")
            continue

        jiheids.append(detgeshu)

        ori_image = xinimage
        annotated_image = default_bbox_annotator.annotate(
            scene=ori_image, detections=detections[detgeshu]
        )
        annotated_image = default_label_annotator.annotate(
            scene=annotated_image, detections=detections[detgeshu]
        )
        jiheanoimages.append(annotated_image)
        jihedet.append(detections[detgeshu])
    print("现在的jiheids是多少，会不会有问题", jiheids)
    jihecenter = []
    jihemask = []
    if len(jiheids) == 0:
        return -1, None, None
    for item in jihedet:
        x, s = get_det_bbox3d_center(sam_predictor, scene_info, xinintermediate_output_dir, toscanid, imageid,
                                     classtarget1, item,
                                     des, xinintermediate_output_dir)
        if x is None:
            continue
        jihemask.append(s)
        jihecenter.append(x)
    if len(jihecenter) == 0:
        return -1, None, None
    bbox_index, zuixiaojuli = zhaodaozuijin(jihecenter, jiheids, chuzhongxindian)
    if bigstuff:
        print("本身物体很大不需要限制距离")
    else:
        if zuixiaojuli<=0.25:
            print("正常完整的在0.2m以内")
        else:
            print("正常完整的超过了0.2m")
            return -1, None, None
    f = jiheids.index(bbox_index)
    detimg = jiheanoimages[f]
    detmask = jihemask[f]

    base64Frames = stitch_and_encode_images_to_base64(
        jiheanoimages, jiheids, intermediate_output_dir=bbox_index_gpt_select_output_dir
    )
    base64 = encode_PIL_image_to_base64(
        resize_image(detimg)
    )

    # 这里我们需要处理的只有一个图像
    # 欧，我想起来了，一个是框，一个是id。
    # 我们可以用format去搞

    multi_prompt = apibbox_prompts.format(
        target=classtarget1,
    )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": multi_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{cankaotuxiangbase}",
                        "detail": "high",
                    },
                },

            ],
        }
    )

    # call VLM to get the result
    cost = 0
    retry = 1
    """while retry <= 2:
        gpt_message = None

        gpt_response = openaigpt.safe_chat_complete(
            messages, response_format={"type": "json_object"}, content_only=True
        )

        cost += gpt_response["cost"]
        gpt_content_json = json.loads(gpt_response["content"])
        gpt_message = {
            "role": "assistant",
            "content": [{"text": gpt_response["content"], "type": "text"}],
        }
        messages.append(gpt_message)
        # save gpt_content_json

        ceshi = gpt_content_json.get("correct", False)
        if ceshi is False:
            print("重新试一下")
            retry += 1
            continue
        else:
            # 防止一直循环
            break
    # 我们看看它怎么选框的
    print("它每次怎么判断这个框对不对，", imageid, " ", gpt_content_json)
    """
    # 如果检测到了我们就可以存了。
    ceshi = True
    if ceshi is False:
        return -1, None, None
    else:
        print("这里的imageid是多少，我们选的bbox_index是多少，", imageid, " ", bbox_index)
        mmengine.mkdir_or_exist(sam_dir)
        annoimgfile = os.path.join(sam_dir, "annotated_image_bbox.jpg")
        maskfile = os.path.join(sam_dir, "maskfilter_image.jpg")
        detimg.save(annoimgfile)
        detmask.save(maskfile)
        return bbox_index, detections[bbox_index], detimg


def apisanhuo(quanimgpathls, quanidls, apibbox_prompts, dirforeachdet,
              classtarget,
              desc, cankaotuxiangbase):
    # 其实这个弄完我们就能尝试一下了，看看效果怎么样，我们主要就是为了拿到较好的图像，这样到时候能挑出来较好的框，以及注意了，我们可以先把
    # 计算框大小大于85的那个框删掉，我们来先写这个。
    # 去选出来新的image里面合适的id
    # ceshiid是为了存文件用的
    # 这个是需要路径的
    bbox_index_gpt_select_output_dir = os.path.join(
        dirforeachdet, "huosange"
    )
    mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)
    quanimg = [Image.open(img) for img in quanimgpathls]
    # 我们不需要标注，这里只需要标上图片的id就可以了
    base64Frames = stitch_and_encode_images_to_base64(
        quanimg, quanidls, intermediate_output_dir=bbox_index_gpt_select_output_dir
    )

    # 这里我们需要处理的只有一个图像
    # 欧，我想起来了，一个是框，一个是id。
    # 我们可以用format去搞

    multi_prompt = apibbox_prompts.format(
        target=classtarget,
        num_images=len(quanimgpathls),
    )
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": multi_prompt},
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                            "detail": "high",
                        },
                    },
                    base64Frames,
                ),
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{cankaotuxiangbase}",
                        "detail": "high",
                    },
                },

            ],
        }
    )

    # call VLM to get the result
    cost = 0
    retry = 1
    while retry <= 2:
        bbox_index = -1
        gpt_message = None

        gpt_response = openaigpt.safe_chat_complete(
            messages, response_format={"type": "json_object"}, content_only=True
        )
        print(gpt_response)
        cost += gpt_response["cost"]
        gpt_content = gpt_response["content"]
        if (gpt_content[0] == '`'):
            gpt_content = gpt_content.strip()[8:-4]
        print(gpt_content)
        gpt_content_json = json.loads(gpt_content)
        gpt_message = {
            "role": "assistant",
            "content": [{"text": gpt_response["content"], "type": "text"}],
        }
        messages.append(gpt_message)
        # save gpt_content_json

        imgidls = gpt_content_json.get("image_ids", [0, 1, 2, 3, 4])
        if imgidls is None: 
            print("重新试一下")
            retry += 1
            continue
        else:
            # 防止一直循环
            break

    if imgidls is None:
        return None, None
    else:
        dedaosanimgls = []
        for wx in imgidls:
            dedaosanimgls.append(quanimgpathls[wx])
        return imgidls, dedaosanimgls


def juedingxubuxuyao(yaobuyaoduiyingimgls, idls, xuyao_prompt, classtarget, duiyingdir):
    xubuxuyaodir = os.path.join(duiyingdir, "xubuxu")
    mmengine.mkdir_or_exist(xubuxuyaodir)
    xubuxuyao = False
    base64Frames = stitch_and_encode_images_to_base64(
        yaobuyaoduiyingimgls, idls, intermediate_output_dir=xubuxuyaodir
    )
    multi_prompt = xuyao_prompt.format(
        target=classtarget,
    )
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": multi_prompt},
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                            "detail": "high",
                        },
                    },
                    base64Frames,
                ),
            ],
        }
    )

    # call VLM to get the result
    cost = 0
    retry = 1
    while retry <= 2:
        gpt_message = None

        gpt_response = openaigpt.safe_chat_complete(
            messages, response_format={"type": "json_object"}, content_only=True
        )

        cost += gpt_response["cost"]
        gpt_content = gpt_response["content"]
        if (gpt_content[0] == '`'):
            gpt_content = gpt_content.strip()[8:-4]
        print(gpt_content)
        gpt_content_json = json.loads(gpt_content)
        gpt_message = {
            "role": "assistant",
            "content": [{"text": gpt_response["content"], "type": "text"}],
        }
        messages.append(gpt_message)
        # save gpt_content_json

        xu = gpt_content_json.get("answer", True)
        print("看看xubuxuyao的判断 ", gpt_content_json)
        if xu is None:
            print("重新试一下")
            retry += 1
            continue
        else:
            # 防止一直循环
            break

    if xu is None:
        return True
    else:
        if xu is True:
            return True
        else:
            return False


def apixuanranweitiaoleigaibboxduiying(imageid, duiyingimgpath,
                                       xinintermediate_output_dir,
                                       classtarget1,
                                       jiancemodel, chuzhongxindian,
                                       sam_predictor, scene_info, toscanid, des):
    # 去选出来新的image里面合适的id
    # ceshiid是为了存文件用的
    # 这个是需要路径的
    xinimage = Image.open(duiyingimgpath)
    detections = jiancemodel.detect(duiyingimgpath, classtarget1)
    detections = detections[detections.confidence > 0.25]
    num_candidate = len(detections)
    if num_candidate == 0:
        return -1, None
    messages = []

    sam_dir = os.path.join(
        xinintermediate_output_dir, "sammask", f"index_{imageid}"
    )
    jiheids = []
    jiheanoimages = []
    jihedet = []
    for detgeshu in range(num_candidate):
        www = detections[detgeshu]
        for xx1, yy1, xx2, yy2 in www.xyxy:
            x1 = xx1
            y1 = yy1
            x2 = xx2
            y2 = yy2
        width = x2 - x1  # xmax - xmin
        height = y2 - y1  # ymax - ymin
        bili = float(width * height / 1254432)
        print("现在的detection比例多少", bili)
        if bili >= 0.75:
            # 一般来说不会这么大
            print("是那种恶心的框全图的情况")
            continue

        jiheids.append(detgeshu)

        ori_image = xinimage
        annotated_image = default_bbox_annotator.annotate(
            scene=ori_image, detections=detections[detgeshu]
        )
        annotated_image = default_label_annotator.annotate(
            scene=annotated_image, detections=detections[detgeshu]
        )
        jiheanoimages.append(annotated_image)
        jihedet.append(detections[detgeshu])
    print("现在的jiheids是多少，会不会有问题", jiheids)
    jihecenter = []
    jihemask = []
    if len(jiheids) == 0:
        return -1, None
    for item in jihedet:
        x, s = get_det_bbox3d_center(sam_predictor, scene_info, xinintermediate_output_dir, toscanid, imageid,
                                     classtarget1, item,
                                     des, xinintermediate_output_dir)
        if x is None:
            continue
        jihemask.append(s)
        jihecenter.append(x)
    if len(jihecenter) == 0:
        return -1, None
    bbox_index = zhaodaozuijin(jihecenter, jiheids, chuzhongxindian)
    f = jiheids.index(bbox_index)
    detimg = jiheanoimages[f]
    detmask = jihemask[f]

    print("这里的imageid是多少，我们选的bbox_index是多少，", imageid, " ", bbox_index)
    mmengine.mkdir_or_exist(sam_dir)
    annoimgfile = os.path.join(sam_dir, "annotated_image_bbox.jpg")
    maskfile = os.path.join(sam_dir, "maskfilter_image.jpg")
    detimg.save(annoimgfile)
    detmask.save(maskfile)
    return bbox_index, detections[bbox_index]


def iou(box11, box22):
    # box1 and box2 are lists or tuples of the form [xmin, ymin, xmax, ymax]
    # Calculate the (x, y)-coordinates of the intersection rectangle
    box1 = []
    box2 = []
    for xx1, yy1, xx2, yy2 in box11:
        x1 = xx1
        y1 = yy1
        x2 = xx2
        y2 = yy2
        box1.append(x1)
        box1.append(y1)
        box1.append(x2)
        box1.append(y2)
    for xx1, yy1, xx2, yy2 in box22:
        x1 = xx1
        y1 = yy1
        x2 = xx2
        y2 = yy2
        box2.append(x1)
        box2.append(y1)
        box2.append(x2)
        box2.append(y2)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Compute the area of both rectangles
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of both areas minus the inter_area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def cesuan2iou(detectionsmao, pred_detection):
    pred_box = pred_detection.xyxy
    max_iou = 0
    max_index = -1

    for index in range(len(detectionsmao)):
        detection = detectionsmao[index]
        box = detection.xyxy
        current_iou = iou(pred_box, box)
        if current_iou > max_iou:
            max_iou = current_iou
            max_index = index

    if max_iou > 0.3:
        return True, max_index, detectionsmao[max_index]
    else:
        return False, max_index, detectionsmao[max_index]


def dedaoneg(supdet, pred_detection, sam_mask_output_dir, sam_predictor, img_path):
    bbox_2dmao = supdet.xyxy
    bbox_2d = pred_detection.xyxy

    mmengine.mkdir_or_exist(sam_mask_output_dir)

    # Get sam mask from 2d bbox
    # sam_predictor does not support batched inference at the moment, so len(results) is always 1
    # bbox_2d can be n * 4 or 1-d array of 4. If 1-d array, masks.data.shape would still be 1, H, W
    # 就是在result的前面对点进行处理。
    result_mao = sam_predictor(img_path, bboxes=bbox_2dmao, verbose=False)[0]
    sam_maskmao = result_mao.masks.data.cpu().numpy()[0]
    mask_numpy = sam_maskmao

    # 获取掩码为 True 的点的坐标
    true_points = np.argwhere(mask_numpy)  # 返回一个二维数组，每行是 (y, x) 坐标
    num_true_points = np.sum(mask_numpy)
    print(num_true_points)
    print("坐标点 (y, x)：")
    print(true_points)
    true_points = true_points[::500]
    points = true_points

    # 如果需要，可以将其转换为 NumPy 数组
    points = np.array(points)

    # 打印结果
    print("points:")
    print(points)

    # 如果需要，可以为这些点创建对应的 labels
    labels = np.zeros(len(points), dtype=int)  # 假设所有点的标签都是 0
    print("labels:")
    print(labels)
    result_mao.save(f"{sam_mask_output_dir}/anchor_sam_raw_mao.jpg")
    return points, labels


def get_det_bbox3d_from_image_idsinglesup(
        sam_predictor,
        scene_info,
        xuanranwenjianjia,
        scene_id,
        image_id,
        pred_target_class,
        pred_detection,
        desc,
        intermediate_output_dir,
        mao,
        jiancemodel,
        bbox_index=None,
        pred_sam_mask=None,
):
    # 这个也得要除了聚类，你这个单独的也需要一起用的。
    """
    Get the 3D bbox from the image_id.

    Args:
        scene_id (str): The scene ID.
        image_id (str): The image ID
        pred_target_class (str): The predicted target class
        pred_detection (Detection): The predicted detection
        gt_object_id (int): The ground truth object ID
        query (str): The query
        matched_image_ids (list): The matched image IDs
        intermediate_output_dir (str): The intermediate output directory
        bbox_index (int): The bbox index, used while ensembling points
        pred_sam_mask (np.ndarray): The predicted segmentation mask

    Raises:
        NotImplementedError: If the point filter type is not implemented.

    Returns:
        tuple: The predicted 3D bbox.
    """
    # ok，image_path弄出来之后，后面就只剩下pred_points那里的事情了

    # image_id是int所以不用变了
    if type(image_id) == str:
        return None
    zhuanhuan = "frame-" + f"{image_id:06d}" + "inv.jpg"
    color1 = "frame-" + f"{image_id:06d}" + ".jpg"
    depth1 = "frame-" + f"{image_id:06d}" + ".pgm"
    pose1 = "frame-" + f"{image_id:06d}" + ".txt"

    img_path = os.path.join(xuanranwenjianjia, zhuanhuan)
    colorfile1 = os.path.join(xuanranwenjianjia, color1)
    depthfile1 = os.path.join(xuanranwenjianjia, depth1)
    posefile1 = os.path.join(xuanranwenjianjia, pose1)
    xinimage = Image.open(img_path)
    detectionsmao = jiancemodel.detect(img_path, mao)
    detectionsmao = detectionsmao[detectionsmao.confidence > 0.25]
    maol = len(detectionsmao)
    if maol == 0:
        xiugai = True
    else:
        xiugai = False
    flag = False
    # 初始我们认为是不需要的
    if not xiugai:
        flag, supindex, supdet = cesuan2iou(detectionsmao, pred_detection)

    sam_mask_output_dir = os.path.join(
        intermediate_output_dir,
        f"baseyesv2sam_mask_{image_id}_{pred_target_class.replace('/', 'or')}_target_{desc}",
    )

    if pred_sam_mask is not None:
        # pred seg
        result = pred_sam_mask
    else:
        # from scratch
        if not flag:
            bbox_2d = pred_detection.xyxy

            mmengine.mkdir_or_exist(sam_mask_output_dir)

            # Get sam mask from 2d bbox
            # sam_predictor does not support batched inference at the moment, so len(results) is always 1
            # bbox_2d can be n * 4 or 1-d array of 4. If 1-d array, masks.data.shape would still be 1, H, W
            result = sam_predictor(img_path, bboxes=bbox_2d, verbose=False)[0]
            # save the mask
            result.save(f"{sam_mask_output_dir}/anchor_sam_raw.jpg")

            mmengine.dump(result, f"{sam_mask_output_dir}/anchor_sam_raw.pkl")
            mmengine.dump(pred_detection, f"{sam_mask_output_dir}/pred_detection.pkl")
            if True:
                result = post_process_mask(result)
                result.save(f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg")

            mmengine.dump(
                result, f"{sam_mask_output_dir}/anchor_sam_ks{kernel_size}.pkl"
            )

            # 暂时不用process了，后面再说

            # annotate the bbox on the ori image
            print(img_path)
            ori_image = Image.open(img_path)
            annotated_image = default_bbox_annotator.annotate(
                scene=ori_image, detections=pred_detection
            )
            annotated_image = default_label_annotator.annotate(
                scene=annotated_image, detections=pred_detection
            )
            # save the image
            annotated_image.save(f"{sam_mask_output_dir}/anchor_bbox.jpg")

            # save the original image for reference
            shutil.copy(img_path, f"{sam_mask_output_dir}/ancho_ori.jpg")

        else:
            bbox_2d = pred_detection.xyxy
            points, labels = dedaoneg(supdet, pred_detection, sam_mask_output_dir, sam_predictor, img_path)

            result = sam_predictor(img_path, bboxes=bbox_2d, points=points, labels=labels, verbose=False)[0]
            # save the mask
            result.save(f"{sam_mask_output_dir}/anchor_sam_raw.jpg")

            mmengine.dump(result, f"{sam_mask_output_dir}/anchor_sam_raw.pkl")
            mmengine.dump(pred_detection, f"{sam_mask_output_dir}/pred_detection.pkl")
            if True:
                result = post_process_mask(result)
                result.save(f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg")

            mmengine.dump(
                result, f"{sam_mask_output_dir}/anchor_sam_ks{kernel_size}.pkl"
            )

            # 暂时不用process了，后面再说

            # annotate the bbox on the ori image
            print(img_path)
            ori_image = Image.open(img_path)
            annotated_image = default_bbox_annotator.annotate(
                scene=ori_image, detections=pred_detection
            )
            annotated_image = default_label_annotator.annotate(
                scene=annotated_image, detections=pred_detection
            )
            # save the image
            annotated_image.save(f"{sam_mask_output_dir}/anchor_bbox.jpg")

            # save the original image for reference
            shutil.copy(img_path, f"{sam_mask_output_dir}/ancho_ori.jpg")

    # ok，获得好sam之后
    # 前面其实获得单张图片的sam也是没有问题的，所以问题关键再ensemble_pred_points这里

    # results[0].masks.data.shape would be torch.Size([n, H, W]), n is the number of the input bbox_2d
    sam_mask = result.masks.data.cpu().numpy()[0]
    if pred_detection is None:
        pred_detection = mmengine.load(f"{sam_mask_output_dir}/pred_detection.pkl")
    # ensemble adjacent images
    with TimeCounter(tag="EnsemblePredPoints", log_interval=60):
        aligned_points_3d = ensemble_pred_points(
            scene_info,
            xuanranwenjianjia,
            posefile1,
            depthfile1,
            colorfile1,
            scene_id,
            image_id,
            pred_target_class,
            sam_mask=sam_mask,
            sam_mask_output_dir=sam_mask_output_dir,
            intermediate_output_dir=intermediate_output_dir,
            detections=pred_detection
        )
    # 这一下的都不用改，我们需要改的就是上面怎么获得3dbbox的
    # 而且是align的
    with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
        if True:
            aligned_points_3d_filtered = remove_statistical_outliers(
                aligned_points_3d,
                nb_neighbors=5,
                std_ratio=1.0
            )
    # save the aligned points 3d
    aligned_points_3d_output_dir = os.path.join(
        intermediate_output_dir, "projected_points"
    )
    mmengine.mkdir_or_exist(aligned_points_3d_output_dir)
    np.save(
        f"{aligned_points_3d_output_dir}/ensemble{1}_matching.npy",
        aligned_points_3d,
    )
    np.save(
        f"{aligned_points_3d_output_dir}/ensemble{1}__matching_filtered.npy",
        aligned_points_3d_filtered,
    )

    # if nan in points, return None
    if (
            aligned_points_3d.shape[0] == 0
            or aligned_points_3d_filtered.shape[0] == 0
            or np.isnan(aligned_points_3d).any()
            or np.isnan(aligned_points_3d_filtered).any()
    ):
        return None, None, None, None

    pred_bbox = calculate_aabb(aligned_points_3d_filtered)
    poseceshi = np.loadtxt(posefile1)
    alignjuzhen = scene_info[scene_id]["axis_align_matrix"]
    poseceshialign = np.dot(alignjuzhen, poseceshi)

    return pred_bbox, poseceshialign, alignjuzhen, aligned_points_3d_filtered

def juedingxianju(
        cankaobase, classtarget
):
    # 去选出来新的image里面合适的id
    # ceshiid是为了存文件用的
    # 这个是需要路径的

    annotated_image_base64 = cankaobase
    # 这里我们需要处理的只有一个图像
    # 欧，我想起来了，一个是框，一个是id。
    # 我们可以用format去搞

    multi_prompt = xianju_prompt.format(
        targetclass = classtarget
    )
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": multi_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{annotated_image_base64}",
                        "detail": "high",
                    },
                },
            ],
        }
    )

    # call VLM to get the result
    cost = 0
    retry = 1
    while retry <= 2:
        bbox_index = -1
        gpt_message = None

        gpt_response = openaigpt.safe_chat_complete(
            messages, response_format={"type": "json_object"}, content_only=True
        )

        cost += gpt_response["cost"]
        gpt_content = gpt_response["content"]
        if (gpt_content[0] == '`'):
            gpt_content = gpt_content.strip()[8:-4]
        print(gpt_content)
        gpt_content_json = json.loads(gpt_content)
        print(gpt_content_json)
        gpt_message = {
            "role": "assistant",
            "content": [{"text": gpt_response["content"], "type": "text"}],
        }
        messages.append(gpt_message)
        # save gpt_content_json

        bigstuff = gpt_content_json.get("limit", True)
        break



    return bigstuff


def zhuhanshu(posels, chushidet, scene_info, detectmodel, chushijpgpath, chushiposepath, yaoqiuwenjianjia, to_scanid,
              godxuanran, classtarget, desc, sam_predictor, cankaotuxiangbase, shibushisu, maomao):
    print("进入投影部分")
    if cankaotuxiangbase is None:
        return posels, None
    cunchulujing = chushijpgpath
    bigstuff = juedingxianju(cankaotuxiangbase, classtarget)

    if isinstance(chushiposepath, str):
        posechushilujing = chushiposepath
        pose = np.loadtxt(posechushilujing)
    else:
        pose = chushiposepath
    ceshijieguo = yaoqiuwenjianjia
    tempfile = os.path.join(ceshijieguo, "tempfile")
    # 这里得idpanduan是在渲染中得target_id
    # 根据这个id获得
    ###没有什么必要得，真正重要的是
    ###targetimage得路径，pose路径，以及初始得det，我们之后要做得就是完成mao那块得内容了

    # pose已经是处理好的，不需要变,这一步必须加，因为你前面处理完之后停的位置不一定在哪
    posels.append(pose)

    # 初始的pose

    # 先对初始的图片进行投影处理,说明pose是正确的
    chushiid = 1
    cunquxuanrantupianyiqi(chushiid, ceshijieguo, godxuanran, pose)
    jiancemodel = detectmodel
    detection = chushidet
    for xx1, yy1, xx2, yy2 in detection.xyxy:
        x1 = xx1
        y1 = yy1
        x2 = xx2
        y2 = yy2
    width = x2 - x1  # xmax - xmin
    height = y2 - y1  # ymax - ymin
    chushidaxiao = float(width * height)
    """if shibushisu == "supported-by":
        mao = ast.literal_eval(maomao)[0]
        bboxpred, _, _, pointschushi = get_det_bbox3d_from_image_idsinglesup(sam_predictor, scene_info, ceshijieguo,
                                                                             to_scanid,
                                                                             1,
                                                                             classtarget, detection, desc, tempfile,
                                                                             mao, jiancemodel)
    else:"""


    bboxpred, _, _, pointschushi = get_det_bbox3d_from_image_idsingle(sam_predictor, scene_info, ceshijieguo,
                                                                      to_scanid,
                                                                      1,
                                                                      classtarget, detection, desc, tempfile)
    if bboxpred is None:
        return posels, None
    cankaocenter = (bboxpred[0], bboxpred[1], bboxpred[2])

    # 从这里开始就不能用chushi的了，得用最先开始的
    w, _, _ = get_det_bbox3d_from_image_idjulei(sam_predictor, scene_info, ceshijieguo, to_scanid, 1, classtarget,
                                                detection, desc,
                                                tempfile)
    print(w)
    # w是多个detec边框
    detnum = 0
    pointslist = []
    # 是这样的每个都要存但是如果detection合适的话，也需要计算一下mask，是这样做的
    print("聚类出来多少个呀", len(w))
    # 我们只尝试一轮就好
    quanidls = []
    quanimgpathls = []
    for j in w:
        detnum += 1
        poses = huodezuijindedian(scene_info, pose, j, to_scanid)
        duiyingposes = duiyinghuodezuijindedian(scene_info, pose, j, to_scanid)
        # 根据每个detect获得poses
        idforeachdet = 0
        ceshidir = f"detbbox_{detnum}"
        dirforeachdet = os.path.join(ceshijieguo, ceshidir)
        if not os.path.exists(dirforeachdet):
            mmengine.mkdir_or_exist(dirforeachdet)
        for pose1 in poses:
            ###说明pose1需要从这里改，并且注意我们
            pose1 = rotate_local_axis(pose1, 'x', -15)
            cunquxuanrantupianyiqi(idforeachdet, dirforeachdet, godxuanran, pose1)
            imgfilepath = os.path.join(dirforeachdet, ("frame-" + f"{int(idforeachdet):06d}inv" + ".jpg"))
            quanidls.append(idforeachdet)
            quanimgpathls.append(imgfilepath)
            posels.append(pose1)
            idforeachdet += 1
        break
    dedaosanidls, dedaosanimgls = apisanhuo(quanimgpathls, quanidls, weitiao_prompt7, dirforeachdet,
                                            classtarget,
                                            desc, cankaotuxiangbase)

    # 我觉得它选的其实还行
    yaobuyaoduiyingimgls = []
    yaobuyaoduiyingidls = []
    for i in range(len(dedaosanidls)):

        num, detection, annofile = apixuanranweitiaoleigaibbox(dedaosanidls[i], dedaosanimgls[i], bijiao_prompt6,
                                                               dirforeachdet,
                                                               classtarget,
                                                               jiancemodel, desc, cankaotuxiangbase, cankaocenter,
                                                               sam_predictor, scene_info, to_scanid, bigstuff)
        # 这里才是开始需要用的时候
        # 因为基础代码已经写好了，这里我们要做的是把额外的函数引进来就好

        if num == -1:
            print("没有框出正确的物体")
            continue
        # 把它放下面
        if annofile is not None:
            yaobuyaoduiyingimgls.append(annofile)
            yaobuyaoduiyingidls.append(dedaosanidls[i])
        if detection is None:
            print("进入这里了么")
            continue
        ####限制距离是给他用的
        ceshipoints = get_det_3dpoints_from_image_idcircle(sam_predictor, scene_info, dirforeachdet, to_scanid,
                                                           dedaosanidls[i],
                                                           classtarget, detection, desc)
        pointslist.append(ceshipoints)

    # 前面最初的
    pointslist.append(pointschushi)

    duiyingdir = os.path.join(ceshijieguo, f"detbbox_1duiying")
    if not os.path.exists(duiyingdir):
        mmengine.mkdir_or_exist(duiyingdir)
    """if len(yaobuyaoduiyingimgls) != 0:
        xubuxuyao = juedingxubuxuyao(yaobuyaoduiyingimgls, yaobuyaoduiyingidls, xuyao_prompt9, classtarget, duiyingdir)
    else:
        xubuxuyao = False"""
    xubuxuyao = False
    # if xubuxuyao:
    # 那个判断是要在前面做
    if xubuxuyao:
        duiyingimgpathls = []
        for ids in dedaosanidls[:1]:
            # 尽量减少误差，只要一个就行了
            duiyingpose = duiyingposes[ids]
            posels.append(duiyingpose)
            cunquxuanrantupianyiqi(ids, duiyingdir, godxuanran, duiyingpose)
            duiyingpath = os.path.join(duiyingdir, ("frame-" + f"{int(ids):06d}inv" + ".jpg"))
            duiyingimgpathls.append(duiyingpath)
            num, detection = apixuanranweitiaoleigaibboxduiying(ids, duiyingpath,
                                                                duiyingdir,
                                                                classtarget,
                                                                jiancemodel, cankaocenter,
                                                                sam_predictor, scene_info, to_scanid, desc)
            # 它不一定能看出来了，所以我们不比较了，因为大概率比较近，不会选错
            # 这里才是开始需要用的时候
            # 因为基础代码已经写好了，这里我们要做的是把额外的函数引进来就好

            if num == -1:
                print("没有框出正确的物体")
                continue
            if detection is None:
                print("进入这里了么")
                continue
            ceshipoints = get_det_3dpoints_from_image_idcircle(sam_predictor, scene_info, duiyingdir, to_scanid,
                                                               ids,
                                                               classtarget, detection, desc)
            # 这个不用改
            pointslist.append(ceshipoints)
    else:
        print("比较完整不需要")

    predbboxchushils = []
    pointslistpro = []
    ######  这里来处理奇异值去掉那些
    ###先获得每团点云获得的的bbox
    ####
    pointslisttemp = []
    for tuan in pointslist:
        if tuan is None:
            continue

        else:
            pred_bbox1 = calculate_aabb(tuan)
            predbboxchushils.append(pred_bbox1)
            pointslisttemp.append(tuan)
    # 这一步是为了防止里面有NOne，导致0，导致错误删除有用的投影
    pointslist = pointslisttemp
    print("最后一步，pointslist有多长，", len(pointslist))
    if len(pointslist) == 0:
        return posels, None

    ###最终决定使用的点云团index
    volumes_with_index = [(i, dx * dy * dz) for i, (_, _, _, dx, dy, dz) in enumerate(predbboxchushils)]
    volumes = [dx * dy * dz for i, (_, _, _, dx, dy, dz) in enumerate(predbboxchushils)]

    # 按体积从大到小排序
    sorted_volumes_with_index = sorted(volumes_with_index, key=lambda x: x[1], reverse=True)

    # 提取排序后的索引
    sorted_indices = [i for i, _ in sorted_volumes_with_index]

    # 提取排序后的框
    sorted_boxes = [predbboxchushils[i] for i in sorted_indices]
    sorted_volumes = [v for _, v in sorted_volumes_with_index]

    # 我们主要用排序后的索引
    # 没关系就先这样做，之后还会有iou来保障一下

    remaining_indices = []
    trash_indices = []
    for i in range(len(sorted_volumes) - 1):
        if sorted_volumes[i] >= sorted_volumes[i + 1] * 2.2:
            trash_indices.append(sorted_indices[i])
        else:
            break
    print(trash_indices)
    for j in sorted_indices:
        if j not in trash_indices:
            remaining_indices.append(j)

    ####   这里暂时不看iou，因为后面还有聚类， 我们相当于把有可能的像雕塑的那种情况排除了
    # 但是后面具体选的对不对另说。
    for shuju in remaining_indices:
        pointslistpro.append(pointslist[shuju])
    pointslist = pointslistpro

    aligned_points_3d = np.concatenate(pointslist, axis=0)
    # save the aligned points 3d
    intermediate_output_dir = os.path.join(ceshijieguo, "zuizhongproject")
    aligned_points_3d_output_dir = os.path.join(
        intermediate_output_dir, "projected_points"
    )
    mmengine.mkdir_or_exist(aligned_points_3d_output_dir)
    np.save(
        f"{aligned_points_3d_output_dir}/ensemble_matching.npy",
        aligned_points_3d,
    )
    ###这里是因为我们上面的代码已经自己filter过了
    np.save(
        f"{aligned_points_3d_output_dir}/ensemble_matching_filtered.npy",
        aligned_points_3d,
    )

    # if nan in points, return None
    if (
            aligned_points_3d.shape[0] == 0
            or aligned_points_3d.shape[0] == 0
            or np.isnan(aligned_points_3d).any()
            or np.isnan(aligned_points_3d).any()
    ):
        print("什麽都沒有")
        # 如果找不到任何框的话
        return posels, None
    # tt = cluster_numpy_array(aligned_points_3d)
    # zuizhongalign = find_largest_cluster(tt)
    # pred_bbox = calculate_aabb(zuizhongalign)
    pred_bbox = calculate_aabb(aligned_points_3d)
    print("最終：", pred_bbox)
    # okk，基本都写好了，就剩下文件的对应了
    return posels, pred_bbox


# 我们首先来看一下iou怎么算的，当框比较大的时候有没有影响。
# 如果框太大的话也是会影响的，我在想最后一下，是不是可以尝试一下聚类，因为说不定因为有五个投影出来的比较多。


"""id = 2
pose1 = huodezuijindedian(pose, bboxpred, to_scanid)
np.save("buzaihanshuli.txt", pose1)
# 看一下最原始弄好的，应该肯定是哪个函数有问题
pose1 = rotate_local_axis(pose1, axis='y', angle_deg=30)
cunquxuanrantupianyiqi(id, ceshijieguo, godxuanran, pose1)
"""

# 我們可以這樣選擇兩個框的并集
# 選擇能把兩個框都框起來的





