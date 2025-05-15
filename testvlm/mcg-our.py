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
from PIL import Image
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
from PIL import Image, ImageDraw, ImageFont

DEFAULT_OPENAIGPT_CONFIG = {"temperature": 1, "top_p": 1, "max_tokens": 4095}
# from fuzhu import huodemaodiandet_paths
from promptpaixu import paixu_prompt2
from promptpaixu import *
from centerbao3 import *

tovg = "to.csv"
yesvg = "yes.csv"
import numpy as np

from zhiwei.nvdiffrast_tool.godmokuai import *

from cunpose import *


def interpolate_translation(ref_pose, a_pose, t):
    """
    将 a_pose 矩阵的平移向量修改为 a_pose 原始平移向量与参考平移向量之间的中间值。

    参数:
        ref_pose (numpy.ndarray): 参考 pose 矩阵 (4x4)
        a_pose (numpy.ndarray): a pose 矩阵 (4x4)
        t (float): 插值比例，范围在 [0, 1] 之间
            - t = 0: 返回 a_pose 的原始平移向量
            - t = 1: 返回 ref_pose 的平移向量
            - 0 < t < 1: 返回两者之间的插值向量

    返回:
        numpy.ndarray: 修改后的 a_pose 矩阵 (4x4)
    """
    # 提取平移向量
    ref_translation = ref_pose[:3, 3]  # 参考平移向量
    a_translation = a_pose[:3, 3]      # a_pose 的原始平移向量

    # 计算插值后的平移向量
    interpolated_translation = (1 - t) * a_translation + t * ref_translation

    # 修改 a_pose 的平移向量
    a_pose[:3, 3] = interpolated_translation

    return a_pose


def jisuancost(pre_po, now_po):
    pre_l1 = pre_po[:3, 3]
    pre_l2 = now_po[:3, 3]
    center_distance = np.linalg.norm(pre_l2 - pre_l1)
    pre_r1 = pre_po[:3, :3]
    pre_r2 = now_po[:3, :3]
    rotation_quaternion1 = R.from_matrix(pre_r1).as_quat()
    rotation_quaternion2 = R.from_matrix(pre_r2).as_quat()

    # 计算四元数之间的夹角（弧度制）
    quaternion_dot_product = np.dot(rotation_quaternion1, rotation_quaternion2)
    quaternion_angle = 2 * np.arccos(np.clip(quaternion_dot_product, -1, 1))

    # 将夹角转换为度
    # quaternion_angle_degrees = np.degrees(quaternion_angle)
    quaternion_angle_degrees = quaternion_angle

    center_distance = center_distance / 0.5

    return center_distance, quaternion_angle_degrees


def patscost(chushipose, ensem_id_pose):
    pre_pose = chushipose
    l_cost = 0
    degree_cost = 0
    for pose in ensem_id_pose:
        l, d = jisuancost(pre_pose, pose)
        l_cost += l
        degree_cost += d
        pre_pose = pose
    return l_cost, degree_cost


class VisualGrounder:
    def __init__(
            self,
            scene_info_path,
            det_info_path,
            vg_file_path,
            yesvg_file_path,
            output_dir,
            openaigpt_type,
            ensemble_num=1,
            prompt_version=1,
            **kwargs,
    ) -> None:
        self.scene_info_path = scene_info_path
        # 这样改一下就行，因为我们做大量图片检查是在yes里面做的
        self.scene_infoall = mmengine.load(self.scene_info_path)
        self.vg_file_path = yesvg_file_path
        self.yesvg_file_path = "../hm3rscandata/xuanran/ceshioutputscan/query_analysisyes/yesterdaymi.csv"
        # 进行多图片检测的是vg，不带to
        self.tovg_file_path = vg_file_path
        self.output_dir = output_dir
        self.openaigpt_type = openaigpt_type
        self.ensemble_num = ensemble_num

        # cuda available
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # set openai
        DEFAULT_OPENAIGPT_CONFIG["temperature"] = kwargs.get("openai_temperature", 1.0)
        DEFAULT_OPENAIGPT_CONFIG["top_p"] = kwargs.get("openai_top_p", 1.0)
        self.openaigpt_config = {"model": openaigpt_type, **DEFAULT_OPENAIGPT_CONFIG}
        self.openaigpt = OpenAIGPT(**self.openaigpt_config)

        # set prompts
        self.prompt_version = prompt_version
        self.system_prompt = SYSTEM_PROMPTS[f"v{prompt_version}"]
        # self.input_prompt = paixu_prompt4
        self.input_prompt = xuanranpaixu_prompt1
        self.bbox_select_user_prompt = BBOX_SELECT_USER_PROMPTS[f"v{prompt_version}"]
        self.image_id_invalid_prompt = IMAGE_ID_INVALID_PROMPTS[f"v{prompt_version}"]
        self.detection_not_exist_prompt = DETECTION_NOT_EXIST_PROMPTS[
            f"v{prompt_version}"
        ]

        self.evaluate_result_func = eval(
            f'self.{kwargs.get("evaluate_function", "evaluate_3diou")}'
        )

        # set sam
        use_sam_huge = kwargs.get("use_sam_huge", False)
        if use_sam_huge:
            assert os.path.exists(
                "../checkpoints/SAM/sam_vit_h_4b8939.pth"
            ), "Error: no checkpoints/SAM/sam_vit_h_4b8939.pth found."
            self.sam_predictor = UltralyticsSAMHuge(
                "../checkpoints/SAM/sam_vit_h_4b8939.pth"
            )
            print(f"Use SAM-Huge.")
        else:
            self.sam_predictor = SAM("sam_b.pt")

        self.scene_infos = SceneInfoHandler(scene_info_path)
        self.forfuncsceneinfo = mmengine.load(scene_info_path)
        self.det_infos = DetInfoHandler(det_info_path)
        self.vg_file = pd.read_csv(self.tovg_file_path)  #

        self.default_bbox_annotator = sv.BoundingBoxAnnotator()
        self.bbox_annotator = sv.BoundingBoxAnnotator(thickness=6, color=Color.BLACK)
        self.default_label_annotator = sv.LabelAnnotator()
        self.label_annotator = sv.LabelAnnotator(
            text_position=Position.CENTER,
            text_thickness=2,
            text_scale=1,
            color=Color.BLACK,
        )

        self.accuracy_calculator = AccuracyCalculator()

        # matching
        matching_info_path = kwargs.get("matching_info_path", None)
        # self.matching_infos = MatchingInfoHandler(
        #    matching_info_path, scene_infos=self.scene_infos
        # )
        self.matching_infos = None

        self.min_matching_num = kwargs.get("min_matching_num", 50)
        self.cd_loss_thres = kwargs.get("cd_loss_thres", 0.1)
        self.use_point_prompt = kwargs.get("use_point_prompt", False)
        self.use_bbox_prompt = kwargs.get("use_bbox_prompt", False)
        self.use_point_prompt_num = kwargs.get("use_point_prompt_num", 1)
        # assert use_bbox_prompt and use_point_prompt cannot be both False
        assert (
                self.use_bbox_prompt or self.use_point_prompt
        ), "use_bbox_prompt and use_point_prompt cannot be both False."

        # Morphological operations
        self.post_process_erosion = kwargs.get("post_process_erosion", True)
        self.post_process_dilation = kwargs.get("post_process_dilation", True)
        self.kernel_size = kwargs.get("kernel_size", 3)
        self.post_process_component = kwargs.get("post_process_component", True)
        self.post_process_component_num = kwargs.get("post_process_component_num", 1)
        if (
                self.post_process_erosion
                or self.post_process_dilation
                or self.post_process_component
        ):
            self.post_process = True
        else:
            self.post_process = False

        self.point_filter_nb = kwargs.get("point_filter_nb", 20)
        self.point_filter_std = kwargs.get("point_filter_std", 1.0)
        self.point_filter_type = kwargs.get("point_filter_type", "statistical")
        self.point_filter_tx = kwargs.get("point_filter_tx", 0.05)
        self.point_filter_ty = kwargs.get("point_filter_ty", 0.05)
        self.point_filter_tz = kwargs.get("point_filter_tz", 0.05)
        self.project_color_image = kwargs.get("project_color_image", False)

        # set VLM parameters
        self.gpt_max_retry = kwargs.get("gpt_max_retry", 5)
        assert (
                self.gpt_max_retry >= 0
        ), "gpt_max_retry should be greater than or equal to 0."
        self.gpt_max_input_images = kwargs.get("gpt_max_input_images", 6)
        self.get_bbox_index_type = kwargs.get("get_bbox_index_type", "constant")
        self.use_all_images = kwargs.get("use_all_images", False)
        self.use_new_detections = kwargs.get("use_new_detections", False)
        self.skip_bbox_selection_when1 = kwargs.get("skip_bbox_selection_when1", False)
        self.image_det_confidence = kwargs.get("image_det_confidence", 0.3)
        self.use_bbox_anno_f_gpt_select_id = kwargs.get(
            "use_bbox_anno_f_gpt_select_id", False
        )
        self.fix_grid2x4 = kwargs.get("fix_grid2x4", False)
        self.dynamic_stitching = kwargs.get("dynamic_stitching", True)
        self.online_detector = kwargs.get("online_detector", "gdino")

        # set online detector
        if self.use_new_detections:
            if "yolo" in self.online_detector.lower():
                detection_model = "../checkpoints/yolov8_world/yolov8x-worldv2.pt"
                self.detection_model = OnlineDetector(
                    detection_model=detection_model, device=self.device
                )
            elif "gdino" in self.online_detector.lower():
                self.detection_model = OnlineDetector(
                    detection_model="Grounding-DINO-1.5Pro", device=self.device
                )
            elif "tidai" in self.online_detector.lower():
                self.detection_model = OnlineDetector(
                    detection_model="tidai", device=self.device
                )

        print(f"Point filter nb: {self.point_filter_nb}, std: {self.point_filter_std}")
        print(
            f"SAM mask processing type, kernel_size: {self.kernel_size}, erosion: {self.post_process_erosion}, dilation: {self.post_process_dilation}, component: {self.post_process_component}, component_num: {self.post_process_component_num}."
        )
        # print(f"Kwargs: {kwargs}.")

        self.output_prefix = f"{os.path.basename(self.vg_file_path.split('.')[0])}_{self.openaigpt_type}_promptv{self.prompt_version}"

    # 这个是针对today的图像的

    def quyiyoudet(self, pred_class, scene_id, image_id):
        image_id = self.convert_imageid(image_id)
        ori_detections = self.det_infos.get_detections_filtered_by_score(
            scene_id, image_id, 0.2
        )
        detections = self.det_infos.get_detections_filtered_by_class_name(
            None, None, pred_class, ori_detections
        )
        return detections

    def huodebboxapitwo(self, to_image, target_class, yes_image):
        to_imagebase64 = encode_PIL_image_to_base64(
            resize_image(to_image)
        )
        yes_imagebase64 = encode_PIL_image_to_base64(
            resize_image(yes_image)
        )
        x = apibbox_prompt2.format(
            targetclass=target_class,
            width=270,
            height=480
        )
        ww = []
        ww.append({
            "role": "user",
            "content": [
                {"type": "text", "text": x},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{to_imagebase64}",
                        "detail": "low",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{yes_imagebase64}",
                        "detail": "low",
                    },
                },
            ],
        })

        gpt_response = self.openaigpt.safe_chat_complete(
            ww, response_format={"type": "json_object"}, content_only=True
        )

        print("结束进行gpt测试")
        cost = gpt_response["cost"]
        gpt_content = gpt_response["content"]
        print(gpt_content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        gpt_content_json = json.loads(gpt_content)
        ifornot = gpt_content_json.get("findornot", "false")
        det = gpt_content_json.get("bbox", "false")

        def double(x):
            return x * 3

        # 使用map()函数将每个元素乘以2
        doubledet = []
        if len(det) != 0:
            for item in det:
                item = list(map(double, item))
                doubledet.append(item)

        return cost, ifornot, doubledet

    def huodebboxapi(self, to_image, target_class):
        to_imagebase64 = encode_PIL_image_to_base64(
            resize_image(to_image)
        )
        x = apibbox_prompt.format(
            targetclass=target_class,
            width=270,
            height=480
        )
        ww = []
        ww.append({
            "role": "user",
            "content": [
                {"type": "text", "text": x},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{to_imagebase64}",
                        "detail": "low",
                    },
                },
            ],
        })

        gpt_response = self.openaigpt.safe_chat_complete(
            ww, response_format={"type": "json_object"}, content_only=True
        )

        print("结束进行gpt测试")
        cost = gpt_response["cost"]
        gpt_content = gpt_response["content"]
        print(gpt_content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        gpt_content_json = json.loads(gpt_content)
        ifornot = gpt_content_json.get("findornot", "false")
        det = gpt_content_json.get("bbox", "false")

        def double(x):
            return x * 2

        # 使用map()函数将每个元素乘以2
        doubledet = []
        if len(det) != 0:
            for item in det:
                item = list(map(double, item))
                doubledet.append(item)

        return cost, ifornot, doubledet

    def panduanshifoujixuyes(self, query, yesid, yesscan_id, anchorclass):
        cost = 0
        ceshiid = self.convert_imageid(yesid)
        to_image_path = "../hm3rscandata/scannet/ceshioutputscan/" + \
                        self.scene_infos.infos[yesscan_id]["images_info"][ceshiid]['image_path']
        to_image = Image.open(to_image_path)
        to_imagebase64 = encode_PIL_image_to_base64(
            resize_image(to_image)
        )
        # to_query_prompt = f"You are an agent who is very good at looking at pictures. Now I'm giving you one picture. You need to determine whether the picture contains an {classtarget} that is fully meets a specific description: {query}. If it contains, return True; if not, return False. Notice, return only False or True."
        to_query_prompt = panduan_prompt3
        to_query_prompt = to_query_prompt.format(
            anchor_object=anchorclass,
            query=query,
        )
        begin_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": to_query_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{to_imagebase64}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ]
        print("开始进行gpt测试")
        gpt_response = self.openaigpt.safe_chat_complete(
            begin_messages, response_format={"type": "json_object"}, content_only=True
        )

        print("结束进行gpt测试")
        cost += gpt_response["cost"]
        gpt_content = gpt_response["content"]
        print(gpt_content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        gpt_content_json = json.loads(gpt_content)
        flag = gpt_content_json.get("anchor_object_may_be_correct", False)
        return flag, cost

    def panduanshifoujixu(self, query, ceshiid, toscan_id, anchorclass):
        cost = 0
        ceshiid = self.convert_imageid(ceshiid)
        to_image_path = "../hm3rscandata/scannet/ceshioutputscan/" + \
                        self.scene_infos.infos[toscan_id]["images_info"][ceshiid]['image_path']
        to_image = Image.open(to_image_path)
        to_imagebase64 = encode_PIL_image_to_base64(
            resize_image(to_image)
        )
        # to_query_prompt = f"You are an agent who is very good at looking at pictures. Now I'm giving you one picture. You need to determine whether the picture contains an {classtarget} that is fully meets a specific description: {query}. If it contains, return True; if not, return False. Notice, return only False or True."
        to_query_prompt = panduan_prompt2
        to_query_prompt = to_query_prompt.format(
            anchor_object=anchorclass,
            query=query,
        )
        begin_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": to_query_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{to_imagebase64}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ]
        print("开始进行gpt测试")
        gpt_response = self.openaigpt.safe_chat_complete(
            begin_messages, response_format={"type": "json_object"}, content_only=True
        )

        print("结束进行gpt测试")
        cost += gpt_response["cost"]
        gpt_content = gpt_response["content"]
        print(gpt_content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        gpt_content_json = json.loads(gpt_content)
        flag = gpt_content_json.get("anchor_object_may_be_correct", False)
        return flag, cost

    def post_process_mask(self, mask):
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
        kernel = np.ones((self.kernel_size * 2 + 1, self.kernel_size * 2 + 1), np.uint8)

        # Apply morphological erosion if requested
        if self.post_process_erosion:
            img = cv2.erode(img, kernel, iterations=1)

        # Apply morphological dilation if requested
        if self.post_process_dilation:
            img = cv2.dilate(img, kernel, iterations=1)

        # Find all connected components
        num_labels, labels_im = cv2.connectedComponents(
            img
        )  # label 0 is background, so start from 1
        if self.post_process_component and num_labels > 1:
            # Calculate the area of each component and sort them, keeping the largest k
            component_areas = [
                (label, np.sum(labels_im == label)) for label in range(1, num_labels)
            ]
            component_areas.sort(key=lambda x: x[1], reverse=True)
            largest_components = [
                x[0] for x in component_areas[: self.post_process_component_num]
            ]
            img = np.isin(labels_im, largest_components).astype(np.uint8)

        # Return the processed image as a boolean mask
        new_mask = img.astype(bool)

        if is_ultralytics:
            new_mask = torch.from_numpy(new_mask[None, :])  # should be tensor
            ultra_result.update(masks=new_mask)
            return ultra_result

        return new_mask

    def evaluate_3diou(self, pred_bbox, vg_row):
        """
        Evaluate 3D bounding box IoU.
        """
        scene_id = vg_row["scan_id"]
        target_id = vg_row["target_id"]
        gt_bbox = self.scene_infos.get_object_gt_bbox(
            scene_id, object_id=target_id, axis_aligned=True
        )
        # * calculate iou
        iou3d = calculate_iou_3d(pred_bbox, gt_bbox)

        default_eval_result = self.init_default_eval_result_dict(vg_row)

        default_eval_result.update(
            {
                "iou3d": iou3d,
                "acc_iou_25": iou3d >= 0.25,
                "acc_iou_50": iou3d >= 0.50,
                "gpt_pred_bbox": pred_bbox.tolist(),
                "gt_bbox": gt_bbox.tolist(),
            }
        )

        return default_eval_result

    def calculate_brightness_opencv(self, image_path):
        image = cv2.imread(image_path)
        # 将图像转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 计算灰度图的平均亮度
        brightness = cv2.mean(gray)[0]
        return brightness

    def get_ensemble_masks_matching(
            self, scene_id, image_id, pred_target_class, sam_mask, sam_mask_output_dir
    ):
        """
        Use feature matching to ensemble masks.

        Args:
            scene_id (int): The ID of the scene.
            image_id (int): The ID of the image.
            pred_target_class (str): The predicted target class.
            sam_mask (Tensor): The segmentation mask generated by the SAM model.
            sam_mask_output_dir (str): The output directory to save the generated masks.

        Returns:
            tuple: A tuple containing the ensemble image IDs and ensemble masks.
        """
        if self.ensemble_num == 1:
            return [image_id], [sam_mask]

        matching_results = self.matching_infos.get_matching_results_by_image_id(
            scene_id, image_id
        )
        # 只要一开始的matching_results不出问题就可以
        # filter out those matching pairs with invalid posed images
        matching_results = {
            pair: matching_result
            for pair, matching_result in matching_results.items()
            if self.scene_infos.is_posed_image_valid(scene_id, pair[0])
               and self.scene_infos.is_posed_image_valid(scene_id, pair[1])
        }

        matching_results = self.matching_infos.filter_matching_results_by_mask(
            scene_id,
            image_id,
            sam_mask,
            matching_results,
            remove_matching=True,
            min_matching_num=self.min_matching_num,
            cd_loss_thres=self.cd_loss_thres,
        )

        # by default, the anchor mask is included, so the other images should be self.ensemble_num - 1
        matching_results = self.matching_infos.select_top_k_matching_results_by_cdloss(
            matching_results, top_k=10000
        )  # 10000 here means take all matching results

        # save these matching results
        image_matching_vis_output_dir = os.path.join(
            sam_mask_output_dir, f"ensemble{self.ensemble_num}", "image_matching"
        )
        mmengine.mkdir_or_exist(image_matching_vis_output_dir)

        skipped_image_output_dir = os.path.join(
            sam_mask_output_dir, f"ensemble{self.ensemble_num}", "skipped_images"
        )
        mmengine.mkdir_or_exist(skipped_image_output_dir)

        # For each tracking_images, find the box and mask
        ensemble_image_ids = []
        ensemble_masks = []
        tracking_output_dir = os.path.join(
            sam_mask_output_dir, f"ensemble{self.ensemble_num}", "tracking"
        )
        mmengine.mkdir_or_exist(tracking_output_dir)
        image_id_str = f"{int(image_id):05d}"
        image_id_str = "frame-0" + image_id_str

        for i, (key, matching_result) in enumerate(matching_results.items()):
            if len(ensemble_masks) == self.ensemble_num - 1:
                # enough masks, break
                break

            # save the matching result visualization
            self.matching_infos.save_matching_results_visualization(
                scene_id,
                {key: matching_result},
                output_dir=image_matching_vis_output_dir,
                is_draw_matches=False,
            )

            if image_id_str == key[0]:
                target_image_id = key[1]
                target_key = "kp1"
            elif image_id_str == key[1]:
                target_image_id = key[0]
                target_key = "kp0"
            else:
                print(
                    f"[Matching Ensemble] Scene id: {scene_id} Image id: {image_id_str} not in matching_pair {key}. Something wrong."
                )
                continue

            target_kps = matching_result[target_key]
            target_image_path = self.scene_infos.get_image_path(
                scene_id, target_image_id
            )

            sam_prompts = {}

            if self.use_bbox_prompt:
                # Get 2d bbox
                ori_detections = self.det_infos.get_detections_filtered_by_score(
                    scene_id, target_image_id, self.image_det_confidence
                )
                detections = self.det_infos.get_detections_filtered_by_class_name(
                    None, None, pred_target_class, ori_detections
                )

                if len(detections) == 0:
                    # no detections
                    all_bboxes_annotated_skipped_image = (
                        self.det_infos.annotate_image_with_detections(
                            scene_id, target_image_id, ori_detections
                        )
                    )
                    all_bboxes_annotated_skipped_image.save(
                        f"{skipped_image_output_dir}/{i}_{target_image_id}_all_bboxes.jpg"
                    )
                    continue
                else:
                    # store the image with target box annotation
                    all_bboxes_annotated_image = (
                        self.det_infos.annotate_image_with_detections(
                            scene_id, target_image_id, detections
                        )
                    )
                    all_bboxes_annotated_image.save(
                        f"{tracking_output_dir}/{i}_{target_image_id}_all_bboxes.jpg"
                    )

                    # first sort the index and then use something like below to get the result
                    # ineed to check if wether iou > 0
                    # use something like detections[0:5][[1, 1, 2, 4]]
                    bbox_areas = detections.box_area  # 1d array

                    # calculate how many keypoints are in the bboxes
                    inliers = [
                        np.sum(
                            (target_kps[:, 1] > x1)
                            & (target_kps[:, 1] < x2)
                            & (target_kps[:, 0] > y1)
                            & (target_kps[:, 0] < y2)
                        )
                        for x1, y1, x2, y2 in detections.xyxy
                    ]

                    iob = inliers / bbox_areas  # intersection over bbox_areas

                    # find the best bbox and nonzero
                    index = np.argmax(iob)

                    if inliers[index] == 0:
                        # print(f"[Matching Ensemble] Scene id: {scene_id}, Target image id: {target_image_id}, Pred target class: {pred_target_class} does not have any keypoints in all bboxes.")
                        continue
                    else:
                        # store the image with target box annotation
                        bbox_annotated_image = (
                            self.det_infos.annotate_image_with_detections(
                                scene_id, target_image_id, detections[[index]]
                            )
                        )
                        bbox_annotated_image.save(
                            f"{tracking_output_dir}/{i}_{target_image_id}_bbox.jpg"
                        )
                        sam_prompts.update(dict(bboxes=detections.xyxy[index]))

            if self.use_point_prompt:
                # selects the self.use_point_prompt_num target_kps with some criteria
                # if use box prompt, then selects the top self.use_point_prompt_num nearest to the bbox center
                # The two versions below are quite similar
                if self.use_bbox_prompt and "bboxes" in sam_prompts:
                    bbox = sam_prompts["bboxes"]
                    bbox_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    # calculate the distance
                    distances = np.linalg.norm(
                        target_kps[:, [1, 0]] - bbox_center, axis=1
                    )
                    # get the index
                    selected_kp_index = np.argsort(distances)[
                                        : self.use_point_prompt_num
                                        ]

                else:
                    # calcualte the center of the kps and select the top_k nearest points
                    kps_center = np.mean(target_kps, axis=0)
                    distances = np.linalg.norm(target_kps - kps_center, axis=1)
                    selected_kp_index = np.argsort(distances)[
                                        : self.use_point_prompt_num
                                        ]

                selected_target_kps = target_kps[selected_kp_index]
                sam_prompts.update(
                    dict(
                        points=selected_target_kps[:, [1, 0]],
                        labels=np.array([1]).repeat(len(selected_target_kps)),
                    )
                )

                # visualize the keypoints on images
                points_annotated_image = self.det_infos.annotate_image_with_points(
                    scene_id, target_image_id, selected_target_kps[:, [1, 0]]
                )
                # save the image
                points_annotated_image.save(
                    f"{tracking_output_dir}/{i}_{target_image_id}_points.jpg"
                )

            if len(sam_prompts) == 0:
                continue

            prompt_suffix = ""
            if "points" in sam_prompts:
                prompt_suffix += "_w_p"
            if "bboxes" in sam_prompts:
                prompt_suffix += "_w_b"

            # start getting mask
            # Use sam to get the mask
            sam_prediction = self.sam_predictor(
                target_image_path, verbose=False, **sam_prompts
            )[0]

            # save the results
            sam_prediction.save(
                f"{tracking_output_dir}/{i}_{target_image_id}{prompt_suffix}_sam.jpg"
            )

            # postprocessing
            if self.post_process:
                sam_prediction = self.post_process_mask(sam_prediction)
                sam_prediction.save(
                    f"{tracking_output_dir}/{i}_{target_image_id}{prompt_suffix}_sam_postprocessed.jpg"
                )

            ensemble_image_ids.append(target_image_id)
            ensemble_masks.append(sam_prediction.masks.data.cpu().numpy()[0])

        # add the anchor mask and id
        ensemble_image_ids.append(image_id)
        ensemble_masks.append(sam_mask)

        return ensemble_image_ids, ensemble_masks

    def ensemble_pred_points(
            self,
            xuanranwenjianjia,
            scene_id,
            image_id,
            pred_target_class,
            sam_mask,
            matched_image_ids,
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

        ensemble_image_ids, ensemble_masks = self.get_ensemble_masks_matching(
            scene_id,
            image_id,
            pred_target_class,
            sam_mask=sam_mask,
            sam_mask_output_dir=sam_mask_output_dir,
        )
        #

        ensemble_points = []
        # projections for all the ids
        for current_image_id, current_mask in zip(ensemble_image_ids, ensemble_masks):
            current_aligned_points_3d = self.scene_infos.project_single_image_to_3d_with_mask(
                scene_id=scene_id,
                image_id=current_image_id,
                xuanrandir=xuanranwenjianjia,
                mask=current_mask,
                with_color=self.project_color_image,
            )

            ensemble_points.append(current_aligned_points_3d)

        aligned_points_3d = np.concatenate(ensemble_points, axis=0)
        aligned_points_3d = uniform_downsampling(aligned_points_3d, 0.5)

        return aligned_points_3d

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

    def get_gpt_select_bbox_indexxin(
            self, ceshiid, xinimage, detections, xinintermediate_output_dir, classtarget, query
    ):
        # 去选出来新的image里面合适的id
        # ceshiid是为了存文件用的
        num_candidate = len(detections)
        messages = []
        bbox_index_gpt_select_output_dir = os.path.join(
            xinintermediate_output_dir, "bbox_index_gpt_select_prompts"
        )
        mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)

        labels = [f"ID:{ID}" for ID in range(len(detections))]
        if self.use_bbox_anno_f_gpt_select_id:
            image = self.bbox_annotator.annotate(scene=xinimage, detections=detections)
        annotated_image = self.label_annotator.annotate(
            scene=xinimage, detections=detections, labels=labels
        )
        # save the annotated_image
        annotated_image.save(
            f"{bbox_index_gpt_select_output_dir}/{ceshiid}_annotated.jpg"
        )

        # convert the annotated image to base64
        annotated_image_base64 = encode_PIL_image_to_base64(
            resize_image(annotated_image)
        )
        # 这里我们需要处理的只有一个图像
        # 欧，我想起来了，一个是框，一个是id。
        # 我们可以用format去搞
        multi_prompt = """You are an agent who is very good at analyzing images. I am now providing you one picture. There are {num_candidate_bboxes} candidate objects shown in the image. I have annotated each object at the center with an object ID in white color text and black background. Please find out which ID marked {classtarget} object in the picture exactly meets the description: {query}.
        Reply using JSON format with two keys "reasoning" and "object_id" like this:
        {{
          "reasoning": "your reasons", // Explain the justification why you select the object ID.
          "object_id": 0 // The object ID you selected. Always give one object ID from the image, which you are the most confident of, even you think the image does not contain the correct object.
        }}"""
        multi_prompt = multi_prompt.format(
            classtarget=classtarget,
            num_candidate_bboxes=num_candidate,
            query=query,

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
                            "detail": "low",
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

            gpt_response = self.openaigpt.safe_chat_complete(
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
        return bbox_index, cost

    def apixuanranbbox(
            self, messages_his, imageid, imagefilepath, apibbox_prompts, xinintermediate_output_dir, mao, classtarget1,
            query,
            jiancemodel, des
    ):
        # 去选出来新的image里面合适的id
        # ceshiid是为了存文件用的
        # 这个是需要路径的
        ###我觉得mis应该还是有用的尤其是横过来

        xinimage = Image.open(imagefilepath)
        detections = jiancemodel.detect(imagefilepath, classtarget1)
        detections = detections[detections.confidence > 0.20]
        num_candidate = len(detections)
        if num_candidate == 0:
            return -1, None
        if num_candidate == 1:
            return 0, detections[0]

        messages = copy.deepcopy(messages_his)
        bbox_index_gpt_select_output_dir = os.path.join(
            xinintermediate_output_dir, "bbox_index_gpt_select_prompts"
        )
        mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)
        zuizhonglabel = []
        zuizhongdet = []
        chuliindex = []
        ###偶对了。我想起来了这里需要增加的就是最前面只有一个的时候就返回这一个
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
            if bili >= 0.80:
                # 一般来说不会这么大
                print("是那种恶心的框全图的情况")
                continue

            zuizhongdet.append(detections[detgeshu])
            zuizhonglabel.append(detgeshu)
            chuliindex.append(detgeshu)

        # labels = [f"ID:{ID}" for ID in range(len(detections))]
        labels = [f"ID:{ID}" for ID in zuizhonglabel]
        # 这样我们把bbox标注的那个搞过来，之后弄一个标注的列表
        # 然后对这个列表按照顺序标注就好了
        if self.use_bbox_anno_f_gpt_select_id:
            image = self.bbox_annotator.annotate(scene=xinimage, detections=detections)
        chouqu = xinimage
        for x in chuliindex:
            ll = f"ID:{x}"
            lll = [ll]
            chouqu = self.label_annotator.annotate(
                scene=chouqu, detections=detections[x], labels=lll
            )
        annotated_image = chouqu
        # save the annotated_image
        annotated_image.save(
            f"{bbox_index_gpt_select_output_dir}/{imageid}_annotated.jpg"
        )
        num_candidate = len(chuliindex)

        if num_candidate == 1:
            return chuliindex[0], detections[chuliindex[0]]
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
            anchorclass=mao,
            classtarget=classtarget1,
            num_candidate_bboxes=num_candidate,
            query=query,
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
                            "detail": "low",
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

            gpt_response = self.openaigpt.safe_chat_complete(
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

            bbox_index = int(gpt_content_json.get("object_id", -1))
            if bbox_index < 0:
                # invalid bbox index
                retry += 1
                continue
            else:
                break

        if bbox_index == -1:
            return -1, None
        else:
            return bbox_index, detections[bbox_index]

    def apixuanranchushileibbox(
            self, messages_his, imageid, imagefilepath, apibbox_prompts, xinintermediate_output_dir, classtarget1,
            query,
            jiancemodel, des
    ):
        # 去选出来新的image里面合适的id
        # ceshiid是为了存文件用的
        # 这个是需要路径的
        xinimage = Image.open(imagefilepath)
        detections = jiancemodel.detect(imagefilepath, classtarget1)
        detections = detections[detections.confidence > 0.20]
        num_candidate = len(detections)
        if num_candidate == 1:
            return 0, detections[0]
        if num_candidate == 0:
            return -1, None
        messages = copy.deepcopy(messages_his)
        bbox_index_gpt_select_output_dir = os.path.join(
            xinintermediate_output_dir, "findfbboxselect"
        )
        mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)

        jiheids = []
        jiheanoimages = []
        for detgeshu in range(num_candidate):
            jiheids.append(detgeshu)

            ori_image = xinimage
            annotated_image = self.default_bbox_annotator.annotate(
                scene=ori_image, detections=detections[detgeshu]
            )
            annotated_image = self.default_label_annotator.annotate(
                scene=annotated_image, detections=detections[detgeshu]
            )
            jiheanoimages.append(annotated_image)
        base64Frames = self.stitch_and_encode_images_to_base64(
            jiheanoimages, jiheids, intermediate_output_dir=bbox_index_gpt_select_output_dir
        )

        # 这里我们需要处理的只有一个图像
        # 欧，我想起来了，一个是框，一个是id。
        # 我们可以用format去搞

        multi_prompt = apibbox_prompts.format(
            target=classtarget1,
            num_candidate_bboxes=num_candidate,
            description=des,
        )
        print("当前的描述是：", des)

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
                                "detail": "low",
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
            bbox_index = -1
            gpt_message = None

            gpt_response = self.openaigpt.safe_chat_complete(
                messages, response_format={"type": "json_object"}, content_only=True
            )

            cost += gpt_response["cost"]
            gpt_content_json = json.loads(gpt_response["content"])
            print("当前怎么选的,", gpt_content_json)
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

        def apixuanranweitiaoleibbox(
                self, imageid, imagefilepath, apibbox_prompts, xinintermediate_output_dir, classtarget1,
                query,
                jiancemodel, des
        ):
            # 去选出来新的image里面合适的id
            # ceshiid是为了存文件用的
            # 这个是需要路径的
            xinimage = Image.open(imagefilepath)
            detections = jiancemodel.detect(imagefilepath, classtarget1)
            detections = detections[detections.confidence > 0.20]
            num_candidate = len(detections)
            if num_candidate == 1:
                return 0, detections[0]
            if num_candidate == 0:
                return -1, None
            messages = copy.deepcopy(messages_his)
            bbox_index_gpt_select_output_dir = os.path.join(
                xinintermediate_output_dir, "findfbboxselect"
            )
            mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)

            jiheids = []
            jiheanoimages = []
            for detgeshu in range(num_candidate):
                jiheids.append(detgeshu)

                ori_image = xinimage
                annotated_image = self.default_bbox_annotator.annotate(
                    scene=ori_image, detections=detections[detgeshu]
                )
                annotated_image = self.default_label_annotator.annotate(
                    scene=annotated_image, detections=detections[detgeshu]
                )
                jiheanoimages.append(annotated_image)
            base64Frames = self.stitch_and_encode_images_to_base64(
                jiheanoimages, jiheids, intermediate_output_dir=bbox_index_gpt_select_output_dir
            )

            # 这里我们需要处理的只有一个图像
            # 欧，我想起来了，一个是框，一个是id。
            # 我们可以用format去搞

            multi_prompt = apibbox_prompts.format(
                target=classtarget1,
                num_candidate_bboxes=num_candidate,
                description=des,
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
                                    "detail": "low",
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
                bbox_index = -1
                gpt_message = None

                gpt_response = self.openaigpt.safe_chat_complete(
                    messages, response_format={"type": "json_object"}, content_only=True
                )

                cost += gpt_response["cost"]
                gpt_content_json = json.loads(gpt_response["content"])
                print("怎么选bbox的，", gpt_content_json)
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

    def baohanwuti(self, img_path, prompt, jiancemod, anchorclass, targetclass):
        img = Image.open(img_path)
        to_imagebase64 = encode_PIL_image_to_base64(
            resize_image(img)
        )
        x = prompt.format(
            anchor_class=anchorclass,
            target_class=targetclass,
        )
        ww = []
        ww.append({
            "role": "user",
            "content": [
                {"type": "text", "text": x},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{to_imagebase64}",
                        "detail": "low",
                    },
                },

            ],
        })
        gpt_response = self.openaigpt.safe_chat_complete(
            ww, response_format={"type": "json_object"}, content_only=True
        )

        print("结束进行gpt测试")
        cost = gpt_response["cost"]
        gpt_content = gpt_response["content"]
        print(gpt_content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        gpt_content_json = json.loads(gpt_content)
        maodianifornot = gpt_content_json.get("anchor_exist", "false")
        targetifornot = gpt_content_json.get("target_exist", "false")

        # mao1 = jiancemod.detect(img_path, anchorclass)
        # mao1 = mao1[mao1.confidence > 0.20]
        # 使用map()函数将每个元素乘以2

        # tar1 = jiancemod.detect(img_path, targetclass)
        # tar1 = tar1[tar1.confidence > 0.20]
        mao1 = 1
        tar1 = 1
        maoexist = maodianifornot
        tarexist = targetifornot

        return maoexist, tarexist, mao1, tar1

    def cunxuanran(self, img, depth, patspose, imgfile, depthfile, invimgfile, posefile):
        data1 = depth * 1000
        data1 = data1.astype(np.uint32)
        downsample_factor = 2

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

    def cunquxuanrantupian(self, xianzaiid, intfiledir, godxuanran, yespose):
        # 先暂时存today的pose
        ###我们只需要把这个函数更改成专门针对todaypose得就好了
        fanhuiimg, fanhuidepth, fanhuipose, patspose = godxuanran.toposeduiyingchuli(yespose)
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
        self.cunxuanran(fanhuiimg, fanhuidepth, patspose, xuanranimagefile, xuanrandepthfile, xuanraninvimagefile,
                        xuanranposefile)

        return xuanranimagefile, xuanrandepthfile, xuanraninvimagefile, nowpose, patspose

    def cunquxuanrantupianforto(self, xianzaiid, intfiledir, godxuanran, topose):
        # 先暂时存today的pose
        fanhuiimg, fanhuidepth, fanhuipose, patspose = godxuanran.toposeduiyingchuli(topose)
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
        self.cunxuanran(fanhuiimg, fanhuidepth, patspose, xuanranimagefile, xuanrandepthfile, xuanraninvimagefile,
                        xuanranposefile)

        return xuanranimagefile, xuanrandepthfile, xuanraninvimagefile, nowpose, patspose


    def move_along_local_axis(self, pose, delta_x=0, delta_y=0, delta_z=0):
        """沿局部坐标系平移（修改平移部分）"""
        new_pose = pose.copy()
        new_pose[:3, 3] += np.array([delta_x, delta_y, delta_z])
        return new_pose

    def rotate_local_axis(self, pose, axis='z', angle_deg=0):
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

    def shiyongaction(self, qishipose, fangxiang):
        if fangxiang == "up":
            xinpose = self.rotate_local_axis(qishipose, "x", 30)
        elif fangxiang == "down":
            xinpose = self.rotate_local_axis(qishipose, "x", -30)
        elif fangxiang == "left":
            xinpose = self.rotate_local_axis(qishipose, "y", -30)
        elif fangxiang == "right":
            xinpose = self.rotate_local_axis(qishipose, "y", 30)
        elif fangxiang == "forward":
            xinpose = self.move_along_local_axis(qishipose, delta_z=0.3)
        else:
            xinpose = self.move_along_local_axis(qishipose, delta_z=-0.3)

    def huodexinqishipose(self, qishipose, toimagebase, shangyiciaction, maodian, mubiao, query):
        x = celue_prompt1.format(
            anchor=maodian,
            query=query,
            target=mubiao,
            lastaction=shangyiciaction,
        )
        ww = []
        ww.append({
            "role": "user",
            "content": [
                {"type": "text", "text": x},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{toimagebase}",
                        "detail": "low",
                    },
                },
            ],
        })
        gpt_response = self.openaigpt.safe_chat_complete(
            ww, response_format={"type": "json_object"}, content_only=True
        )

        print("结束进行gpt测试")
        cost = gpt_response["cost"]
        gpt_content = gpt_response["content"]
        print(gpt_content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        gpt_content_json = json.loads(gpt_content)
        yidong = gpt_content_json.get("move", "down")
        xinpose = self.shiyongaction(yidong)
        return xinpose, yidong

    def genjuguanxichanshengpose(self, alignfuzhu, jianceweizipose, guanxi, zengjia):
        ###ok,都可以保证
        zongpose = []
        if zengjia:
            zongpose.append(jianceweizipose)
        align_pro = chulialign(alignfuzhu)
        qishipose = update_y_kaojin(jianceweizipose, align_pro)
        # 我们在每种类型下都增加一个俯视的轨迹


        if guanxi == 'down':
            qishipose = self.move_along_local_axis(qishipose, delta_z=-0.45)
            for y_ang in [-90, -45, 0, 45, 90]:
                dangqianpose = self.rotate_local_axis(qishipose, 'y', y_ang)
                for x_ang in [0, -18, -36, -54]:
                    zongpose.append(self.rotate_local_axis(dangqianpose, 'x', x_ang))

            return zongpose
        elif guanxi == 'up':
            qishipose = self.move_along_local_axis(qishipose, delta_z=-0.45)
            for y_ang in [-90, -45, 0, 45, 90]:
                dangqianpose = self.rotate_local_axis(qishipose, 'y', y_ang)
                for x_ang in [0, -18, -36, -54]:
                    zongpose.append(self.rotate_local_axis(dangqianpose, 'x', x_ang))
            return zongpose
        else:
            t = -1
            w = -1
            qishipose = self.move_along_local_axis(qishipose, delta_z=-0.15)
            qishipose = interpolate_translation(align_pro, qishipose, 0.2)

            for y_ang in [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342]:
                dangqianpose = self.rotate_local_axis(qishipose, 'y', y_ang)
                zongpose.append(self.rotate_local_axis(dangqianpose, 'x', w * 25))
            return zongpose

    def apixuanranimage(self, maodangqianid, idls, imagels, apixuanran_promptw, intermediate_output_dir, query, mao,
                        tar, condition):
        # yesid唯一得作用就是做这个
        if len(idls) == 0:
            print("全是白色图像，没法弄")
            return [], "", -1, "", "", ""
        image_id_gpt_select_output_dir = os.path.join(
            intermediate_output_dir, f"zheyilundijige{maodangqianid}grid_image"
        )
        mmengine.mkdir_or_exist(image_id_gpt_select_output_dir)
        for img_path in imagels:
            print(img_path)
        images = [Image.open(img_path) for img_path in imagels]

        base64Frames = self.stitch_and_encode_images_to_base64(
            images, idls, intermediate_output_dir=image_id_gpt_select_output_dir
        )
        #
        ###### * End of loading images and creating grid images

        ###### * Format the prompt for VLM
        input_prompt_dict = {
            "query": query,
            "anchorclass": mao,
            "targetclass": tar,
            "num_view_selections": len(idls),
            "condition": condition,
        }

        gpt_input = apixuanran_promptw.format_map(input_prompt_dict)
        system_prompt = self.system_prompt

        print(gpt_input)
        begin_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_input},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                ],
            },
        ]
        messages = begin_messages

        cost = 0
        retry = 1
        pred_image_id = -1
        gpt_content_json = ""
        while retry <= 1:
            ###### * Get pred_image_id use VLM
            try:
                # save the prompt
                mmengine.dump(
                    filter_image_url(messages),
                    f"{image_id_gpt_select_output_dir}/gpt_prompts_retry{retry}.json",
                    indent=2,
                )
                # call VLM api
                with TimeCounter(tag="pred_image_id VLM Time", log_interval=10):
                    gpt_response = self.openaigpt.safe_chat_complete(
                        messages,
                        response_format={"type": "json_object"},
                        content_only=True,
                    )
                    gpt_content = gpt_response["content"]
                    gpt_content_json = json.loads(gpt_content)
                    print(gpt_content_json)

            except Exception as e:
                print(
                    f"[VLMPredImageID] {scene_id}: [{query[:20]}] VLM failed to return a response. Error: {e} Retrying..."
                )
                retry += 1
                continue

            # Extract and process the relevant information from the response
            cost += gpt_response["cost"]
            pred_image_id = gpt_content_json.get("target_image_id", -1)
            refer_ids = gpt_content_json.get("reference_image_ids", [pred_image_id])
            des = gpt_content_json.get("extended_description", "")
            desp = gpt_content_json.get("extended_description_withposition", "")

            # save the response
            gpt_message = {
                "role": "assistant",
                "content": [{"text": gpt_content, "type": "text"}],
            }
            mmengine.dump(
                filter_image_url(messages + [gpt_message]),
                f"{image_id_gpt_select_output_dir}/gpt_responses_retry{retry}.json",
                indent=2,
            )
            if pred_image_id is None:
                pred_image_id = -1
            pred_image_id = int(pred_image_id)  # str -> int, get a valid image ID
            break
            ###### * End of get pred_image_id using VLM

        # 返回int, 以及pred_imagefile
        # 所以我们看看标注图像是怎么弄的
        if pred_image_id == -1:
            return [], "", -1, "", "", ""
        if pred_image_id != -1:
            messages.append(gpt_message)
        messages_his = messages
        ind = idls.index(pred_image_id)
        return refer_ids, messages_his, pred_image_id, imagels[ind], desp, des

    def biaozhurefer(self, imagefile, detection, cankaodir):
        cankaofile = os.path.join(cankaodir, "cankaobbox.jpg")
        xinimage = Image.open(imagefile)

        """ori_image = xinimage
        annotated_image = self.default_bbox_annotator.annotate(
            scene=ori_image, detections=detection
        )
        annotated_image = self.default_label_annotator.annotate(
            scene=annotated_image, detections=detection
        )"""
        # 创建一个可以在给定图像上绘图的对象
        wwww = detection.xyxy
        for xx1, yy1, xx2, yy2 in detection.xyxy:
            x1 = xx1
            y1 = yy1
            x2 = xx2
            y2 = yy2
        annotated_image = xinimage
        draw = ImageDraw.Draw(annotated_image)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=4)  # 红色边框，宽度为3

        # 设置字体和字体大小
        # 如果没有指定字体文件，可以使用默认字体
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 字体文件路径
        font = ImageFont.truetype(font_path, 60)  # 字体大小为 24

        # 设置文本内容和位置
        text = "refer"
        text_position = (10, 10)  # 左上角的位置，可以根据需要调整偏移量

        # 设置文本颜色
        text_color = (255, 0, 0)  # 白色，也可以根据需要调整颜色

        # 在图像上绘制文本
        draw.text(text_position, text, font=font, fill=text_color)
        """annotated_image = cv2.imread(imagefile)
        wwww = detection.xyxy
        for xx1, yy1, xx2, yy2 in detection.xyxy:
            x1 = xx1
            y1 = yy1
            x2 = xx2
            y2 = yy2


        # 绘制矩形框
        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

        #这个是框标注完了，接下来是refer的标注，这个参考学长给出的。
        #或者我们看一下id0怎么标的，我们只需要大概位置就可以标了

        # 这样我们把bbox标注的那个搞过来，之后弄一个标注的列表
        # 然后对这个列表按照顺序标注就好了
        #ceshiarray = np.array(annotated_image)

        cv2.putText(annotated_image, "refer", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)"""

        # convert the annotated image to base64
        # annotated_image = Image.fromarray(annotated_image)
        annotated_image.save(cankaofile)
        annotated_image = Image.open(cankaofile)
        annotated_image_base64 = encode_PIL_image_to_base64(
            resize_image(annotated_image)
        )
        return annotated_image_base64

    def findfitimage(self, todayvgrow, yes_scanid, to_scanid, reference_poses1, reference_pathes1):
        # 这个函数，只负责找id，detection或者bbox什么的等会写一个别的函数去找。
        # 注意了，我这里的多种的是从bboxapi选择里的函数里返回的，所以绝对有id，我们现在只需要按照bbox那里的东西把咱们获得的新的image里面的东西选出来就好
        # 注意了，这个不能从什么已经弄好的detection里面找东西，这个是不行的，因为你不能提前知道这个东西
        # 注意了，我们是需要找到包含target物体，且query正确的id哟。
        # 对了，其实只需要给referencesid就足够了，我可以一个一个的找

        # 是这样的maodian物体我们直接用过去的好了
        # get_yesimage_align_pose(self, scan_id, image_id):
        # detect应该没啥问题，可以直接用
        # 如果是between关系
        # 我们应该单独写一个函数进行寻找两个pose

        ###主要是要看一下reference_ids起到了什么作用
        # 所有输入到findfi函数里的
        ###我们统一一下所有得reference_poses，注意传入得是列表
        if isinstance(reference_poses1, list):
            reference_poses = reference_poses1
        else:
            reference_poses = [reference_poses1]
        if isinstance(reference_pathes1, list):
            reference_pathes = reference_pathes1
        else:
            reference_pathes = [reference_pathes1]

        hmposels = []
        scene_info = mmengine.load(self.scene_info_path)
        # cankaoyes = cankaoyes.split('.')[0]  # "frame-000045.color"原来是这个，现在我们不需要去重复的ids了
        maodianclasses = todayvgrow["anchors_types"]
        maodianclasses = ast.literal_eval(maodianclasses)
        conditions = eval(todayvgrow["attributes"]) + eval(todayvgrow["conditions"])
        classtarget = todayvgrow["pred_target_class"]
        actioncost = 0
        linecost = 0
        degreecost = 0
        jisuanalign = scene_info[yes_scanid]["axis_align_matrix"]

        def yzzhuanhuan(view3):
            view3[:, 2] = -view3[:, 2]
            view3[:, [1, 2]] = view3[:, [2, 1]]
            view3[2, 3] += 0.55
            return view3

        pre_pose = yzzhuanhuan(jisuanalign)  # 昨天调整
        pre_pose = huodealignpose(pre_pose, to_scanid, yes_scanid)  # 取今天对齐
        pre_pose = np.linalg.inv(pre_pose)  # 今天alignpose位置

    

        cost = 0
        # 现在是这样的我们应该只需要进行一个align之后就完成了，所以不需要这个god说实在话，但是我们仍然需要
        godxuanran = godmodule(to_scanid, yes_scanid)
        query = todayvgrow["utterance"]
        xuanrancuntupiandir = os.path.join(self.dangqianoutdir, "xuanrantupianintdir")  # 换一下位置
        mmengine.mkdir_or_exist(xuanrancuntupiandir)
        xuanrancuntupianid = 0

        # to_query_prompt = f"You are an agent who is very good at looking at pictures. Now I'm giving you one picture. You need to determine whether the picture contains an {classtarget} that is fully meets a specific description: {query}. If it contains, return True; if not, return False. Notice, return only False or True."
        fenxi_query_prompt = """You are an agent who is very good at analyzing spatial relationships. Given you a query : {query} , a target object : {classtarget1} and anchor object : {anchorclass} , your task is to determine the spatial relationship of the target object relative to anchor object according to the query content. The possible spatial relationships are as follows: 
                            - up: the target object is above the anchor object // the target object that is lying on the anchor object // the target object that is on top of the anchor object.
                            - down: the target object is below the anchor object // the target object that is supporting the anchor object // the target object with anchor object on top // the target object with anchor object on its top.
                            - near: the target object is close to the anchor object.
                            - far: the target object is far from the anchor object.
                            - between: the target object is between multiple anchor objects.

                            Please simply return the spatial relationship of the target object relative to the anchor object.

                            Reply using JSON format with one key "reasoning" like this:
                            {{
                              "reasoning": "up" // Return the spatial relationship type (up, down, near, far, or between) that you determine for the target object relative to the anchor object.
                            }}

                            """
        ###是这样的，为了保持准确度。我们只用target和anchor
        fenxi_query_prompt = fenxi_query_prompt.format(
            classtarget1=classtarget,
            anchorclass=maodianclasses,
            query=query,
        )
        fenxi = []
        fenxi.append({
            "role": "user",
            "content": [
                {"type": "text", "text": fenxi_query_prompt},
            ],
        })
        response1 = self.openaigpt.safe_chat_complete(
            fenxi, response_format={"type": "json_object"}, content_only=True
        )
        cost += response1["cost"]
        content = response1["content"]
        print(content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        content_json = json.loads(content)
        guanxi = content_json.get("reasoning", "near")
        # guanxi = "near"
        # 获得关系这块没有问题
        # zhunmengeibetween = []
        # posezhuanmen1 = []
        # posezhuanmen2 = []

        # 参考img和分析关系已经准备好了。
        # cankao的也已经准备好
        # between不要专门搞，让mem去处理就好
        # 之后这个循环最好保证每次都一样
        # okk，在这里欧

        ###我们看看yes_id得作用是什么
        for yes_id in range(len(reference_pathes)):
            # 现在注意了这个可能比较少了，可能只有三四个
            mis = []
            # 前面卡住可能是因为mis的原因，mis应该最好每次都是单独的，因为我们只是判断。
            # if yes_id[0] == "f":
            #    yes_id = int(yes_id.split('-')[1])
            # elif yes_id[0] == '0':
            #    yes_id = int(yes_id)
            # else:
            #    yes_id = yes_id
            # yespose不需要对过齐

            # 旋转是在获得今天的pose之后进行的，只不过是在求逆之前
            # yesid_zhunhuan = "frame-0" + f"{int(yes_id):05d}"
            # yesposeformem = self.scene_infos.infos[yes_scanid]["images_info"][yesid_zhunhuan]['extrinsic_matrix']
            # yespath = "../hm3rscandata/xuanran/ceshioutputscan/" + \
            #          self.scene_infos.infos[yes_scanid]["images_info"][yesid_zhunhuan]['image_path']  # 参考图片的路径
            yesposeformem = reference_poses[yes_id]
            if isinstance(yesposeformem, str):
                yesposeformem1 = np.loadtxt(yesposeformem)
                yesposeformem = yesposeformem1
            yespath = reference_pathes[yes_id]
            jiancemodel = self.detection_model
            ###确定没变才给的

            # 我们应该获取的是今天的位姿
            # 我们要取得今天的detection才对，但是之前的存着也不是不行
            # 注意我们的pose都是求过逆的
            # 只要返回了一次图片，就应该记录一次
            # ceshiid = "frame-000022"

            toscan_id = to_scanid
            query = todayvgrow["utterance"]
            classtarget = todayvgrow["pred_target_class"]
            # Image.fromarray(img
            # 这里我们不在存
            mao_exist, tar_exist, maodet, tardet = self.baohanwuti(yespath, baohan_prompt2, jiancemodel,
                                                                   maodianclasses[0], classtarget
                                                                   )  # 我们看看这个包含有没有做到，我们需要做的，并且应该能重复用，就是检测
            # 这个是只有vlm和detec同时发现东西才是算有东西
            # to_query_prompt = f"You are an agent who is very good at looking at pictures. Now I'm giving you one picture. You need to determine whether the picture contains an {classtarget} that is fully meets a specific description: {query}. If it contains, return True; if not, return False. Notice, return only False or True."
            to_query_prompt = """You are an agent who is very good at looking at pictures. Now I'm giving you one picture. You need to determine whether the picture contains an {classtarget} that is fully meets a specific description: {query}. 
                        Reply using JSON format with two keys "reasoning", "image_correct_or_not" like this:
                        {{
                          "reasoning": "your reasons", // Explain the justification why you think the image is correct, or why the image is not correct.
                          "image_correct_or_not": true, // true if you think the image is correct, false if you think the image is wrong
                        }}"""
            to_query_prompt = to_query_prompt.format(
                classtarget=classtarget,
                query=query,
            )
            panduan_prompt = """You are an agent who is very good at looking at pictures. Now I'm giving you one picture. You need to determine whether the anchor object {anchor_object} in the picture is almost completely captured.
            If the anchor object is almost completely captured in this picture, return True. If only a part of the anchor object is captured and you feel that the shooting perspective is very close to the object, return False.
            Reply using JSON format with two keys "reasoning", "image_correct_or_not" like this:
            {{
            "reasoning": "your reasons", // Explain the justification why you think the image meets the condition or not.
            "image_correct_or_not": true // true if the anchor object is almost completely captured, false if only a part is captured and the shooting perspective is very close
            }}"""
            panduan_prompt = panduan_prompt.format(
                anchor_object=maodianclasses
            )

            # 暂时不用路径，如果后面需要的话，我们再在后面加上
            # 这个是多classes的么，
            # 记得要改呀，不改也行看情况·，本来昨天的就不准
            alignfuzhu = self.scene_infos.infos[to_scanid]["axis_align_matrix"]
            # 我们batch的时候存的就是求过逆的所以，欧我明白了这步是对的，因为这个才是真正横平数直的那个
            alignfuzhu = np.linalg.inv(alignfuzhu)

            # 我们在这里存一下

            # 假设先尝试一下都为正
            # tar_exist = True
            # mao_exist = True
            # 我们要将yesposeformem转变成今天的pose
            hmyespose2to = yesposeformem
            hmposels.append(hmyespose2to)
            nowpose = yesposeformem
            ###现在是都渲染好了。就等着看呢，因为我们只有一个，修改一下prompt好了。



            
            # 注意，现在昨天和今天有transformer是完全能够对齐的，所以我们
            chanshengpose = self.genjuguanxichanshengpose(alignfuzhu, yesposeformem, guanxi, True)
            # 这个函数等会去处理，假设已经处理好了
            # 唯一需要注意的是需要记录一下wangxiang
            
            query = todayvgrow["utterance"]
            classtarget = todayvgrow["pred_target_class"]
            imagels = []
            posels = []
            idls = []
            for item in chanshengpose:
                xuanranimagefile, xuanrandepthfile, xuanraninvimagefile, fanhuipose, costpose = self.cunquxuanrantupian(
                    xuanrancuntupianid, xuanrancuntupiandir, godxuanran, item)
                # 对渲染图片，pose等等进行存储
                # 注意就算它是高度白化识别不出来东西，你也需要存储这个pose
                hmposels.append(costpose)
                if self.calculate_brightness_opencv(xuanraninvimagefile) < 200:
                    imagels.append(xuanraninvimagefile)
                    posels.append(item)
                    idls.append(xuanrancuntupianid)
                nowpose = fanhuipose  # 返回的是取过逆的to中的pose
                xuanrancuntupianid += 1
                # xuanrantupianintdir = os.path.join(self.dangqianoutdir, toscan_id, "xuanrantupianintdir")

                actioncost += 1
                # 注意这里的初始位姿应该在没有对齐的to世界中的做过变换的东西
                l, r = jisuancost(pre_pose, nowpose)
                linecost += l
                degreecost += r
                # 只要返回了一次图片，就应该记录一次
                # ceshiid = "frame-000022"
                # xinid = 'frame-000112'

                pre_pose = nowpose
            if len(idls) == 0:
                continue

            refer, mis, pred_imageidapi, pred_imagefileapi, des, desp = self.apixuanranimage(yes_id, idls,
                                                                                             imagels,
                                                                                             apixuanran_prompt5,
                                                                                             xuanrancuntupiandir,
                                                                                             query,
                                                                                             maodianclasses[0],
                                                                                             classtarget,
                                                                                             conditions)
            # file因为一开始弄成inv了了。
            if pred_imageidapi == -1:
                continue
            else:
                bbox_idapi, detectionapi = self.apixuanranbbox(mis, pred_imageidapi, pred_imagefileapi,
                                                               apibbox_prompt3,
                                                               xuanrancuntupiandir, maodianclasses[0],
                                                               classtarget, query, jiancemodel, des)
                # bbox_idapi, detectionapi = self.apixuanranchushileibbox(mis, pred_imageidapi, pred_imagefileapi,
                #                                                                                weitiao_prompt5,
                #                                                                               xuanrancuntupiandir,
                #                                                                                classtarget, query, jiancemodel, des)
                if bbox_idapi == -1:
                    continue
                else:
                    outfile = os.path.join(xuanrancuntupiandir, "cankao.txt")
                    with open(outfile, 'w', encoding='utf-8') as file:
                        for element in refer:
                            file.write(str(element) + '\n')
                    refer = []
                    refer.append(pred_imageidapi)
                    refer = list(set(refer))
                    cankaobase = self.biaozhurefer(pred_imagefileapi, detectionapi, self.zuizhongpatsdir)
                    posezhuanhuan = "frame-" + f"{int(pred_imageidapi):06d}" + ".txt"
                    posefile = os.path.join(xuanrancuntupiandir, posezhuanhuan)
                    return cankaobase, des, desp, hmposels, mis, godxuanran, xuanrancuntupianid, refer, pred_imageidapi, pred_imagefileapi, posefile, detectionapi, nowpose, cost, actioncost, linecost, degreecost, xuanrancuntupiandir


        return None, None, None, hmposels, None, None, -1, [], -1, None, None, None, nowpose, cost, actioncost, linecost, degreecost, xuanrancuntupiandir
        # 你判断的时候

    def xuanranbijiaodongmeidong(self, todayvgrow, yes_scanid, to_scanid,
                                 reference_imageids, scene_infos, jiancemodel, yaoquedingdeclass):
        hmposels = []
        if reference_imageids == -1:
            print("dongmeidong 因为id为-1退出, 这个不需要它获得pose")
            None, None, None
        scene_info = scene_infos
        # cankaoyes = cankaoyes.split('.')[0]  # "frame-000045.color"原来是这个，现在我们不需要去重复的ids了
        # maodianclasses = todayvgrow["anchors_types"]
        # maodianclasses = ast.literal_eval(maodianclasses)
        classtarget = todayvgrow["pred_target_class"]
        if isinstance(yaoquedingdeclass, list):
            panduanclass = yaoquedingdeclass[0]
        else:
            panduanclass = yaoquedingdeclass

        cost = 0
        # 现在是这样的我们应该只需要进行一个align之后就完成了，所以不需要这个god说实在话，但是我们仍然需要
        godxuanran = godmodule(to_scanid, yes_scanid)
        query = todayvgrow["utterance"]
        xuanrancuntupiandir = os.path.join(self.dangqianoutdir, "dong4", "panduan")  # 换一下位置
        mmengine.mkdir_or_exist(xuanrancuntupiandir)

        ###我们看看yes_id得作用是什么
        ###不要列表了
        yes_id = reference_imageids
        # 现在注意了这个可能比较少了，可能只有三四个
        mis = []
        # 前面卡住可能是因为mis的原因，mis应该最好每次都是单独的，因为我们只是判断。
        if type(yes_id) is not str:
            yes_id = yes_id
        elif yes_id[0] == "f":
            yes_id = int(yes_id.split('-')[1])
        elif yes_id[0] == '0':
            yes_id = int(yes_id)
        else:
            yes_id = yes_id
        # yespose不需要对过齐

        # 旋转是在获得今天的pose之后进行的，只不过是在求逆之前
        yesid_zhunhuan = "frame-0" + f"{int(yes_id):05d}"
        yesposeformem = self.scene_infos.infos[yes_scanid]["images_info"][yesid_zhunhuan]['extrinsic_matrix']
        yespath = "../hm3rscandata/xuanran/ceshioutputscan/" + \
                  self.scene_infos.infos[yes_scanid]["images_info"][yesid_zhunhuan]['image_path']  # 参考图片的路径

        ###这是原本的，我们再去找渲染的

        toscan_id = to_scanid
        query = todayvgrow["utterance"]
        classtarget = todayvgrow["pred_target_class"]

        ####这里我们判断么，判断一下吧，但是我们不用api多麻烦。
        ####我们下一步是去找怎么根据yesterdaypose去渲染图片的。

        detections = jiancemodel.detect(yespath, panduanclass)
        detections = detections[detections.confidence > 0.20]
        if len(detections) == 0:
            ###比较的话这里也要处理一下下
            ###注意是这样的，这里说明锚点选错了。我们不进行今天场所的探索。而是直接想办法在center中找新锚点
            print("动没动因为什么都检测不到退出， 但是这里必须保存pose")
            return None, None, None

        ###这里已经处理好了，转变成topose了所以还好
        yesposeformem = huodealignpose(yesposeformem, toscan_id, yes_scanid)

        xuanranimagefile, xuanrandepthfile, xuanraninvimagefile, fanhuipose, costpose = self.cunquxuanrantupian(
            yes_id, xuanrancuntupiandir, godxuanran, yesposeformem)

        ###注意了，我们好像改了，所以你传入昨天的位姿好像是不对的
        ###注意fanhuipose是ni，costpose应该是topose
        ###我们直接把那个todaypose的东西拿出来
        posemingzi = "frame-" + f"{int(yes_id):06d}" + ".txt"
        xuanranposefile = os.path.join(xuanrancuntupiandir, posemingzi)

        # convert the annotated image to base64
        image1 = Image.open(yespath)
        cankaopath = os.path.join(xuanrancuntupiandir, f"cankao{yes_id}.jpg")
        image1.save(cankaopath)
        image1_base64 = encode_PIL_image_to_base64(
            resize_image(image1)
        )
        image2 = Image.open(xuanranimagefile)
        image2_base64 = encode_PIL_image_to_base64(
            resize_image(image2)
        )

        fenxi = []
        input_prompt = same_prompt1.format(
            target_class=panduanclass,
        )
        fenxi.append({
            "role": "user",
            "content": [
                {"type": "text", "text": input_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image1_base64}",
                        "detail": "low",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image2_base64}",
                        "detail": "low",
                    },
                },
            ],
        })
        response1 = self.openaigpt.safe_chat_complete(
            fenxi, response_format={"type": "json_object"}, content_only=True
        )
        cost += response1["cost"]
        content = response1["content"]

        print(content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        content_json = json.loads(content)
        yiyangme = content_json.get("images_same_or_not", False)
        w = np.loadtxt(xuanranposefile)
        ###注意了，它这个pose是从渲染中间提取出来的
        ###所以存渲染的时候存的是哪一个呢
        return yiyangme, w, xuanranimagefile

    def zhaomaodianforhbe(self, images, matched_image_ids, end_class, pred_target_class, intermediate_output_dir,
                          total_cost, yes_scanid, to_scanid, tovg_row, scene_infos):
        zhaomaopose = []
        target_class = tovg_row["instance_type"]
        query = target_class
        mao_class = tovg_row["anchors_types"]
        mao_class = ast.literal_eval(mao_class)
        jiancemodel = self.detection_model
        ###用之前的来弄
        base64Frames = self.stitch_and_encode_images_to_base64(
            images, matched_image_ids, intermediate_output_dir=intermediate_output_dir
        )
        #
        ###### * End of loading images and creating grid images

        ###### * Format the prompt for VLM

        ####这里注意了，我们需要根据类行来决定具体是用哪个来处理，这里写成代码
        ####hor和betwe的直接原来的，找锚点物体，其余的直接进行query查询这里注意两个函数直接决定，东西还在不在原地了
        input_prompt_dict = {
            "num_view_selections": len(matched_image_ids),
            "anchor_class": mao_class[0],
            "targetclass": target_class,
        }

        if self.prompt_version <= 2:
            gpt_input = self.input_prompt.format_map(input_prompt_dict)
            system_prompt = ""
        elif self.prompt_version <= 3:  # support
            gpt_input = self.input_prompt.format_map(input_prompt_dict)
            system_prompt = self.system_prompt
        else:
            raise NotImplementedError

        print(gpt_input)
        begin_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_input},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                ],
            },
        ]
        self.dangqianoutdir = intermediate_output_dir
        image_id_gpt_select_output_dir = os.path.join(
            intermediate_output_dir, "image_index_gpt_select_prompts"
        )
        mmengine.mkdir_or_exist(image_id_gpt_select_output_dir)
        ###### * End of format the prompt for VLM

        retry = 1
        pred_image_id_retry = 0

        messages = begin_messages

        # * main loop
        # 我们需要更改一下messgaes的信息
        # 把messages的意思改成要求图片中存在要求的锚点物体，且比较清晰（如果有多个，尽量包括多个）
        # 但是由于图片整个的不确定性，我不知道场景全不全，所以我们第三重保险，看看能不能检测到锚点物体，能检测到一个就行。并且要求返回refer_image_id的时候
        # 能检测到多个的放前面
        # 也不对，我们是为了位姿，不需要重新detection，前面的detection已经足够了
        # 我们只需要写好这个的
        ff = False
        # 也就是说我们只需要更改以下这个函数所需要的prompt就好了
        # 同样我们需要更改一下matchmaoid的数量，太多了也不行
        scene_id = yes_scanid
        while retry <= self.gpt_max_retry:
            ###### * Get pred_image_id using VLM
            # 需要增加一个retry
            try:
                refer_ids, pred_image_id, gpt_content_json, message_his, cost, retry_times, isolated_id = (
                    self.get_gpt_pred_image_idyes(
                        yes_scanid, query, messages, intermediate_output_dir
                    )
                )
            except Exception as e:
                retry += 1
                print("同样的问题，识别有问题，需要重新测试")
                continue
            # 这里是我们预想的他给出的
            # retry_times = 1
            # refer_ids = ['000109', '000116', '000054']
            # pred_image_id = 109
            # gpt_content_json = {
            #    'reasoning': "First, I identified the images that are too dark to see the objects clearly. None of the images are too dark, so all images are considered. Next, I checked for similar images and selected the clearest ones. Images 000021, 000022, and 000020 are very similar, so I selected the clearest one, which is 000021. Images 000036, 000037, and 000039 are also similar, so I selected the clearest one, which is 000036. Images 000042, 000043, 000044, and 000045 are similar, so I selected the clearest one, which is 000045. Images 000019 and 000013 are similar, so I selected the clearest one, which is 000019. Images 000025 and 000026 are similar, so I selected the clearest one, which is 000025. Images 000034 and 000031 are similar, so I selected the clearest one, which is 000034. After filtering, the remaining images are 000004, 000019, 000021, 000025, 000034, 000036, 000045, and 000048. I then sorted the images based on the presence of the end object (washing machine) and the target object (shelf). Images 000025, 000026, 000034, and 000031 contain the washing machine, so they are ranked at the front. Among these, 000034 is the clearest, followed by 000025. Next, I sorted the remaining images based on the presence of the target object (shelf). Image 000048 contains the shelf and is the clearest. The remaining images are sorted based on the camera's proximity to the objects. The final sorted order is 000034, 000025, 000048, 000045, 000036, 000021, 000019, and 000004.",
            #    'sorted_image_ids': ['000034', '000025', '000048', '000045', '000036', '000021', '000019', '000004']}
            # cost = 0
            # message_his = [{'role': 'system',
            #                'content': 'You are good at finding objects specified by user queries in indoor rooms by watching the videos scanning the rooms.'},
            #               {'role': 'user', 'content': [...]}, {'role': 'assistant', 'content': [...]            ff = True
            total_cost += cost
            pred_image_id_retry += retry_times
            print(type(refer_ids))
            print(refer_ids)
            if refer_ids == None:
                refer_ids = []
            if len(refer_ids) == 0:
                break
            if message_his is not None:
                print("是从这里取得么")
                print(pred_image_id)
                print("pred_image_id的类型是啥,", type(pred_image_id))
                # successfully get a valid image id
                image_path = self.scene_infos.get_image_path(scene_id, pred_image_id)
                print("这里取到image_path了么？", image_path)
                messages = message_his  # include stitched images
            elif message_his is None:
                # not complete, retry
                retry += 1
                print(
                    f"[GetImageIDBboxVanilla] {scene_id}: [{query[:20]}] VLM predict image id module is not complete. This may because VLM doet not respond or the responded image ID is invalid. Retrying from getting a image id."
                )
                continue
            ###### * End of get pred_image_id using VLM
            # okk，从这里开始，需要介入今天的信息,不，应该从下面，至少检测到东西再开始
            shifouunique = False
            # 以下的不需要
            ###### * Get pred_bbox_id

            # 注意，这里的话不需要再进行vlm选择bbox_id了，针对昨天的东西，我们只需要判断到底有没有存在就行了

            # god_module = Theallknow(self.scene_info_path, yes_id, yes_scanid, to_scanid)
            reference_imageids = refer_ids[0]
            ###到这里说明不等于零，我们拿过来看一下。
            break
        if len(refer_ids) != 0:
            ###我们只取第一个

            ###我们学findfi去把渲染的给处理出来
            # 以列表的形式
            maodongmeidong, poses, imagepathes = self.xuanranbijiaodongmeidong(
                tovg_row, yes_scanid, to_scanid,
                reference_imageids, scene_infos, jiancemodel, mao_class)

            # 这个应该在上一步中渲染或者什么的决定，所以先不急。
            ###返回的是是否same，所以是反着的

            if maodongmeidong is None:

                #不需要加入这个东西zhaomaopose.append(poses)
                return None, None, None, zhaomaopose, isolated_id
            elif maodongmeidong:

                zhaomaopose.append(poses)
                ###如果没动的话直接返回咱这个mao的poses,以及图像path就好了吧。
                return poses, imagepathes, True, zhaomaopose, isolated_id

            else:
                ###所以是动了，这里我们需要考虑应该想什么别的方法去判断。
                ###锚点物体动了我们不管这个，我们先去把v1那个给写了。
                ###这里如果动了的话直接一口气我们不在这个函数里面弄，我们单独弄一个新的。

                zhaomaopose.append(poses)
                return poses, imagepathes, False, zhaomaopose, isolated_id
        else:
            ###在记忆中没有找到锚点
            return None, None, None, zhaomaopose, isolated_id
        # 如果为none的化我们就需要考虑直接转一圈直接结束结果。

    """def v1shunbianmaotarget(self):
        ###用之前的来弄
        base64Frames = self.stitch_and_encode_images_to_base64(
            images, matched_image_ids, intermediate_output_dir=intermediate_output_dir
        )
        #
        ###### * End of loading images and creating grid images

        ###### * Format the prompt for VLM

        ####这里注意了，我们需要根据类行来决定具体是用哪个来处理，这里写成代码
        ####hor和betwe的直接原来的，找锚点物体，其余的直接进行query查询这里注意两个函数直接决定，东西还在不在原地了
        input_prompt_dict = {
            "pred_target_class": end_class,
            "num_view_selections": len(matched_image_ids),
            "anchor_class": pred_target_class[0],
        }

        if self.prompt_version <= 2:
            gpt_input = self.input_prompt.format_map(input_prompt_dict)
            system_prompt = ""
        elif self.prompt_version <= 3:  # support
            gpt_input = self.input_prompt.format_map(input_prompt_dict)
            system_prompt = self.system_prompt
        else:
            raise NotImplementedError

        print(gpt_input)
    """

    def xianzhaomaoforhbe(self, vg_row, tovg_row):
        total_cost = 0
        scene_id = vg_row["scan_id"]
        query = tovg_row["anchors_types"]
        end_class = tovg_row["instance_type"]
        shibushisu = tovg_row["coarse_reference_type"]
        pred_target_class = tovg_row["anchors_types"]
        pred_target_class = ast.literal_eval(pred_target_class)
        join_list = ",".join(pred_target_class)
        self.dangqianoutdir = vg_row["intermediate_output_dir"]
        # 不过这玩意是个列表类型的，我们后面考虑一下怎么处理
        # 目标物体什么的都要变
        conditions = eval(vg_row["attributes"]) + eval(vg_row["conditions"])
        ####我们要求mao和target它都要
        ###只要这里变动一下就可以了。
        """if shibushisu == "horizontal" or shibushisu == "between":
            matched_image_idsmao = eval(
                vg_row[f"matched_image_ids_confidence{0.3}mao"]
            )
            matched_image_ids = matched_image_idsmao

        else:
            matched_image_idsmao = eval(
                vg_row[f"matched_image_ids_confidence{0.3}mao"]
            )
            matched_image_idstarget = eval(
                vg_row[f"matched_image_ids_confidence{0.2}"]
            )
            matched_image_ids = matched_image_idsmao + matched_image_idstarget
            matched_image_ids = list(set(matched_image_ids))"""
        matched_image_idsmao = eval(
                vg_row[f"matched_image_ids_confidence{0.3}mao"]
            )
        matched_image_idstarget = eval(
            vg_row[f"matched_image_ids_confidence{0.2}"]
        )
        matched_image_ids = matched_image_idsmao + matched_image_idstarget
        matched_image_ids = list(set(matched_image_ids))
        new_list = []
        if len(matched_image_ids) > 160:
            for i in range(0, len(matched_image_ids), 3):
                new_list.append(matched_image_ids[i])
            matched_image_ids = new_list
        elif len(matched_image_ids) > 80:
            for i in range(0, len(matched_image_ids), 2):
                new_list.append(matched_image_ids[i])
            matched_image_ids = new_list
        else:
            matched_image_ids = matched_image_ids
        print("现在macthid有多长,", len(matched_image_ids))

        # ok,以上是参考目标物体用的，之后我们去修改轨迹那里，我们应该是不需要用到轨迹了
        flag = False

        allmaodian = False
        # 以下这一步不用管，我们要求它不能为0
        if len(matched_image_ids) == 0:
            print("是否一开始")
            cankaoids = []
            t = 0
            for key, value in self.scene_infos.infos[scene_id]["images_info"].items():
                if t % 3 == 0:
                    cankaoids.append(key)
                t += 1
            # maodianyes, cost = self.juedingcankao(cankaoids, scene_id, pred_target_class, self.system_prompt)
            maodianyes = "frame-000060.color"
            cost = 0
            total_cost += cost
            if len(maodianyes) != 0:
                allmaodian = True
        scene_infos = mmengine.load(self.scene_info_path)
        intermediate_output_dir = vg_row["intermediate_output_dir"]
        # 我们最后还是需要用det文件去取得
        ###### * Load images and create grid images
        # views_pre_selection_paths = huodemaodiandet_paths(self.scene_info_path, matched_image_ids, pred_target_class)
        # 我的建议是直接多个物体，返回的detection两者的并集而不是交集。之后让vlm判断

        #######注意，这上面这步就是把存在锚点物体的图片都搞出来
        # 我们需要单独写一个函数去搞
        # match_id不是所有的么，根据这些id应该直接得到views_selection的path。
        # 这里其实比较好的一点是，它是去取pose，所以这个det是谁都不重要，并且我们其实只需要做一个图片的id判断，这个的代码已经写好。
        # 这里要从昨天开始测试起来。
        # 所以
        yesdet_infosfile = "../hm3rscandata/xuanran/ceshioutputscan/image_instance_detectoryes/yolo_prompt_v2_updated_yesterday_250_updated2_relations/chunk30/detection.pkl"
        yesdet_infos = DetInfoHandler(yesdet_infosfile)

        print("检查一下为什么出现f.colory", matched_image_ids)
        if not allmaodian:
            views_pre_selection_paths = [
                yesdet_infos.get_image_path(scene_id, image_id)
                for image_id in matched_image_ids
            ]
        else:
            views_pre_selection_paths = [
                yesdet_infos.get_image_path(scene_id, image_id)
                for image_id in cankaoids
            ]
            matched_image_ids = cankaoids

        # 我们到时候看看是个咋回事就好。
        ww = []
        for item in views_pre_selection_paths:
            if not item.endswith("f.color.jpg"):
                ww.append(item)
        views_pre_selection_paths = ww

        matched_image_ids = [id.split(".")[0].split("-")[1] for id in matched_image_ids]
        indexed_matched_image_ids = list(enumerate(matched_image_ids))
        indexed_matched_image_ids.sort(key=lambda x: x[1])
        sorted_views_pre_selection_paths = [views_pre_selection_paths[i] for i, _ in indexed_matched_image_ids]
        sorted_matched_image_ids = [matched_image_ids[i] for i, _ in indexed_matched_image_ids]
        matched_image_ids = sorted_matched_image_ids
        views_pre_selection_paths = sorted_views_pre_selection_paths
        ###到这里之后排个顺序
        images = [Image.open(img_path) for img_path in views_pre_selection_paths]
        yes_scanid = scene_id
        to_scanid = tovg_row["scan_id"]

        maoposepath, maoimgpath, zhaodaomaomeiyou, zhaomaopose, isolated_id = self.zhaomaodianforhbe(images, matched_image_ids,
                                                                                        end_class, pred_target_class,
                                                                                        intermediate_output_dir,
                                                                                        total_cost, yes_scanid,
                                                                                        to_scanid, tovg_row,
                                                                                        scene_infos)

        return maoposepath, maoimgpath, zhaodaomaomeiyou, zhaomaopose, isolated_id

    def get_det_bbox3d_from_image_idsingle(
            self,
            xuanranwenjianjia,
            scene_id,
            image_id,
            pred_target_class,
            pred_detection,
            gt_object_id,
            query,
            matched_image_ids,  # supposed to have all valid images and not an empty list
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

        if type(image_id) == str:
            return None
        zhuanhuan = "frame-" + f"{image_id:06d}" + ".jpg"
        img_path = os.path.join(xuanranwenjianjia, zhuanhuan)

        sam_mask_output_dir = os.path.join(
            intermediate_output_dir,
            f"baseyesv2sam_mask_{image_id}_{pred_target_class.replace('/', 'or')}_target_{gt_object_id}",
        )

        if pred_sam_mask is not None:
            # pred seg
            result = pred_sam_mask
        else:
            # from scratch
            if pred_detection is None:
                if bbox_index is None:
                    return None
                else:
                    # Get 2d bbox
                    detections = self.det_infos.get_detections_filtered_by_score(
                        scene_id, image_id, self.image_det_confidence
                    )
                    detections = self.det_infos.get_detections_filtered_by_class_name(
                        None, None, pred_target_class, detections
                    )

                    if len(detections) == 0:
                        print(
                            f"[Get Anchor Bbox]The predicted image {image_id} has no detections for {scene_id}: {query}. pred_taget_class: {pred_target_class}."
                        )
                        return None

                    pred_detection = detections[bbox_index]

            bbox_2d = pred_detection.xyxy

            mmengine.mkdir_or_exist(sam_mask_output_dir)

            # Get sam mask from 2d bbox
            # sam_predictor does not support batched inference at the moment, so len(results) is always 1
            # bbox_2d can be n * 4 or 1-d array of 4. If 1-d array, masks.data.shape would still be 1, H, W
            result = self.sam_predictor(img_path, bboxes=bbox_2d, verbose=False)[0]
            # save the mask
            result.save(f"{sam_mask_output_dir}/anchor_sam_raw.jpg")

            mmengine.dump(result, f"{sam_mask_output_dir}/anchor_sam_raw.pkl")
            mmengine.dump(pred_detection, f"{sam_mask_output_dir}/pred_detection.pkl")

            if self.post_process:
                result = self.post_process_mask(result)
                result.save(f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg")

            mmengine.dump(
                result, f"{sam_mask_output_dir}/anchor_sam_ks{self.kernel_size}.pkl"
            )

            # annotate the bbox on the ori image
            print(img_path)
            ori_image = Image.open(img_path)
            annotated_image = self.default_bbox_annotator.annotate(
                scene=ori_image, detections=pred_detection
            )
            annotated_image = self.default_label_annotator.annotate(
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
            aligned_points_3d = self.ensemble_pred_points(
                xuanranwenjianjia,
                scene_id,
                image_id,
                pred_target_class,
                sam_mask=sam_mask,
                matched_image_ids=matched_image_ids,
                sam_mask_output_dir=sam_mask_output_dir,
                intermediate_output_dir=intermediate_output_dir,
                detections=pred_detection,
            )
        # 这一下的都不用改，我们需要改的就是上面怎么获得3dbbox的
        # 而且是align的
        with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
            if self.point_filter_type == "statistical":
                aligned_points_3d_filtered = remove_statistical_outliers(
                    aligned_points_3d,
                    nb_neighbors=self.point_filter_nb,
                    std_ratio=self.point_filter_std,
                )
            elif self.point_filter_type == "truncated":
                aligned_points_3d_filtered = remove_truncated_outliers(
                    aligned_points_3d,
                    tx=self.point_filter_tx,
                    ty=self.point_filter_ty,
                    tz=self.point_filter_tz,
                )
            elif self.point_filter_type == "none":
                aligned_points_3d_filtered = aligned_points_3d
            else:
                raise NotImplementedError(
                    f"Point filter type {self.point_filter_type} is not implemented."
                )

        # save the aligned points 3d
        aligned_points_3d_output_dir = os.path.join(
            intermediate_output_dir, "projected_points"
        )
        mmengine.mkdir_or_exist(aligned_points_3d_output_dir)
        np.save(
            f"{aligned_points_3d_output_dir}/ensemble{self.ensemble_num}_matching.npy",
            aligned_points_3d,
        )
        np.save(
            f"{aligned_points_3d_output_dir}/ensemble{self.ensemble_num}__matching_filtered.npy",
            aligned_points_3d_filtered,
        )

        # if nan in points, return None
        if (
                aligned_points_3d.shape[0] == 0
                or aligned_points_3d_filtered.shape[0] == 0
                or np.isnan(aligned_points_3d).any()
                or np.isnan(aligned_points_3d_filtered).any()
        ):
            print(
                f"[Box projection] Aligned_points_3d (filtered) is empty or has nan for {scene_id}: {query}. The VLM predicted image id is {image_id}.\
                matched_image_ids: {matched_image_ids}. pred_target_class: {pred_target_class}."
            )
            return None

        pred_bbox = calculate_aabb(aligned_points_3d_filtered)

        return pred_bbox

    def get_det_bbox3d_from_image_idformulti(
            self,
            xuanranwenjianjia,
            scene_id,
            image_id,
            pred_target_class,
            pred_detection,
            gt_object_id,
            query,
            matched_image_ids,  # supposed to have all valid images and not an empty list
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
        if type(image_id) == str:
            return None
        zhuanhuan = "frame-" + f"{image_id:06d}" + ".jpg"
        img_path = os.path.join(xuanranwenjianjia, zhuanhuan)

        sam_mask_output_dir = os.path.join(
            intermediate_output_dir,
            "tempmask",
            f"temp_sam_mask_{image_id}_{pred_target_class.replace('/', 'or')}_target_{gt_object_id}",
        )

        if pred_sam_mask is not None:
            # pred seg
            result = pred_sam_mask
        else:
            # from scratch
            if pred_detection is None:
                if bbox_index is None:
                    return None
                else:
                    # Get 2d bbox
                    detections = self.det_infos.get_detections_filtered_by_score(
                        scene_id, image_id, self.image_det_confidence
                    )
                    detections = self.det_infos.get_detections_filtered_by_class_name(
                        None, None, pred_target_class, detections
                    )

                    if len(detections) == 0:
                        print(
                            f"[Get Anchor Bbox]The predicted image {image_id} has no detections for {scene_id}: {query}. pred_taget_class: {pred_target_class}."
                        )
                        return None

                    pred_detection = detections[bbox_index]

            bbox_2d = pred_detection.xyxy

            mmengine.mkdir_or_exist(sam_mask_output_dir)

            # Get sam mask from 2d bbox
            # sam_predictor does not support batched inference at the moment, so len(results) is always 1
            # bbox_2d can be n * 4 or 1-d array of 4. If 1-d array, masks.data.shape would still be 1, H, W
            result = self.sam_predictor(img_path, bboxes=bbox_2d, verbose=False)[0]
            # save the mask
            result.save(f"{sam_mask_output_dir}/anchor_sam_raw.jpg")

            if self.post_process:
                result = self.post_process_mask(result)
                result.save(f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg")

            # annotate the bbox on the ori image
            print(img_path)
            ori_image = Image.open(img_path)
            annotated_image = self.default_bbox_annotator.annotate(
                scene=ori_image, detections=pred_detection
            )
            annotated_image = self.default_label_annotator.annotate(
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

        # ensemble adjacent images
        with TimeCounter(tag="EnsemblePredPoints", log_interval=60):
            aligned_points_3d = self.ensemble_pred_points(
                xuanranwenjianjia,
                scene_id,
                image_id,
                pred_target_class,
                sam_mask=sam_mask,
                matched_image_ids=matched_image_ids,
                sam_mask_output_dir=sam_mask_output_dir,
                intermediate_output_dir=intermediate_output_dir,
                detections=pred_detection,
            )
        # 这一下的都不用改，我们需要改的就是上面怎么获得3dbbox的
        # 而且是align的
        with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
            if self.point_filter_type == "statistical":
                aligned_points_3d_filtered = remove_statistical_outliers(
                    aligned_points_3d,
                    nb_neighbors=self.point_filter_nb,
                    std_ratio=self.point_filter_std,
                )
            elif self.point_filter_type == "truncated":
                aligned_points_3d_filtered = remove_truncated_outliers(
                    aligned_points_3d,
                    tx=self.point_filter_tx,
                    ty=self.point_filter_ty,
                    tz=self.point_filter_tz,
                )
            elif self.point_filter_type == "none":
                aligned_points_3d_filtered = aligned_points_3d
            else:
                raise NotImplementedError(
                    f"Point filter type {self.point_filter_type} is not implemented."
                )

        # if nan in points, return None
        if (
                aligned_points_3d.shape[0] == 0
                or aligned_points_3d_filtered.shape[0] == 0
                or np.isnan(aligned_points_3d).any()
                or np.isnan(aligned_points_3d_filtered).any()
        ):
            print(
                f"[Box projection] Aligned_points_3d (filtered) is empty or has nan for {scene_id}: {query}. The VLM predicted image id is {image_id}.\
                matched_image_ids: {matched_image_ids}. pred_target_class: {pred_target_class}."
            )
            return None

        return aligned_points_3d_filtered

    def shengchengcankaopose(self, refer, xuanrandir):
        todayposels = []

        for id in refer:
            nowposetxt = "frame-" + f"{int(id):06d}" + ".txt"
            xuanranposefile = os.path.join(xuanrandir, nowposetxt)
            nowpose = np.loadtxt(xuanranposefile)
            nowpose1 = self.rotate_local_axis(nowpose, 'x', 35)
            nowpose1 = self.move_along_local_axis(nowpose1, delta_y=0.2)
            nowpose1 = self.move_along_local_axis(nowpose1, delta_z=-0.6)
            todayposels.append(nowpose1)
            nowpose2 = self.rotate_local_axis(nowpose1, 'y', -20)
            nowpose3 = self.rotate_local_axis(nowpose1, 'y', 20)
            todayposels.append(nowpose2)
            todayposels.append(nowpose3)

        return todayposels

    def referxuanrancun(self, xuanranid, xuanrandir, refer, godxuan):
        xuanranid = xuanranid + 1
        todayposels = self.shengchengcankaopose(refer, xuanrandir)
        patsxuanrandir = xuanrandir
        mmengine.mkdir_or_exist(patsxuanrandir)
        idls = []
        imagels = []
        posels = []
        for item in todayposels:
            xuanranimagefile, xuanrandepthfile, xuanraninvimagefile, fanhuipose = self.cunquxuanrantupian(
                xuanranid, patsxuanrandir, godxuan, item)
            # 对渲染图片，pose等等进行存储
            if self.calculate_brightness_opencv(xuanranimagefile) < 200:
                imagels.append(xuanraninvimagefile)
                posels.append(item)
                idls.append(xuanranid)
            nowpose = fanhuipose  # 返回的是取过逆的to中的pose
            xuanranid += 1
        # 把上一层的refer的都复制过来
        """for id in refer:
            nowimage = "frame-" + f"{int(id):06d}" + ".jpg"
            nowdepth = "frame-" + f"{int(id):06d}" + ".pgm"
            nowposetxt = "frame-" + f"{int(id):06d}" + ".txt"
            srcimagefile = os.path.join(xuanrandir, nowimage)
            srcdepthfile = os.path.join(xuanrandir, nowdepth)
            srcposefile = os.path.join(xuanrandir, nowposetxt)
            dstimagefile = os.path.join(patsxuanrandir, nowimage)
            dstdepthfile = os.path.join(patsxuanrandir, nowdepth)
            dstposefile = os.path.join(patsxuanrandir, nowposetxt)
            shutil.copy(srcimagefile,dstimagefile)
            shutil.copy(srcdepthfile, dstdepthfile)
            shutil.copy(srcposefile, dstposefile)"""
        return idls, imagels, patsxuanrandir

    def apixuanranpats(self, messages_his, xuanranid, xuanrandir, cankao_path, refer, pred_target_class,
                       apixuanranpats_prompt1, jiancemodel, godxuan):
        hmidls, hmimagels, hmpatsdir = self.referxuanrancun(xuanranid, xuanrandir, refer, godxuan)
        annoimagepathls = []
        # 每张图片做检测，并且anno之后，
        bbox_index_gpt_select_output_dir = os.path.join(
            xuanrandir, "pats_bboxchuli"
        )
        mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)
        messages = copy.deepcopy(messages_his)
        detls = []
        fanhuiimgidls = []
        for i in range(len(hmidls)):
            imagefilepath = hmimagels[i]
            imageid = hmidls[i]
            xinimage = Image.open(imagefilepath)

            detections = jiancemodel.detect(imagefilepath, pred_target_class)
            detections = detections[detections.confidence > 0.20]
            num_candidate = len(detections)
            # 这里的我们也要，一个是描述，第二个多加一个判断好吧
            if num_candidate > 0:
                mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)

                labels = [f"ID:{ID}" for ID in range(len(detections))]
                if self.use_bbox_anno_f_gpt_select_id:
                    image = self.bbox_annotator.annotate(scene=xinimage, detections=detections)
                annotated_image = self.label_annotator.annotate(
                    scene=xinimage, detections=detections, labels=labels
                )
                # save the annotated_image
                annotated_image.save(
                    f"{bbox_index_gpt_select_output_dir}/{imageid}_annotated.jpg"
                )
                annoimagepathls.append(f"{bbox_index_gpt_select_output_dir}/{imageid}_annotated.jpg")
                fanhuiimgidls.append(imageid)
                detls.append(detections)
        if len(detls) == 0:
            return [], {}, None

        image_id_gpt_select_output_dir = os.path.join(
            xuanrandir, "pats_bboxchuli", "grid_image"
        )
        mmengine.mkdir_or_exist(image_id_gpt_select_output_dir)
        images = [Image.open(img_path) for img_path in annoimagepathls]

        base64Frames = self.stitch_and_encode_images_to_base64(
            images, fanhuiimgidls, intermediate_output_dir=image_id_gpt_select_output_dir
        )
        # convert the annotated image to base64
        cankaodakai = Image.open(cankao_path)
        annotated_image_base64 = encode_PIL_image_to_base64(
            resize_image(cankaodakai)
        )
        # 这里我们需要处理的只有一个图像
        # 欧，我想起来了，一个是框，一个是id。
        # 我们可以用format去搞

        multi_prompt = apixuanranpats_prompt1.format(
            classtarget=pred_target_class,
            daixuan=len(annoimagepathls),
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
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{annotated_image_base64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        )

        # call VLM to get the result
        cost = 0
        retry = 1
        while retry <= 3:
            try:
                bbox_index = -1
                gpt_message = None

                gpt_response = self.openaigpt.safe_chat_complete(
                    messages, response_format={"type": "json_object"}, content_only=True
                )

                cost += gpt_response["cost"]
                print(gpt_response["content"])
                gpt_content_json = json.loads(gpt_response["content"])
                gpt_message = {
                    "role": "assistant",
                    "content": [{"text": gpt_response["content"], "type": "text"}],
                }

                bbox_index = gpt_content_json.get("object_id_dict", {})
            except:
                retry += 1
                continue

            messages.append(gpt_message)
            # save gpt_content_json
            if bbox_index is None:
                # invalid bbox index
                retry += 1
                continue
            elif bbox_index is {}:
                retry += 1
                continue
            else:
                break
        if retry > 2:
            return [], {}, None
        flag = False
        for key, value in bbox_index.items():
            if value != -1:
                flag = True
        if not flag:
            return [], {}, None
        else:
            return fanhuiimgidls, bbox_index, detls

    def get_det_bbox3d_from_image_idmulti(
            self,
            messages_his,
            godxuan,
            xuanranid,
            refer,
            xuanranwenjianjia,
            scene_id,
            image_id,
            pred_target_class,
            pred_detection,
            gt_object_id,
            query,
            matched_image_ids,  # supposed to have all valid images and not an empty list
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
        refer1 = refer[:1]
        refer = refer1
        if type(image_id) == str:
            return None
        zhuanhuan = "frame-" + f"{image_id:06d}" + ".jpg"

        img_path = os.path.join(xuanranwenjianjia, zhuanhuan)

        sam_mask_output_dir = os.path.join(
            intermediate_output_dir,
            f"baseyesv2sam_mask_final_{pred_target_class.replace('/', 'or')}_target_{gt_object_id}",
        )
        # 先存好这张图片的cankao形式，为后面的detection做准备

        bbox_2d = pred_detection.xyxy

        mmengine.mkdir_or_exist(sam_mask_output_dir)

        # annotate the bbox on the ori image
        print(img_path)
        ori_image = Image.open(img_path)
        annotated_image = self.default_bbox_annotator.annotate(
            scene=ori_image, detections=pred_detection
        )
        annotated_image = self.default_label_annotator.annotate(
            scene=annotated_image, detections=pred_detection
        )
        # save the image
        annotated_image.save(f"{sam_mask_output_dir}/cankao_bbox.jpg")
        cankao_path = f"{sam_mask_output_dir}/cankao_bbox.jpg"
        jiancemodel = self.detection_model
        imgls, bboxzhishi, detls = self.apixuanranpats(messages_his, xuanranid, xuanranwenjianjia, cankao_path, refer,
                                                       pred_target_class, apixuanranpats_prompt1, jiancemodel, godxuan)

        if detls is None:
            ensemble_points = []
            idimg = image_id
            iddet = pred_detection
            t = self.get_det_bbox3d_from_image_idformulti(
                xuanranwenjianjia,
                scene_id,
                idimg,
                pred_target_class,
                iddet,
                gt_object_id=gt_object_id,
                query=query,
                matched_image_ids=matched_image_ids,
                intermediate_output_dir=intermediate_output_dir, )
            ensemble_points.append(t)
            aligned_points_3d = np.concatenate(ensemble_points, axis=0)
            aligned_points_3d = uniform_downsampling(aligned_points_3d, 0.1)

            with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
                if self.point_filter_type == "statistical":
                    aligned_points_3d_filtered = remove_statistical_outliers(
                        aligned_points_3d,
                        nb_neighbors=self.point_filter_nb,
                        std_ratio=self.point_filter_std,
                    )
                elif self.point_filter_type == "truncated":
                    aligned_points_3d_filtered = remove_truncated_outliers(
                        aligned_points_3d,
                        tx=self.point_filter_tx,
                        ty=self.point_filter_ty,
                        tz=self.point_filter_tz,
                    )
                elif self.point_filter_type == "none":
                    aligned_points_3d_filtered = aligned_points_3d
                else:
                    raise NotImplementedError(
                        f"Point filter type {self.point_filter_type} is not implemented."
                    )

            # save the aligned points 3d
            aligned_points_3d_output_dir = os.path.join(
                intermediate_output_dir, "projected_points"
            )
            mmengine.mkdir_or_exist(aligned_points_3d_output_dir)
            np.save(
                f"{aligned_points_3d_output_dir}/ensemble{self.ensemble_num}_matching.npy",
                aligned_points_3d,
            )
            np.save(
                f"{aligned_points_3d_output_dir}/ensemble{self.ensemble_num}__matching_filtered.npy",
                aligned_points_3d_filtered,
            )

            # if nan in points, return None
            if (
                    aligned_points_3d.shape[0] == 0
                    or aligned_points_3d_filtered.shape[0] == 0
                    or np.isnan(aligned_points_3d).any()
                    or np.isnan(aligned_points_3d_filtered).any()
            ):
                print(
                    f"[Box projection] Aligned_points_3d (filtered) is empty or has nan for {scene_id}: {query}. The VLM predicted image id is {image_id}.\
                                matched_image_ids: {matched_image_ids}. pred_target_class: {pred_target_class}."
                )
                return None

            pred_bbox = calculate_aabb(aligned_points_3d_filtered)

            return pred_bbox


        # 这一下的都不用改，我们需要改的就是上面怎么获得3dbbox的
        # 而且是align的
        else:
            ensemble_points = []
            idimg = image_id
            iddet = pred_detection
            t = self.get_det_bbox3d_from_image_idformulti(
                xuanranwenjianjia,
                scene_id,
                idimg,
                pred_target_class,
                iddet,
                gt_object_id=gt_object_id,
                query=query,
                matched_image_ids=matched_image_ids,
                intermediate_output_dir=intermediate_output_dir, )
            ensemble_points.append(t)
            # 参考的也存一下
            for i in range(len(imgls)):
                if bboxzhishi == -1:
                    continue
                idimg = imgls[i]
                zhishi = bboxzhishi[str(idimg)]
                iddet = detls[i][zhishi]
                t = self.get_det_bbox3d_from_image_idformulti(
                    xuanranwenjianjia,
                    scene_id,
                    idimg,
                    pred_target_class,
                    iddet,
                    gt_object_id=gt_object_id,
                    query=query,
                    matched_image_ids=matched_image_ids,
                    intermediate_output_dir=intermediate_output_dir, )
                ensemble_points.append(t)
            aligned_points_3d = np.concatenate(ensemble_points, axis=0)
            aligned_points_3d = uniform_downsampling(aligned_points_3d, 0.1)

            with TimeCounter(tag="RemoveStatisticalOutliers", log_interval=60):
                if self.point_filter_type == "statistical":
                    aligned_points_3d_filtered = remove_statistical_outliers(
                        aligned_points_3d,
                        nb_neighbors=self.point_filter_nb,
                        std_ratio=self.point_filter_std,
                    )
                elif self.point_filter_type == "truncated":
                    aligned_points_3d_filtered = remove_truncated_outliers(
                        aligned_points_3d,
                        tx=self.point_filter_tx,
                        ty=self.point_filter_ty,
                        tz=self.point_filter_tz,
                    )
                elif self.point_filter_type == "none":
                    aligned_points_3d_filtered = aligned_points_3d
                else:
                    raise NotImplementedError(
                        f"Point filter type {self.point_filter_type} is not implemented."
                    )

            # save the aligned points 3d
            aligned_points_3d_output_dir = os.path.join(
                intermediate_output_dir, "projected_points"
            )
            mmengine.mkdir_or_exist(aligned_points_3d_output_dir)
            np.save(
                f"{aligned_points_3d_output_dir}/ensemble{self.ensemble_num}_matching.npy",
                aligned_points_3d,
            )
            np.save(
                f"{aligned_points_3d_output_dir}/ensemble{self.ensemble_num}__matching_filtered.npy",
                aligned_points_3d_filtered,
            )

            # if nan in points, return None
            if (
                    aligned_points_3d.shape[0] == 0
                    or aligned_points_3d_filtered.shape[0] == 0
                    or np.isnan(aligned_points_3d).any()
                    or np.isnan(aligned_points_3d_filtered).any()
            ):
                print(
                    f"[Box projection] Aligned_points_3d (filtered) is empty or has nan for {scene_id}: {query}. The VLM predicted image id is {image_id}.\
                    matched_image_ids: {matched_image_ids}. pred_target_class: {pred_target_class}."
                )
                return None

            pred_bbox = calculate_aabb(aligned_points_3d_filtered)

            return pred_bbox

    def init_default_eval_result_dict(self, vg_row):
        """
        Initializes and returns a default evaluation result dictionary.

        Args:
            vg_row (dict): A dictionary containing the necessary information for evaluation.

        Returns:
            dict: A dictionary with default evaluation results.

        """
        is_unique_scanrefer = vg_row.get("is_unique_scanrefer", False)
        is_multi_scanrefer = not is_unique_scanrefer
        scannet18_class = vg_row.get("scannet18_class", None)

        is_unique_category = vg_row.get("is_unique_category", False)
        is_multi_category = not is_unique_category
        category = vg_row.get("category", None)

        is_easy_referit3d = vg_row.get("is_easy_referit3d", False)
        is_hard_referit3d = not is_easy_referit3d
        distractor_number = vg_row.get("distractor_number", None)

        is_vd_referit3d = vg_row.get("is_vd_referit3d", False)
        is_vid_referit3d = not is_vd_referit3d
        vd_referit3d = vg_row.get("vd_referit3d", None)

        default_eval_result = {
            "iou3d": -1,
            "acc_iou_25": False,
            "acc_iou_50": False,
            "is_unique_scanrefer": is_unique_scanrefer,
            "is_multi_scanrefer": is_multi_scanrefer,
            "scannet18_class": scannet18_class,
            "is_unique_category": is_unique_category,
            "is_multi_category": is_multi_category,
            "category": category,
            "is_easy_referit3d": is_easy_referit3d,
            "is_hard_referit3d": is_hard_referit3d,
            "distractor_number": distractor_number,
            "is_vd_referit3d": is_vd_referit3d,
            "is_vid_referit3d": is_vid_referit3d,
            "vd_referit3d": vd_referit3d,
            "gpt_pred_bbox": None,
        }

        return default_eval_result

    def get_bbox_index_constant(self):
        return 0

    def get_gpt_select_bbox_index(
            self, messages, scene_id, image_id, query, detections, intermediate_output_dir
    ):
        """
        Use VLM to choose which bounding box is the target object in one image.

        Parameters:
            messages (list): A list of messages exchanged between the user and the assistant.
            scene_id (int): The ID of the scene.
            image_id (int): The ID of the image.
            query (str): The query string.
            detections (list): A list of bounding box detections.
            intermediate_output_dir (str): The directory to store intermediate output.

        Returns:
            tuple: A tuple containing the following elements:
                - bbox_index (int): The index of the selected bounding box.
                - message_his (list): The history of messages exchanged between the user and the assistant.
                - cost (int): The cost of the VLM operation.
                - retry (int): The number of retries performed.

        Raises:
            NotImplementedError: If the prompt version is greater than 3.

        """
        num_candidate_bboxes = len(detections)
        if self.prompt_version <= 2:
            bbox_select_user_prompt = self.bbox_select_user_prompt.format(
                num_candidate_bboxes=num_candidate_bboxes,
                query=query,
                image_id=image_id,
            )
        elif self.prompt_version <= 3:
            bbox_select_user_prompt = self.bbox_select_user_prompt.format(
                num_candidate_bboxes=num_candidate_bboxes
            )
        else:
            raise NotImplementedError

        # Annotate the image with detectionss
        image_path = self.scene_infos.get_image_path(scene_id, image_id)
        assert image_path, "\t[GPTSelectBBoxID] The image path is None when letting VLM to select bbounding index. scene_id: {scene_id}, image_id: {image_id}."

        bbox_index_gpt_select_output_dir = os.path.join(
            intermediate_output_dir, "bbox_index_gpt_select_prompts"
        )
        mmengine.mkdir_or_exist(bbox_index_gpt_select_output_dir)
        print(image_path)
        image = Image.open(image_path)

        labels = [f"ID:{ID}" for ID in range(len(detections))]
        if self.use_bbox_anno_f_gpt_select_id:
            image = self.bbox_annotator.annotate(scene=image, detections=detections)
        annotated_image = self.label_annotator.annotate(
            scene=image, detections=detections, labels=labels
        )
        # save the annotated_image
        annotated_image.save(
            f"{bbox_index_gpt_select_output_dir}/{image_id:05d}_annotated.jpg"
        )
        fanhuipath = f"{bbox_index_gpt_select_output_dir}/{image_id:05d}_annotated.jpg"

        # convert the annotated image to base64
        annotated_image_base64 = encode_PIL_image_to_base64(
            resize_image(annotated_image)
        )

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": bbox_select_user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{annotated_image_base64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        )

        # call VLM to get the result
        cost = 0
        retry = 1
        while retry <= self.gpt_max_retry:
            bbox_index = -1
            gpt_message = None
            try:
                mmengine.dump(
                    filter_image_url(messages),
                    f"{bbox_index_gpt_select_output_dir}/{image_id:05d}_gpt_prompts_retry{retry}.json",
                    indent=2,
                )
                gpt_response = self.openaigpt.safe_chat_complete(
                    messages, response_format={"type": "json_object"}, content_only=True
                )
            except Exception as e:
                print(
                    f"\t[GPTSelectBBoxID] {scene_id}: [{query[:20]}] VLM failed to return a response. Error: {e} Retrying..."
                )
                retry += 1
                continue

            cost += gpt_response["cost"]
            gpt_content_json = json.loads(gpt_response["content"])
            gpt_message = {
                "role": "assistant",
                "content": [{"text": gpt_response["content"], "type": "text"}],
            }
            messages.append(gpt_message)
            # save gpt_content_json
            mmengine.dump(
                filter_image_url(messages),
                f"{bbox_index_gpt_select_output_dir}/{image_id:05d}_gpt_responses_retry{retry}.json",
                indent=2,
            )

            bbox_index = int(gpt_content_json.get("object_id", -1))
            if bbox_index < 0 or bbox_index >= num_candidate_bboxes:
                # invalid bbox index
                retry += 1
                bbox_invalid_prompt = f"""Your selected bounding box ID: {bbox_index} is invalid, it should start from 0 and there are only {num_candidate_bboxes} candidate objects in the image. Now try again to select one. Remember to reply using JSON format with the required keys."""

                print(
                    f"\t[VLMSelectBBoxID] {scene_id}: [{query[:20]}] VLM gives an invalid bbox ID {bbox_index}. Append the response and retrying..."
                )
                messages.append({"role": "user", "content": bbox_invalid_prompt})
                print(
                    f"\t[VLMSelectBBoxID] {scene_id}: [{query[:20]}] Here are the messages."
                )
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
        return fanhuipath, bbox_index, message_his, cost, retry

    def get_gpt_pred_image_idyes(self, scene_id, query, messages, intermediate_output_dir):

        image_id_gpt_select_output_dir = os.path.join(
            intermediate_output_dir, "image_index_gpt_select_prompts"
        )
        mmengine.mkdir_or_exist(image_id_gpt_select_output_dir)

        cost = 0
        retry = 1
        pred_image_id = -1
        gpt_content_json = ""
        while retry <= 3:
            ###### * Get pred_image_id use VLM
            try:
                # save the prompt
                mmengine.dump(
                    filter_image_url(messages),
                    f"{image_id_gpt_select_output_dir}/gpt_prompts_retry{retry}.json",
                    indent=2,
                )
                # call VLM api
                with TimeCounter(tag="pred_image_id VLM Time", log_interval=10):
                    gpt_response = self.openaigpt.safe_chat_complete(
                        messages,
                        response_format={"type": "json_object"},
                        content_only=True,
                    )
                    gpt_content = gpt_response["content"]
                    print(gpt_content)
                    gpt_content_json = json.loads(gpt_content)

            except Exception as e:
                print(
                    f"[VLMPredImageID] {scene_id}: [{query[:20]}] VLM failed to return a response. Error: {e} Retrying..."
                )
                retry += 1
                continue

            # Extract and process the relevant information from the response
            cost += gpt_response["cost"]
            pred_image_id = "00045"
            refer_ids = gpt_content_json.get("selected_image_ids", [])
            isolated_id = gpt_content_json.get("unique_question", -1)

            # save the response
            gpt_message = {
                "role": "assistant",
                "content": [{"text": gpt_content, "type": "text"}],
            }
            mmengine.dump(
                filter_image_url(messages + [gpt_message]),
                f"{image_id_gpt_select_output_dir}/gpt_responses_retry{retry}.json",
                indent=2,
            )

            messages.append(gpt_message)
            pred_image_id = int(pred_image_id)  # str -> int, get a valid image ID
            break
            ###### * End of get pred_image_id using VLM

        if isinstance(messages[-1], dict) and messages[-1]["role"] == "user":
            # means the process is not complete, whether VLM produces no response or VLM provides invalid image id
            message_his = None
        else:
            # means VLM gives a valid image id
            message_his = messages
        print("能正常返回么")
        return refer_ids, pred_image_id, gpt_content_json, message_his, cost, retry, isolated_id

    def get_gpt_pred_image_id(self, scene_id, query, messages, intermediate_output_dir):
        """
        Use VLM to select the image containing the target object.

        Args:
            scene_id (str): The ID of the scene.
            query (str): The query string.
            messages (list): A list of messages exchanged between the user and the assistant.
            intermediate_output_dir (str): The directory to store intermediate output.

        Returns:
            tuple: A tuple containing the following elements:
                - pred_image_id (int): The predicted image ID.
                - gpt_content_json (dict): The JSON content of the GPT response.
                - message_his (list): The updated list of messages exchanged between the user and the assistant.
                - cost (int): The total cost of the GPT responses.
                - retry (int): The number of retries made.

        Raises:
            Exception: If VLM fails to return a response.

        """
        image_id_gpt_select_output_dir = os.path.join(
            intermediate_output_dir, "image_index_gpt_select_prompts"
        )
        mmengine.mkdir_or_exist(image_id_gpt_select_output_dir)

        cost = 0
        retry = 1
        pred_image_id = -1
        gpt_content_json = ""
        while retry <= self.gpt_max_retry:
            ###### * Get pred_image_id use VLM
            try:
                # save the prompt
                mmengine.dump(
                    filter_image_url(messages),
                    f"{image_id_gpt_select_output_dir}/gpt_prompts_retry{retry}.json",
                    indent=2,
                )
                # call VLM api
                with TimeCounter(tag="pred_image_id VLM Time", log_interval=10):
                    gpt_response = self.openaigpt.safe_chat_complete(
                        messages,
                        response_format={"type": "json_object"},
                        content_only=True,
                    )
            except Exception as e:
                print(
                    f"[VLMPredImageID] {scene_id}: [{query[:20]}] VLM failed to return a response. Error: {e} Retrying..."
                )
                retry += 1
                continue

            # Extract and process the relevant information from the response
            cost += gpt_response["cost"]
            gpt_content = gpt_response["content"]
            gpt_content_json = json.loads(gpt_content)
            pred_image_id = gpt_content_json.get("target_image_id", -1)
            refer_ids = gpt_content_json.get("reference_image_ids", [])

            # save the response
            gpt_message = {
                "role": "assistant",
                "content": [{"text": gpt_content, "type": "text"}],
            }
            mmengine.dump(
                filter_image_url(messages + [gpt_message]),
                f"{image_id_gpt_select_output_dir}/gpt_responses_retry{retry}.json",
                indent=2,
            )

            image_path = self.scene_infos.get_image_path(scene_id, pred_image_id)
            if image_path is None:
                # image ID invalid
                retry += 1
                print(
                    f"[VLMPredImageID] {scene_id}: [{query[:20]}] VLM failed to find a valid image ID. Pred image id is {pred_image_id}. Retrying..."
                )

                # append the gpt response
                if self.image_id_invalid_prompt is not None:
                    messages.append(gpt_message)
                    image_id_invalid_prompt = self.image_id_invalid_prompt.format(
                        image_id=pred_image_id
                    )
                    messages.append(
                        {"role": "user", "content": image_id_invalid_prompt}
                    )
                continue

            messages.append(gpt_message)
            pred_image_id = int(pred_image_id)  # str -> int, get a valid image ID
            break
            ###### * End of get pred_image_id using VLM

        if isinstance(messages[-1], dict) and messages[-1]["role"] == "user":
            # means the process is not complete, whether VLM produces no response or VLM provides invalid image id
            message_his = None
        else:
            # means VLM gives a valid image id
            message_his = messages
        return refer_ids, pred_image_id, gpt_content_json, message_his, cost, retry

    def juedingcankao(self, cankaoidls, yesscene_id, targetobject, system_prompt):
        # 应该取出最有可能的目标物体
        to_image_path1 = ["../hm3rscandata/scannet/ceshioutputscan/" + \
                          self.scene_infos.infos[yesscene_id]["images_info"][xinid.split(".")[0]]['image_path'] for
                          xinid in cankaoidls]
        images = [Image.open(img_path) for img_path in to_image_path1]
        cuncunfile = os.path.join(self.dangqianoutdir, "juedingcankao")
        base64Frames = self.stitch_and_encode_images_to_base64(
            images, cankaoidls, intermediate_output_dir=cuncunfile
        )
        gpt_input = cankao_prompt1.format(
            num_view_selections=len(to_image_path1),
            targetclass=targetobject,
        )
        begin_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_input},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                ],
            },
        ]
        gpt_response = self.openaigpt.safe_chat_complete(
            begin_messages, response_format={"type": "json_object"}, content_only=True
        )

        print("结束进行gpt测试")
        cost = gpt_response["cost"]
        gpt_content = gpt_response["content"]
        print(gpt_content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        gpt_content_json = json.loads(gpt_content)
        idc = gpt_content_json.get("selected_image_id", cankaoidls[0])
        return idc, cost

    # 为了解决没有检测到锚点物体的这个问题
    def juedingmaodian(self, cankaoidls, yesscene_id, targetobject, system_prompt):
        to_image_path1 = ["../hm3rscandata/scannet/ceshioutputscan/" + \
                          self.scene_infos.infos[yesscene_id]["images_info"][xinid.split(".")[0]]['image_path'] for
                          xinid in cankaoidls]
        images = [Image.open(img_path) for img_path in to_image_path1]
        cuncunfile = os.path.join(self.dangqianoutdir, "juedingmaodian")
        base64Frames = self.stitch_and_encode_images_to_base64(
            images, cankaoidls, intermediate_output_dir=cuncunfile
        )
        gpt_input = maodian_prompt1.format(
            num_view_selections=len(to_image_path1),
            targetclass=targetobject,
        )
        begin_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_input},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                ],
            },
        ]
        gpt_response = self.openaigpt.safe_chat_complete(
            begin_messages, response_format={"type": "json_object"}, content_only=True
        )

        print("结束进行gpt测试")
        cost = gpt_response["cost"]
        gpt_content = gpt_response["content"]
        print(gpt_content)
        # 我们一次性把box啥的也给做了，cost我等会单独写一个模块去搞。
        gpt_content_json = json.loads(gpt_content)
        idc = gpt_content_json.get("selected_image_id", cankaoidls[0])
        return idc, cost

    def centerzhijiezhaotargetupdownjianshao(self, tovg_row, referposes, godxuanran):
        ###target这种都是优先看到mao点物体
        ###okk，所以之前的pose是存的已经变成to的了，所以可以直接用
        ###所以这里不需要找taregt
        ####所以这里我们需要提供一些pose
        zongpose = []
        ####注意了
        # 第一步把alignfuzhu拿出来
        ###以及确定一下那个pose是不是今天的了
        toscan_id = tovg_row["scan_id"]
        alignfuzhu = self.scene_infos.infos[toscan_id]["axis_align_matrix"]
        # 我们batch的时候存的就是求过逆的所以，欧我明白了这步是对的，因为这个才是真正横平数直的那个
        alignfuzhu = np.linalg.inv(alignfuzhu)
        align_pro = chulialign(alignfuzhu)
        if referposes is None:
            qishipose = align_pro
        else:
            #qishipose = update_y_kaojin(referposes, align_pro)
            qishipose = align_pro
        # 我们在每种类型下都增加一个俯视的轨迹
        ###我们需要弄一个topose的存储的东西
        qishipose = self.move_along_local_axis(qishipose, delta_z=-0.35)

        t = -1
        w = -1
        for y_ang in [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342]:
            dangqianpose = self.rotate_local_axis(qishipose, 'y', y_ang)
            zongpose.append(self.rotate_local_axis(dangqianpose, 'x', w * 25))

        raomaodian = os.path.join(self.dangqianoutdir, "updowndexuanrancankao")
        mmengine.mkdir_or_exist(raomaodian)

        ###这里我们的prompt要改一下，改成最先看到的，以及尝试再12id之前分析

        # 下面这几个是俯视图
        id = 0
        imagels = []
        idls = []
        for id in range(len(zongpose)):
            xuanranimagefile, xuanrandepthfile, xuanraninvimagefile, fanhuipose, costpose = self.cunquxuanrantupianforto(
                id, raomaodian, self.godxuanran, zongpose[id])
            if self.calculate_brightness_opencv(xuanraninvimagefile) < 200:
                imagels.append(xuanraninvimagefile)
                idls.append(id)
            id += 1

        # 之后直接进行v1的query寻找
        ###我们想要它返回图像的id，以及bbox，这个是一起的，我们尝试用findf的东西来处理
        ###这里的id只会影响名字，所以我们使用
        ###我们之后修改一下prompt，因为不是每张都有可能有目标物体的

        ###我们需要改一下，因为它直接去找目标物体了

        ###我们要按照新修的v1去改输出，这弄完就没什么事情了，改一下文件输入输出就好
        ###对，存xuanran上面已经做了，我们只需要做下面的
        target_class = tovg_row["instance_type"]
        mao_class = tovg_row["anchors_types"]
        mao_class = ast.literal_eval(mao_class)
        query = tovg_row["utterance"]
        conditions = eval(tovg_row["attributes"]) + eval(tovg_row["conditions"])
        ##这里不是动没动了

        jiancemodel = self.detection_model
        ###用之前的来弄
        base64Frames = self.stitch_and_encode_images_to_base64(
            imagels, idls, intermediate_output_dir=raomaodian
        )
        #
        ###### * End of loading images and creating grid images

        ###### * Format the prompt for VLM

        ####这里注意了，我们需要根据类行来决定具体是用哪个来处理，这里写成代码
        ####hor和betwe的直接原来的，找锚点物体，其余的直接进行query查询这里注意两个函数直接决定，东西还在不在原地了
        input_prompt_dict = {
            "targetclass": target_class,
            "num_view_selections": len(idls),
            "anchorclass": mao_class,
            "query": query,
            "condition": conditions,
        }

        gpt_input = updownmaotar_prompt.format_map(input_prompt_dict)
        system_prompt = self.system_prompt

        print(gpt_input)
        begin_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_input},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                ],
            },
        ]
        messages = begin_messages
        cost = 0
        retry = 1

        # call VLM api
        with TimeCounter(tag="pred_image_id VLM Time", log_interval=10):
            gpt_response = self.openaigpt.safe_chat_complete(
                messages,
                response_format={"type": "json_object"},
                content_only=True,
            )
            gpt_content = gpt_response["content"]
            gpt_content_json = json.loads(gpt_content)
            print(gpt_content_json)

        # Extract and process the relevant information from the response
        cost += gpt_response["cost"]
        gpt_history = {
            "role": "assistant",
            "content": [{"text": gpt_content, "type": "text"}],
        }
        messages.append(gpt_history)
        mishis = messages

        ###注意，这里的话找到没有找到指的是锚点物体，咱们如果连锚点都找不到那么返回-1就好了

        ###找到了，才弄下一步
        mao_id = gpt_content_json.get("anchor_image_id", -1)
        target_id = gpt_content_json.get("target_image_id", -1)
        desp = gpt_content_json.get("extended_description", None)
        ###我们不需要复杂的只需要这种desp的就可以了
        ###这里的下一步是找bbox了，我们先把这个给写了
        # 这里应该先判断动了没有
        ###这个就没有动没动有动这一说了
        ###target都是对的，那么肯定maodin是在的

        if target_id != -1 and target_id <= mao_id:
            ###直接去测试bbox
            ###我们下一步就是把这里给写好
            ####这里我们标注和参考呢单独写好
            nowimage = "frame-" + f"{int(target_id):06d}" + ".jpg"
            tarimagepathes = os.path.join(raomaodian, nowimage)

            bbox_idapi, detectionapi = self.apixuanranbbox(mishis, target_id, tarimagepathes,
                                                           apibbox_prompt3,
                                                           raomaodian, mao_class,
                                                           target_class, query, self.detection_model, desp)
            ###这里没什么直接
            dierbuxuanrandir = os.path.join(self.dangqianoutdir, "xuanranforupdown")
            zuihoushuliang = mao_id + 1
            hmposels = zongpose[:zuihoushuliang]
            if detectionapi is None:
                return (-1, hmposels)
            cankaobase = self.biaozhurefer(tarimagepathes, detectionapi, self.zuizhongpatsdir)
            nowpose = "frame-" + f"{int(target_id):06d}" + ".txt"
            tarposepathes = os.path.join(raomaodian, nowpose)
            

            return (cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)

        elif target_id == -1 and mao_id != -1:
            ###返回mao去做findf
            ###来做这个
            ###我们可以返回一个元组，然后再拆
            maoimage = "frame-" + f"{int(mao_id):06d}" + ".jpg"
            maoposes = "frame-" + f"{int(mao_id):06d}" + ".txt"
            imgpath = os.path.join(raomaodian, maoimage)
            posepath = os.path.join(raomaodian, maoposes)

            zuihoushuliang = mao_id + 1
            hmposels = zongpose[:zuihoushuliang]
            return (posepath, imgpath, hmposels)
        elif mao_id == -1:
            ####去找一遍mao顺便找target
            ###这是最后一个了，弄完这个之后就可以考虑调整文件debug了，之后注意了，我们需要最先找到的，跟nearfar不太一样。
            ###但是代码是一样的，关键是这里我们怎么选择开始旋转的点，这样吧我们还是获取这里的pose之后想办法退后几步再进行旋转。
            ###okk，我们来写吧
            # mao_id,以及yes_scanid我们再找对应
            hmposels = zongpose
            return (-1, hmposels)
            ###没啥好说的
            ###注意了，它要保证pose已经是今天了，你想一像

    def centerzhijiezhaotargetforhobet(self, vg_row, tovg_row, referposes, referimagespath):
        ###okk，所以之前的pose是存的已经变成to的了，所以可以直接用
        ###这个函数是总函数所以给posels
        posels = []
        zongpose = []
        ####注意了
        # 第一步把alignfuzhu拿出来
        ###以及确定一下那个pose是不是今天的了
        toscan_id = tovg_row["scan_id"]
        classtarget = tovg_row["instance_type"]
        maodianclasses = tovg_row["anchors_types"]
        zuihouquery = tovg_row["utterance"]
        conditions = eval(tovg_row["attributes"]) + eval(tovg_row["conditions"])

        maodianclasses = ast.literal_eval(maodianclasses)
        alignfuzhu = self.scene_infos.infos[toscan_id]["axis_align_matrix"]
        # 我们batch的时候存的就是求过逆的所以，欧我明白了这步是对的，因为这个才是真正横平数直的那个
        alignfuzhu = np.linalg.inv(alignfuzhu)
        align_pro = chulialign(alignfuzhu)
        ####这里不会重复，因为有个旋转往下的步骤
        if referposes is not None:
            #qishipose = update_y_kaojin(referposes, align_pro)
            qishipose = align_pro
        else:
            qishipose = align_pro
        # 我们在每种类型下都增加一个俯视的轨迹
        ###我们需要弄一个topose的存储的东西

        t = -1
        w = -1
        for y_ang in [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342]:
            dangqianpose = self.rotate_local_axis(qishipose, 'y', y_ang)
            zongpose.append(self.rotate_local_axis(dangqianpose, 'x', w * 25))

        raomaodian = os.path.join(self.dangqianoutdir, "horbetdexuanrancankao")
        mmengine.mkdir_or_exist(raomaodian)

        # 下面这几个是俯视图
        id = 0
        imagels = []
        idls = []
        ###可以尝试存储两份。之后就是为甚么会有大量的白色呢
        imagelsbeiyong = []
        idlsbeiyong = []
        for id in range(len(zongpose)):
            xuanranimagefile, xuanrandepthfile, xuanraninvimagefile, fanhuipose, costpose = self.cunquxuanrantupianforto(
                id, raomaodian, self.godxuanran, zongpose[id])
            if self.calculate_brightness_opencv(xuanraninvimagefile) < 200:
                imagels.append(xuanraninvimagefile)
                idls.append(id)
            if self.calculate_brightness_opencv(xuanraninvimagefile) < 225:
                imagelsbeiyong.append(xuanraninvimagefile)
                idlsbeiyong.append(id)
            posels.append(zongpose[id])
            id += 1
        if len(idls)==0:
            idls = idlsbeiyong
            imagels = imagelsbeiyong
        # 之后直接进行v1的query寻找
        ###我们想要它返回图像的id，以及bbox，这个是一起的，我们尝试用findf的东西来处理
        ###这里的id只会影响名字，所以我们使用
        ###我们之后修改一下prompt，因为不是每张都有可能有目标物体的
        ###这个是nearfar，直接找target_id物体得
        refer, mis, pred_imageidapi, pred_imagefileapi, des, desp = self.apixuanranimage(id, idls,
                                                                                         imagels,
                                                                                         apixuanran_promptdong,
                                                                                         raomaodian,
                                                                                         zuihouquery,
                                                                                         maodianclasses[0],
                                                                                         classtarget,
                                                                                         conditions)
        ###这个是hori所以必须旋转完毕，所以所有的我们都要
        # file因为一开始弄成inv了了。
        if pred_imageidapi == -1:
            return -1, -1, -1, -1, -1, -1, posels, -1
        else:
            bbox_idapi, detectionapi = self.apixuanranbbox(mis, pred_imageidapi, pred_imagefileapi,
                                                           apibbox_prompt3,
                                                           raomaodian, maodianclasses[0],
                                                           classtarget, zuihouquery, self.detection_model, des)
            # bbox_idapi, detectionapi = self.apixuanranchushileibbox(mis, pred_imageidapi, pred_imagefileapi,
            #                                                                                weitiao_prompt5,
            #                                                                               xuanrancuntupiandir,
            #                                                                                classtarget, query, jiancemodel, des)
            if bbox_idapi == -1:
                return -1, -1, -1, -1, -1, -1, posels, -1,
            else:
                outfile = os.path.join(raomaodian, "cankao.txt")
                with open(outfile, 'w', encoding='utf-8') as file:
                    for element in refer:
                        file.write(str(element) + '\n')
                cankaobase = self.biaozhurefer(pred_imagefileapi, detectionapi, self.zuizhongpatsdir)
                posezhuanhuan = "frame-" + f"{int(pred_imageidapi):06d}" + ".txt"
                posefile = os.path.join(raomaodian, posezhuanhuan)

                return cankaobase, pred_imageidapi, pred_imagefileapi, posefile, detectionapi, raomaodian, posels, des

    def zhaomaodianforupdownshunbianquerydi1bu(self, images, matched_image_ids, end_class,
                                               pred_target_class, intermediate_output_dir,
                                               total_cost, yes_scanid, to_scanid, tovg_row,
                                               scene_infos):
        # 我们仍然使用哪个findf里面那个，但是我们简单修改一下，因为我们需要它返回锚点和非锚点，以及找不到目标物体的时候，只返回锚点物体，这个需要改一下。
        ###单独弄，因为我们的prompt要改的
        # 原来的inter呢也可以用吧，嗯
        posels = []
        target_class = tovg_row["instance_type"]
        mao_class = tovg_row["anchors_types"]
        mao_class = ast.literal_eval(mao_class)
        query = tovg_row["utterance"]
        godxuanran = godmodule(to_scanid, yes_scanid)
        conditions = eval(tovg_row["attributes"]) + eval(tovg_row["conditions"])

        dongmeidongdir = os.path.join(self.dangqianoutdir, "dong4", "panduan")

        jiancemodel = self.detection_model
        ###用之前的来弄
        base64Frames = self.stitch_and_encode_images_to_base64(
            images, matched_image_ids, intermediate_output_dir=dongmeidongdir
        )
        #
        ###### * End of loading images and creating grid images

        ###### * Format the prompt for VLM

        ####这里注意了，我们需要根据类行来决定具体是用哪个来处理，这里写成代码
        ####hor和betwe的直接原来的，找锚点物体，其余的直接进行query查询这里注意两个函数直接决定，东西还在不在原地了
        input_prompt_dict = {
            "targetclass": target_class,
            "num_view_selections": len(matched_image_ids),
            "anchorclass": mao_class,
            "query": query,
            "condition": conditions,
        }

        gpt_input = juedingmaoandtar_prompt.format_map(input_prompt_dict)
        system_prompt = self.system_prompt

        print(gpt_input)
        begin_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_input},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                ],
            },
        ]
        messages = begin_messages
        cost = 0

        # call VLM api
        with TimeCounter(tag="pred_image_id VLM Time", log_interval=10):
            gpt_response = self.openaigpt.safe_chat_complete(
                messages,
                response_format={"type": "json_object"},
                content_only=True,
            )
            gpt_content = gpt_response["content"]
            gpt_content_json = json.loads(gpt_content)
            print(gpt_content_json)

        # Extract and process the relevant information from the response
        cost += gpt_response["cost"]
        gpt_history = {
            "role": "assistant",
            "content": [{"text": gpt_content, "type": "text"}],
        }
        messages.append(gpt_history)
        mishis = messages
        targetfind = gpt_content_json.get("find_or_not", False)
        ###这是做了v1，我们来慢慢弄一下
        ###开始找了
        isolated_id = gpt_content_json.get("unique_question", -1)
        if isolated_id!=-1:
            aa = "frame-0" + f"{int(isolated_id):05d}"
            aa = self.scene_infos.infos[yes_scanid]["images_info"][aa]['extrinsic_matrix']
            isolated_pose = huodealignpose(aa, to_scanid, yes_scanid)

        if targetfind:
            ###找到了，才弄下一步
            mao_id = gpt_content_json.get("anchor_image_id", -1)
            target_id = gpt_content_json.get("target_image_id", -1)
            desp = gpt_content_json.get("extended_description", "")
            ###我们不需要复杂的只需要这种desp的就可以了
            ###这里的下一步是找bbox了，我们先把这个给写了
            # 这里应该先判断动了没有
            ###它这里检查detection是对
            maodongmeidong, poses, imagepathes = self.xuanranbijiaodongmeidong(
                tovg_row, yes_scanid, to_scanid,
                mao_id, scene_infos, jiancemodel, mao_class)

            tardongmeidong, tarposes, tarimagepathes = self.xuanranbijiaodongmeidong(
                tovg_row, yes_scanid, to_scanid,
                target_id, scene_infos, jiancemodel, target_class)
            if poses is not None:
                posels.append(poses)
            if poses is not None and tarposes is not None:
                posels.append(tarposes)
            if maodongmeidong is None:
                toposeformao = None
                ###这里给出posels
                zongde = self.centerzhijiezhaotargetupdownjianshao(tovg_row, toposeformao, godxuanran)
                ###前面得弄好了，来弄这个,下面跟这个一样得
                # return (cankaobase, tarimagepathes, tarposepathes, detectionapi, desp)
                # (posepath, imgpath)
                # (-1)
                ###这里的-1是转一圈都没有找到，所以这个就正常返回-1就行了
                if len(zongde) == 2:
                    posels += zongde[1]
                    if isolated_id!=-1:
                        baodi = self.zuihoubaodi(tovg_row, isolated_pose, self.godxuanran)
                        if len(baodi) == 2:
                            posels += baodi[1]
                            return (-1, posels)
                        else:
                            ###(cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)
                            posels += baodi[5]
                            cankaobase = baodi[0]
                            desp = baodi[4]
                            tarimagepathes = baodi[1]
                            tarposes = baodi[2]
                            detectionapi = baodi[3]
                            return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)
                    else:
                        return (-1, posels)

                elif len(zongde) == 3:
                    posels += zongde[2]
                    poses = zongde[0]
                    imagepathes = zongde[1]
                    return ([poses], [imagepathes], posels)
                else:
                    posels += zongde[5]
                    cankaobase = zongde[0]
                    desp = zongde[4]
                    tarimagepathes = zongde[1]
                    tarposes = zongde[2]
                    detectionapi = zongde[3]
                    return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)
            elif tardongmeidong is None:
                if maodongmeidong:

                    return ([poses], [imagepathes], posels)
                else:
                    ###okk，我们来写吧
                    # mao_id,以及yes_scanid我们再找对应
                    maoid_zhunhuan = "frame-0" + f"{int(mao_id):05d}"
                    yesposeformao = self.scene_infos.infos[yes_scanid]["images_info"][maoid_zhunhuan][
                        'extrinsic_matrix']
                    toposeformao = huodealignpose(yesposeformao, to_scanid, yes_scanid)
                    zongde = self.centerzhijiezhaotargetupdownjianshao(tovg_row, toposeformao, godxuanran)
                    ###前面得弄好了，来弄这个,下面跟这个一样得
                    # return (cankaobase, tarimagepathes, tarposepathes, detectionapi, desp)
                    # (posepath, imgpath)
                    # (-1)
                    ###这里的-1是转一圈都没有找到，所以这个就正常返回-1就行了
                    if len(zongde) == 2:
                        posels+=zongde[1]
                        if isolated_id!=-1:
                            baodi = self.zuihoubaodi(tovg_row, isolated_pose, self.godxuanran)
                            if len(baodi) == 2:
                                posels += baodi[1]
                                return (-1, posels)
                            else:
                                ###(cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)
                                posels += baodi[5]
                                cankaobase = baodi[0]
                                desp = baodi[4]
                                tarimagepathes = baodi[1]
                                tarposes = baodi[2]
                                detectionapi = baodi[3]
                                return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)
                        else:
                            return (-1, posels)
                    elif len(zongde) == 3:
                        posels += zongde[2]
                        poses = zongde[0]
                        imagepathes = zongde[1]
                        return ([poses], [imagepathes], posels)
                    else:
                        posels += zongde[5]
                        cankaobase = zongde[0]
                        desp = zongde[4]
                        tarimagepathes = zongde[1]
                        tarposes = zongde[2]
                        detectionapi = zongde[3]
                        return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)



            elif tardongmeidong and maodongmeidong:
                ###直接去测试bbox
                ###我们下一步就是把这里给写好
                ####这里我们标注和参考呢单独写好
                bbox_idapi, detectionapi = self.apixuanranbbox(mishis, target_id, tarimagepathes,
                                                               apibbox_prompt3,
                                                               dongmeidongdir, mao_class,
                                                               target_class, query, self.detection_model, desp)
                ###这里没什么直接
                dierbuxuanrandir = os.path.join(self.dangqianoutdir, "xuanranforupdown")
                if detectionapi is None:
                    return (-1, posels)
                cankaobase = self.biaozhurefer(tarimagepathes, detectionapi, self.zuizhongpatsdir)

                return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)

            elif not tardongmeidong and maodongmeidong:
                ###返回mao去做findf
                ###来做这个
                ###我们可以返回一个元组，然后再拆
                ###这里就是要找到这个mao_id对应图像得pose以及路径
                ###这已经是列表了
                return ([poses], [imagepathes], posels)
            elif not maodongmeidong:
                ####去找一遍mao顺便找target
                ###这是最后一个了，弄完这个之后就可以考虑调整文件debug了，之后注意了，我们需要最先找到的，跟nearfar不太一样。
                ###但是代码是一样的，关键是这里我们怎么选择开始旋转的点，这样吧我们还是获取这里的pose之后想办法退后几步再进行旋转。
                ###okk，我们来写吧
                # mao_id,以及yes_scanid我们再找对应
                maoid_zhunhuan = "frame-0" + f"{int(mao_id):05d}"
                yesposeformao = self.scene_infos.infos[yes_scanid]["images_info"][maoid_zhunhuan][
                    'extrinsic_matrix']
                toposeformao = huodealignpose(yesposeformao, to_scanid, yes_scanid)
                zongde = self.centerzhijiezhaotargetupdownjianshao(tovg_row, toposeformao, godxuanran)
                ###前面得弄好了，来弄这个,下面跟这个一样得
                # return (cankaobase, tarimagepathes, tarposepathes, detectionapi, desp)
                # (posepath, imgpath)
                # (-1)
                ###这里的-1是转一圈都没有找到，所以这个就正常返回-1就行了
                if len(zongde) == 2:
                    posels += zongde[1]
                    if isolated_id!=-1:
                        baodi = self.zuihoubaodi(tovg_row, isolated_pose, self.godxuanran)
                        if len(baodi) == 2:
                            posels += baodi[1]
                            return (-1, posels)
                        else:
                            ###(cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)
                            posels += baodi[5]
                            cankaobase = baodi[0]
                            desp = baodi[4]
                            tarimagepathes = baodi[1]
                            tarposes = baodi[2]
                            detectionapi = baodi[3]
                            return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)
                    else:
                        return (-1, posels)

                elif len(zongde) == 3:
                    posels += zongde[2]
                    poses = zongde[0]
                    imagepathes = zongde[1]
                    return ([poses], [imagepathes], posels)
                else:
                    posels += zongde[5]
                    cankaobase = zongde[0]
                    desp = zongde[4]
                    tarimagepathes = zongde[1]
                    tarposes = zongde[2]
                    detectionapi = zongde[3]
                    return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)

                ###注意了，它要保证pose已经是今天了，你想一像







        else:
            ###前面是target找到了
            ###这里是target没有找到，并且mao点物体都没有
            mao_id = gpt_content_json.get("anchor_image_id", -1)
            if mao_id == -1:
                ###连锚点也没找到
                ###这里我们应该使用之前找锚点的方案，转一圈顺便找物体
                ###尝试一下
                toposeformao = None

                zongde = self.centerzhijiezhaotargetupdownjianshao(tovg_row, toposeformao, godxuanran)
                ###前面得弄好了，来弄这个,下面跟这个一样得
                # return (cankaobase, tarimagepathes, tarposepathes, detectionapi, desp)
                # (posepath, imgpath)
                # (-1)
                ###这里的-1是转一圈都没有找到，所以这个就正常返回-1就行了
                if len(zongde) == 2:
                    posels += zongde[1]
                    if isolated_id!=-1:
                        baodi = self.zuihoubaodi(tovg_row, isolated_pose, self.godxuanran)
                        if len(baodi) == 2:
                            posels += baodi[1]
                            return (-1, posels)
                        else:
                            ###(cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)
                            posels += baodi[5]
                            cankaobase = baodi[0]
                            desp = baodi[4]
                            tarimagepathes = baodi[1]
                            tarposes = baodi[2]
                            detectionapi = baodi[3]
                            return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)
                    else:
                        return (-1, posels)
                elif len(zongde) == 3:
                    posels += zongde[2]
                    poses = zongde[0]
                    imagepathes = zongde[1]
                    return ([poses], [imagepathes], posels)
                else:
                    posels += zongde[5]
                    cankaobase = zongde[0]
                    desp = zongde[4]
                    tarimagepathes = zongde[1]
                    tarposes = zongde[2]
                    detectionapi = zongde[3]
                    return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)

            else:
                ###找到锚点了
                ###步骤是差不多得
                ###找到了，才弄下一步
                mao_id = gpt_content_json.get("anchor_image_id", -1)
                ###我们不需要复杂的只需要这种desp的就可以了
                ###这里的下一步是找bbox了，我们先把这个给写了
                # 这里应该先判断动了没有
                maodongmeidong, poses, imagepathes = self.xuanranbijiaodongmeidong(
                    tovg_row, yes_scanid, to_scanid,
                    mao_id, scene_infos, jiancemodel, mao_class)
                if maodongmeidong is not None:
                    posels.append(poses)
                ###如果mao啥也没有检测到的话,就同样转圈圈
                if maodongmeidong is None:
                    toposeformao = None

                    zongde = self.centerzhijiezhaotargetupdownjianshao(tovg_row, toposeformao, godxuanran)
                    ###前面得弄好了，来弄这个,下面跟这个一样得
                    # return (cankaobase, tarimagepathes, tarposepathes, detectionapi, desp)
                    # (posepath, imgpath)
                    # (-1)
                    ###这里的-1是转一圈都没有找到，所以这个就正常返回-1就行了
                    if len(zongde) == 2:
                        posels += zongde[1]
                        if isolated_id!=-1:
                            baodi = self.zuihoubaodi(tovg_row, isolated_pose, self.godxuanran)
                            if len(baodi) == 2:
                                posels += baodi[1]
                                return (-1, posels)
                            else:
                                ###(cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)
                                posels += baodi[5]
                                cankaobase = baodi[0]
                                desp = baodi[4]
                                tarimagepathes = baodi[1]
                                tarposes = baodi[2]
                                detectionapi = baodi[3]
                                return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)
                        else:
                            return (-1, posels)
                    elif len(zongde) == 3:
                        posels += zongde[2]
                        poses = zongde[0]
                        imagepathes = zongde[1]
                        return ([poses], [imagepathes], posels)
                    else:
                        posels += zongde[5]
                        cankaobase = zongde[0]
                        desp = zongde[4]
                        tarimagepathes = zongde[1]
                        tarposes = zongde[2]
                        detectionapi = zongde[3]
                        return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)

                elif maodongmeidong:
                    ###返回mao去做findf
                    ###来做这个
                    ###我们可以返回一个元组，然后再拆
                    ###这里就是要找到这个mao_id对应图像得pose以及路径
                    ###这已经是列表了
                    return ([poses], [imagepathes], posels)
                elif not maodongmeidong:
                    ####去找一遍mao顺便找target
                    ###这是最后一个了，弄完这个之后就可以考虑调整文件debug了，之后注意了，我们需要最先找到的，跟nearfar不太一样。
                    ###但是代码是一样的，关键是这里我们怎么选择开始旋转的点，这样吧我们还是获取这里的pose之后想办法退后几步再进行旋转。
                    ###okk，我们来写吧
                    # mao_id,以及yes_scanid我们再找对应
                    maoid_zhunhuan = "frame-0" + f"{int(mao_id):05d}"
                    yesposeformao = self.scene_infos.infos[yes_scanid]["images_info"][maoid_zhunhuan][
                        'extrinsic_matrix']
                    toposeformao = huodealignpose(yesposeformao, to_scanid, yes_scanid)
                    zongde = self.centerzhijiezhaotargetupdownjianshao(
                        tovg_row, toposeformao, godxuanran)
                    if len(zongde) == 2:
                        posels += zongde[1]
                        if isolated_id!=-1:
                            baodi = self.zuihoubaodi(tovg_row, isolated_pose, self.godxuanran)
                            if len(baodi) == 2:
                                posels += baodi[1]
                                return (-1, posels)
                            else:
                                ###(cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)
                                posels += baodi[5]
                                cankaobase = baodi[0]
                                desp = baodi[4]
                                tarimagepathes = baodi[1]
                                tarposes = baodi[2]
                                detectionapi = baodi[3]
                                return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)
                        else:
                            return (-1, posels)
                    elif len(zongde) == 3:
                        posels += zongde[2]
                        poses = zongde[0]
                        imagepathes = zongde[1]
                        return ([poses], [imagepathes], posels)
                    else:
                        posels += zongde[5]
                        cankaobase = zongde[0]
                        desp = zongde[4]
                        tarimagepathes = zongde[1]
                        tarposes = zongde[2]
                        detectionapi = zongde[3]
                        return (cankaobase, desp, tarimagepathes, tarposes, detectionapi, posels)

    ###到这里可能就结束了，但是有一个严重的问题是我们要全部存储今天的pose才行
    ###我们主要关注的四findfi，以及dongmeidong给的是不是，以及center是不是，就是这样,center肯定是的，我们看一下那个biji

    def xianzhaomaoforupdownshunbianqueryzong(self, vg_row, tovg_row):
        total_cost = 0
        scene_id = vg_row["scan_id"]
        query = tovg_row["anchors_types"]
        end_class = tovg_row["instance_type"]
        shibushisu = tovg_row["coarse_reference_type"]
        pred_target_class = tovg_row["anchors_types"]
        pred_target_class = ast.literal_eval(pred_target_class)
        join_list = ",".join(pred_target_class)
        self.dangqianoutdir = vg_row["intermediate_output_dir"]
        # 不过这玩意是个列表类型的，我们后面考虑一下怎么处理
        # 目标物体什么的都要变
        conditions = eval(vg_row["attributes"]) + eval(vg_row["conditions"])
        ####我们要求mao和target它都要
        ###只要这里变动一下就可以了。
        if shibushisu == "horizontal" or shibushisu == "between":
            matched_image_idsmao = eval(
                vg_row[f"matched_image_ids_confidence{0.3}mao"]
            )
            matched_image_ids = matched_image_idsmao

        else:
            matched_image_idsmao = eval(
                vg_row[f"matched_image_ids_confidence{0.3}mao"]
            )
            matched_image_idstarget = eval(
                vg_row[f"matched_image_ids_confidence{0.2}"]
            )
            matched_image_ids = matched_image_idsmao + matched_image_idstarget
            matched_image_ids = list(set(matched_image_ids))

        new_list = []
        if len(matched_image_ids) > 160:
            for i in range(0, len(matched_image_ids), 3):
                new_list.append(matched_image_ids[i])
            matched_image_ids = new_list
        elif len(matched_image_ids) > 80:
            for i in range(0, len(matched_image_ids), 2):
                new_list.append(matched_image_ids[i])
            matched_image_ids = new_list
        else:
            matched_image_ids = matched_image_ids
        print("现在macthid有多长,", len(matched_image_ids))

        # ok,以上是参考目标物体用的，之后我们去修改轨迹那里，我们应该是不需要用到轨迹了
        flag = False

        allmaodian = False
        # 以下这一步不用管，我们要求它不能为0
        if len(matched_image_ids) == 0:
            print("是否一开始")
            cankaoids = []
            t = 0
            for key, value in self.scene_infos.infos[scene_id]["images_info"].items():
                if t % 3 == 0:
                    cankaoids.append(key)
                t += 1
            # maodianyes, cost = self.juedingcankao(cankaoids, scene_id, pred_target_class, self.system_prompt)
            maodianyes = "frame-000060.color"
            cost = 0
            total_cost += cost
            if len(maodianyes) != 0:
                allmaodian = True
        scene_infos = mmengine.load(self.scene_info_path)
        intermediate_output_dir = vg_row["intermediate_output_dir"]
        # 我们最后还是需要用det文件去取得
        ###### * Load images and create grid images
        # views_pre_selection_paths = huodemaodiandet_paths(self.scene_info_path, matched_image_ids, pred_target_class)
        # 我的建议是直接多个物体，返回的detection两者的并集而不是交集。之后让vlm判断

        #######注意，这上面这步就是把存在锚点物体的图片都搞出来
        # 我们需要单独写一个函数去搞
        # match_id不是所有的么，根据这些id应该直接得到views_selection的path。
        # 这里其实比较好的一点是，它是去取pose，所以这个det是谁都不重要，并且我们其实只需要做一个图片的id判断，这个的代码已经写好。
        # 这里要从昨天开始测试起来。
        # 所以
        yesdet_infosfile = "../hm3rscandata/xuanran/ceshioutputscan/image_instance_detectoryes/yolo_prompt_v2_updated_yesterday250_updated2_relations/chunk30/detection.pkl"
        yesdet_infos = DetInfoHandler(yesdet_infosfile)

        print("检查一下为什么出现f.colory", matched_image_ids)
        if not allmaodian:
            views_pre_selection_paths = [
                yesdet_infos.get_image_path(scene_id, image_id)
                for image_id in matched_image_ids
            ]
        else:
            views_pre_selection_paths = [
                yesdet_infos.get_image_path(scene_id, image_id)
                for image_id in cankaoids
            ]
            matched_image_ids = cankaoids

        # 我们到时候看看是个咋回事就好。
        ww = []
        for item in views_pre_selection_paths:
            if not item.endswith("f.color.jpg"):
                ww.append(item)
        views_pre_selection_paths = ww

        matched_image_ids = [id.split(".")[0].split("-")[1] for id in matched_image_ids]
        indexed_matched_image_ids = list(enumerate(matched_image_ids))
        indexed_matched_image_ids.sort(key=lambda x: x[1])
        sorted_views_pre_selection_paths = [views_pre_selection_paths[i] for i, _ in indexed_matched_image_ids]
        sorted_matched_image_ids = [matched_image_ids[i] for i, _ in indexed_matched_image_ids]
        matched_image_ids = sorted_matched_image_ids
        views_pre_selection_paths = sorted_views_pre_selection_paths
        images = [Image.open(img_path) for img_path in views_pre_selection_paths]
        ###到这里之后排个顺序
        yes_scanid = scene_id
        to_scanid = tovg_row["scan_id"]
        ###这步是在做v1+找锚点别弄混了
        self.dangqianoutdir = intermediate_output_dir
        ###我们把posels放在daifenxi里面
        hmposels = []
        daifenxi = self.zhaomaodianforupdownshunbianquerydi1bu(images, matched_image_ids, end_class,
                                                               pred_target_class, intermediate_output_dir,
                                                               total_cost, yes_scanid, to_scanid, tovg_row,
                                                               scene_infos)

        if len(daifenxi) == 2:
            hmposels += daifenxi[1]
            return -1, -1, -1, -1, -1, -1, hmposels

        elif len(daifenxi) == 3:
            ###这里的化
            hmposels += daifenxi[2]
            ###也就是说daifenxi[1]是-1的path。为甚么会这样的
            cankaoceshi, descrip, descri, hmposels1, meshis, godxuan, xuanranid, hmrefer, jieguoid, jieguofile, jieguoposefile, jieguodetection, nowyespose, cc, action_cost, line_cost, degree_cost, xuanrandir = self.findfitimage(
                tovg_row, yes_scanid, to_scanid, reference_poses1=daifenxi[0], reference_pathes1=daifenxi[1])
            hmposels += hmposels1
            if jieguoid == -1:
                return -1, -1, -1, -1, -1, -1, hmposels
            else:
                return 1, cankaoceshi, descri, jieguofile, jieguoposefile, jieguodetection, hmposels



        else:
            hmposels += daifenxi[5]
            return 1, daifenxi[0], daifenxi[1], daifenxi[2], daifenxi[3], daifenxi[4], hmposels
            # return (cankaobase, desp, tarimagepathes, tarposes, detectionapi)

    def zuihoubaodi(self, tovg_row, referposes, godxuanran):
        ###target这种都是优先看到mao点物体
        ###okk，所以之前的pose是存的已经变成to的了，所以可以直接用
        ###所以这里不需要找taregt
        ####所以这里我们需要提供一些pose
        zongpose = []
        ####注意了
        # 第一步把alignfuzhu拿出来
        ###以及确定一下那个pose是不是今天的了
        toscan_id = tovg_row["scan_id"]
        alignfuzhu = self.scene_infos.infos[toscan_id]["axis_align_matrix"]
        # 我们batch的时候存的就是求过逆的所以，欧我明白了这步是对的，因为这个才是真正横平数直的那个
        alignfuzhu = np.linalg.inv(alignfuzhu)
        align_pro = chulialign(alignfuzhu)
        if referposes is None:
            qishipose = align_pro
        else:
            qishipose = update_y_kaojin(referposes, align_pro)
        # 我们在每种类型下都增加一个俯视的轨迹
        ###我们需要弄一个topose的存储的东西
        qishipose = self.move_along_local_axis(qishipose, delta_z=-0.25)

        t = -1
        w = -1
        for y_ang in [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342]:
            dangqianpose = self.rotate_local_axis(qishipose, 'y', y_ang)
            zongpose.append(self.rotate_local_axis(dangqianpose, 'x', w * 25))

        raomaodian = os.path.join(self.dangqianoutdir, "zuihoubaodi")
        mmengine.mkdir_or_exist(raomaodian)

        ###这里我们的prompt要改一下，改成最先看到的，以及尝试再12id之前分析

        # 下面这几个是俯视图
        id = 0
        imagels = []
        idls = []
        for id in range(len(zongpose)):
            xuanranimagefile, xuanrandepthfile, xuanraninvimagefile, fanhuipose, costpose = self.cunquxuanrantupianforto(
                id, raomaodian, self.godxuanran, zongpose[id])
            if self.calculate_brightness_opencv(xuanraninvimagefile) < 200:
                imagels.append(xuanraninvimagefile)
                idls.append(id)
            id += 1

        # 之后直接进行v1的query寻找
        ###我们想要它返回图像的id，以及bbox，这个是一起的，我们尝试用findf的东西来处理
        ###这里的id只会影响名字，所以我们使用
        ###我们之后修改一下prompt，因为不是每张都有可能有目标物体的

        ###我们需要改一下，因为它直接去找目标物体了

        ###我们要按照新修的v1去改输出，这弄完就没什么事情了，改一下文件输入输出就好
        ###对，存xuanran上面已经做了，我们只需要做下面的
        target_class = tovg_row["instance_type"]
        mao_class = tovg_row["anchors_types"]
        mao_class = ast.literal_eval(mao_class)
        query = tovg_row["utterance"]
        conditions = eval(tovg_row["attributes"]) + eval(tovg_row["conditions"])
        ##这里不是动没动了

        jiancemodel = self.detection_model
        ###用之前的来弄
        base64Frames = self.stitch_and_encode_images_to_base64(
            imagels, idls, intermediate_output_dir=raomaodian
        )
        #
        ###### * End of loading images and creating grid images

        ###### * Format the prompt for VLM

        ####这里注意了，我们需要根据类行来决定具体是用哪个来处理，这里写成代码
        ####hor和betwe的直接原来的，找锚点物体，其余的直接进行query查询这里注意两个函数直接决定，东西还在不在原地了
        input_prompt_dict = {
            "targetclass": target_class,
            "num_view_selections": len(idls),
            "query": query,
        }

        gpt_input = zhijiexuan_prompt.format_map(input_prompt_dict)
        system_prompt = self.system_prompt

        print(gpt_input)
        begin_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_input},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                ],
            },
        ]
        messages = begin_messages
        cost = 0
        retry = 1

        # call VLM api
        with TimeCounter(tag="pred_image_id VLM Time", log_interval=10):
            gpt_response = self.openaigpt.safe_chat_complete(
                messages,
                response_format={"type": "json_object"},
                content_only=True,
            )
            gpt_content = gpt_response["content"]
            gpt_content_json = json.loads(gpt_content)
            print(gpt_content_json)

        # Extract and process the relevant information from the response
        cost += gpt_response["cost"]
        gpt_history = {
            "role": "assistant",
            "content": [{"text": gpt_content, "type": "text"}],
        }
        messages.append(gpt_history)
        mishis = messages

        ###注意，这里的话找到没有找到指的是锚点物体，咱们如果连锚点都找不到那么返回-1就好了

        ###找到了，才弄下一步
        ###还有一点
        v1= gpt_content_json.get("match_query_id", -1)
        v2 = gpt_content_json.get("object_image_id", -1)
        desp = gpt_content_json.get("extended_description", None)
        if v1==-1:
            target_id = v2
        else:
            target_id = v1
        ###我们不需要复杂的只需要这种desp的就可以了
        ###这里的下一步是找bbox了，我们先把这个给写了
        # 这里应该先判断动了没有
        ###这个就没有动没动有动这一说了
        ###target都是对的，那么肯定maodin是在的

        if target_id != -1:
            ###直接去测试bbox
            ###我们下一步就是把这里给写好
            ####这里我们标注和参考呢单独写好
            nowimage = "frame-" + f"{int(target_id):06d}" + ".jpg"
            tarimagepathes = os.path.join(raomaodian, nowimage)

            bbox_idapi, detectionapi = self.apixuanranbbox(mishis, target_id, tarimagepathes,
                                                           apibbox_prompt3,
                                                           raomaodian, mao_class,
                                                           target_class, query, self.detection_model, desp)
            ###这里没什么直接
            dierbuxuanrandir = os.path.join(self.dangqianoutdir, "xuanranforupdown")
            hmposels = zongpose
            if detectionapi is None:
                return (-1, hmposels)
            cankaobase = self.biaozhurefer(tarimagepathes, detectionapi, self.zuizhongpatsdir)
            nowpose = "frame-" + f"{int(target_id):06d}" + ".txt"
            tarposepathes = os.path.join(raomaodian, nowpose)
            

            return (cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)

        else:
            ####去找一遍mao顺便找target
            ###这是最后一个了，弄完这个之后就可以考虑调整文件debug了，之后注意了，我们需要最先找到的，跟nearfar不太一样。
            ###但是代码是一样的，关键是这里我们怎么选择开始旋转的点，这样吧我们还是获取这里的pose之后想办法退后几步再进行旋转。
            ###okk，我们来写吧
            # mao_id,以及yes_scanid我们再找对应
            hmposels = zongpose
            return (-1, hmposels)
            ###没啥好说的
            ###注意了，它要保证pose已经是今天了，你想一像

    def get_image_id_bbox_vanilla(self, vg_row, tovg_row):
        """
        Main process, use VLM to get the target object's bounding box.

        Args:
            vg_row (dict): A dictionary containing the information related to the visual grounding task.

        Returns:
            dict: A dictionary containing the following information:
                - done (bool): Indicates whether the VLM process was successful or not.
                - pred_image_id (int): The predicted image ID.
                - bbox_index (int): The index of the predicted bounding box.
                - gpt_input (str): The input prompt for the VLM model.
                - gpt_content (str): The JSON string containing the content generated by the VLM model.
                - gpt_cost (float): The total cost of the VLM process.
                - pred_detection (dict): The predicted detection information for the target object.
                - other_infos (dict): Additional information about the VLM process, including the number of iterations in the main loop, the number of times the predicted image ID was obtained, and more.
        """
        # Extract scan_id and query from the vg_row

        total_cost = 0
        scene_id = vg_row["scan_id"]
        query = tovg_row["anchors_types"]
        end_class = tovg_row["instance_type"]
        shibushisu = tovg_row["coarse_reference_type"]
        yes_scanid = scene_id
        to_scanid = tovg_row["scan_id"]
        # god_module = Theallknow(self.scene_info_path, yes_id, yes_scanid, to_scanid)
        self.dangqianoutdir = vg_row["intermediate_output_dir"]
        self.godxuanran = godmodule(to_scanid, yes_scanid)
        ####不能一起做而是应该在每个里面直接处理好
        ###我们统一要求jieguoid是一个判断指标，如果为-1说明的确是找不到东西了
        ###之后如果有targetid得话我们给出，如果没有给1，在找到得情况下，以及参考图像要存到一个最终得zuizhongpats文件里，
        self.zuizhongpatsdir = os.path.join(self.dangqianoutdir, "zuizhongpats")
        mmengine.mkdir_or_exist(self.zuizhongpatsdir)
        hmposels = []
        ###我们还是从头来一遍
        if shibushisu == "horizontal" or shibushisu == "between":
            ###不管找没有找到都先把之前的那个pose送回来
            ###这里注意要按照老师讲的弄一下
            referposes, referimagespath, zhoameizhaodao, zhaomaopose, isolated_id = self.xianzhaomaoforhbe(vg_row, tovg_row)
            if isolated_id!=-1:
                aa = "frame-0" + f"{int(isolated_id):05d}"
                yesposeformem = self.scene_infos.infos[yes_scanid]["images_info"][aa]['extrinsic_matrix']
                toposeformem = huodealignpose(yesposeformem, to_scanid, yes_scanid)
            if zhaomaopose is not None:
                hmposels += zhaomaopose
            ###这样把我们返回pose不返回path，只有imgpath有用
            if zhoameizhaodao is None:
                ###也就是说一开始找不到，之后我们绕中心转一圈
                ###okk,处理好了
                cankaobase, pred_imageidapi, pred_imagefileapi, posefile, detectionapi, raomaodian, hmposels1, descri = self.centerzhijiezhaotargetforhobet(
                    vg_row, tovg_row, referposes, referimagespath)
                hmposels += hmposels1
                # 我们过，直接处理updawnm的
                if pred_imageidapi == -1:
                    # VLM Fail in the above process
                    if isolated_id!=-1:
                        baodi = self.zuihoubaodi(tovg_row, toposeformem, self.godxuanran)
                        if len(baodi)==2:
                            hmposels+=baodi[1]
                            # random choose one
                            zuizhongdet = None
                            zuizhongid = "-1"
                            zuizhongimagefile = ""
                            zuizhongposefile = ""
                            zuizhongdir = ""
                            nowyespose = None
                            action_cost = 0
                            line_cost = 0
                            degree_cost = 0
                            posels = hmposels
                            zuizhongcankao = None
                            zuizhongdes = ""
                        else:
                            ###(cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)
                            hmposels += baodi[5]
                            cankaobase = baodi[0]
                            desp = baodi[4]
                            tarimagepathes = baodi[1]
                            tarposes = baodi[2]
                            detectionapi = baodi[3]

                            zuizhongdet = detectionapi
                            zuizhongimagefile = tarimagepathes
                            zuizhongposefile = tarposes
                            zuizhongcankao = cankaobase
                            zuizhongdir = self.zuizhongpatsdir
                            zuizhongdes = desp
                            zuizhongid = 1
                    else:
                        zuizhongdet = None
                        zuizhongid = "-1"
                        zuizhongimagefile = ""
                        zuizhongposefile = ""
                        zuizhongdir = ""
                        nowyespose = None
                        action_cost = 0
                        line_cost = 0
                        degree_cost = 0
                        posels = hmposels
                        zuizhongcankao = None
                        zuizhongdes = ""






                else:

                    zuizhongdet = detectionapi
                    zuizhongimagefile = pred_imagefileapi
                    zuizhongposefile = posefile
                    zuizhongcankao = cankaobase
                    zuizhongdir = self.zuizhongpatsdir
                    zuizhongdes = descri
                    zuizhongid = pred_imageidapi


            else:
                if zhoameizhaodao:
                    cankaoceshi, descrip, descri, hmposels1, meshis, godxuan, xuanranid, hmrefer, jieguoid, jieguofile, jieguoposefile, jieguodetection, nowyespose, cc, action_cost, line_cost, degree_cost, xuanrandir = self.findfitimage(
                        tovg_row, yes_scanid, to_scanid, reference_poses1=referposes, reference_pathes1=referimagespath)
                    # 已经获得好了
                    # 我们过，直接处理updawnm的
                    hmposels += hmposels1
                    if jieguoid == -1:
                        if isolated_id!=-1:
                            baodi = self.zuihoubaodi(tovg_row, toposeformem, self.godxuanran)
                            if len(baodi)==2:
                                hmposels+=baodi[1]
                                # random choose one
                                zuizhongdet = None
                                zuizhongid = "-1"
                                zuizhongimagefile = ""
                                zuizhongposefile = ""
                                zuizhongdir = ""
                                nowyespose = None
                                action_cost = 0
                                line_cost = 0
                                degree_cost = 0
                                posels = hmposels
                                zuizhongcankao = None
                                zuizhongdes = ""
                            else:
                                ###(cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)
                                hmposels += baodi[5]
                                cankaobase = baodi[0]
                                desp = baodi[4]
                                tarimagepathes = baodi[1]
                                tarposes = baodi[2]
                                detectionapi = baodi[3]

                                zuizhongdet = detectionapi
                                zuizhongimagefile = tarimagepathes
                                zuizhongposefile = tarposes
                                zuizhongcankao = cankaobase
                                zuizhongdir = self.zuizhongpatsdir
                                zuizhongdes = desp
                                zuizhongid = 1
                        else:
                            zuizhongdet = None
                            zuizhongid = "-1"
                            zuizhongimagefile = ""
                            zuizhongposefile = ""
                            zuizhongdir = ""
                            nowyespose = None
                            action_cost = 0
                            line_cost = 0
                            degree_cost = 0
                            posels = hmposels
                            zuizhongcankao = None
                            zuizhongdes = ""
                    else:
                        zuizhongdet = jieguodetection
                        zuizhongimagefile = jieguofile
                        zuizhongposefile = jieguoposefile
                        zuizhongcankao = cankaoceshi
                        zuizhongdir = self.zuizhongpatsdir
                        zuizhongdes = descri
                        zuizhongid = jieguoid


                else:
                    # 我们直接根据那个某个中心点转一圈找到目标物体pose以及图像就好。
                    # 其实我们就根据那个genjuch
                    # 传给它的东西都是一样的
                    # 这样吧，省的传来传去了，我们直接弄一个scene_info去用
                    cankaobase, pred_imageidapi, pred_imagefileapi, posefile, detectionapi, raomaodian, hmposels1, descri = self.centerzhijiezhaotargetforhobet(
                        vg_row, tovg_row, referposes, referimagespath)
                    # 我们过，直接处理updawnm的
                    hmposels += hmposels1

                    if pred_imageidapi == -1:
                        # VLM Fail in the above process
                        if isolated_id!=-1:
                            baodi = self.zuihoubaodi(tovg_row, toposeformem, self.godxuanran)
                            if len(baodi) == 2:
                                hmposels += baodi[1]
                                # random choose one
                                zuizhongdet = None
                                zuizhongid = "-1"
                                zuizhongimagefile = ""
                                zuizhongposefile = ""
                                zuizhongdir = ""
                                nowyespose = None
                                action_cost = 0
                                line_cost = 0
                                degree_cost = 0
                                posels = hmposels
                                zuizhongcankao = None
                                zuizhongdes = ""
                            else:
                                ###(cankaobase, tarimagepathes, tarposepathes, detectionapi, desp, hmposels)
                                hmposels += baodi[5]
                                cankaobase = baodi[0]
                                desp = baodi[4]
                                tarimagepathes = baodi[1]
                                tarposes = baodi[2]
                                detectionapi = baodi[3]

                                zuizhongdet = detectionapi
                                zuizhongimagefile = tarimagepathes
                                zuizhongposefile = tarposes
                                zuizhongcankao = cankaobase
                                zuizhongdir = self.zuizhongpatsdir
                                zuizhongdes = desp
                                zuizhongid = 1
                        else:
                            zuizhongdet = None
                            zuizhongid = "-1"
                            zuizhongimagefile = ""
                            zuizhongposefile = ""
                            zuizhongdir = ""
                            nowyespose = None
                            action_cost = 0
                            line_cost = 0
                            degree_cost = 0
                            posels = hmposels
                            zuizhongcankao = None
                            zuizhongdes = ""
                    else:
                        zuizhongdet = detectionapi
                        zuizhongimagefile = pred_imagefileapi
                        zuizhongposefile = posefile
                        zuizhongcankao = cankaobase
                        zuizhongdir = self.zuizhongpatsdir
                        zuizhongdes = descri
                        zuizhongid = pred_imageidapi

        else:
            ###这里是可以用v1了。
            # 这里先判断，不行了再找锚点处理
            # 如果找锚点的过程中顺便query，没定到返回锚点坐标，锚点都没有定到返回-1
            ###这个我们等会慢慢弄，3点必须开始弄那个论文以及ppt。
            ###我们现在要处理的就是updown的posels
            pred_imageidapi, cankaobase, descri, pred_imagefileapi, posefile, detectionapi, hmposels1 = self.xianzhaomaoforupdownshunbianqueryzong(
                vg_row, tovg_row)
            hmposels += hmposels1
            if pred_imageidapi == -1:
                # VLM Fail in the above process

                # random choose one
                zuizhongdet = None
                zuizhongid = "-1"
                zuizhongimagefile = ""
                zuizhongposefile = ""
                zuizhongdir = ""
                nowyespose = None
                action_cost = 0
                line_cost = 0
                degree_cost = 0
                posels = hmposels
                zuizhongcankao = None
                zuizhongdes = ""
            else:
                zuizhongdet = detectionapi
                zuizhongimagefile = pred_imagefileapi
                zuizhongposefile = posefile
                zuizhongcankao = cankaobase
                zuizhongdir = self.zuizhongpatsdir
                zuizhongdes = descri
                zuizhongid = pred_imageidapi

        # 有可能是因为retry次数导致的出来
        # judge if success or not for the VLM process

        ###xuanrandir和id是干啥的呢
        data_dict = {
            "done": True,
            "des": zuizhongdes,
            "cankao": zuizhongcankao,
            "pred_image_id": zuizhongid,
            "pred_image_file": zuizhongimagefile,
            "pred_det": zuizhongdet,
            "posefile": zuizhongposefile,
            "xuanrandir": zuizhongdir,
            "gpt_cost": 0,
            "action_cost": 0,
            "line_cost": 0,
            "degree_cost": 0,
            "now_pose": 0,
            "godxuan": self.godxuanran,
            "posels": hmposels,
        }
        return data_dict

    @TimeCounter(tag="stitch_and_encode_images_to_base64 Time", log_interval=50)
    def stitch_and_encode_images_to_base64(
            self, images, images_id, intermediate_output_dir=None
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
        if self.fix_grid2x4:
            # fix layout
            row = 2
            column = 4
        else:
            # square stitching
            num_views_pre_selection = len(images)
            grid_nums = math.ceil(num_views_pre_selection / self.gpt_max_input_images)
            column = math.ceil(grid_nums ** 0.5)
            row = math.ceil(grid_nums / column)
        grid_layout = (row, column)

        # stitch images
        if self.dynamic_stitching:
            grid_images = dynamic_stitch_images_fix_v2(
                images,
                fix_num=self.gpt_max_input_images,
                ID_array=images_id,
                ID_color="red",
                annotate_id=True,
            )
        else:
            grid_images = stitch_images(
                images,
                grid_dims=grid_layout,
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

    def eval_worker_image_only(self, vg_row1):
        # 我们主要在这里进行操作
        """
        Evaluate the worker for image-only grounding.

        Args:
            vg_row (pandas.Series): The row containing the information for evaluation.

        Returns:
            dict: A dictionary containing the evaluation results.

        """
        i, vg_row = vg_row1
        # 今天的sceneid
        scene_id = vg_row["scan_id"]
        # 这里的query需要改成锚点物体
        # 但是target 物体不用变，因为是一一对应的
        zuihouquery = vg_row["utterance"]
        query = vg_row["anchors_types"]
        shibushisu = vg_row["coarse_reference_type"]
        maomao = vg_row["anchors_types"]
        target_id = vg_row["target_id"]
        pred_target_class = vg_row["pred_target_class"]
        ddfile = pd.read_csv(self.yesvg_file_path)
        # 这里没有关系，因为这里的target最后给的是最终结果
        for jj, jow in ddfile.iterrows():
            if jj == i:
                duiyingyesvgrow = jow
                break

        # 这个并不影响，待一会删掉就行了，我们需要得是

        # duiyingtovgrow = qucotovgrow(vg_row, self.vg_file_path, self.tovg_file_path)
        # target_id = duiyingtovgrow["target_id"]
        yesscene_id = duiyingyesvgrow["scan_id"]

        # 这里是需要的，因为这是针对今天的scan的目标物体所弄得id
        # 不对，不需要，你本身today的信息就不能用，我们需要得是to得vg去做pred_bbox，做pats。
        file_prefix = f"{zuihouquery.replace('/', 'or').replace(' ', '_')[:50]}"
        file_prefix = file_prefix + scene_id + "*" + yesscene_id + "*" + str(target_id)
        intermediate_output_dir = os.path.join(
            self.output_dir, "intermediate_results", scene_id, file_prefix
        )
        duiyingyesvgrow["intermediate_output_dir"] = intermediate_output_dir

        default_result_dict = {
            "done": False,
            "xiufu_id": yesscene_id,
            "scene_id": yesscene_id,
            "query": query,
            "pred_target_class": pred_target_class,
            "gpt_input": None,
            "gpt_content": None,
            "gpt_pred_image_id": None,
            "target_id": target_id,
            "gpt_cost": 0,
            "other_infos": None,
            "eval_result": self.init_default_eval_result_dict(vg_row),
            "cost": 0,
            "action_cost": 0,
            "line_cost": 0,
            "degree_cost": 0,
            "jieguo_pairs": None,
            "now_pose": None
        }
        # 但是你最终是按照今天的测试正确率，所以evalresult给vg_row

        # The input dict for the grounding modules
        data_dict = self.get_image_id_bbox_vanilla(duiyingyesvgrow, vg_row)
        # 之后就去这个函数里面搞事情就行了

        # Extract the relevant information from the data_dict
        pred_image_id = data_dict["pred_image_id"]
        pred_imagefile = data_dict["pred_image_file"]
        pred_detection = data_dict["pred_det"]
        # bbox_index = data_dict["bbox_index"]
        gpt_cost = data_dict["gpt_cost"]
        # pred_detection = data_dict["pred_detection"]
        done = data_dict["done"]
        action_cost = data_dict["action_cost"]
        line_cost = data_dict["line_cost"]
        degree_cost = data_dict["degree_cost"]
        posefile = data_dict["posefile"]

        godxuan = data_dict["godxuan"]

        desc = data_dict["des"]
        cankaotuxiangbase = data_dict["cankao"]
        posels = data_dict["posels"]

        default_result_dict.update(
            {
                "done": done,
                "gpt_pred_image_id": pred_image_id,
                # "bbox_index": bbox_index,
                "gpt_cost": gpt_cost,
                "action_cost": action_cost,
                "line_cost": line_cost,
                "degree_cost": degree_cost,
            }
        )
        # 这里我们的dict已经返回的是正确得了
        # * get the pred 3d bouding box from pred_image_id
        # 该进行测试了，这时候要用vg_row

        to_scene_id = vg_row["scan_id"]

        """xinmatched_image_ids = eval(
            vg_row[f"matched_image_ids_confidence{self.image_det_confidence}"]
        )

        if len(xinmatched_image_ids) == 0:
            print("yolo有问题，直接推出")
            xinmatched_image_ids = [1]
        # filter out those with invalid extrinsics, not possible that all match_image_ids are invalid
        xinmatched_image_ids = [
            match_image_id
            for match_image_id in xinmatched_image_ids
            if self.scene_infos.is_posed_image_valid(vg_row["scan_id"], match_image_id)
        ]
        assert (
                len(xinmatched_image_ids) > 0
        ), f"All matched_image_ids are invalid in . This is very strange!"
        """

        to_scene_id = vg_row["scan_id"]
        # 我们不做别的想法，只是为了尽量满足其它的函数好吧
        # 注意我们要求findf里面当然可以存在原始版本，是这样的我们不是已经改了么，都改成
        # 需要转的了
        baoposefile = posefile
        patsdir = os.path.join(self.zuizhongpatsdir, "weitiao")
        mmengine.mkdir_or_exist(patsdir)
        print(desc)
        # desc描述需要再给一遍，pred_detection需要主函数来给
        idpanduan = int(pred_image_id)
        if idpanduan == -1:
            poselspro = posels
            pred_bbox = None
        ###idpanduan是为了用来判断得，其次就是detetion再其次就是参考base，以及目标物体得pose路径和image路径，以及描述，maomao前面给了不用管就这样
        poselspro, pred_bbox = zhuhanshu(posels, pred_detection, self.forfuncsceneinfo, self.detection_model,
                                         pred_imagefile,
                                         baoposefile, patsdir, to_scene_id, godxuan, pred_target_class, desc,
                                         self.sam_predictor, cankaotuxiangbase, shibushisu, maomao)

        # 所以上面如果它收到的-1，它应该直接返回none就好了
        if pred_bbox is None:
            # 这里返回之前要更新一下
            tt = self.init_default_eval_result_dict(vg_row)

            tt.update({
                "iou3d": 0,
                "acc_iou_25": False,
                "acc_iou_50": False,
                "gpt_pred_bbox": [],
                "gt_bbox": [],
            })
            default_result_dict.update({"eval_result": tt})

            default_result_dict.update({"query": zuihouquery})

            default_result_dict.update({"scene_id": vg_row["scan_id"]})

            default_result_dict.update({"action_cost": action_cost})

            default_result_dict.update({"line_cost": line_cost})

            default_result_dict.update({"degree_cost": degree_cost})

            xuanrandir = self.zuizhongpatsdir


            print("此sample的query+scanid+xiufuid ", zuihouquery, "*", vg_row["scan_id"],  "*", yesscene_id)
            if poselspro is None:
                print("为什么会这样的")
            else:
                print("poselspro不是None")
                cunchuliebiao(poselspro, xuanrandir)

            # 注意，这只是处理了一个而已。

            return default_result_dict

        eval_result = self.evaluate_result_func(pred_bbox, vg_row)

        default_result_dict.update({"eval_result": eval_result})

        default_result_dict.update({"query": zuihouquery})

        default_result_dict.update({"scene_id": vg_row["scan_id"]})  # 只不过是最后又更新回来了

        default_result_dict.update({"action_cost": action_cost})

        default_result_dict.update({"line_cost": line_cost})

        default_result_dict.update({"degree_cost": degree_cost})
        xuanrandir = self.zuizhongpatsdir


        print("此sample的query+scanid+xiufuid ", zuihouquery, "*", vg_row["scan_id"],  "*", yesscene_id)
        if poselspro is None:
            print("为什么会这样的")
        else:
            print("poselspro不是None")
            cunchuliebiao(poselspro, xuanrandir)

        # 注意，这只是处理了一个而已。
        return default_result_dict

    def save_progress(self, results, output_file):
        """
        Save temp result to file.
        """
        mmengine.mkdir_or_exist(self.output_dir)
        output_path = os.path.join(self.output_dir, output_file)
        mmengine.dump(results, output_path, indent=2)

        print(f"Results saved to {output_path}")

    def load_progress(self, temp_output_file):
        """
        Load temp result from file.
        """
        temp_output_path = os.path.join(self.output_dir, temp_output_file)
        if os.path.exists(temp_output_path):
            temp_results = mmengine.load(temp_output_path)
            if "results" in temp_results:
                return temp_results["results"]
            else:
                return temp_results
        return None

    def remove_temp(self, temp_output_file):
        """
        Remove temp result file.
        """
        temp_output_path = os.path.join(self.output_dir, temp_output_file)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            print(f"Removed {temp_output_path}")

    def eval(self):
        """
        Evaluate the visual grounder.

        This method performs the evaluation of the visual grounder by processing tasks and generating results.
        It checks if the results already exist and returns if they do.
        If there are resumed results from a previous evaluation, it resumes from there.
        Otherwise, it starts a new evaluation.
        The evaluation is performed in parallel for multiple tasks using tqdm for progress tracking.
        The progress is periodically saved to a temporary file.
        After the evaluation is complete, the final results are calculated and saved to the output file.
        If the evaluation is interrupted, the progress is saved to the temporary file.

        Returns:
            final_results (list): The final results of the evaluation.
        """
        print(
            f"Start evaluating {self.tovg_file_path}. From yes_v2 version use yesterday information {self.vg_file_path}")
        temp_output_file = f"{self.output_prefix}_temp_results.json"
        output_file = f"{self.output_prefix}_results.json"
        # * if output_file already exists, means done
        if mmengine.exists(os.path.join(self.output_dir, output_file)):
            print(f"Results already exist in {output_file}.")
            return
        resumed_results = self.load_progress(temp_output_file)
        self.eval_worker = self.eval_worker_image_only
        xiufu_FILE = "../hm3rscandata/xuanran/ceshioutputscan/query_analysisyes/yesterdaymi.csv"
        xiufufile = pd.read_csv(xiufu_FILE)

        if resumed_results:
            # assert resumed_results should be a list
            assert isinstance(
                resumed_results, list
            ), "resumed_results should be a list."
            resumed_results = [
                result for result in resumed_results if result["done"]
            ]  # only include done results
            print(
                f"Resuming from saved progress in {temp_output_file}. Already processed {len(resumed_results)} results."
            )
        else:
            print("Starting new evaluation")
            resumed_results = []

        completed_pairs = set(
            (result["scene_id"] + "*" + result["xiufu_id"] + "*" + str(result["target_id"]), result["query"]) for result
            in
            resumed_results
        )
        # 注意vg_file已经变成了今天的vg_file。所有row什么的都变成了今天了
        # Tasks that need to be processed are those not in completed_pairs
        # vg_file还是今天的没问题
        tasks = []
        for i, row in self.vg_file.iterrows():
            for j, jow in xiufufile.iterrows():
                if j == i:
                    if (row["scan_id"] + "*" + jow["scan_id"] + "*" + str(row["target_id"]),
                        row["utterance"]) not in completed_pairs:
                        pair = (i, row)
                        tasks.append(pair)

        """tasks = [
            row
            for _, row in self.vg_file.iterrows()
            if (row["scan_id"], row["utterance"]) not in completed_pairs
        ]"""
        print(f"{len(tasks)} tasks remaining to be processed.")

        random.shuffle(tasks)
        # 注意只要有row，我们可以通过函数导出来，不用担心

        try:  # * we want to save temp results, so cannot use mmengine.track_parallel_progress
            with tqdm(total=len(tasks)) as pbar:
                for task in tasks:
                    result = self.eval_worker(task)
                    resumed_results.append(result)
                    pbar.update()

                    # Periodically save progress
                    if pbar.n % 2 == 0:
                        self.save_progress(resumed_results, temp_output_file)

            final_results, is_complete = self.calculate_final_results(resumed_results)
            if is_complete:
                self.save_progress(final_results, output_file)
                # Remove the temporary file after successful completion
                self.remove_temp(temp_output_file)
            else:
                self.save_progress(final_results, temp_output_file)

            return final_results

        except (Exception, KeyboardInterrupt) as e:
            print(
                f"Error of type: {type(e).__name__}, message: {e} occurred during evaluation."
            )
            traceback.print_exc()
            final_results, _ = self.calculate_final_results(resumed_results)
            self.save_progress(final_results, temp_output_file)
            print(f"Progress is saved.")

    def calculate_final_results(self, results):
        """
        Calculate final accuracy.
        """
        results, is_complete = self.validate_final_results(results)

        accuracy_stat, accuracy_summary = self.accuracy_calculator.compute_accuracy(
            results
        )

        self.accuracy_calculator.print_statistics()

        total_cost = self.get_costs(results)

        return {
            "accuracy_summary": ",".join(accuracy_summary),
            "accuracy_stat": accuracy_stat,
            "total_gpt_cost": total_cost,
            "results": results,
        }, is_complete

    def validate_final_results(self, results):
        """
        Verify results, check if all task is complete, also remove extra results.
        """
        print("Verifying results...")
        # Get all unique scan_id and query pairs from the vg file
        # vg_pairs = set(
        #    self.vg_file[["scan_id", "utterance"]].itertuples(index=False, name=None)
        # )
        vg_pairs = []
        xiufu_FILE = "../hm3rscandata/xuanran/ceshioutputscan/query_analysisyes/yesterdaymi.csv"
        xiufufile = pd.read_csv(xiufu_FILE)
        for i, row in self.vg_file.iterrows():
            for j, jow in xiufufile.iterrows():
                if j == i:
                    vg_pairs.append(
                        (row["scan_id"] + "*" + jow["scan_id"] + "*" + str(row["target_id"]), row["utterance"]))
        vg_pairs = set(vg_pairs)
        # Get all unique scan_id and query pairs from the results
        result_pairs = set(
            (result["scene_id"] + "*" + result["xiufu_id"] + "*" + str(result["target_id"]), result["query"]) for result
            in
            results)

        is_complete = True

        # Check if all pairs in the vg file are in the results
        if vg_pairs != result_pairs:
            missing_pairs = vg_pairs - result_pairs
            extra_pairs = result_pairs - vg_pairs

            # Report missing and extra pairs
            if missing_pairs:
                print(
                    f"Missing pairs: {missing_pairs}. {len(missing_pairs)} pairs are missed."
                )
                is_complete = False
            if extra_pairs:
                print(
                    f"Extra pairs: {extra_pairs}. {len(extra_pairs)} extra pairs will be removed."
                )

            # Remove extra pairs from results
            filtered_results = [
                result
                for result in results
                if (result["scene_id"] + "*" + result["xiufu_id"] + "*" + str(result["target_id"]),
                    result["query"]) not in extra_pairs
            ]

            # Optionally, update results list
            results[:] = filtered_results

            print(
                f"Results have been updated to remove extra pairs. Now it has {len(results)} results."
            )
        else:
            print(
                "Verified Success. All scan_id and query pairs in the vg file were matched."
            )

        # sort the resuts according to "scene_id"
        results = sorted(results, key=lambda x: x["scene_id"])

        return results, is_complete

    def get_costs(self, results):
        """
        Get total VLM cost.
        """
        cost = 0
        for result in results:
            if type(result["gpt_cost"]) == float:
                cost += result["gpt_cost"]

        return cost


def get_csv_row(vg_file, scene_id, query):
    matching_row = vg_file[
        (vg_file["scan_id"] == scene_id) & (vg_file["utterance"] == query)
        ]

    if not matching_row.empty:
        return matching_row.iloc[0]
    else:
        return "No matching row found."


class Tee:
    """
    Show log in terminal and log file.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="outputs/visual_groundingxuanran0324"
    )
    parser.add_argument(
        "--scene_info_path",
        type=str,
        default="../hm3rscandata/xuanran/ceshioutputscan/3rscan_instance_data/scenes_train_val_info_w_images.pkl",
    )  # * if change this, need to disable global cache of tools
    parser.add_argument(
        "--det_info_path",
        type=str,
        default="../hm3rscandata/scannet/ceshioutputscan/image_instance_detectorto/*/chunk*/detection.pkl",
    )
    parser.add_argument(
        "--vg_file_path",
        type=str,
        default="../hm3rscandata/scannet/ceshioutputscan/query_analysisto/prompt_v2_updated_today250_updated2_relations_with_images_selected_diffconf_and_pkl.csv",
    )
    parser.add_argument(
        "--yesvg_file_path",
        type=str,
        default="../hm3rscandata/xuanran/ceshioutputscan/query_analysisyes/yesterdaymi.csv",
    )
    parser.add_argument(
        "--matching_info_path",
        type=str,
        default="../hm3rscandata/scannet/ceshioutputscan/match_result_200/exhaustive_matching_ceshi.pkl",
    )

    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)

    # sam setting
    parser.add_argument("--use_sam_huge", action="store_true")
    parser.add_argument(
        "--evaluate_function",
        type=str,
        default="evaluate_3diou",
        choices=["evaluate_3diou"],
    )
    # openai setting
    parser.add_argument("--openaigpt_type", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--gpt_max_retry", type=int, default=3)
    parser.add_argument("--gpt_max_input_images", type=int, default=6)
    parser.add_argument("--use_all_images", action="store_true",
                        help="Use all images for grounding without view_preselection.")
    parser.add_argument("--ensemble_num", type=int, default=1)  # 1 means no ensemble
    parser.add_argument("--openai_top_p", type=float, default=0.3)
    parser.add_argument("--openai_temperature", type=float, default=0.1)

    # post_processing
    parser.add_argument("--use_bbox_prompt", action="store_true")
    parser.add_argument("--use_point_prompt", action="store_true")
    parser.add_argument("--use_point_prompt_num", type=int, default=1)
    parser.add_argument("--post_process_erosion", action="store_true")
    parser.add_argument("--post_process_dilation", action="store_true")
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--post_process_component", action="store_true")
    parser.add_argument("--post_process_component_num", type=int, default=2)
    parser.add_argument("--point_filter_nb", type=int, default=5)
    parser.add_argument("--point_filter_std", type=float, default=1.0)
    parser.add_argument(
        "--point_filter_type",
        type=str,
        default="statistical",
        choices=["statistical", "truncated", "none"],
    )
    parser.add_argument("--point_filter_tx", type=float, default=0.05)
    parser.add_argument("--point_filter_ty", type=float, default=0.05)
    parser.add_argument("--point_filter_tz", type=float, default=0.05)
    parser.add_argument("--project_color_image", default=True, action="store_true")
    parser.add_argument("--use_bbox_anno_f_gpt_select_id", action="store_true")

    # For ensemble experiments
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--search_ensemble", action="store_true",
                        help="Directly use the grounding intermediate outputs and perform more ensemble projection with different ensemble number.")

    # exp name
    parser.add_argument("--exp_name", default=None, type=str)

    # For get bbox index type
    parser.add_argument("--skip_bbox_selection_when1", action="store_true")
    parser.add_argument("--prompt_version", type=int, default=3)
    parser.add_argument("--image_det_confidence", type=float, default=0.3)
    parser.add_argument("--fix_grid2x4", action="store_true")
    parser.add_argument("--dynamic_stitching", action="store_true")
    parser.add_argument("--online_detector", type=str, default="gdino")
    parser.add_argument("--use_new_detections", action="store_true")
    parser.add_argument(
        "--get_bbox_index_type",
        type=str,
        default="vlm_select",
        choices=["constant", "vlm_select"],
    )
    print(parser)
    args, unknown = parser.parse_known_args()
    #args = parser.parse_args()
    print(args)

    other_configs = {
        "evaluate_function": args.evaluate_function,
        "post_process_erosion": args.post_process_erosion,
        "post_process_dilation": args.post_process_dilation,
        "kernel_size": args.kernel_size,
        "post_process_component": args.post_process_component,
        "post_process_component_num": args.post_process_component_num,
        "point_filter_nb": args.point_filter_nb,
        "point_filter_std": args.point_filter_std,
        "use_sam_huge": args.use_sam_huge,
        "matching_info_path": args.matching_info_path,
        "use_bbox_prompt": args.use_bbox_prompt,
        "use_point_prompt": args.use_point_prompt,
        "point_filter_type": args.point_filter_type,
        "point_filter_tx": args.point_filter_tx,
        "point_filter_ty": args.point_filter_ty,
        "point_filter_tz": args.point_filter_tz,
        "project_color_image": args.project_color_image,
        "use_point_prompt_num": args.use_point_prompt_num,
        "gpt_max_retry": args.gpt_max_retry,
        "gpt_max_input_images": args.gpt_max_input_images,
        "get_bbox_index_type": args.get_bbox_index_type,
        "use_all_images": args.use_all_images,
        "prompt_version": args.prompt_version,
        "use_new_detections": args.use_new_detections,
        "skip_bbox_selection_when1": args.skip_bbox_selection_when1,
        "openai_temperature": args.openai_temperature,
        "openai_top_p": args.openai_top_p,
        "image_det_confidence": args.image_det_confidence,
        "use_bbox_anno_f_gpt_select_id": args.use_bbox_anno_f_gpt_select_id,
        "fix_grid2x4": args.fix_grid2x4,
        "dynamic_stitching": args.dynamic_stitching,
        "online_detector": args.online_detector,
    }

    FROM_SCRATCH = args.from_scratch
    SEARCH_ENSEMBLE = args.search_ensemble

    assert (
            FROM_SCRATCH or SEARCH_ENSEMBLE
    ), "If not from scratch or not searching ensemble, nothing happens."

    assert args.prompt_version == 3, "Prompt version should be larger than 3. Version 1 and 2 are just for ablation study during the development phase."

    if FROM_SCRATCH:
        # * Add the date and time to the output_dir
        # the priority of resume_from > exp_name
        if args.resume_from is not None:
            args.output_dir = args.resume_from
        else:
            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")

            if args.exp_name is not None:
                output_dir = os.path.join(args.output_dir, f"{args.exp_name}")
                args.output_dir = output_dir
            else:
                formatted_now = now.strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )  # Example format: 2024-02-26_15-30-00
                args.output_dir = os.path.join(args.output_dir, formatted_now)

        mmengine.mkdir_or_exist(args.output_dir)

    if SEARCH_ENSEMBLE:
        if not FROM_SCRATCH and args.result_path is not None:
            result_path = args.result_path  # * set this to directly doing ensemble
            args.output_dir = os.path.dirname(result_path)
            # load the params from the config file in args.output_dir
            args_yaml_path = os.path.join(args.output_dir, "args.yaml")
            resumed_args = mmengine.load(args_yaml_path)
            prompt_version = resumed_args["prompt_version"]
            gpt_version = resumed_args["openaigpt_type"]

            vg_file_path = f"outputs/query_analysis/{os.path.basename(result_path).replace(f'_{gpt_version}_promptv{prompt_version}_results.json', '.csv')}"

            args.vg_file_path = vg_file_path
        else:
            # if None, then the result path is calculated from vg_file
            # This usually comes from doing ensemble after from scratch
            result_path = f"{args.output_dir}/{os.path.basename(args.vg_file_path.split('.')[0])}_{args.openaigpt_type}_promptv{args.prompt_version}_results.json"

    # * load the args.yaml to update args
    args_yaml_path = os.path.join(args.output_dir, "args.yaml")
    if mmengine.exists(args_yaml_path):
        print(f"Loading resuming args from {args_yaml_path}.")
        resumed_args = mmengine.load(args_yaml_path)
        # * update args with resumed_args
        for key, value in resumed_args.items():
            # * if different, print
            if key in args and getattr(args, key) != value and key != "resume_from":
                print(f"Updated {key} from {getattr(args, key)} to {value}")
                setattr(args, key, value)

    # Save the parser arguments to the output_dir as a YAML file
    args_yaml_path = os.path.join(args.output_dir, "args.yaml")
    mmengine.dump(vars(args), args_yaml_path)

    # Optionally, you can print the arguments to verify them
    print("Configuration parameters:")
    print(vars(args))

    log_file = open(os.path.join(args.output_dir, "log.txt"), "a")
    sys.stdout = Tee(sys.stdout, log_file)
    print("--------------Starting a new run-----------------")
    # print time, args, and whether is from scratch
    print(f"Time: {datetime.datetime.now()}")
    print(f"Args: {vars(args)}")
    print(f"From Scratch: {FROM_SCRATCH}")

    grounder = VisualGrounder(
        scene_info_path=args.scene_info_path,
        det_info_path=args.det_info_path,
        vg_file_path=args.vg_file_path,
        yesvg_file_path=args.yesvg_file_path,
        output_dir=args.output_dir,
        openaigpt_type=args.openaigpt_type,
        ensemble_num=args.ensemble_num,
        **other_configs,
    )

    if FROM_SCRATCH:
        grounder.eval()  # inferenceing from scratch

    if SEARCH_ENSEMBLE:
        if not mmengine.exists(result_path):
            print(
                f"{result_path} does not exist, which mainly means the from scratch process is not complete."
            )
            exit()

        vg_file = pd.read_csv(args.vg_file_path)
        results = mmengine.load(result_path)["results"]

        ensemble_nums = [1, 2, 3, 4, 5, 6, 7]
        # remove args.ensemble_num in ensemble_nums
        if args.ensemble_num in ensemble_nums:
            ensemble_nums.remove(args.ensemble_num)

        # save the args values
        args_yaml_path = os.path.join(grounder.output_dir, "args.yaml")
        mmengine.dump(vars(args), args_yaml_path)

        for ensemble_num in ensemble_nums:
            new_result_path = result_path.replace(
                ".json", f"_matching_ks{args.kernel_size}_ensemble{ensemble_num}.json"
            )
            # if already exists, skip
            if mmengine.exists(new_result_path):
                print(f"{new_result_path} already exists. Skip.")
                continue

            grounder.ensemble_num = ensemble_num
            print(
                f"Start processing ensemble_num: {ensemble_num}, using tracking: matching."
            )
            for sample in tqdm(results):
                gpt_pred_image_id = sample["gpt_pred_image_id"]
                scene_id = sample["scene_id"]
                query = sample["query"]
                target_id = sample["target_id"]
                gpt_pred_image_id = sample["gpt_pred_image_id"]

                row = get_csv_row(vg_file, scene_id, query)
                pred_target_class = row["pred_target_class"]
                matched_image_ids = eval(
                    row[f"matched_image_ids_confidence{grounder.image_det_confidence}"]
                )
                file_prefix = f"{query.replace('/', 'or').replace(' ', '_')[:60]}"
                intermediate_output_dir = os.path.join(
                    grounder.output_dir, "intermediate_results", scene_id, file_prefix
                )

                # load the sam mask
                sam_mask_output_dir = f"{intermediate_output_dir}/sam_mask_{gpt_pred_image_id:05d}_{pred_target_class.replace('/', 'or')}_target_{target_id}"
                sam_mask_output_path = (
                    f"{sam_mask_output_dir}/anchor_sam_ks{args.kernel_size}.pkl"
                )

                if len(matched_image_ids) == 0:
                    # Use all images
                    num_posed_images = grounder.scene_infos.get_num_posed_images(
                        scene_id
                    )
                    print(
                        f"No matched images for {scene_id}: {query}. Using all {num_posed_images} images."
                    )
                    matched_image_ids = [f"{i:05d}" for i in range(num_posed_images)]

                # filter out those with invalid extrinsics, not possible that all match_image_ids are invalid
                matched_image_ids = [
                    match_image_id
                    for match_image_id in matched_image_ids
                    if grounder.scene_infos.is_posed_image_valid(
                        scene_id, match_image_id
                    )
                ]
                assert (
                        len(matched_image_ids) > 0
                ), f"All matched_image_ids are invalid in {scene_id}. This is very strange!"

                if mmengine.exists(sam_mask_output_path):
                    pred_sam_mask = mmengine.load(sam_mask_output_path)
                elif mmengine.exists(
                        sam_mask_output_path.replace(f"ks{args.kernel_size}.pkl", "raw.pkl")
                ):
                    # read mask and process from raw
                    print("process raw mask with kernel size", args.kernel_size)
                    pred_sam_mask = mmengine.load(
                        sam_mask_output_path.replace(
                            f"ks{args.kernel_size}.pkl", "raw.pkl"
                        )
                    )
                    if grounder.post_process:
                        pred_sam_mask = grounder.post_process_mask(pred_sam_mask)
                        pred_sam_mask.save(
                            f"{sam_mask_output_dir}/anchor_sam_postprocessed.jpg"
                        )
                    mmengine.dump(
                        pred_sam_mask,
                        f"{sam_mask_output_dir}/anchor_sam_ks{args.kernel_size}.pkl",
                    )
                else:
                    print("no raw mask!")
                    pred_sam_mask = None

                if grounder.use_new_detections and pred_sam_mask is None:
                    pred_bbox = None
                else:
                    pred_bbox = grounder.get_det_bbox3d_from_image_id(
                        scene_id=scene_id,
                        image_id=gpt_pred_image_id,
                        pred_target_class=pred_target_class,
                        pred_detection=None,
                        gt_object_id=target_id,
                        query=query,
                        matched_image_ids=matched_image_ids,
                        intermediate_output_dir=intermediate_output_dir,
                        bbox_index=sample.get("bbox_index", 0),
                        pred_sam_mask=pred_sam_mask,
                    )

                if pred_bbox is None:
                    if sample["eval_result"]["gpt_pred_bbox"] is not None:
                        print(
                            f"Ensemble pred_bbox is None. The original pred_bbox is {sample['eval_result']['gpt_pred_bbox']}."
                        )
                    sample["eval_result"]["iou3d"] = -1
                    sample["eval_result"]["acc_iou_25"] = False
                    sample["eval_result"]["acc_iou_50"] = False
                    sample["eval_result"]["gpt_pred_bbox"] = None
                    continue
                else:
                    eval_result = grounder.evaluate_result_func(pred_bbox, row)
                sample["eval_result"] = eval_result

            # save the new results
            final_results, is_complete = grounder.calculate_final_results(results)
            if is_complete:
                # add ensemble_num at the suffix
                mmengine.dump(final_results, new_result_path, indent=2)
            else:
                print("The results are not complete.")

# 这里的话，只要前面findfimage找到东西之后，我们就立即让weitiaobao去处理，直接得到bbox就好
