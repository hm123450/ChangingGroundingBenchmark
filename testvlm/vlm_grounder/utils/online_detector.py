import os
import random
import time

import cv2
import mmengine
import numpy as np
import supervision as sv
from PIL.Image import Image
from requests.exceptions import ProxyError, ReadTimeout
from ultralytics import YOLO



def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 40,
    max_delay: int = 30,
    errors: tuple = (ProxyError, ReadTimeout, RuntimeError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                # * print the error info
                num_retries += 1
                if num_retries > max_retries:
                    print(
                        f"[DETECTOR] Encounter error of type: {type(e).__name__}, message: {e}"
                    )
                    raise Exception(
                        f"[DETECTOR] Maximum number of retries ({max_retries}) exceeded."
                    )

                print(
                    f"[DETECTOR] Retrying after {delay} seconds due to error of type: {type(e).__name__}, message: {e}"
                )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(min(delay, max_delay))
            except Exception as e:
                print(
                    f"[DETECTOR] Unkown error of type: {type(e).__name__}, message: {e}"
                )
                raise e

    return wrapper
#/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/3rdparty/GroundingDINO/weights
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from torchvision.ops import box_convert
import torch
def tidaizhuanhuan(image_source, boxes):
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return xyxy


model = load_model("/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/3rdparty/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/3rdparty/GroundingDINO/weights/groundingdino_swint_ogc.pth")
BOX_TRESHOLD = 0.20
TEXT_TRESHOLD = 0.25



class OnlineDetector:
    def __init__(
        self,
        detection_model="Grounding-DINO-1.5Pro",
        device="cuda:0",
        global_cache_dir="./outputs/global_cache/gdino_cache",
    ):
        self.device = device
        self.global_cache_dir = global_cache_dir
        mmengine.mkdir_or_exist(self.global_cache_dir)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        if "yolo" in detection_model:
            self.detection_model = YOLO(detection_model).to(device)
            self.detect = self.detect_yolo
        elif "tidai" in detection_model:
            self.detection_model = load_model("/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/3rdparty/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/3rdparty/GroundingDINO/weights/groundingdino_swint_ogc.pth")
            self.detect = self.detect_tidai 
     

    def detect_yolo(self, image_path, category):
        """
        Detect use yolo-world model
        """
        self.detection_model.set_classes([category])
        image = cv2.imread(image_path)
        c_det_results = self.detection_model(
            [image], conf=0.01, verbose=False
        )  # * list of B images
        c_sv_result = sv.Detections.from_ultralytics(
            c_det_results[0]
        )  # * use supervsion to annotate and save the image
        return c_sv_result

    def detect_tidai(self, image_path, category):
        """
        Detect use yolo-world model
        """
        if isinstance(category, list):
            categoryls = category
        else:
            categoryls = [category]

        IMAGE_PATH = image_path
        TEXT_PROMPT = '.'.join(categoryls)
        image_source, image = load_image(IMAGE_PATH)

        boxes, logits, phrases = predict(
            model=self.detection_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        boxes = tidaizhuanhuan(image_source, boxes)
        scores = np.array(logits)
        valid_indices = [i for i, phrase in enumerate(phrases) if phrase in categoryls]
        phrasesvalid = [phrases[i] for i in valid_indices]

        categorys = np.array(phrasesvalid)
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        class_id = []
        for item in categorys:
            class_id.append(categoryls.index(item))

        class_id = np.array(class_id)
        if boxes.shape[0] == 0:
            print(f"Tidai Detect nothing for {image_path}.")
            c_sv_result = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty((0)),
                class_id=np.empty((0), dtype=np.int64),
                data={"class_name": np.empty((0), dtype=np.str_)},
            )
        else:
            c_sv_result = sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=class_id,
                data={"class_name": categorys},
            )
        return c_sv_result

    def visualize(self, image_path, sv_result, output_path):
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(
                sv_result.data["class_name"], sv_result.confidence
            )
        ]
        det_image = Image.open(image_path)
        annotated_image = self.bounding_box_annotator.annotate(det_image, sv_result)
        annotated_image = self.label_annotator.annotate(
            annotated_image, sv_result, labels
        )

        # * save this image
        cv2.imwrite(output_path, annotated_image)

    def convert_PILimage_2_mask(self, masks):
        """
        Extract the alpha channel in the PIL.Image object as a mask
        """
        mask_data_list = []
        for mask in masks:
            mask_data = np.array(mask)[np.newaxis, :, :, -1]
            mask_data_list.append(mask_data)
        if len(mask_data_list) > 0:
            res_masks = np.concatenate(mask_data_list, axis=0).astype(bool)
        else:
            res_masks = None
        return res_masks
