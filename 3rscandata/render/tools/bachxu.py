# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_scannet_data.py
"""

import argparse
import os
import pickle
from multiprocessing import Pool
from os import path as osp

import numpy as np
import scannet_utils
import csv
import pandas as pd


def chulimap(file_path):

        # 读取 TSV 文件
    df = pd.read_csv(file_path, sep='\t')



    # 将第二列和第三列转换为字典
    result_dict = dict(zip(df.iloc[:, 1], df.iloc[:, 2]))
    return result_dict
"""def chulimap(filename):
    # 初始化一个空字典来存储Label和NYU40 Mapping的映射关系
    label_to_nyu40_dict = {}

    # 打开CSV文件
    with open(filename, mode='r', encoding='utf-8') as file:
        # 创建CSV读取器
        reader = csv.reader(file)

        # 跳过标题行
        #next(reader)

        # 遍历CSV文件中的每一行
        t=0
        for row in reader:
            t += 1
            if t==1 or t==2:
                continue

            # 检查行是否有足够的列
            print(row)
            print(type(row[2]))
            if len(row) >= 3:
                # 将第二列（Label）和第三列（NYU40 Mapping）添加到字典中
                label_to_nyu40_dict[row[1]] = int(row[2])
    return label_to_nyu40_dict

    # 打印字典以验证结果
    #for label, nyu40 in label_to_nyu40_dict.items():
    #    print(f"{label}: {nyu40}")"""
DONOTCARE_CLASS_IDS = np.array([])
# OBJ_CLASS_IDS = np.array(
# [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
# OBJ_CLASS_IDS = np.array([])

#label是指那个tsv文件
def export(align_file, mesh_file, agg_file, seg_file, meta_file, label_map_file, test_mode=False):
    """Export original files to vert, ins_label, sem_label and bbox file.

    Args:
        mesh_file (str): Path of the mesh_file.
        agg_file (str): Path of the agg_file.
        seg_file (str): Path of the seg_file.
        meta_file (str): Path of the meta_file.
        label_map_file (str): Path of the label_map_file.
        test_mode (bool): Whether is generating test data without labels.
            Default: False.

    It returns a tuple, which contains the the following things:
        np.ndarray: Vertices of points data.
        np.ndarray: Indexes of label.
        np.ndarray: Indexes of instance.
        np.ndarray: Instance bboxes.
        dict: Map from object_id to label_id.
    """
    #我们时csv文件，应该要改成
    #label_map = scannet_utils.read_label_mapping(
    #    label_map_file, label_from="raw_category", label_to="nyu40id"
    #)
    label_map = chulimap(label_map_file)
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    #这一步应该是没有用的。
    #lines = open(meta_file).readlines()
    alignlines = open(align_file)
    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)# already单位阵
    #我们把这个搞开以及这里的文件
    for line in alignlines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    print("看看转换成功了么？ ", axis_align_matrix)

    # perform global alignment of mesh vertices
    ### 我记得这里是需要取个逆的。我们先看看原版有没有
    axis_align_matrix = np.linalg.inv(axis_align_matrix)
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]], axis=1)

    # Load semantic and instance labels
    if not test_mode:
        #look, ok,same
        object_id_to_segs, label_to_segs = scannet_utils.read_aggregation(
            agg_file
        )  # * return dicts with id(int) or label(str) to lists of seg ids, object ids are 1-indexed
        seg_to_verts, num_verts = scannet_utils.read_segmentation(seg_file)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
        raw_categories = np.array([None] * num_verts)  # Array to store raw categories

        object_id_to_label_id = {}
        object_id_to_raw_category = {}
        for raw_category, segs in label_to_segs.items():
            label_id = label_map[raw_category]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id
                raw_categories[verts] = raw_category  # Assign raw category

        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][
                        0
                    ]  # * obj_id: int
                if object_id not in object_id_to_raw_category:
                    object_id_to_raw_category[object_id] = raw_categories[verts][
                        0
                    ]  # * obj_id: str, note, the obj_id is 1-indexed
        unaligned_bboxes, unaligned_obj_point_clouds, o2n, n2o = scannet_utils.extract_bbox(
            mesh_vertices, object_id_to_segs, object_id_to_label_id, instance_ids
        )#这一步我们用我们之前的方法,但是如果等于0，我们要加一些。
        aligned_bboxes, aligned_obj_point_clouds, o2n, n2o = scannet_utils.extract_bbox(
            aligned_mesh_vertices,
            object_id_to_segs,
            object_id_to_label_id,
            instance_ids,
        )
    else:
        label_ids = None
        raw_categories = None
        instance_ids = None
        unaligned_bboxes = None
        aligned_bboxes = None
        object_id_to_label_id = None
        aligned_obj_point_clouds = None
        unaligned_obj_point_clouds = None
        object_id_to_raw_category = None

    return (
        mesh_vertices,
        aligned_mesh_vertices,
        label_ids,
        raw_categories,
        instance_ids,
        unaligned_bboxes,
        aligned_bboxes,
        unaligned_obj_point_clouds,
        aligned_obj_point_clouds,
        object_id_to_raw_category,
        object_id_to_label_id,
        axis_align_matrix,
        o2n,
        n2o
    )


def export_one_scan(
    align_dir,
    scan_name,
    output_filename_prefix,
    max_num_point,
    label_map_file,
    scannet_dir,
    exposed_dir,
    test_mode=False,
):
    if not osp.exists(output_filename_prefix):
        os.makedirs(output_filename_prefix)

    mesh_file = osp.join(scannet_dir, scan_name, "labels.instances.annotated.v2.ply")
    agg_file = osp.join(scannet_dir, scan_name, "semseg.v2.json")#相当于segroups那个，但是咱们那个有bbox
    seg_file = osp.join(
        scannet_dir, scan_name, "mesh.refined.0.010000.segs.v2.json"#相当于有很多数字的那个
    )
    align_file = osp.join(align_dir, scan_name, "axis_alignment.txt")
    # includes axisAlignment info for the train set scans.
    #就是每个info.txt，咱们到时候去export_posed文件里面去找
    meta_file = osp.join(exposed_dir, scan_name, f"_info.txt")
    (
        mesh_vertices,
        aligned_mesh_vertices,
        semantic_labels,
        raw_categories,
        instance_labels,
        unaligned_bboxes,
        aligned_bboxes,
        unaligned_obj_point_clouds,
        aligned_obj_point_clouds,
        object_id_to_raw_category,
        object_id_to_label_id,
        axis_align_matrix,
        o2n,
        n2o,
    ) = export(align_file, mesh_file, agg_file, seg_file, meta_file, label_map_file, test_mode)
    #export才是主要的处理文件
    if not test_mode:
        # mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        # mesh_vertices = mesh_vertices[mask, :]
        # semantic_labels = semantic_labels[mask]
        # instance_labels = instance_labels[mask]
        # raw_categories = raw_categories[mask]

        num_instances = len(np.unique(instance_labels))
        print(f"Num of instances: {num_instances - 1}")

        # bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS) # * keep all instances
        # unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        # bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        # aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f"Num of care instances: {unaligned_bboxes.shape[0]}")

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]
                raw_categories = raw_categories[choices]

    # Save points, semantic_labels, instance_labels as .npy files
    np.save(f"{output_filename_prefix}/unaligned_points.npy", mesh_vertices)
    np.save(f"{output_filename_prefix}/aligned_points.npy", aligned_mesh_vertices)
    scene_info = {}  # Dictionary to hold scene information

    if not test_mode:
        np.save(f"{output_filename_prefix}/semantic_mask.npy", semantic_labels)
        np.save(f"{output_filename_prefix}/instance_mask.npy", instance_labels)
        np.save(f"{output_filename_prefix}/raw_category_mask.npy", raw_categories)

        # * assert these four npy have the same length
        assert (
            len(semantic_labels)
            == len(instance_labels)
            == len(raw_categories)
            == len(mesh_vertices)
        ), "Lengths of semantic_labels, instance_labels, raw_categories, and mesh_vertices are not equal."

        # Save bounding boxes and raw category names in a dict
        for obj_id, (aligned_bbox, unaligned_bbox) in enumerate(
            zip(aligned_bboxes, unaligned_bboxes)
        ):
            print("zip出来是啥，", obj_id)
            print("需要从1开始的啥：", object_id_to_raw_category)#这个是按照真实序号来的知道吧
            cunde = obj_id
            obj_id = n2o[obj_id] - 1
            raw_category_name = object_id_to_raw_category.get(
                obj_id + 1, "None"
            )  # * object_id_to_raw_category is 1 indexed
            if raw_category_name == "None":
                print(
                    f"Something wrong for the raw category name of object {obj_id} in scan {scan_name}."
                )
                exit(0)
            scene_info[obj_id] = {
                "aligned_bbox": aligned_bbox,
                "unaligned_bbox": unaligned_bbox,
                "raw_category": raw_category_name,
            }

            # * save aligned and unaligned points
            # * first check if the two types of points have the same shape

            np.save(
                f"{output_filename_prefix}/object_{obj_id}_aligned_points.npy",
                aligned_obj_point_clouds[cunde],
            )
            np.save(
                f"{output_filename_prefix}/object_{obj_id}_unaligned_points.npy",
                unaligned_obj_point_clouds[cunde],
            )

        scene_info["axis_align_matrix"] = axis_align_matrix
        # * store the object number
        scene_info["num_objects"] = len(aligned_bboxes)

    return {scan_name: scene_info}


def worker(args):
    (
        align_dir,
        scan_name,
        output_filename_prefix,
        max_num_point,
        label_map_file,
        scannet_dir,
        exposed_dir,
        test_mode,
    ) = args
    print("-" * 20 + f"begin for {scan_name}.")
    return export_one_scan(
        align_dir,
        scan_name,
        output_filename_prefix,
        max_num_point,
        label_map_file,
        scannet_dir,
        exposed_dir,
        test_mode,
    )


def batch_export(
    beginhm,
    endhm,
    max_num_point,
    output_folder,
    scan_names_file,
    label_map_file,
    scannet_dir,
    exposed_dir,
    test_mode=False,
    num_workers=20,
):
    if test_mode and not os.path.exists(scannet_dir):
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    scan_names = [line.rstrip() for line in open(scan_names_file)]
    #人家会自动读的
    # * sort scan_names
    scan_names.sort()
    ###这里是这样的，因为这里很快我们不需要那个分开
    #scan_name就是子文件夹名字，这个不用管
    #scan_names = scan_names[beginhm, endhm]
    align_dir = "../ceshioutputscan/axis_align"
    args = [
        (
            align_dir,
            scan_name,
            osp.join(output_folder, scan_name),
            max_num_point,
            label_map_file,
            scannet_dir,
            exposed_dir,
            test_mode,
        )
        for scan_name in scan_names
    ]

    all_scene_info = {}
    with Pool(num_workers) as p:
        results = p.map(worker, args)
        for result in results:
            all_scene_info.update(result)

    # Save the combined scene information
    if test_mode:
        file_name = "scenes_test_info.pkl"
    else:
        file_name = f"scenes_train_val_info_range_{beginhm}_{endhm}.pkl"
    with open(osp.join(output_folder, file_name), "wb") as f:
        pickle.dump(all_scene_info, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_num_point", default=None, help="The maximum number of the points."
    )

    parser.add_argument(
        "--output_folder",
        default="../ceshioutputscan/xu3rscan_instance_data",
        help="output folder of the result.",
    )
    parser.add_argument(
        "--train_scannet_dir", default="../../../../../3rscandata/data", help="scannet data directory."
    )
    parser.add_argument(
        "--test_scannet_dir", default="scans_test", help="scannet data directory."
    )
    ###它这里的label_file应该指的是那个process的file才对

    parser.add_argument(
        "--label_map_file",
        default="meta/processed_file1.tsv",
        help="The path of label map file.",
    )
    parser.add_argument(
        "--train_scan_names_file",
        default="meta/valid.txt",
        help="The path of the file that stores the scan names.",
    )
    parser.add_argument(
        "--test_scan_names_file",
        default="meta_data/scannetv2_test.txt",
        help="The path of the file that stores the scan names.",
    )
    parser.add_argument('--scene_range_begin', type=int, default=0)
    parser.add_argument('--scene_range_end', type=int,
                        default=-1)  # * -1 means to the end
    """
    parser.add_argument(
        "--max_num_point", default=None, help="The maximum number of the points."
    )

    parser.add_argument(
        "--output_folder",
        default="ceshioutput360/3rscan_instance_data",
        help="output folder of the result.",
    )
    parser.add_argument(
        "--train_scannet_dir", default="ceshiscan", help="scannet data directory."
    )
    parser.add_argument(
        "--test_scannet_dir", default="scans_test", help="scannet data directory."
    )
    parser.add_argument(
        "--label_map_file",
        default="3rscan.csv",
        help="The path of label map file.",
    )
    parser.add_argument(
        "--train_scan_names_file",
        default="ceshi.txt",
        help="The path of the file that stores the scan names.",
    )
    parser.add_argument(
        "--test_scan_names_file",
        default="meta_data/scannetv2_test.txt",
        help="The path of the file that stores the scan names.",
    )"""
    args = parser.parse_args()
    scene_range_begin = args.scene_range_begin
    if args.scene_range_end == -1:
        scene_range_end = 1335
    else:
        scene_range_end = args.scene_range_end
    exposed_dir = "../ceshioutputscan/posed_imageschushi"
    batch_export(
        scene_range_begin,
        scene_range_end,
        args.max_num_point,
        args.output_folder,
        args.train_scan_names_file,
        args.label_map_file,
        args.train_scannet_dir,
        exposed_dir,
        test_mode=False,
    )
    # * change output folder for test
    """args.output_folder = args.output_folder.replace("scannet", "scannet_test")
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.test_scan_names_file,
        args.label_map_file,
        args.test_scannet_dir,
        exposed_dir,
        test_mode=True,
    )"""


if __name__ == "__main__":
    main()
