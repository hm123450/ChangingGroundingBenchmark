# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/scannet_utils.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts"""

import csv
import json
import os

import numpy as np
from plyfile import PlyData


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            object_id = (
                data["segGroups"][i]["objectId"] + 1
            )  # instance ids should be 1-indexed
            label = data["segGroups"][i]["label"]
            segs = data["segGroups"][i]["segments"]
            object_id_to_segs[object_id] = segs  # * segs are is list
            if label in label_to_segs:
                label_to_segs[label].extend(segs)  # * for semantic segmentation
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data["segIndices"])
        for i in range(num_verts):
            seg_id = data["segIndices"][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def extract_bbox(mesh_vertices, object_id_to_segs, object_id_to_label_id, instance_ids):
    """
    Extracts bounding boxes and point clouds for each instance.

    Parameters:
    mesh_vertices (numpy.ndarray): The mesh vertices.
    object_id_to_segs (dict): Mapping of object IDs to segments.
    object_id_to_label_id (dict): Mapping of object IDs to label IDs.
    instance_ids (numpy.ndarray): Array of instance IDs.

    Returns:
    numpy.ndarray: An array containing bounding boxes for each instance.
                   The ID (1-index) of each bounding box is its index in the returned array plus 1.
    list of numpy.ndarray: A list containing point clouds for each instance,
                           corresponding to the bounding boxes. 'None' for instances without points.
    """
    print(object_id_to_segs)
    print(object_id_to_label_id)
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    t = 0
    object2num = {}
    num2object = {}
    for i in list(object_id_to_segs.keys()):
        print(type(i))
        object2num[i] = t
        num2object[t] = i
        t += 1
    print(object2num)
    #注意应该只有最后一步存的时候会有问题
    instance_bboxes = np.zeros((num_instances, 7))
    print("type是啥", instance_bboxes.shape)
    instance_pcs = [None] * num_instances  # Initialize list with None

    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        obj_pc_rgb = mesh_vertices[instance_ids == obj_id, :]
        if len(obj_pc) == 0:
            print(
                f"WARNING: object id {obj_id} does not have points. Corresponding entry is set to None."
            )
            continue

        xyz_min = np.min(obj_pc, axis=0)
        xyz_max = np.max(obj_pc, axis=0)
        tt = xyz_max - xyz_min
        if tt[0]==0:
            tt[0]+=0.000001
        if tt[1]==0:
            tt[1]+=0.000001
        if tt[2]==0:
            tt[2]+=0.000001
        bbox = np.concatenate(
            [(xyz_min + xyz_max) / 2.0, tt, np.array([label_id])]
        )
        #为了保证数量一样我们再返回一个字典
        shuzi = object2num[obj_id]
        instance_bboxes[shuzi, :] = bbox
        instance_pcs[shuzi] = (
            obj_pc_rgb  # Store the point cloud at the appropriate index
        )

    return instance_bboxes, instance_pcs, object2num, num2object


def represents_int(s):
    """Judge whether string s represents an int.

    Args:
        s(str): The input string to be judged.

    Returns:
        bool: Whether s represents int or not.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping



def read_mesh_vertices(filename):
    """Read XYZ for each vertex.

    Args:
        filename(str): The name of the mesh vertices file.

    Returns:
        ndarray: Vertices.
    """
    assert os.path.isfile(filename)
    #is ok args ok
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
    return vertices


def read_mesh_vertices_rgb(filename):
    """Read XYZ and RGB for each vertex.

    Args:
        filename(str): The name of the mesh vertices file.

    Returns:
        Vertices. Note that RGB values are in 0-255.
    """
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["red"]
        vertices[:, 4] = plydata["vertex"].data["green"]
        vertices[:, 5] = plydata["vertex"].data["blue"]
    return vertices
