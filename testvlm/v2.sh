#!/bin/bash

# initial visual grounding
# TO CHANGE
VG_FILE=/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/hm3rscandata/xuanran/ceshioutputscan/query_analysisto/todaymi.csv

# TO CHANGE
DET_INFO=/mnt/afs/rsxu/humiao/vlmg1/vlm-grounder1/hm3rscandata/xuanran/ceshioutputscan/image_instance_detectorto/yolo_prompt_v2_updated_today_250_updated2_relations/chunk30/detection.pkl

# TO CHANGE
MATCH_INFO=data/scannet/scannet_match_data/*.pkl

DATE=2025-04-23
#EXP_NAME=v2_ceshi_bufen_ceshi_1_2
EXP_NAME=v2_ceshi_posels_weibolu_1

GPT_TYPE=gpt-4.1-2025-04-14
PROMPT_VERSION=3

#-m debugpy --listen localhost:11086 --wait-for-client
python \
    v2.py \
  --from_scratch \
  --do_ensemble \
  --post_process_component \
  --post_process_erosion \
  --use_sam_huge \
  --use_bbox_prompt \
  --vg_file_path ${VG_FILE} \
  --exp_name ${DATE}_${EXP_NAME} \
  --prompt_version ${PROMPT_VERSION} \
  --openaigpt_type ${GPT_TYPE} \
  --skip_bbox_selection_when1 \
  --det_info_path ${DET_INFO} \
  --matching_info_path ${MATCH_INFO} \
  --use_new_detections \
  --dynamic_stitching \
  --kernel_size 7 \
  --online_detector tidai > log.txt \

