# ChangingGroundingBenchmark
## Installation
First, clone the repository with submodules:
```bash
git clone https://github.com/hm123450/ChangingGroundingBenchmark
cd ChangingGroundingBenchmark
```
Install submodules:
- GroundingDINO
- pats @ 98d2e03

## Enviroment
This project is ran on Python 3.10.11:
```bash
conda create -n "changinggrounding" python=3.10.11
conda activate changinggrounding
```

Install basic library:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Then, install the required Python packages and PyTorch3D:
```bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

**SAM-Huge:** Download the SAM-Huge weight file from [here](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth) and place it in the `checkpoints/SAM` folder.

**PATS:** For image matching, we use PATS. Download the required [weights](https://drive.google.com/drive/folders/1SEz5oXVH1MQ2Q9lzLmz_6qQUoe6TAJL_?usp=sharing) and place them in the `3rdparty/pats/weights` folder.

**GroundingDINO:** For 2D detector, we use GroundingDINO. Download the required [weights](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) and place them in the `3rdparty/GroundingDINO/weights` folder.

Install the tensor-resize module:
```bash
cd 3rdparty/pats/setup
python setup.py install
cd ../../..
```

The PATS weights folder structure should look like this:
```
pats
├── data
└── weights
    ├── indoor_coarse.pt
    ├── indoor_fine.pt
    ├── indoor_third.pt
    ├── outdoor_coarse.pt
    ├── outdoor_fine.pt
    └── outdoor_third.pt
```

### Setup API Keys

Set your OpenAI API key in all `.../vlm_grounder/utils/my_openai.py`:
```python
api_key = "your_openai_api_key"  # sk-******
```

### Data Preparation

Navigate to the dataset folder:
```bash
cd 3rscandata/render
```

#### 1. Prepare Data

Follow the method [ChangingGroundingBenchmark](https://huggingface.co/datasets/miao1108316/ChangingGrounding) to download ChangingGroundingBenchmark. 
Make sure you download the metafile 3RScan.json.
Install the Nvidiffrast_tool in ChangingGroundingBenchmark/testvlm dir

#### 2. Posed Images

Follow the method [ChangingGroundingBenchmark](https://huggingface.co/datasets/miao1108316/ChangingGrounding) to extract and rerender images and infos.
Put the output in `outputscan/posed_images`
```bash
mkdir -p 3rscandata/render/outputscan/posed_images
```

#### 3. Generate Dataset Info File

Run the script to batch load ScanNet data:
```bash
python tools/batch_load_scannet_data.py
```

This will export the data to the `3rscandata/render/outputscan/3rscan_instance_data` folder.

Update the info file with posed images information:
```bash
python tools/update_info_file_with_images.py
```

### Benchmark test

We release the test data used in our paper in the `outputs/query_analysis` folder today.csv and yesterday.csv [cache](https://huggingface.co/datasets/miao1108316/changinggroundingcache)  


#### 1. Prepare test samples

Choose samples from Changing_Grounding.csv and process them , process for both yesterday.csv and today.csv:
```bash
python choose.py
python update.sh
```

Calculate the fine-grained categories (e.g., Unique, Multi). Also, do it for today.csv and yesterday.csv:
```bash
python 3rscandata/render/tools/pre_compute_category.py 
```

Noted you should set the output path by your willing

#### 2. Exhaustive Matching in the Scene

Use PATS to obtain exhaustive matching data. This procession is for the baselines. If you only want run MCG, you don't need to run it:
```bash
python 3rscandata/render/tools/vlm_grounder/tools/exhaustive_matching.py
```

This will generate `/3rscandata/render/outputscan/match_results/result.pkl`



#### 3. Query Analysis

Run the QueryAnalysis module to analyze each query and get the predicted target class and conditions for today.csv and yesterday.csv:
```bash
python 3rscandata/render/tools/vlm_grounder/tools/querysisto.py
```

The output will be in the `/3rscandata/render/outputscan/query_analysisto` folder. Predicted target class accuracy typically exceeds 99%.

#### 4. Instance Detection

Run the ImageInstanceDetector module to detect target class objects for each image. We suggest to use Yolov8-world for object detection for low cost and speed. If using YOLO, `checkpoints/yolov8_world/yolov8x-worldv2.pt` will be downloaded automatically. Noted that if you want to run MCG, you need to run it for today and yesterday, otherwise, baseline methods only need procession on today.
```bash
python 3rscandata/render/tools/vlm_grounder/tools/detectionto.py
```

Output results will be in the `/3rscandata/render/outputscan/image_instance_detectorto` folder.


#### 5. View Pre-Selection

Run the ViewPreSelection module to locate all images containing the predicted target class. Noted that if you want to run MCG, you need to run it for today and yesterday, otherwise, baseline methods only need procession on today.
```bash
python 3rscandata/render/tools/vlm_grounder/tools/viewto.py
```

A new CSV file will be produced in the QueryAnalysis output directory, with the suffix `_with_images_selected_diffconf_and_pkl` appended.

#### 6. Benchmark test

Run the baselines and MCG. Intermediate results with visualization will be saved in `testvlm/outputs/`.

Enter testvlm dir, you can see baseline exe files, `wg.sh`, `crg.sh`, `mog.sh`, and our framework exe file mcg-our.sh

Please change the `VG_FILE`, `DET_INFO`, `MATCH_INFO`, `DATE`, and `EXP_NAME` variables accordingly to baseline exe files.

```bash
#!/usr/bin/zsh
source ~/.zshrc

# Initial visual grounding
VG_FILE=outputs/query_analysis/*_relations_with_images_selected_diffconf_and_pkl.csv

DET_INFO=outputs/image_instance_detector/*/chunk*/detection.pkl

MATCH_INFO=data/scannet/scannet_match_data/*.pkl

DATE=2024-06-21
EXP_NAME=test 

GPT_TYPE=gpt-4o-2024-05-13
PROMPT_VERSION=3

python ./vlm_grounder/grounder/visual_grouder.py \
  --from_scratch \
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
  --online_detector [yolo|gdino]
```

As for mcg-our.sh, except for `VG_FILE`, `DET_INFO`, `MATCH_INFO`, `DATE`, and `EXP_NAME` variables, you also need to update your yesVG_FILE in mcg-our.py.

#### 7. Calculate cost and accuracy
Noted you need to update your own output path in files.
```bash
python testvlm/mcgcost.py
```

#### Reminder
If you encounter any issues, feel free to open an issue at any time.

