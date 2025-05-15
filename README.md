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
cd data/scannet/
```

#### 1. Prepare ScanNet Data

Download the [ScanNet dataset](https://github.com/ScanNet/ScanNet) and organize the data folder structure as follows: 
```
data/
└── scannet
    ├── grounding
    ├── meta_data
    ├── scans  # Place ScanNet data here
    │   ├── scene0000_00
    │   ├── scene0000_01
    │   ...
    │
    └── tools
```

#### 2. Sample Posed Images

We extract one frame out of every 20, requiring approximately 850 seconds and 27GB of disk space:
```bash
python tools/extract_posed_images.py --frame_skip 20 --nproc 8  # using 8 processes
```

This will generate the `data/scannet/posed_images` folder.

#### 3. Generate Dataset Info File

Run the script to batch load ScanNet data:
```bash
python tools/batch_load_scannet_data.py
```

This will export the ScanNet data to the `data/scannet/scannet_instance_data` folder.

Update the info file with posed images information:
```bash
python tools/update_info_file_with_images.py
```

### Run VLM-Grounder

First, set the path environment variable:
```bash
cd path/to/VLMGrounder
export PYTHONPATH=$PYTHONPATH:path/to/VLMGrounder
```

