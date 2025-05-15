import argparse
from os.path import basename

import mmengine
import pandas as pd
from tqdm import tqdm
import ast
from vlm_grounder.utils import CategoryJudger

category_judger = CategoryJudger(openai_model="gpt-3.5-turbo-0125")

import pandas as pd
df = pd.read_csv("../../meta/processed_file1.tsv", sep='\t')
duiying = dict(zip(df['Label'], df['category']))
duiying["unknown"] = "unknown"
duiying["drink"] = "drink"


def load_detections(pkl_file):
    return mmengine.load(pkl_file)

def filter_and_match_detections(scene_instance, target_class, confidence_threshold):
    matched_images = []
    for image_path, detection_results in scene_instance.items():
        detections_filtered = detection_results[
            detection_results.confidence > confidence_threshold
        ]
        target_class = duiying[target_class]

        for detection in detections_filtered:
            if category_judger.is_same_category(
                detection[-1]["class_name"], target_class
            ):  # detection[-1] is data
                matched_images.append(basename(image_path).replace(".jpg", ""))
                break
    return matched_images


def find_matching_images(
    csv_file, pkl_file, confidence_thresholds, n=None, save_interval=1
):
    """
    Find matching images for each row in a CSV file based on specified criteria.

    Args:
        csv_file (str): The path to the CSV file containing the data.
        pkl_file (str): The path to the pickle file containing instance memory.
        confidence_thresholds (list): A list of confidence thresholds to use for matching.
        n (int, optional): The maximum number of rows to process. Defaults to None.
        save_interval (int, optional): The interval at which to save the intermediate CSV files. Defaults to 1.

    Returns:
        pandas.DataFrame: The modified DataFrame with matched image IDs and paths.

    """
    data = pd.read_csv(csv_file)
    if n is not None and n < len(data):
        data = data.head(n)
        file_suffix = f"_top{n}"
    else:
        file_suffix = ""

    instance_memory = load_detections(pkl_file)

    for index, row in tqdm(data.iterrows(), total=len(data)):
        scene_id = row["scan_id"]

        mao_class = data.loc[index, "anchors_types"]
        mao_class = ast.literal_eval(mao_class)
        mao_match = []
        for item in mao_class:
            for confidence_threshold in confidence_thresholds:
                if scene_id in instance_memory:
                    scene_instance = instance_memory[scene_id]
                    matched_images = filter_and_match_detections(
                        scene_instance, item, confidence_threshold
                    )
                    mao_match+=matched_images
                else:
                    matched_images = []
                    mao_match+=matched_images

                data.loc[index, f"matched_image_ids_confidence{confidence_threshold}mao"] = (
                    str(mao_match)
                )
        data.loc[index, "instance_pkl_path"] = pkl_file

        if (index + 1) % save_interval == 0:
            new_csv_file = csv_file.replace(
                ".csv", f"_with_images_selected_diffconf_and_pkl_top{index + 1}mao.csv"
            )
            data.loc[0:index].to_csv(
                new_csv_file, index=False
            )  # here data.loc[index] is also saved

    # save again the whole file
    new_csv_file = csv_file.replace(
        ".csv", f"_with_images_selected_diffconf_and_pkl{file_suffix}mao.csv"
    )
    data.to_csv(new_csv_file, index=False)
    print("new_csv_file:", new_csv_file)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vg_file",
        type=str,
        default="../../../ceshioutputscan/query_analysisto/today.csv",
    )  # * no used
    parser.add_argument(
        "--det_file",
        type=str,
        default="../../../ceshioutputscan/image_instance_detectorto/yolo_prompt_v2_updated_today_250_updated2_relations/chunk30/detection.pkl",
    )
    parser.add_argument("--sample_num", type=int, default=250)
    args = parser.parse_args()
    confidence_thresholds = [0.2, 0.3, 0.4, 0.5]
    # change this if used gdino detector
    det_file = args.det_file
    csv_file = args.vg_file
    n = args.sample_num  # process the first n samples

    print(csv_file)
    find_matching_images(csv_file, det_file, confidence_thresholds, n, save_interval=250)
