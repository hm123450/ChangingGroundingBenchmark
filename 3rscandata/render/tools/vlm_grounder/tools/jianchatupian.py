import json


json_file_path = "en_1k_fanal_version_5_5.json"

from PIL import Image

def check_images_width_greater_than_height(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            images = []
            for part in data:
                all_images_files = [
                    hm for hm in part["images"]
                ]
                images+=all_images_files
            
            for image_name in images:
                image_path = image_name
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        if width <= height:
                            #print(f"Image {image_name} with width {width} and height {height} is not wider than it is taller.")
                            return False
                        else:
                            print(f"Image {image_name} with width {width} and height {height} is wider than it is taller.")
                except FileNotFoundError:
                    print(f"Image {image_name} not found.")
                    return False
            return True
    except FileNotFoundError:
        print("The file was not found.")
        return False
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        return False
def unique(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        images = []
        ww = []
        for part in data:
            all_images_files = [
                hm for hm in part["images"]
            ]
            images+=all_images_files
        for part in data:
            x= f"{part['id']}"
            ww.append(x)

    print(len(ww))
    ww = list(set(ww))
    print(len(ww))
    images = list(set(images))
    print(len(images))
    with open("tupianoutput.txt", "w", encoding="utf-8") as file:
        for item in images:
            file.write(f"{item}\n")  # 加换行符\n

# 替换 'path_to_your_json_file.json' 和 'path_to_images_directory' 为你的 JSON 文件路径和图像目录

#result = check_images_width_greater_than_height(json_file_path)
#if result:
#    print("All images in the JSON file are wider than they are taller.")
#else:
#    print("Some images in the JSON file are not wider than they are taller.")
unique(json_file_path)
###所以是有重合的