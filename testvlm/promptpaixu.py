paixu_prompt = """Given a series of images from a video scanning an indoor room and a query describing a specific object in the room, you need to analyze the images to locate the object mentioned in the query within the images.

You will be provided with multiple images, and the top-left corner of each image will have an ID indicating the order in which it appears in the video. Adjacent images have adjacent IDs. Please note that to save space, multiple images have been combined into one image with dynamic layouts. Note that I have filtered the video to remove some images that do not contain objects of the target class. You will also receive a list of target objects, and please sort these images according to this list of target objects. The sorting rules are as follows: 1. When there is only one type of object in the target object list, the clearer the image, the more it should be ranked at the front. 2. When there are multiple types of objects in the target object list, the more types of target objects an image contains, the more it should be ranked at the front. Conversely, the fewer types of target objects an image contains, the more it should be ranked at the back. If several images contain the same number of types of target objects, then the clearer the image, the more it should be ranked at the front. Finally you should return the sorted image ids.

Reply using JSON format with one keys "sorted_image_ids" like this:
{{
    "sorted_image_ids": ["00045", "00002", "00013", "00004", ......], // A list of IDs of images sorted by the rules
}}

Now start the task:
Target Objects: {pred_target_class}

Here are the {num_view_selections} images for you to sort .

"""

paixu_prompt1 = """Given a series of images from a video scanning an indoor room and a query describing a specific object in the room, you need to analyze the images to locate the object mentioned in the query within the images.

You will be provided with multiple images, and the top-left corner of each image will have an ID indicating the order in which it appears in the video. Adjacent images have adjacent IDs. Please note that to save space, multiple images have been combined into one image with dynamic layouts. Note that I have filtered the video to remove some images that do not contain objects of the target class. You will also receive a list of target objects, and please sort these images according to this list of target objects. The sorting rules are as follows: 1. When there is only one type of object in the target object list, if the image gives you the impression that the camera was closer to the target object when the photo was taken, the more it should be ranked at the front. 2. When there are multiple types of objects in the target object list, the more types of target objects an image contains, the more it should be ranked at the front. Conversely, the fewer types of target objects an image contains, the more it should be ranked at the back. If several images contain the same number of types of target objects, the image should be ranked higher if it gives the impression that the camera was closer to the target objects when the image was taken. Finally you should return the sorted image ids.

Reply using JSON format with one keys "sorted_image_ids" like this:
{{
    "sorted_image_ids": ["00045", "00002", "00013", "00004", ......], // A list of IDs of images sorted by the rules
}}

Now start the task:
Target Objects: {pred_target_class}

Here are the {num_view_selections} images for you to sort .

"""

paixu_prompt2 = """Given a series of images from a video scanning an indoor room and a query describing a specific object in the room, you need to analyze the images to locate the object mentioned in the query within the images.

You will be provided with multiple images, and the top-left corner of each image will have an ID indicating the order in which it appears in the video. Adjacent images have adjacent IDs. Please note that to save space, multiple images have been combined into one image with dynamic layouts. Note that I have filtered the video to remove some images that do not contain objects of the target class. You will also receive a list of target objects, and please sort these images according to this list of target objects. First, pick out the images that you think are too blurry to be recognized. These images will not participate in the sorting. After that, you will sort the remaining images according to the following rules. The sorting rules are as follows: 1. When there is only one type of object in the target object list, if the image gives you the impression that the camera was closer to the target object when the photo was taken, the more it should be ranked at the front. 2. When there are multiple types of objects in the target object list, the more types of target objects an image contains, the more it should be ranked at the front. Conversely, the fewer types of target objects an image contains, the more it should be ranked at the back. If several images contain the same number of types of target objects, the image should be ranked higher if it gives the impression that the camera was closer to the target objects when the image was taken. Notice, please try not to return null values. If you think it is impossible to distinguish the priority order of these images, please output them in the order in which I input them. Finally you should return the sorted image ids.

Reply using JSON format with keys "sorted_image_ids" and "discarded_image_ids" like this:
{{
    "sorted_image_ids": ["00045", "00002", "00013", "00004", ......], // A list of IDs of images sorted by the rules, note that the returned serial numbers should be in the form like "00045", not "00045.color", do not add any suffix after numbers.
    "discarded_image_ids": ["00023", "00056", ......],  // A list of IDs of images that are too blurry or cannot identify the target objects
}}

Then start the task:
Target objects are {pred_target_class} . Here are the {num_view_selections} images for you to sort .

"""

tongyi_prompt = """You are an agent who is very good at analyzing pictures. Now I have a target object: {targetobject}. You will be provided with multiple images, and the top-left corner of each image will have an ID.
If there is only one image, you just need to return its ID.
If there are multiple images, I would like you to analyze whether the target object {targetobject} in these pictures belong to the same one (because only a part of the target object may be captured in some pictures). You may find that some pictures are of the same target object while others are of a different but also consistent target object.
After the analysis, please return to me the IDs of the pictures grouped by the target objects they capture, and give the reasons.
Your response should be formatted as a JSON object with two keys "reasoning", "same_object_image_ids_list" like this:
{{
"reasoning": "your reasoning process", // Explain why you think certain pictures captured the same target object. If there is only one image, this can be a simple statement like "Only one image provided."
"same_object_image_ids_list": [["00001"]] or [["00001", "00042", "00003"], ["00114", "00005"]] // A list of lists, where each inner list contains the IDs of images that you analyzed to have captured the same target object. The inner lists should be sorted in descending order of their length. If there is only one image, this will be a list with a single inner list containing that one ID.
}}
"""

paixu_prompt3 = """You are an agent who is very good at analyzing pictures. Given a series of indoor room images from a video, you need to analyze the images to rank them. Each image has an ID in the top-left corner indicating its order in the video. Multiple images may be combined.
You'll get a list of target objects. First, pick out the blurry images that can't be recognized, they won't be sorted. If some images are very similar, only select the clearest one to participate in the sorting.
Sort the remaining images by these rules:
If the end object is detected in an image, this image should be ranked at the front.
If there's one target object type, images where the camera seems closer to the object should be ranked higher.
If there are multiple target object types, images with more target object types should be ranked higher. If several images have the same number of target object types, the one where the camera seems closer should be ranked higher.
If you can't distinguish the order, output the images in the input order.
Reply in JSON with "reasoning" and "sorted_image_ids":
{{
"reasoning": "your reasoning process" // your thinking process about the sorting task
"sorted_image_ids": ["00045", "00002", ...] // A list of IDs of images sorted by the rules, note that the returned serial numbers should be in the form like "00045", not "00045.color", do not add any suffix after numbers.
}}
Start the task:
Target objects are {pred_target_class}. Here are {num_view_selections} images to sort.
"""

paixu_prompt4 = """You are an agent who is very good at analyzing pictures. Given a series of indoor room images from a video, you need to analyze the images to rank them. Each image has an ID in the top-left corner indicating its order in the video. Multiple images may be combined.
You'll get a list of target objects and a end object. First, pick out the images that are so dark that the objects cannot be clearly seen, they won't be sorted. If some images are very similar, only select the clearest one to participate in the sorting.
Sort the remaining images by these rules:
If the end object is detected in an image, this image should be ranked at the front.
If there's one target object type, images where the camera seems closer to the object should be ranked higher.
If there are multiple target object types, images with more target object types should be ranked higher. If several images have the same number of target object types, the one where the camera seems closer should be ranked higher.
If you can't distinguish the order, output the images in the input order.
Reply in JSON with "reasoning" and "sorted_image_ids":
{{
"reasoning": "your reasoning process" // your thinking process about the sorting task
"sorted_image_ids": ["00045", "00002", ...] // A list of IDs of images sorted by the rules, note that the returned serial numbers should be in the form like "00045", not "00045.color", do not add any suffix after numbers.
}}
Start the task:
End object is {end_class}. Target objects are {pred_target_class}. Here are {num_view_selections} images to sort.
"""

panduan_prompt2 = """You are an agent who is very good at looking at pictures. Now I'm giving you one picture. You need to determine whether the provided picture may contain an anchor object that meets a specific description by a query. This picture probably already contains an anchor object, but the target object may not  be captured. However, we are not sure if the contained anchor object is correct. You need to judge whether the anchor object in the picture has possibility to be correct based on the query and the photo.
Reply using JSON format with two keys "reasoning", "anchor_object_may_be_correct" like this:
{{
"reasoning": "your reasons", // Explain the justification why you think the anchor object may be correct or not.
"anchor_object_may_be_correct": True // True if you think the anchor object may be correct, False if you think it may not be correct
}}

Start the task:
Anchor object is {anchor_object}. Query is {query}.

"""

paixu_prompt5 = """You are an agent who is very good at analyzing pictures. Given a series of indoor room images from a video, you need to analyze the images to rank them. Each image has an ID in the top-left corner indicating its order in the video. Multiple images may be combined.
You'll get a list of target objects, a end object and a query. First, pick out the images that are so dark that the objects cannot be clearly seen, they won't be sorted. If some images are very similar, only select the clearest one to participate in the sorting.
Sort the remaining images by these rules:
If the end object is detected in an image, this image should be ranked at the front.
If there's one target object type, images where the camera seems closer to the object should be ranked higher.
If there are multiple target object types, images with more target object types should be ranked higher. If several images have the same number of target object types, the one where the camera seems closer should be ranked higher.
If you can't distinguish the order, output the images in the input order.
After the above sorting, move the images that you guess are most likely to contain the target object that can meet the query to the front(note, don't change the end object images order).
Reply in JSON with "reasoning" and "sorted_image_ids":
{{
"reasoning": "your reasoning process" // your thinking process about the sorting task
"sorted_image_ids": ["00045", "00002", ...] // A list of IDs of images sorted by the rules, note that the returned serial numbers should be in the form like "00045", not "00045.color", do not add any suffix after numbers.
}}
Start the task:
End object is {end_class}. Target objects are {pred_target_class}. Query is {query}. Here are {num_view_selections} images to sort.
"""

panduan_prompt3 = """You are an agent who is very good at looking at pictures. Now I'm giving you one picture. It's highly likely that our picture cannot capture both the target object and the anchor object simultaneously, so it's difficult to determine whether the query is satisfied. What I need is for you to judge whether the anchor object in the picture has possibility to meet the query.
For example, if the query is 'a washing machine under the shelf', but in a certain picture, the shelf is very close to the ground, and it's obvious that there's no way for a washing machine to be under it. In this case, you should return False. Note, if the query does not describe a vertical (up-down) relationship, return True.
Reply using JSON format with two keys "reasoning" and "anchor_object_may_be_correct" like this:
{{
"reasoning": "your reasons", // Explain the justification why you think the anchor object may or may not meet the query.
"anchor_object_may_be_correct": true // True if you think the anchor object may meet the query, False if you think it may not.
}}
Start the task:
Anchor object is {anchor_object}. Query is {query}.
"""

apibbox_prompt = """You are a professional computer vision assistant. You need to analyze the attached image (WIDTH={width}px × HEIGHT={height}px) and:
1. Detect all visible target object: {targetclass}
2. Generate xyxy-format bounding boxes for each detection
3. Output pixel coordinates relative to the image's original dimensions

Reply using JSON format with four keys  "reasoning", "findornot", "bbox","confidence" like this:
{{
"reasoning": "thinking process"// Explanation of how the detection was made and why certain areas were identified as target objects or not
"findornot": true or false, // True if you find target object in the image, False if you don't find target object in the image
"bbox": [[object1_x_min, object1_y_min, object1_x_max, object1_y_max], [object2_x_min, object2_y_min, object2_x_max, object2_y_max]] or [], // int values [0 ≤ x < WIDTH, 0 ≤ y < HEIGHT], empty list if no target object is found
"confidence": [confidence_score1, confidence_score2] or [] // float values between 0 and 1 representing your confidence level for each detection, empty list if no target object is found
}}
"""

apibbox_prompt2 = """You are a professional computer vision assistant. You will be provided with two images. The first image has dimensions (WIDTH={width}px × HEIGHT={height}px), and the second image (the reference image) has dimensions (WIDTH={width}px × HEIGHT={height}px). Your task is to:
1. Detect all visible target object {targetclass} in the first image. If no such target objects are detected in the first image, skip steps 2, 3, and 4.
2. Then, determine whether the target objects {targetclass} detected in the first image roughly similiar with the {targetclass} in the second image. Due to shooting perspectives, you may see the target objects from different angles. Also, because of scene changes, the target objects in the first image may have undergone transformations or have something added or removed. So, we only need to compare if they roughly belong to the same class of {targetclass}, for example, don't confuse a chair with a sofa. if not, skip steps 3, 4.
3. Generate xyxy-format bounding boxes for each detected similar object in the first image.  Note that the origin of the coordinate system is at the top - left corner of the image, where the x - axis extends horizontally to the right and the y - axis extends vertically downward.
4. Output pixel coordinates relative to the original dimensions of the first image.

Reply using JSON format with four keys "reasoning", "findornot", "bbox", "confidence" like this:
{{
"reasoning": "thinking process", // Explanation of how you think
"findornot": true or false, // True if you find similar target objects in the first image compared to the second image's target object, False otherwise
"bbox": [[object1_x_min, object1_y_min, object1_x_max, object1_y_max], [object2_x_min, object2_y_min, object2_x_max, object2_y_max]] or [], // int values [0 ≤ x < WIDTH1, 0 ≤ y < HEIGHT1], empty list if no similar target object is found
"confidence": [confidence_score1, confidence_score2] or [] // float values between 0 and 1 representing your confidence level for each detection, empty list if no similar target object is found
}}
"""

cankao_prompt1 = """ You are an agent who is very good at analyzing pictures. Given a series of indoor room images from a video, you need to analyze the images to select the one that most clearly shows the target objects {targetclass}. Each image has an ID in the top - left corner indicating its order in the video.

You'll get some pictures. First, pick out the images that are so dark that the objects cannot be clearly seen; these images will not be considered for selection. If some images are very similar, only select the clearest one.

Among the remaining images, select the one that most clearly shows the target object {targetclass}.

Reply in JSON with "reasoning" and "selected_image_id":
{{
"reasoning": "your reasoning process", // your thinking process about the selection task
"selected_image_id": "frame-000045.color" // The ID of the image that most clearly shows the target objects. Note that the returned serial number should be in the form like "frame-000045.color" or "frame-000120.color" or "frame-000001.color", not "00045.color", do not add any suffix after numbers.
}}

Start the task:
Target object is {targetclass}. Here are {num_view_selections} images to analyze."""

maodian_prompt1 = """ You are an agent who is very good at analyzing pictures. Given a series of indoor room images from a video, you need to analyze the images to select the one that most clearly shows the target objects {targetclass}. Each image has an ID in the top - left corner indicating its order in the video.

You'll get some pictures. First, pick out the images that are so dark that the objects cannot be clearly seen; these images will not be considered for selection. If some images are very similar, only select the clearest one.

Among the remaining images, select two images that most clearly show the target object {targetclass}.

Reply in JSON with "reasoning" and "selected_image_id":
{{
"reasoning": "your reasoning process", // your thinking process about the selection task
"selected_image_id": ["frame-000045", "frame-000120"] // The ID of the image that most clearly shows the target objects. Note that the returned serial number should be in the form like "frame-000045" or "frame-000120" or "frame-000001", not "00045.color", do not add any suffix after numbers.
}}

Start the task:
Target object is {targetclass}. Here are {num_view_selections} images to analyze."""

buchonghe_prompt1 = """You are an agent who is very good at analyzing pictures. Given a series of indoor room images from a video, you need to analyze these images. Each image has an ID in the top - left corner indicating its order in the video. Multiple images may be combined.
Your task is to go through the images. If there are some images that are very similar, select only the clearest one. If an image is not similar to the others, keep it as well. Finally, output the selected images in the order of their IDs.
Reply in JSON with "reasoning" and "selected_image_ids":
{{
"reasoning": "your reasoning process for image selection",
"selected_image_ids": ["frame-000002", "frame-000004", "frame-000010"...] // A list of IDs of selected images, note that the returned serial numbers should be in the form like "frame-000045", not "00045.color", do not add any suffix after numbers.
}}
Start the task: Here are {num_view_selections} images for you to process.
"""

baohan_prompt2 = """You are an agent who is very good at analyzing pictures. Given a single indoor room image, your task is to perform analysis of the content of the image.
First step: Determine whether the image contains anchor object: {anchor_class}. 
Second step: Determine whether the image contains taregt object: {target_class}. 
Reply in JSON with "anchor_exist", "target_exist" and "reasoning":
{{
"reasoning": "Your overall reasoning process for the two-step analysis of the image",
"anchor_exist": True or False, // True if you find anchor object in the image, False if you don't find.
"target_exist": True or False, // True if you find target object in the image, False if you don't find.
}}"""

celue_prompt1 = """You are an intelligent agent specialized in analyzing images and planning camera movements to locate a target object. You will be provided with a single image that contains an anchor object: {anchor}, a query: {query} describing a target object: {target} you need to find, and the last camera movement made: {last_move}.
Your task is to thoroughly examine the provided image, take note of the position and features of the anchor object, and consider the query. When deciding the single most appropriate movement the camera should make to either find the target object described in the query or to increase the likelihood of finding it, you must avoid suggesting the opposite movement of the last one. For example, if the last movement was "down", do not suggest "up" as it implies revisiting an area that has already been seen. Base your decision on the relationship between the anchor object and the expected location of the target object.
{{
"reasoning": "Your detailed reasoning process for determining the single - step camera movement, explaining how this movement will help find the object or scene in the query based on the analysis of the image.",
"move": "up" or "down" or "left" or "right" or "forward", "backward" // One most possible camera movements: "up", "down", "left", "right", "forward", "backward".
}}
"""

baohan_prompt3 = """You are an agent who is very good at analyzing pictures. Given a single indoor room image, your task is to perform analysis of the content of the images. You will be provided with two images.
Step 1: Determine whether the first image contains anchor object: {anchor_class}. 
Step 2: Determine whether the first image contains taregt object: {target_class}. If not, skip step 3.
Step 3: Determine whether the target objects {target_class} detected in the first image are roughly similar to the {target_class} in the second image. Due to shooting perspectives, you may see the target objects from different angles. Also, because of scene changes, the target objects in the first image may have undergone transformations or have something added or removed. So, we only need to compare if they roughly belong to the same class of {target_class}, for example, don't confuse a chair with a sofa. 
Reply in JSON with "anchor_exist", "target_exist" and "reasoning":
{{
"reasoning": "Your overall reasoning process for the two-step analysis of the image",
"anchor_exist": True or False, // True if you find anchor object in the image, False if you don't find.
"target_exist": True or False, // True if you find target object in the image, False if you don't find.
}}"""

xuanranpaixu_prompt1old = """You are an agent who is very good at analyzing pictures. Given a series of indoor room images from a video, you need to analyze the images to select the best 3 images. Each image has an ID in the top-left corner indicating its order in the video. Multiple images may be combined.

The target object is {pred_target_class} and anchor object is {anchor_class}. First, pick out the images that are so dark that the objects cannot be clearly seen, they won't be considered for selection. If some images are very similar, only select the clearest one to participate in the further selection process.

Select the best 3 images from the remaining images by these rules:
Rule 1: If there are images among the remaining ones that can clearly show both the anchor object and the target objects at the same time, these images should be selected. If the number of images selected by this rule reaches 3, stop the task and return the result. Otherwise, proceed to Rule 2.
Rule 2: Based on the number of images already selected in Rule 1, check the remaining images. Select the images that can show the anchor object relatively clearly and sompletely until the total number of selected images reaches 3.

Reply in JSON with "reasoning" and "selected_image_ids":
{{
"reasoning": "your reasoning process" // your thinking process about the selection task
"selected_image_ids": ["00045", "00002", ...] // A list of IDs of the best 3 images selected by the rules, note that the returned serial numbers should be in the form like "00045", not "00045.color", do not add any suffix after numbers.
}}
Now start the task:
Here are {num_view_selections} images to select from.
"""

xuanranpaixu_prompt1 = """You are an intelligent assistant proficient in analyzing images. Given a series of indoor room images from a video, you need to analyze these images and select the best 3 images. Each image has an ID in the upper left corner indicating its sequence in the video. Multiple images may be combined and displayed together to save the place.
The anchor object is {anchor_class}. If there are some images that are very similar, only select the clearest one to participate in the further selection process.
Select the best 3 images from the remaining images according to the following rule:
Rule 1: Select those images from the remaining ones that can clearly display the anchor object until the total number of selected images reaches 3.
Please reply in json format, including "reasoning" and "selected_image_ids":
{{
"reasoning": "Your reasoning process" // Your thinking process regarding the selection task
"selected_image_ids": ["00045", "00002", ...] // A list of the IDs of the best 3 images selected according to the rules. Note that the returned IDs should be in the form of "00045", not "00045.color", and do not add any suffix after the numbers.
"unique_question": 6, // This is an independent question. Regardless of any other factors, only look for which image among all those provided captures the object {targetclass} most clearly. If none is found, return -1.
}}
Now start the task:
There are {num_view_selections} images for you to select from.
"""


qumem_prompt1 = """You are an intelligent assistant proficient in analyzing images. Given a series of indoor room images from a video, you need to analyze these images and select some of them. Each image has an ID in the upper left corner indicating its sequence in the video. Multiple images may be combined and displayed together to save space.
The anchor object is {anchor_class}. If there are some images that are very similar, only select the clearest one to participate in the further selection process.
Please reply in json format, including "reasoning" and "selected_image_ids":
{{
"reasoning": "Your reasoning process" // Your thinking process regarding the selection task
"selected_image_idsback": ["00045"] or [] // Find which image among all those with an ID after {center_id} captures the object {anchor_class} most clearly. If none is found, return empty list. Note that the returned ID should be in the form of "00045", not "00045.color", and do not add any suffix after the numbers.
"selected_image_idsfront": ["00010"] or [] // Find which image among all those with an ID before {center_id} captures the object {anchor_class} most clearly. If none is found, return empty list. Note that the returned ID should be in the form of "00045", not "00045.color", and do not add any suffix after the numbers.
"unique_questionback": 45 or -1 // This is an independent question. Regardless of any other factors, only look for which image among all those with an ID after {center_id} captures the object {targetclass} most clearly. If none is found, return -1.
"unique_questionfront": 6 or -1 // This is an independent question. Regardless of any other factors, only look for which image among all those with an ID before {center_id} captures the object {targetclass} most clearly. If none is found, return -1.
}}
Now start the task:
There are {num_view_selections} images for you to select from.
"""




apixuanran_prompt4 = """Imagine you are in a room and are asked to find a specific object. You already know the query: {query}, the anchor object: {anchorclass}, and the target object: {targetclass}. The images you will be provided are taken as the camera rotates 360 degrees around the anchor object as the origin.
Given a series of images from a video scanning an indoor room (captured during the 360-degree rotation around the anchor object) and a query describing a specific object in the room, you need to analyze the images to locate the object mentioned in the query within the images.
You will be provided with multiple images, and the top-left corner of each image will have an ID indicating the order in which it appears in the video. Adjacent images have adjacent IDs. Please note that to save space, multiple images have been combined into one image with dynamic layouts. You will also be provided with a query sentence describing the object that needs to be found, as well as a parsed version of this query describing the target class of the object to be found and the conditions that this object must satisfy. Please find the ID of the image containing this object based on these conditions. Note that I have filtered the video to remove some images that do not contain objects of the target class. Also note that the pictures I am now providing is taken by the camera when it is very close to the anchor object. So, the anchor object may or may not appear in the picture, but you should remember that the anchor object is located very close to the camera. To locate the target object, you need to consider multiple images from different perspectives (resulting from the 360-degree rotation) and determine which image contains the object that meets the conditions. Note that each condition might not be judged based on just one image alone. Also, the conditions may not be accurate, so it's reasonable for the correct object not to meet all the conditions. You need to find the most possible object based on the query. If you think multiple objects are correct, simply return the one you are most confident of. If you think no objects are meeting the conditions, make a guess to avoid returning nothing. Usually the correct object is visible in multiple images, and you should return the image in which the object is most clearly observed.
Your response should be formatted as a JSON object with only one key "target_image_id", like this:
{{
"reasoning": "your reasoning process" // Explain the process of how you identified and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
"target_image_id": 1 // Replace with the actual image ID (only one ID) annotated on the image that contains the target object.
}}
Here are the {num_view_selections} images for your reference."""

apibbox_prompt4 = """Great! Here is the detailed version of your selected image. You already know the query: {query}, the anchor object: {anchorclass}, and the target object: {classtarget}. The picture you choose is taken by the camera when it is very close to the anchor object. So, even the anchor object may not appear in the picture, you should understand that the anchor object is located very close to the camera.
There are {num_candidate_bboxes} candidate objects shown in the image. I have annotated each object at the center with an object ID in white color text and black background. Please find out which ID marked {classtarget} object in the picture exactly meets the description: {query}.
You need to think through two steps. First, the candidate objects provided to you are not necessarily all of the target object class. You must first identify which of the candidate objects are of the target object class. Second, after the first step of analysis, select the object ID among the found candidate IDs of the target object class that is most likely to satisfy the description in the query.
Reply using JSON format with two keys "reasoning" and "object_id" like this:
{{
"reasoning": "your reasons", // Explain the justification why you select the object ID. Describe your two-step thinking process, including how you identified the target object class candidates and how you made the final selection among them.
"object_id": 0 // The object ID you selected. Always give one object ID from the image, which you are the most confident of, even you think the image does not contain the correct object.
}}
"""

apixuanran_prompt3 = """Imagine you are in a room and are asked to find a specific object. You already know the query: {query}, the anchor object: {anchorclass}, and the target object: {targetclass}. The images you will be provided are taken as the camera rotates 360 degrees around the anchor object as the origin.
Given a series of images from a video scanning an indoor room (captured during the 360 - degree rotation around the anchor object) and a query describing a specific object in the room, you need to analyze the images to locate the object mentioned in the query within the images.
You will be provided with multiple images, and the top - left corner of each image will have an ID indicating the order in which it appears in the video. Adjacent images have adjacent IDs. Please note that to save space, multiple images have been combined into one image with dynamic layouts. You will also be provided with a query sentence describing the object that needs to be found, as well as a parsed version of this query describing the target class of the object to be found and the conditions that this object must satisfy.
Before attempting to locate the target object, first visually inspect each image. If an image with a particular ID looks like it's just a solid white expanse, that is, there are hardly any distinguishable features or non - white elements, then exclude this image with that specific ID from any further analysis. This ensures that we focus only on images that have the potential to contain relevant information for identifying the target object.
After excluding such "all - white" images, find the ID of the image containing this object based on the given conditions. Note that I have already filtered the video to remove some images that do not contain objects of the target class. Also note that the pictures I am now providing are taken by the camera when it is very close to the anchor object. So, the anchor object may or may not appear in the picture, but you should remember that the anchor object is located very close to the camera. To locate the target object, you need to consider multiple images from different perspectives (resulting from the 360 - degree rotation) and determine which image contains the object that meets the conditions. Note that each condition might not be judged based on just one image alone. Also, the conditions may not be accurate, so it's reasonable for the correct object not to meet all the conditions. You need to find the most possible object based on the query. If you think multiple objects are correct, simply return the one you are most confident of. If you think no objects are meeting the conditions, make a guess to avoid returning nothing. Usually the correct object is visible in multiple images, and you should return the image in which the object is most clearly observed.
Your response should be formatted as a JSON object with only one key "target_image_id", like this:
{{
"reasoning": "your reasoning process" // Explain the process of how you identified and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
"target_image_id": 1 // Replace with the actual image ID (only one ID) annotated on the image that contains the target object.
"white_images": [4, 6, 11] // This is a list that contains the IDs of the images that look almost entirely white and were excluded from the analysis.
}}
Here are the {num_view_selections} images for your reference."""

apixuanran_prompt4 = """Imagine you are in a room and are asked to find a specific object. You already know the query: {query}, the anchor object: {anchorclass}, and the target object: {targetclass}. The images you will be provided are taken as the camera rotates 360 degrees around the anchor object as the origin.
Given a series of images from a video scanning an indoor room (captured during the 360 - degree rotation around the anchor object) and a query describing a specific object in the room, you need to analyze the images to locate the object mentioned in the query within the images.
You will be provided with multiple images, and the top - left corner of each image will have an ID indicating the order in which it appears in the video. Adjacent images have adjacent IDs. Please note that to save space, multiple images have been combined into one image with dynamic layouts. You will also be provided with a query sentence describing the object that needs to be found, as well as a parsed version of this query describing the target class of the object to be found and the conditions that this object must satisfy.
Before attempting to locate the target object, first visually inspect each image. If an image with a particular ID presents in such a way that the human eye can't discern any distinguishable features due to it being either a solid white expanse with hardly any non - white elements or having an extremely high brightness level where the details are completely washed out and invisible to the human eye, then exclude this image with that specific ID from any further analysis. This ensures that we focus only on images that have the potential to contain relevant information for identifying the target object.
After excluding such images with undiscernible content due to whiteness or high brightness, find the ID of the image containing this object based on the given conditions. Note that I have already filtered the video to remove some images that do not contain objects of the target class. Also note that the pictures I am now providing are taken by the camera when it is very close to the anchor object. So, the anchor object may or may not appear in the picture, but you should remember that the anchor object is located very close to the camera. To locate the target object, you need to consider multiple images from different perspectives (resulting from the 360 - degree rotation) and determine which image contains the object that meets the conditions. Note that each condition might not be judged based on just one image alone. Also, the conditions may not be accurate, so it's reasonable for the correct object not to meet all the conditions. You need to find the most possible object based on the query. If you think multiple objects are correct, simply return the one you are most confident of. If you think no objects are meeting the conditions, make a guess to avoid returning nothing. Usually the correct object is visible in multiple images, and you should return the image in which the object is most clearly observed.
Your response should be formatted as a JSON object with the following keys:
{{
"reasoning": "your reasoning process" // Explain the process of how you identified and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
"target_image_id": 1, // Replace with the actual image ID (only one ID) annotated on the image that contains the target object.
"white_images": [4, 6, 11] // This is a list that contains the IDs of the images that are either almost entirely white or have an extremely high brightness making them indiscernible to the human eye and were excluded from the analysis.
}}
Here are the {num_view_selections} images for your reference."""

xuanranpaixu_prompt2 = """You are an agent who is very good at analyzing pictures. Given a series of indoor room images from a video, you need to analyze the images to select the best 3 images. Each image has an ID in the top-left corner indicating its order in the video. Multiple images may be combined.

The target object is {pred_target_class} and anchor object is {anchor_class}. First, pick out the images that are so dark that the objects cannot be clearly seen, they won't be considered for selection. If some images are very similar, only select the clearest one to participate in the further selection process.

Select the best 3 images from the remaining images by these rules:
Rule 1: If there are images among the remaining ones that can clearly show both the anchor object and the target objects at the same time, select one of these images which you think the camera is closest to the captured target object and anchor object during shooting, then proceed to Rule 2. If you can't find any images that meet the rule, proceed to Rule 2.
Rule 2: Based on the number of images already selected in Rule 1 (which is 0 or 1 at this time), check the remaining images. Select images that can show the anchor object relatively completely and in which you feel the camera was relatively close to the anchor object during shooting, until the total number of selected images reaches 3.

Reply in JSON with "reasoning" and "selected_image_ids":
{{
"reasoning": "your reasoning process" // your thinking process about the selection task
"selected_image_ids": ["00045", "00002", ...] // A list of IDs of the best 3 images selected by the rules, note that the returned serial numbers should be in the form like "00045", not "00045.color", do not add any suffix after numbers.
}}
Now start the task:
Here are {num_view_selections} images to select from.
"""

apixuanran_prompt6 = """Imagine you're in a room tasked with finding a specific object. You alreday know the query about an object in the room: {query}, anchor object class: {anchorclass}, and target object class: {targetclass}. The provided images are taken as the camera rotates 360 degrees around the anchor object. You should analyze the images to locate the queried object.

The multiple images I give you have IDs in the top - left corner indicating their order in the video. Adjacent images have adjacent IDs. Note that multiple images are combined for space - saving. 

Find the image ID with the target object based on the given query. The video has been pre - filtered to remove images without the target class. Remember, the pictures are taken close to the anchor object, which may or may not be in the picture. 

Conditions from the query may not be judged by one image alone and can be inaccurate, so the correct object may not meet all. Find the most likely object. If multiple seem correct, return the one you're most confident in. If no object meets conditions, make a guess. Usually, the correct object is visible in multiple images; return the one where it's most clearly and completely seen.

Your response should be a JSON object with:
{{
"reasoning": "your reasoning process" // Explain the process of how you identified and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
"target_image_id": 1, // Replace with the actual image ID (only one ID) annotated on the image that contains the target object most completely.
"reference_image_ids": [1, 2, ...] //  A list of IDs of images that you think also contain the target object but from different perspectives as much as possible.
}}
Here are the {num_view_selections} images for reference."""

###尽量返回完整的物体图片。

apixuanranpats_prompt1 = """Great! You will be provided with {daixuan} images to analyze and an additional reference image. I will place the reference image in the end and the reference image has the target object: {classtarget} marked with a purple bounding box. For the {daixuan} images you need to analyze, each has an red ID in the top - left corner indicating its order in the video. Please note that to save space, multiple images have been combined into one image with dynamic layouts. Also, for each object in these multiple images, I have annotated each object at the center with an object ID in white color text on a black background.
Your task is to determine, for each of these multiple images, which object ID corresponds to the same target object as the one framed in the reference image. Due to differences in shooting positions and angles, the target object {classtarget} in each image may be viewed from a different perspective compared to the reference image. You need to carefully judge whether they are the same target object.
You need to think through the following steps for each image. First, analyze the characteristics of the target object in the reference image, such as its shape, size, color, and any unique features. Then, compare these characteristics with each object in the image with an object ID. Try to find the object ID in the image whose corresponding object has the most consistent characteristics with the target object in the reference image.
Reply using JSON like this:
{{
"reasoning": "your reasoning process" // Explain the justification why you select the object ID of the images
"object_id_dict": {{image1_id: 0, image2_id: -1, ....}} //Replace image1_id with the actual ID of the first image, and 0 with the object ID you selected from this image. If you judge that there is no matching object ID in this image, use -1. Similarly, for other images.
}}
"""

apixuanran_prompt9 = """Imagine you're in a room tasked with finding a specific object. You alreday know the query: {query}, anchor object class: {anchorclass}, and target object class: {targetclass}. The provided images are taken as the camera rotates 360 degrees around the anchor object.

Given images from an indoor - scanning video (during the 360 - degree rotation) and a query about an object in the room, analyze the images to locate the queried object.

You'll get multiple images with IDs in the top - left corner indicating their order in the video. Adjacent images have adjacent IDs. Note that multiple images are combined for space - saving. You'll also receive the query sentence and its parsed version, stating the target class and conditions.

Before finding the target, visually inspect each image. Exclude any image with an ID where it's a solid white expanse or has extreme brightness, making features indiscernible. This focuses analysis on potentially useful images.

After exclusion, find the image ID with the target object based on the given conditions. The video has been pre - filtered to remove images without the target class. Remember, the pictures are taken close to the anchor object, which may or may not be in the picture. Consider multiple - perspective images from the 360 - degree rotation. If you find a target - containing image, check others. If another image captures the target more completely (both having the target meeting conditions), return the ID of the more complete one.

Conditions may not be judged by one image alone and can be inaccurate, so the correct object may not meet all. Find the most likely object. If multiple seem correct, return the one you're most confident in. If no object meets conditions, make a guess. Usually, the correct object is visible in multiple images; return the one where it's most clearly and completely seen.

Your response should be a JSON object with:
{{
"reasoning": "your reasoning process" // Explain the process of how you identified and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
"target_image_id": 1, // Replace with the actual image ID (only one ID) annotated on the image that contains the target object most completely.
"white_images": [4, 6, 11] // This is a list that contains the IDs of the images that are either almost entirely white or have an extremely high brightness making them indiscernible to the human eye and were excluded from the analysis.
"reference_image_ids": [1, 2, ...] //  A list of IDs of images that you think also contain the target object.
}}
Here are the {num_view_selections} images for reference."""

apixuanran_prompt5 = """Imagine you're in a room tasked with finding a specific object. You alreday know the anchor object class: {anchorclass}, and target object class: {targetclass},  and the query that target object should match: {query}. The provided images are taken as the camera rotates 360 degrees around the anchor object.

Given images from an indoor - scanning video (during the 360 - degree rotation) and a query about an object in the room, analyze the images to locate the target object according to the query content.

You'll get multiple images with IDs in the top - left corner indicating their order in the video. Adjacent images have adjacent IDs. Note that multiple images are combined for space - saving. You'll also receive the parse version of query, stating the conditions target object need to satisfy.

After exclusion, find the image ID with the target object based on the given query content and conditions. Conditions may not be judged by one image alone and can be inaccurate, so the correct object may not meet all. Find the most likely object. If multiple seem correct, return the one you're most confident in. If no object meets query content, make a guess. Usually, the correct object is visible in multiple images; return the one where it's most clearly seen.

Your response should be a JSON object with:
{{
"reasoning": "your reasoning process" // Explain the process of how you identified and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
"target_image_id": 1, // Replace with the actual image ID (only one ID) annotated on the image that contains the target object most completely.
"reference_image_ids": [1, 2, ...] //  A list of IDs of images that you think also contain the target object.
"extended_description": "The target object is a red-colored box. It has a black stripe across the middle." //Describe the target object in the image with the target_image_id. Focus on its color, and mention any other features. Note that the description does not require stating any position of the object.
"extended_description_withposition": "The target object is a red-colored box located in the lower left corner of the image. " //Describe the target object in the image with the target_image_id. Focus on its color, and position of the target object in the image.
}}
Here are the {num_view_selections} images for reference.
Here is the condition for target object: {condition}
"""

apibbox_prompt33 = """Great! Here is the detailed version of your selected image. You already know the query: {query}, the anchor object: {anchorclass}, and the target object: {classtarget}. The picture you choose is taken by the camera when it is very close to the anchor object. So, even the anchor object may not appear in the picture, you should understand that the anchor object is located very close to the camera.
There are {num_candidate_bboxes} candidate objects shown in the image. I have annotated each object at the center with an object ID in white color text and black background. Additionally, you will be provided with an extended description that includes details such as the target object's color, its position in the picture, and other relevant features. Please find out which ID marked {classtarget} object in the picture exactly meets the description: {query} and the details in the extended description.
You need to think through three steps. First, the candidate objects provided to you are not necessarily all of the target object class. You must first identify which of the candidate objects are of the target object class. Second, cross - reference the identified target - class candidates with the content in the extended description. Third, after the first two steps of analysis, select the object ID among the found candidate IDs of the target object class that is most likely to satisfy the extended description and the query.
Here is the extended description: {description}
Reply using JSON format with two keys "reasoning" and "object_id" like this:
{{
"reasoning": "your reasons", // Explain the justification why you select the object ID. Describe your three - step thinking process, including how you identified the target object class candidates, how you cross - referenced them with the extended description, and how you made the final selection among them.
"object_id": 0 // The object ID you selected. Always give one object ID from the image, which you are the most confident of, even you think the image does not contain the correct object.
}}
"""

weitiao_prompt5 = """You are an excellent image analysis expert. Now, I will provide you with some images, and in each image, there is a target object enclosed by a purple bounding box. Your task is to find the ID of the image in which the purple bounding box most precisely frames the target object that conforms to the given description. You already know the specific description: {description} and the category of the target object: {target}.
Each image has an ID marked in the upper left corner. Please note that, to save space, sometimes multiple images are combined and displayed together. Also, you need to exclude certain images. For these images, the purple bounding box merely encloses most or all of the content of the image, instead of precisely framing the target object that conforms to the description. After excluding these images, based on the given description: {description}, especially considering the object location in image which description mentions, find the ID of the image in which the purple bounding box most precisely frames the target object that matches to the given description: {description}.
Your response should be a JSON object containing the following content:
{{
"reasoning_process": "Your reasoning process" // Explain how you evaluated whether the purple bounding box in each image precisely frames the target object based on the description {description}.
"object_id": 1, // Replace it with the actual image ID (only one ID), and the image corresponding to this ID most accurately contains the target object that conforms to the description within the purple bounding box.
}}
Here are {num_candidate_bboxes} images for your reference."""

weitiao_prompt8 = """You are an excellent image analysis expert. Now, I will provide you with some images, and in each image, there is a target object enclosed by a purple bounding box. In addition, I will also provide you with a reference image at the last. In the reference image, there is also a target object enclosed by a green bounding box, and the word "refer" is marked in red in the upper left corner of the reference image. Your task is to find the ID of an image in which the purple bounding box most precisely encloses the target object that conforms to the given description. You already know that the specific description is: {description} and the category of the target object is: {target}.
Each image has an ID marked in the upper left corner. Please note that, to save space, sometimes multiple images are combined and displayed together. Also, you need to exclude certain images. For these images, the purple bounding box merely encloses most or all of the content of the image, rather than precisely enclosing the target object that conforms to the description. After excluding these images, based on the specified description and the target object enclosed by the purple bounding box in the reference image (please note that the target objects enclosed in the reference image and the candidate images may be the same object but from different perspectives), find the ID of an image in which the purple bounding box most precisely encloses the target object that not only conforms to the given description but is also the same object as the target object enclosed in the reference image.
Your response should be a JSON object containing the following content:
{{
"reasoning_process": "Your reasoning process" // Explain how you evaluate whether the purple bounding box in each image precisely encloses the target object based on the description and the target object enclosed in the reference image.
"object_id": 1, // Replace it with the actual image ID (only one ID), and the image corresponding to this ID most accurately contains the target object that conforms to the description and is also the same object as the target object enclosed in the reference image within the purple bounding box. If no image ID that meets the requirements is found, return -1
}}
Here are {num_candidate_bboxes} images and one reference image for your selection."""

#"extended_description_withposition": "The target object is a red-colored box close to a chair. It has a black stripe across the middle and is in the lower left corner of the image. " //Describe the target object in the image with the target_image_id. Focus on its color, and mention any other features such as position of the target object in the image or other features which can help to uniquely ensure the detection of this object in the image.

weitiao_prompt7 = """You are an excellent image analysis expert. Now, I will provide you with some images that have an ID marked in the upper left corner. These images are all taken by rotating around the target object {target} framed by a green border in the reference image. I will also provide you with the reference image. In the reference image, there is a target object {target} enclosed by a green bounding box, and the word "refer" is marked in red in the upper left corner of the reference image. Your task is to find out which three images among those with IDs marked in the upper left corner capture the target object {target} framed in the reference image most clearly (at most four images can be selected), and return the IDs of these images. Please note that, in order to save space, sometimes multiple images are combined and displayed together.
Your response should be a JSON object containing the following content:
{{
"reasoning_process": "Your reasoning process" // Explain how you select the images that capture the target object framed in the reference image most clearly and completely.
"image_ids": [2, 4, 5, 7], // Replace them with the actual image IDs. Return the four IDs of the images that, in your opinion, capture the target object framed in the reference image most clearly and completely among all the images. 
}}
There are {num_images} images and one reference image for you to choose from.
"""


weitiao_prompt6 = """You are an excellent image analysis expert. Now, I will provide you with some images. In each image, there is an object enclosed by a red bounding box, and an ID is marked in the upper left corner of each image. Please note that, to save space, sometimes multiple images are combined and displayed together. In addition, I will also provide you with a reference image at the end. In the reference image, there is a target object {target} enclosed by a green bounding box, and the word "refer" is marked in red in the upper left corner of the reference image. Your task is to find the ID of an image in which the red bounding box can most precisely enclose an object that is consistent with the target object enclosed by the green bounding box in the reference image.
Your response should be a JSON object containing the following content:
{{
"reasoning_process": "Your reasoning process" // Explain how you evaluate whether the red bounding box in each image precisely encloses the same object based on the target object enclosed in the reference image.
"object_id": 1, // Replace it with the actual image ID (only one ID). The red bounding box in the image corresponding to this ID most accurately contains the most similiar object as the target object enclosed by the green bounding box in the reference image. If no image ID that meets the requirements is found, return -1.
}}
Here are {num_candidate_bboxes} images and one reference image for your selection."""



bijiao_prompt6 = """You are an excellent image analysis expert. Now, I will first provide you with an image. In this image, there is an object enclosed by a red bounding box. In addition, I will also provide you with a reference image at the end. In the reference image, there is a target object {target} enclosed by a green bounding box, and the word "refer" is marked in red in the upper left corner of the reference image. Your task is to determine whether the red bounding box in the first image encloses an object that is very similar to the target object enclosed by the green bounding box in the reference image. The judgment mainly involves comparing the color, shape, category, and other characteristics.
Your response should be a JSON object containing the following content:
{{
"reasoning_process": "Your reasoning process" // Explain how you determine whether the red bounding box in the first image encloses an object that is very similar to the target object enclosed in the reference image.
"correct": True or False, // If you determine that the red bounding box in the first image contains an object that is very similar to the target object enclosed by the green bounding box in the reference image, then return True. Otherwise, return False.
}}"""


xuyao_prompt9 = """You are an excellent image analysis expert. Now, I will first provide you with some images. In these images, there is a target object {target} enclosed by a red bounding box. Note that, to save space, I may splice these several pictures together. Your task is to determine whether the objects enclosed by the red bounding boxes in these images only capture the upper part of the objects. For example, when you see that the objects enclosed by the red bounding boxes in these several photos are a table, but in these three photos, only the upper surface of the table is captured. That is, only the tabletop is captured without obviously capturing the lower part of the table, such as the table legs. Then you should determine that these pictures only capture the upper part of the table.
Your response should be a JSON object containing the following content:
{{
"reasoning_process": "Your reasoning process" // Explain how you determine whether the objects enclosed by the red bounding boxes in these images only capture the upper part of the target objects.
"answer": True or False, // If you determine that the objects enclosed by the red bounding boxes in these images only capture the upper part of the target objects, then return "True". Otherwise, return "False".
}}"""

apibbox_prompt3 = """Great! Here is the detailed version of the picture you've selected. There are {num_candidate_bboxes} candidate objects shown in the picture. I have annotated an object ID at the center of each object with white text on a black background. You already know the query content: {query}, the anchor object: {anchorclass}, and the target object: {classtarget}. In addition, you will be provided with an extended description: {description}, which includes the position of the target object in the picture. Please find out which ID among the marked ones corresponds to the {classtarget} object that exactly matches the query content and the content in the extended description.
You need to think through two steps. First, the candidate objects provided to you are not necessarily all of the target object class {classtarget}. You must first determine which of the candidate objects belong to the target object class {classtarget}. Second, after the analysis of the first two steps, select the object from the found candidate IDs of the target object class that is most likely to satisfy the extended description and the query content.
Please reply in JSON format with two keys "reasoning" and "object_id" as follows:
{{
"reasoning": "Your reasoning processing", // Explain the reasons why you select the object ID. Describe your three-step thinking process, including how you identified the candidate objects of the target object class, how you cross-referenced them with the extended description, and how you made the final selection among these candidate objects.
"object_id": 0 // The object ID you select. Always provide one object ID from the picture that you are most certain of, even if you think the picture does not contain the correct object.
}}
"""

same_prompt2 = """You are an agent who is very good at looking at pictures. Now I'm giving you two pictures. You need to determine whether the shooting content of these two pictures is the same.
Reply using JSON format with two keys "reasoning", "images_same_or_not" like this:
{{
"reasoning": "your reasons", // Explain the justification why you think the images are the same, or why the images are not the same.
"images_same_or_not": True, // true if you think the shooting content of the two images is the same, false if you think the shooting content of the two images is different
}}"""

same_prompt1 = """You are an intelligent assistant who is extremely proficient in examining images. You already know the target object category: {target_class}. Now I will provide you with two images. You need to determine whether the target objects captured in these two images are in the exact same position. Since these two images are taken from the same pose, you only need to check whether the target objects are in the same position within the images. (For example, if the target object is a table and you can clearly see that the table is located in the middle of both images, then the target objects captured in these two images are in the same position.)
Please reply in JSON format with two keys, "Reasoning Process" and "Whether the Contents of the Images are the Same", in the following format:
{{
"reasoning": "Your reasons", // Explain the basis for your judgment on whether the target objects captured in these two images are in the same position.
"images_same_or_not": true, // It should be true if you think the target objects captured in the two images are in the same position. If you find that the positions of the target objects captured in the two images are different, or if the target object is captured in the first image but not in the second, then it should be false.
}}
"""


apixuanran_promptdong = """Imagine that you are in a room and tasked with finding a specific object. You already know the query content: {query}, the anchor object class: {anchorclass}, and the target object class: {targetclass}. The provided images are taken as the camera rotates 360 degrees around a certain point. Your task is to analyze these images to locate the target object described in the query.

You will receive multiple images, each with an ID marked in the upper left corner indicating their order in the video. Adjacent images have adjacent IDs. Please note that, to save space, multiple images are combined and displayed together. You will also receive the query statement and its parsed version, which specifies the target class and conditions. Find the image ID that contains the target object meeting the requirements according to the given query statement and conditions.

The conditions may not be judged by just one image and may be inaccurate, so the correct object may not meet all the conditions. Try to find the object that is most likely to meet the conditions. If multiple objects seem to be correct, return the image ID corresponding to the object you are most certain about. If no object meets the conditions, make a guess. Consider the images from multiple perspectives taken during the 360-degree rotation. If you find an image containing the target, also check other images. If another image captures the target more completely, return the ID of the more complete image. Return the ID of the image where the object can be seen most clearly and completely.

Your response should be a JSON object containing the following content:
{{
    "reasoning": "Your reasoning process" // Explain the process of how you identified and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
    "target_image_id": 1, // Replace it with the actual image ID (only one ID), which is marked on the image that contains the target object most clearly.
    "reference_image_ids": [1, 2, ...] // A list of image IDs that you think also contain the target object.
    "extended_description": "The target object is a red box. It has a black stripe in the middle." // Describe the target object in the image with the target_image_id. Focus on its color and mention any other features. Note that the description does not require stating the position of the object.
    "extended_description_withposition": "The target object is a red box located in the lower left corner of the image." // Describe the target object in the image with the target_image_id. Focus on its color and the position of the target object in the image.
}}
Here are {num_view_selections} images for your reference.
Here is the condition for target object: {condition}"""

juedingmaoandtar_prompt = """Imagine that you are in a room and tasked with finding a specific object. You already know the query content: {query}, the anchor object class: {anchorclass}, and the target object class: {targetclass}. These provided images are obtained by frame extraction from a video. Your task is to analyze these images to locate the target object described in the query.

You will receive multiple images, each with an ID marked in the upper left corner to indicate its order in the video. Adjacent images have adjacent IDs. Please note that, to save space, multiple images are combined and displayed together. You will also receive the query statement and its parsed version, which specifies the target object class and conditions. Your task is mainly divided into two steps. In the first step, based on the given query statement and conditions, determine whether there are any images among the provided ones that contain the target object meeting the requirements. If not found, return the image ID as -1. In the second step, if you can find the image that contains the target object meeting the requirements, then return the ID of the image that contains the target object most clearly.

Please note that the query statement and conditions I provided may not be judged by just one image, and they may not be accurate. So the correct object may not meet all the conditions. Try your best to find the object that is most likely to meet these conditions. If multiple objects seem to meet the conditions, then return the ID of the image corresponding to the object you are most certain about.

Your response should be a JSON object containing the following content:
{{
"reasoning": "Your reasoning process" // Explain the process of how you judged and located the target object. If reasoning across different images is needed, explain which images were used and how you reasoned with them.
"find_or_not": True  // Return true if you judge that there is a suitable image containing the target object that matches the query content. Return false if you can't find any image that meets the requirements.
"target_image_id": 4, // If you judge that there is a suitable image containing the target object that matches the query content and conditions, replace it with the actual image ID (only one ID), which is marked on the image that contains the target object most clearly. Return -1 if no image meets the requirements.
"anchor_image_id": 6 // Please return the ID of the image in which you can see the anchor object most clearly among these images.
"extended_description": "The target object is a red box located in the lower left corner of the image." // Describe the target object in the image with the target_image_id. Focus on its color and the position of the target object in the image.
"unique_question": 6, // This is an independent question. Regardless of any other factors, only look for which image among all those provided captures the object {targetclass} most clearly. If none is found, return -1.
}}
Here are {num_view_selections} images for your reference.
The following are the conditions for the target object: {condition}"""

juedingqumem_prompt = """You are an intelligent assistant proficient in analyzing images. Given a series of indoor room images from a video, you need to analyze these images and select some of them. Each image has an ID in the upper left corner indicating its sequence in the video. Multiple images may be combined and displayed together to save space.
The anchor object is {anchor_class}. If there are some images that are very similar, only select the clearest one to participate in the further selection process.
Please reply in json format, including "reasoning" and "selected_image_ids":
{{
"reasoning": "Your reasoning process" // Your thinking process regarding the selection task
"selected_image_idsback": ["00045"] or [] // Find which image among all those with an ID after {center_id} captures the object {anchor_class} most clearly. If none is found, return empty list. Note that the returned ID should be in the form of "00045", not "00045.color", and do not add any suffix after the numbers.
"selected_image_idsfront": ["00010"] or [] // Find which image among all those with an ID before {center_id} captures the object {anchor_class} most clearly. If none is found, return empty list. Note that the returned ID should be in the form of "00045", not "00045.color", and do not add any suffix after the numbers.
"unique_questionback": 45 or -1 // This is an independent question. Regardless of any other factors, only look for which image among all those with an ID after {center_id} captures the object {targetclass} most clearly. If none is found, return -1.
"unique_questionfront": 6 or -1 // This is an independent question. Regardless of any other factors, only look for which image among all those with an ID before {center_id} captures the object {targetclass} most clearly. If none is found, return -1.
}}
Now start the task:
There are {num_view_selections} images for you to select from.
"""



updownmaotar_prompt = """Imagine that you are in a room with the task of finding some specific objects. You already know the query content: {query}, the anchor object category: {anchorclass}, and the target object category: {targetclass}. These provided images are frames extracted from a video that rotates around a certain point. You will receive multiple images, each with an ID marked in the top - left corner to indicate its sequence in the video. Adjacent images have adjacent IDs. Note that, to save space, multiple images will be combined and displayed together. You will also receive the query statement and its parsed version, which clearly define the target object category as well as the anchor object category and conditions.
Your task mainly consists of three steps. In the first step: Based on the given anchor object category, determine whether there are images among the provided ones that clearly capture the anchor object. If there are, proceed to the second step; otherwise, return -1 directly. The second step: Since there are such images, give the smallest image ID (min_ID) that clearly captures the anchor object. The third step: You need to try to find an image that contains the target object meeting the requirements according to the query statement and conditions among the images with IDs from 0 to min_ID, and this photo should clearly capture the target object. If you can find it, then return the corresponding image ID; if not, return target_image as -1. Please note that the query statement and conditions I provided may not be determinable by just one image and may not be accurate. So the correct object may not meet all the conditions. Do your best to find the object that is most likely to meet these conditions. If multiple objects seem to meet the conditions, then return the image ID of the object you are most certain about. (Here is an example. In the first step, you find that images 12, 13, 14, and 15 all clearly capture the anchor object, so you proceed to the second step. In the second step, ID 12 is the smallest, so min_ID is 12. In the third step, you find that among the images with IDs from 0 to 13, no image captures the target object that meets the query statement and conditions, so return target_image_id as -1.)
Your response should be a JSON object containing the following:
{{
"reasoning": "Your reasoning process" // Explain the reasoning process of the three steps in your task reasoning. If you need to reason by integrating different images, explain which images were used and how you reasoned about them.
"anchor_image_id": 12 // Return the smallest image ID that clearly captures the anchor object. If you cannot find any image that captures the anchor object, return -1.
"target_image_id": 4, // If anchor_image_id is -1, then return -1 directly. Otherwise, try to find an image that contains the target object meeting the requirements according to the query statement and conditions among the images with IDs from 0 to anchor_image_id. If you can find it, then return the corresponding image ID; if not, return target_image as -1.
"extended_description": "The target object is a red box located in the lower - left corner of the image." // If target_image_id is -1, then return None directly. Otherwise, in the third step you can find the image ID of the target object that meets the requirements, describe the target object in the image (whose ID is target_image_id). Focus on describing its color and the position of the target object in the image.
"unique_question": 6, // This is an independent question. Regardless of any other factors, only look for which image among all those provided captures the object {targetclass} most clearly. If none is found, return -1.
}}
There are {num_view_selections} images for your reference.
Here are the conditions for the target object: {condition}"""

zhijiexuan_prompt = """Imagine you are in a room tasked with finding a certain object. You already know the query content: {query}, and the target object category: {targetclass}. These provided images are frames extracted from a video rotating around a specific point. You will receive multiple images, each marked with an ID in the top-left corner to indicate its sequence in the video. Adjacent images have consecutive IDs. Note that multiple images are combined and displayed together to save space.
Your task consists of two main steps:
Step 1: Locate an image that contains the target object meeting the requirements of the query statement and conditions, and the image must clearly and completely capture the target object. If found, directly return the corresponding image ID without proceeding to Step 2. If not found, proceed to Step 2.
Step 2: If no qualifying image is found in Step 1, ignore all query content requirements in this step. Directly check all provided images to see if any clearly capture the object {targetclass}. If such an image is found, return its ID; otherwise, return -1.
Your response should be a JSON object containing the following:
{{
"reasoning": "Your reasoning process" // Explain the reasoning for both steps of your task.
"match_query_id": 12 // Return the image ID that meets the query requirements in Step 1; return -1 if not found.
"object_image_id": 4, // If Step 1 is successful, directly return -1. If Step 1 fails, return the image ID that clearly captures the object {targetclass} in Step 2; return -1 if not found.
"extended_description": "The target object is a red box located in the lower-left corner of the image." // Describe the target object in the found image (by ID), focusing on its color and position in the image.
}}
There are {num_view_selections} images for your reference.
"""

xianju_prompt = """Suppose you are an image expert who is very good at analysis. Now I will provide you with the category of a target object {targetclass} and a picture. You need to make judgments according to the following two-step rules:
Step1: If the category of the target object is a relatively large object, such as a bed, a sofa, a closet, a cabinet, a shelf etc., directly return True and do not proceed to the second step.
Step2: Judge according to the picture I provided whether the target object in the picture is photographed relatively completely. If it is complete, return False; if you feel it is not photographed completely, return True.
Return the result in JSON format as follows:
{{
"reasoning": "Your reasoning process" // Explain the reasoning process for your task.
"limit": False // If you determine in the first step that the target object is a large object, then directly return True and do not proceed to the second step. If you proceed to the second step and find that the target object is almost completely captured in the image, then return False. Otherwise, return True.
}}
"""

xianju_prompt1 = """Suppose you are an image expert who is very good at analysis. Now I will provide you with the category of a target object {targetclass} and a picture. You need to make judgments according to the following rule:
Judge according to the picture I provided whether the target object in the picture is photographed relatively completely. If it is complete, return False; if you feel it is not photographed completely, return True.
Return the result in JSON format as follows:
{{
"reasoning": "Your reasoning process" // Explain the reasoning process for your task.
"limit": False // If you find that the target object is almost completely captured in the image, then return False. Otherwise, return True.
}}
"""


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