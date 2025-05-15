import pandas as pd
import json
import os
import random
import ast
json_file = "3RScan.json"
with open(json_file, 'r') as file:
    data = json.load(file)

sample_num = 1
df = pd.read_csv('3rscan.csv')
data1 = []
for item in data:
    if item["type"] == "train" :
        data1.append(item)
#for i in range(sample_num):
print(len(data1))


#filtered_df = df[(df['scan_id'] == "scene0264_00") & (df['target_id'] == 25)]
exist = []

t = 0
while t < 1:
    random_scene0 = random.choice(data1)
    random_initscan = random_scene0["reference"]
    A = random_scene0["ambiguity"]
    random_scene1 = random.choice(random_scene0["scans"])
    random_rescan = random_scene1["reference"]
    if len(list(random_scene1["rigid"]))==0:

        continue
    random_remove_id = random.choice(random_scene1["rigid"])["instance_reference"]
    filtered_df0 = df[(df['scan_id'] == random_initscan) & (df['target_id'] == random_remove_id)]
    filtered_df1 = df[(df['scan_id'] == random_rescan) & (df['target_id'] == random_remove_id)]
    if (not filtered_df0.empty) and (not filtered_df1.empty):
        filtered_df0_ran = filtered_df0.sample(1)
        filtered_df1_ran = filtered_df1.sample(1)
    else: continue
    print(ast.literal_eval(filtered_df0_ran["distractor_ids"].iloc[0]))
    if len(ast.literal_eval(filtered_df0_ran["distractor_ids"].iloc[0]))==0:

        continue
    if len(ast.literal_eval(filtered_df1_ran["distractor_ids"].iloc[0]))==0:

        continue
    #print(type(random_remove_id))
    str1 = random_initscan + "*" + random_rescan + "*" + str(random_remove_id)
    #print(str1)
    if not os.path.exists('today_5.csv'):
        filtered_df1_ran.to_csv('today_5.csv', mode='a', index=False, header=True)
        exist.append(str1)
    else:#
        #dfceshi1 = pd.read_csv('today_250.csv')
        #dfceshi0 = pd.read_csv('yesterday_250.csv')
        if str not in exist:
            filtered_df1_ran.to_csv('today_5.csv', mode='a', index=False, header=False)
            exist.append(str1)
        else: continue
    if not os.path.exists('yesterday_5.csv'):
        filtered_df0_ran.to_csv('yesterday_5.csv', mode='a', index=False, header=True)
    else:
        filtered_df0_ran.to_csv('yesterday_5.csv', mode='a', index=False, header=False)

    t += 1
print(exist)
