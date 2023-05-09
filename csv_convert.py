import csv
import numpy as np
import os
import pandas as pd

def folder_create():
    folder_name = "datas"

    if not os.path.isdir(folder_name):
        os.makedirs('datas')
        path1 = os.path.join('datas', 'train')
        path2 = os.path.join('datas', 'test')
        os.makedirs(path1)
        os.makedirs(path2)

    else:
        print(f"Folder '{folder_name}' already exists.")

def write_to_csv(data):


    folder_create()
    filename = "datas/train/" + input("Please enter the file name:") + ".csv"
    label = input("Please enter the label:")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['nose_x','nose_y','nose_score','left_eye_x','left_eye_y','left_eye_score','right_eye_x','right_eye_y','right_eye_score','left_ear_x','left_ear_y','left_ear_score','right_ear_x','right_ear_y','right_ear_score','left_shoulder_x','left_shoulder_y','left_shoulder_score','right_shoulder_x','right_shoulder_y','right_shoulder_score','left_elbow_x','left_elbow_y','left_elbow_score','right_elbow_x','right_elbow_y','right_elbow_score','left_wrist_x','left_wrist_y','left_wrist_score','right_wrist_x','right_wrist_y','right_wrist_score','left_hip_x','left_hip_y','left_hip_score','right_hip_x','right_hip_y','right_hip_score','left_knee_x','left_knee_y','left_knee_score','right_knee_x','right_knee_y','right_knee_score','left_ankle_x','left_ankle_y','left_ankle_score','right_ankle_x','right_ankle_y','right_ankle_score','class_no'
])  # 写入表头
        for row in data:
            row = np.append(row, label)
            writer.writerow(row)
        f.close()
    if input("Do you want to merge all datas?(y/n)") == 'y':
        merge_csv()

def merge_csv():
    folder_path = "datas/train/"
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

   
    dfs = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # 将所有 DataFrame 合并为一个，并保存到 CSV 文件
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv("datas/train/train.csv", index=False)
# data =np.random.rand(1,51)
# write_to_csv(data)
#merge_csv()
