import os
import json

# 定义文件路径
rumor_path = 'F:/project/毕业设计/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost/'
non_rumor_path = 'F:/project/毕业设计/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost/'
ori_microblog_path = 'F:/project/毕业设计/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/'
data_path = 'F:/project/毕业设计/rumor_data.txt'

rumor_class = os.listdir(rumor_path)
non_rumor_class = os.listdir(non_rumor_path)

rumor_label = "0"
non_rumor_label = "1"

total_non_rumor_list, non_rumor_num = [], 0
total_rumor_list, rumor_num = [], 0

# 遍历谣言并解析
for rumor_class_idx in rumor_class:
    if rumor_class_idx != '.DS_Store':
        try:
            with open(os.path.join(ori_microblog_path, rumor_class_idx), 'r', encoding='UTF-8') as json_file:
                content = json.load(json_file)
                total_rumor_list.append(rumor_label + '\t' + content["text"].strip() + '\n')
                rumor_num += 1
        except FileNotFoundError:
            print("文件不存在：", rumor_class_idx)

# 遍历非谣言并解析
for non_rumor_class_idx in non_rumor_class:
    if non_rumor_class_idx != '.DS_Store':
        try:
            with open(os.path.join(ori_microblog_path, non_rumor_class_idx), 'r', encoding='UTF-8') as json_file:
                content = json.load(json_file)
                total_non_rumor_list.append(non_rumor_label + '\t' + content["text"].strip() + '\n')
                non_rumor_num += 1
        except FileNotFoundError:
            print("文件不存在：", non_rumor_class_idx)

# 写入数据集到文件
data_list = total_rumor_list + total_non_rumor_list
with open(data_path, 'w', encoding='UTF-8') as f:
    for data_idx in data_list:
        f.write(data_idx)

print(f"谣言数量为:{rumor_num}")
print(f"非谣言数量为:{non_rumor_num}")
print("rumor_data.txt输入完成")