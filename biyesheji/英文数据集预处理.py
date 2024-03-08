import os
import json

"""提取数据集"""
politifact_fake_path = 'F:/WorkProject/Pythonproject/毕业设计/FakeNewsNet-master/Data/PolitiFact/FakeNewsContent'
politifact_real_path = 'F:/WorkProject/PythonProject/毕业设计/FakeNewsNet-master/Data/PolitiFact/RealNewsContent'
buzzfeed_fake_path = 'F:/WorkProject/PythonProject/毕业设计/FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent'
buzzfeed_real_path = 'F:/WorkProject/PythonProject/毕业设计/FakeNewsNet-master/Data/BuzzFeed/RealNewsContent'

fake_news_paths = [politifact_fake_path, buzzfeed_fake_path]  # 提取谣言数据集路径列表
real_news_paths = [politifact_real_path, buzzfeed_real_path]  # 提取非谣言数据集路径列表

fake_label = "0"
real_label = "1"

total_fake_list, fake_num = [], 0
total_real_list, real_num = [], 0

for news_path in fake_news_paths:  # 遍历虚假新闻并解析
    for root, dirs, files in os.walk(news_path):
        for idx in files:
            with open(os.path.join(root, idx), 'r', encoding='UTF-8') as json_file:

                content = json_file.read()
                text = json.loads(content)["text"].replace('\n', '').replace("\r", '')
                total_fake_list.append(f"{fake_label}\t{text.strip().lower()}\n")
                fake_num += 1

for news_path in real_news_paths:  # 遍历真实新闻并解析
    for root, dirs, files in os.walk(news_path):
        for idx in files:
            with open(os.path.join(root, idx), 'r', encoding='UTF-8') as json_file:
                content = json_file.read()
                text = json.loads(content)["text"].replace('\r', '').replace('\n', '')
                total_real_list.append(f"{real_label}\t{text.strip().lower()}\n")
                real_num += 1

data_path = 'F:/WorkProject/PythonProject/毕业设计/news_data.txt'
data_list = total_real_list + total_fake_list
with open(data_path, 'w', encoding='UTF-8') as f:
    for data_idx in data_list:
        f.write(data_idx)

print(f"虚假新闻数量为:{fake_num}")
print(f"真实新闻数量为:{real_num}")
print("news_data.txt输入完成")
