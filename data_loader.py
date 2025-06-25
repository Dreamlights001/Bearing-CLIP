# data_loader.py
import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 提取文件名中的标签
def get_label_from_filename(filename):
    # 例如: health20_0_train.csv -> health
    # inner_race_fault30_1.csv -> inner_race_fault
    match = re.match(r"([a-zA-Z_]+)\d+_\d+_(train|test)\.csv", filename)
    if match:
        return match.group(1).replace('_', ' ').strip()  # 'inner_race_fault' -> 'inner race fault'
    return None


class BearingDataset(Dataset):
    def __init__(self, data_dir, prompt_template="A {} bearing"):
        super().__init__()
        self.data_dir = data_dir
        self.prompt_template = prompt_template
        self.samples = []
        self.labels = []

        # 扫描目录，加载所有样本
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                label_name = get_label_from_filename(filename)
                if label_name:
                    file_path = os.path.join(data_dir, filename)
                    df = pd.read_csv(file_path, header=None)
                    # 每一列是一个样本
                    for col in df.columns:
                        # 将一列数据（1024个点）作为一个样本
                        self.samples.append(df[col].values)
                        self.labels.append(label_name)

        # 创建标签到索引的映射，以及文本提示
        self.unique_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.prompts = [self.prompt_template.format(label) for label in self.unique_labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取振动数据和标签索引
        vibration_data = self.samples[idx].astype(np.float32)
        label_name = self.labels[idx]
        label_idx = self.label_to_idx[label_name]

        # 增加一个通道维度，以适应Conv1D的输入 (N, C, L)
        vibration_tensor = torch.from_numpy(vibration_data).unsqueeze(0)  # Shape: (1, 1024)

        return vibration_tensor, label_idx


if __name__ == '__main__':
    # 测试数据加载器
    train_dataset = BearingDataset(data_dir='./bearingset/train_set/')
    test_dataset = BearingDataset(data_dir='./bearingset/test_set/')

    print(f"训练样本总数: {len(train_dataset)}")
    print(f"测试样本总数: {len(test_dataset)}")
    print("-----")
    print(f"发现的标签: {train_dataset.unique_labels}")
    print(f"标签到索引的映射: {train_dataset.label_to_idx}")
    print(f"生成的文本提示 (Prompts): {train_dataset.prompts}")
    print("-----")

    # 获取一个样本
    vibration, label = train_dataset[0]
    print(f"样本振动数据形状: {vibration.shape}")  # torch.Size([1, 1024])
    print(f"样本标签索引: {label}")