import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import re
patterns = [
    r'[a-zA-Z0-9]*:*([/\\]+[^/\\\s]+)+[/\\]*',  # 文件路径
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # 中间一定要有数字  数字和字母和 . 或 : 或 - 的组合
    # r'[a-zA-Z0-9]+\.[a-zA-Z0-9]+',
]

# 合并所有模式
combined_pattern = '|'.join(patterns)

# 替换函数
def replace_patterns(text):
    return re.sub(combined_pattern, '<*>', text)


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        contents = df['Content'].apply(replace_patterns).values     #pre processing
        # contents = df['EventTemplate'].values
        self.sequences = np.array([content.split(' ;-; ') for content in contents], dtype=object)
        self.labels = df['Label'].values

        self.num_normal = (self.labels == 0).sum()
        self.num_anomalous = (self.labels == 1).sum()

        self.normal_weight = max(self.num_anomalous / self.num_normal,1)
        self.anomalous_weight = max(self.num_normal / self.num_anomalous,1)

        if self.num_normal >  self.num_anomalous:
            self.less_indexes = np.where(self.labels == 1)[0]
            self.num_majority = self.num_normal
            self.num_less = self.num_anomalous
        else:
            self.less_indexes = np.where(self.labels == 0)[0]
            self.num_majority = self.num_anomalous
            self.num_less = self.num_normal



    def __len__(self):
        return len(self.labels)

    def get_batch(self, indexes):
        this_batch_seqs = self.sequences[indexes]
        temp =  self.labels[indexes]
        this_batch_labels = temp.astype(object)
        this_batch_labels[temp == 0] = 'normal'
        this_batch_labels[temp == 1] = 'anomalous'
        return this_batch_seqs, this_batch_labels

    def get_label(self):
        return self.labels
