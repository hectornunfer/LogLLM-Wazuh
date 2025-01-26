import os
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from model import LogLLM
from customDataset import CustomDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

max_content_len = 100
max_seq_len = 128
dataset_name = 'HDFS_v1'   # 'Thunderbird' 'HDFS_v1'  'BGL'  'Liberty‘
data_path = r'/content/{}/train.csv'.format(dataset_name)

Bert_path = r"google-bert/bert-base-uncased"
Llama_path = r"meta-llama/Meta-Llama-3-8B"

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

device = torch.device("cuda:0")

print(
f'dataset_name: {dataset_name}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'device: {device}')


def evalModel(model, dataset):
    model.eval()
    pre = 0

    preds = []

    with torch.no_grad():
        for i in range(len(dataset)):
            this_batch_seq, _ = dataset.get_batch(i)
            outputs_id = model(this_batch_seq)
            output = model.Llama_tokenizer.batch_decode(outputs_id)
            matches = re.findall(r' (.*?)\.<|end_of_text|>', output)
            if len(matches) > 0:
                preds.append(matches[0])
                print(dataset.get_batch(i) + ': ' + matches[0])
            else:
                preds.append('')

if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path)
    model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)
    evalModel(model, dataset)