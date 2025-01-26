import os
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from model import LogLLM
from myDataset import myDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

max_content_len = 100
max_seq_len = 128
batch_size = 32
dataset_name = 'Thunderbird'   # 'Thunderbird' 'HDFS_v1'  'BGL'  'Liberty‘
data_path = r'/content/{}/formatted.csv'.format(dataset_name)

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
        indexes = [i for i in range(len(dataset))]
        for bathc_i in tqdm(range(batch_size, len(indexes) + batch_size, batch_size)):
            if bathc_i <= len(indexes):
                this_batch_indexes = list(range(pre, bathc_i))
            else:
                this_batch_indexes = list(range(pre, len(indexes)))
            pre = bathc_i

            this_batch_seqs, _ = dataset.get_batch(this_batch_indexes)
            outputs_ids = model(this_batch_seqs)
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids)

            # print(outputs)

            for text in outputs:
                matches = re.findall(r' (.*?)\.<|end_of_text|>', text)
                if len(matches) > 0:
                    preds.append(matches[0])
                else:
                    preds.append('')
            
            print(dataset.get_batch(this_batch_indexes) + ':' + preds[0])

if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = myDataset(data_path)
    model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)
    evalModel(model, dataset)