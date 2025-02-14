import os.path
import os
import re
from pathlib import Path
import torch
from tqdm import tqdm
from model import LogLLM
from myCustomDataset import CustomDataset
import numpy as np
import pandas as pd
from prepareData.helper import structure_log,fixedSize_window

#### for Thunderbird, Liberty, BGL

dataset_name = 'Thunderbird'   # 'Thunderbird' 'HDFS_v1'  'BGL'  'Liberty‘
data_path = r'/content/{}/thunderbird_prueba.log'.format(dataset_name)
data_path_1 = r'/content/{}/formatted.csv'.format(dataset_name)


Bert_path = r"google-bert/bert-base-uncased"
Llama_path = r"meta-llama/Meta-Llama-3-8B"

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))


data_dir = r'/content/Thunderbird'
log_name = "thunderbird_prueba.log"

start_line = 0
end_line = None


output_dir = data_dir

if __name__ == '__main__':
    window_size = 20
    step_size = 15

    if 'thunderbird' in log_name.lower() or 'spirit' in log_name.lower() or 'liberty' in log_name.lower():
        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'   #thunderbird  , spirit, liberty
    elif 'bgl' in log_name.lower():
        log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'  #bgl
    else:
        raise Exception('missing valid log format')
    print(f'Auto log_format: {log_format}')

    structure_log(data_dir, output_dir, log_name, log_format, start_line = start_line, end_line = end_line)

    df = pd.read_csv(os.path.join(output_dir,f'{log_name}_structured.csv'))

    print(len(df))
    
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

    df_aux = df.reset_index(drop=True)

    print('Start grouping.')

    session_df = fixedSize_window(
        df_aux[['Content', 'Label']],
        window_size=window_size, step_size=step_size
    )

    col = ['Content', 'Label','item_Label']
    spliter=' ;-; '

    session_df = session_df[col]
    session_df['session_length'] = session_df["Content"].apply(len)
    session_df["Content"] = session_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_len = session_df['session_length'].mean()
    max_session_len = session_df['session_length'].max()
    num_anomalous= session_df['Label'].sum()
    num_normal = len(session_df['Label']) - session_df['Label'].sum()

    session_df.to_csv(os.path.join(output_dir, 'formatted.csv'),index=False)

    print('Dataset info:')
    print(f"max session length: {max_session_len}; mean session length: {mean_session_len}\n")
    print(f"number of anomalous sessions: {num_anomalous}; number of normal sessions: {num_normal}; number of total sessions: {len(session_df['Label'])}\n")

    with open(os.path.join(output_dir, 'formatted_info.txt'), 'w') as file:
        file.write(f"max session length: {max_session_len}; mean session length: {mean_session_len}\n")
        file.write(f"number of anomalous sessions: {num_anomalous}; number of normal sessions: {num_normal}; number of total sessions: {len(session_df['Label'])}\n")
    
    dataset = CustomDataset(data_path_1)
    model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False)
    model.eval()

    batch_size = 20  # Tamaño del lote
    logs = df["Content"].tolist()  # Convertir la columna "Content" a una lista
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
            outputs_ids = model(this_batch_seqs)  # Pasar el lote al modelo
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids)
            for text in outputs:
                matches = re.findall(r' (.*?)\.<|end_of_text|>', text)
                if len(matches) > 0:
                    preds.append(matches[0])
                else:
                    preds.append('')

    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy,dtype=int)
    preds[preds_copy == 'anomalous'] = 1
    preds[preds_copy != 'anomalous'] = 0
    gt = dataset.get_label()
    print(preds_copy)