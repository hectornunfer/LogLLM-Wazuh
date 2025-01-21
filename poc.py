import os
import re
from pathlib import Path
import torch
from model import LogLLM
from customDataset import replace_patterns
import numpy as np

# Configuración
dataset_name = 'HDFS_v1'  # Asegúrate de que esto coincida con tu modelo entrenado
Bert_path = r"google-bert/bert-base-uncased"
Llama_path = r"meta-llama/Meta-Llama-3-8B"

max_content_len = 100
max_seq_len = 128

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def analyze_log(log_message, model):
    """Analiza un mensaje de log usando el modelo LLM."""
    processed_log = replace_patterns(log_message)
    sequences = np.array([processed_log.split(' ;-; ')], dtype=object)
    
    model.eval()
    with torch.no_grad():
        outputs_ids = model(sequences)
        outputs = model.Llama_tokenizer.batch_decode(outputs_ids)
        
        for text in outputs:
            matches = re.findall(r' (.*?)\.<|end_of_text|>', text)
            if len(matches) > 0:
                prediction = matches[0]
            else:
                prediction = ''
            
        if prediction == 'anomalous':
            return "anomalous"
        else:
            return "normal"


if __name__ == '__main__':
    # Cargar el modelo
    model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)

    # Log de ejemplo (reemplaza con tu log real)
    log_message = "150610 14:45:20 12420 INFO dfs.DataNode:  Receiving block blk_-1608704935189617140 src: /10.250.18.18:50010 dest: /10.250.18.18:55786"
    # log_message = "150610 14:45:20 12420 INFO dfs.DataNode:  Receiving block blk_-1608704935189617140 src: /10.250.18.18:50010 dest: /10.250.18.18:55786 ;-; 150610 14:45:20 12420 INFO dfs.DataNode:  Receiving block blk_-1608704935189617140 src: /10.250.18.18:50010 dest: /10.250.18.18:55786 ;-; 150610 14:45:20 12420 INFO dfs.DataNode:  Receiving block blk_-1608704935189617140 src: /10.250.18.18:50010 dest: /10.250.18.18:55786"

    # Analizar el log
    prediction = analyze_log(log_message, model)

    # Imprimir el resultado
    print(f"Log: {log_message}")
    print(f"Predicción del modelo: {prediction}")