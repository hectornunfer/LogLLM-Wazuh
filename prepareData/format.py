import os.path


import numpy as np
import pandas as pd
from helper import sliding_window, fixedSize_window, structure_log

#### for Thunderbird, Liberty, BGL


data_dir = r'/content/Thunderbird'
log_name = "thunderbird_prueba.log"

start_line = 0
end_line = None

# # Liberty
# start_line = 40000000
# end_line = 45000000

# # thunderbird
# start_line = 160000000
# end_line = 170000000

output_dir = data_dir



if __name__ == '__main__':
    # group_type = 'time_sliding'

    window_size = 100
    step_size = 100

    if 'thunderbird' in log_name.lower() or 'spirit' in log_name.lower() or 'liberty' in log_name.lower():
        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'   #thunderbird  , spirit, liberty
    elif 'bgl' in log_name.lower():
        log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'  #bgl
    else:
        raise Exception('missing valid log format')
    print(f'Auto log_format: {log_format}')

    structure_log(data_dir, output_dir, log_name, log_format, start_line = start_line, end_line = end_line)

    print(f'window_size: {window_size}; step_size: {step_size}')


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


