import os
import pandas as pd
from typing import Literal, Optional
from datetime import timedelta
from tqdm.auto import tqdm

def preprocess_ibm(num_obs: Optional[int] = None,
                   scale: Literal['small', 'medium', 'large'] = 'small',
                   num_pieces: Optional[int] = None,
                   default_path: str = None):
    """
    Preprocess the IBM dataset to create edges based on transaction timestamps.
    Args:
        num_obs (int, optional): Number of observations to consider from the end of the dataset.
                                 If None, uses all available rows after filtering.
        scale (str): Scale of the dataset to use ('small', 'medium', 'large').
        num_pieces (int, optional): Number of chunks to split the data into for batching the join.
                                    If None, processes in a single piece.
        default_path (str, optional): Base directory for the CSV. If None, defaults to 'datasets/ibm/'.
    """
    date_format = '%Y/%m/%d %H:%M'

    if default_path is None:
        data_path = os.path.join('datasets', 'ibm', f'HI-{scale.capitalize()}_Trans.csv')
    else:
        data_path = os.path.join(default_path, f'HI-{scale.capitalize()}_Trans.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data_df = pd.read_csv(data_path, encoding='utf-8')
    data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'], format=date_format)
    data_df.sort_values('Timestamp', inplace=True)
    data_df = data_df[data_df['Account'] != data_df['Account.1']]

    total_rows = len(data_df)
    if num_obs is None or num_obs > total_rows:
        num_obs = total_rows

    data_df = data_df.tail(num_obs).reset_index(drop=True)
    data_df.reset_index(inplace=True)

    if num_pieces is None or num_pieces < 1:
        num_pieces = 1

    data_df_accounts = data_df[['index', 'Account', 'Account.1', 'Timestamp']]
    delta = 4 * 60

    print('Number of observations:', len(data_df_accounts))
    print('Splitting into', num_pieces, 'piece(s)')

    source = []
    target = []

    for i in tqdm(range(num_pieces), desc="Building edges"):
        start = i * num_obs // num_pieces
        end   = (i + 1) * num_obs // num_pieces
        slice_right = data_df_accounts.iloc[start:end]
        if slice_right.empty:
            continue

        t_min = slice_right['Timestamp'].iloc[0]
        t_max = slice_right['Timestamp'].iloc[-1]

        slice_left = data_df_accounts[
            (data_df_accounts['Timestamp'] >= t_min - timedelta(minutes=delta)) &
            (data_df_accounts['Timestamp'] <= t_max)
        ]

        joined = slice_left.merge(
            slice_right,
            left_on='Account.1',
            right_on='Account',
            suffixes=('_1', '_2'),
            how='inner'
        )

        for _, row in joined.iterrows():
            dt = row['Timestamp_2'] - row['Timestamp_1']
            minutes = dt.days * 24 * 60 + dt.seconds / 60
            if 0 <= minutes <= delta:
                source.append(row['index_1'])
                target.append(row['index_2'])

    save_path = default_path or 'datasets/ibm'
    pd.DataFrame({'txId1': source, 'txId2': target}).to_csv(
        os.path.join(save_path, 'edges.csv'), index=False
    )
