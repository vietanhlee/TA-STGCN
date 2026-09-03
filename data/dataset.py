import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def add_rich_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Constructs cyclical time-of-day features and normalized hour features.

    Args:
        timestamps (pd.DatetimeIndex): Datetime index array.

    Returns:
        np.ndarray: Matrix of shape (N_samples, 3) containing [sin(tod), cos(tod), hour_norm].
    """
    tod = timestamps.hour * 60 + timestamps.minute
    tod_rad = 2 * np.pi * tod / 1440.0
    hour_norm = timestamps.hour / 24.0
    features = np.stack([np.sin(tod_rad), np.cos(tod_rad), hour_norm], axis=1)
    return features


def load_timeseries_double_rolling(
    csv_path: str,
    node_list: list = None,
    window1: int = 3,
    window2: int = 5,
    step_minutes: int = 5
) -> pd.DataFrame:
    """
    Loads raw traffic count CSV, applies double rolling smoothing (Rolling Mean + EMA),
    and resamples to regular step_minutes interval.

    Args:
        csv_path (str): Path to traffic count CSV file.
        node_list (list, optional): List of node IDs to select.
        window1 (int): First window size for rolling mean smoothing (default: 3).
        window2 (int): Second window span for EMA smoothing (default: 5).
        step_minutes (int): Target resampling timestep resolution in minutes (default: 5).

    Returns:
        pd.DataFrame: MultiIndex pivoted DataFrame of shape (Timesteps, Nodes * Features).
    """
    print(f"   Reading CSV: {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    if node_list is not None:
        df = df[df['STT'].isin(node_list)]
    else:
        node_list = sorted(df['STT'].unique())

    feature_cols = ['Car Count', 'Bike Count']
    pivot = df.pivot_table(index='Timestamp', columns='STT', values=feature_cols, aggfunc='mean')
    pivot = pivot.swaplevel(0, 1, axis=1)
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([node_list, feature_cols]))

    pivot_1min = pivot.resample('1min').mean().interpolate(method='linear', limit=30).fillna(0.0)

    smooth_1 = pivot_1min.rolling(window=window1, center=False, min_periods=1).mean()
    smooth_2 = smooth_1.ewm(span=window2, adjust=False).mean()

    resample_rule = f'{step_minutes}min'
    pivot_final = smooth_2.asfreq(resample_rule).fillna(0.0)
    pivot_final.columns = pivot_final.columns.set_names(['Node', 'Feature'])

    print(f"   Data loaded successfully. Shape: {pivot_final.shape}")
    return pivot_final


class MultiStepDataset(Dataset):
    """
    PyTorch Dataset for multi-horizon spatio-temporal graph forecasting.
    
    Generates historical sliding windows of length T_in and forecast target windows of length Horizon,
    applying Z-score feature scaling and appending rich time features.
    
    Args:
        data_df (pd.DataFrame): Pivoted MultiIndex DataFrame of shape (Timesteps, Nodes * Features).
        node_order (list): Ordered list of node IDs.
        T_in (int): Historical observation length (default: 24).
        Horizon (int): Forecasting target horizon (default: 6).
        scaler (dict, optional): Scaler statistics {'mean': np.ndarray, 'std': np.ndarray}.
    """
    def __init__(self, data_df: pd.DataFrame, node_order: list, T_in: int = 24, Horizon: int = 6, scaler: dict = None):
        self.T_in = T_in
        self.Horizon = Horizon
        self.node_order = node_order

        df_sorted = data_df.sort_index(axis=1, level='Node')
        desired_cols = pd.MultiIndex.from_product([node_order, ['Car Count', 'Bike Count']], names=['Node', 'Feature'])
        self.df = df_sorted.reindex(columns=desired_cols)

        self.timestamps = self.df.index
        self.N = len(node_order)

        self.values = self.df.values.astype(float).reshape(-1, self.N, 2)
        self.time_feats = add_rich_time_features(self.timestamps)

        if scaler is None:
            self.means = np.mean(self.values, axis=0, keepdims=True)
            self.stds = np.std(self.values, axis=0, keepdims=True) + 1e-6
        else:
            self.means = scaler['mean']
            self.stds = scaler['std']

        self.valid_len = self.values.shape[0] - self.T_in - self.Horizon + 1

    def __len__(self):
        return max(0, self.valid_len)

    def __getitem__(self, idx: int):
        x_node = self.values[idx : idx + self.T_in]
        y_node = self.values[idx + self.T_in : idx + self.T_in + self.Horizon]

        x_node = (x_node - self.means) / self.stds
        y_node = (y_node - self.means) / self.stds

        t_in_feats = self.time_feats[idx : idx + self.T_in]
        t_in_expanded = np.tile(np.expand_dims(t_in_feats, axis=1), (1, self.N, 1))

        x_final = np.concatenate([x_node, t_in_expanded], axis=-1)
        return torch.from_numpy(x_final.astype(np.float32)), torch.from_numpy(y_node.astype(np.float32))
