import argparse
import os
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import MultiStepDataset, load_timeseries_double_rolling
from models.ta_stgcn import TASTGCNModel
from utils.graph_utils import compute_scaled_laplacian, load_adj_from_excel
from utils.metrics import compute_metrics, print_metrics_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate TA-STGCN Trained Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ta_stgcn_best.pth", help="Path to model checkpoint")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to traffic count CSV file")
    parser.add_argument("--adj_path", type=str, default=None, help="Path to graph adjacency Excel file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Load YAML config fallback
    cfg_dict = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"\n🔍 Loading Checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    ckpt_config = checkpoint.get('config', {})
    scaler = checkpoint.get('scaler')
    nodes = checkpoint.get('nodes')
    csv_path = args.csv_path or ckpt_config.get("data", {}).get("csv_path", "traffic_volume_timeseries_1min_608nodes.csv")
    adj_path = args.adj_path or ckpt_config.get("data", {}).get("adj_path", "road_network_distance_608nodes.xlsx")

    if not os.path.exists(csv_path) and os.path.exists(csv_path + ".gz"):
        csv_path = csv_path + ".gz"

    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"Adjacency matrix file not found: '{adj_path}'")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Traffic count CSV file not found: '{csv_path}'")

    # Reconstruct Graph Laplacian
    A_raw, node_ids = load_adj_from_excel(adj_path)
    if nodes is None:
        nodes = node_ids
    L_tilde = compute_scaled_laplacian(A_raw)

    T_in = ckpt_config.get("data", {}).get("history_minutes", 120) // ckpt_config.get("data", {}).get("time_step_minutes", 5)
    horizon = ckpt_config.get("data", {}).get("horizon", 6)
    step_minutes = ckpt_config.get("data", {}).get("time_step_minutes", 5)

    print(f"   Building TA-STGCN Model for N={len(nodes)} nodes, T_in={T_in}, Horizon={horizon}...")
    model = TASTGCNModel(
        num_nodes=len(nodes),
        in_feat=5,
        block_hidden=ckpt_config.get("model", {}).get("block_hidden", 80),
        num_blocks=ckpt_config.get("model", {}).get("num_blocks", 2),
        T_in=T_in,
        cheb_K=ckpt_config.get("model", {}).get("cheb_K", 3),
        horizon=horizon,
        output_feat=2,
        L_tilde=L_tilde,
        dropout=ckpt_config.get("model", {}).get("dropout", 0.25),
        use_temporal_attention=ckpt_config.get("model", {}).get("use_temporal_attention", True),
        attn_num_heads=ckpt_config.get("model", {}).get("attn_num_heads", 4),
        attn_dropout=ckpt_config.get("model", {}).get("attn_dropout", 0.1),
        attn_position=ckpt_config.get("model", {}).get("attn_position", "middle")
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"   Loading Test Data from: {csv_path}...")
    df_all = load_timeseries_double_rolling(
        csv_path=csv_path,
        node_list=nodes,
        window1=ckpt_config.get("data", {}).get("window1", 3),
        window2=ckpt_config.get("data", {}).get("window2", 5),
        step_minutes=step_minutes
    )

    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    df_test = df_all.iloc[n_train + n_val:]

    test_ds = MultiStepDataset(df_test, nodes, T_in, horizon, scaler)
    test_loader = DataLoader(test_ds, batch_size=ckpt_config.get("training", {}).get("batch_size", 64), shuffle=False)

    print(f"   Evaluating on Test Set ({len(test_ds)} samples)...")
    means = torch.tensor(scaler['mean'], device=device)
    stds = torch.tensor(scaler['std'], device=device)

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            y_true = Y * stds + means
            y_pred = pred * stds + means

            all_y_true.append(y_true.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    metrics = compute_metrics(all_y_true, all_y_pred)
    print_metrics_table(metrics, horizon=horizon, step_minutes=step_minutes)


if __name__ == "__main__":
    main()
