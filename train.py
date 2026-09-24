import argparse
import copy
import gc
import os
import random
import sys
import time
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.dataset import MultiStepDataset, load_timeseries_double_rolling
from models.ta_stgcn import TASTGCNModel
from utils.ema import ModelEMA
from utils.graph_utils import compute_scaled_laplacian, load_adj_from_excel
from utils.logger import TeeLogger
from utils.loss import HuberSmoothLoss
from utils.metrics import compute_metrics, print_metrics_table


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_scaler: torch.amp.GradScaler,
    scaler_stats: dict,
    ema: ModelEMA = None,
    grad_clip_norm: float = 5.0
):
    model.train()
    total_loss, total_mae, total_mse = 0.0, 0.0, 0.0
    count_batches = 0

    means = torch.tensor(scaler_stats['mean'], device=device)
    stds = torch.tensor(scaler_stats['std'], device=device)

    pbar = tqdm(loader, desc="   Training", leave=False)
    for X, Y in pbar:
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            pred = model(X)
            loss = loss_fn(pred, Y)

        if device.type == 'cuda':
            grad_scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()

        with torch.no_grad():
            y_true = Y * stds + means
            y_pred = pred * stds + means
            err = y_true - y_pred
            mae_batch = torch.abs(err).mean().item()
            mse_batch = (err ** 2).mean().item()

            total_mae += mae_batch
            total_mse += mse_batch

        count_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae_batch:.2f}")

    avg_loss = total_loss / max(1, count_batches)
    avg_mae = total_mae / max(1, count_batches)
    avg_mse = total_mse / max(1, count_batches)
    avg_rmse = np.sqrt(avg_mse)
    return avg_loss, avg_mae, avg_mse, avg_rmse


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler_stats: dict,
    loss_fn: nn.Module = None
):
    model.eval()
    total_mae, total_mse, total_loss = 0.0, 0.0, 0.0
    count_batches = 0

    means = torch.tensor(scaler_stats['mean'], device=device)
    stds = torch.tensor(scaler_stats['std'], device=device)

    all_y_true = []
    all_y_pred = []

    pbar = tqdm(loader, desc="   Evaluating", leave=False)
    with torch.no_grad():
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            if loss_fn is not None:
                loss_val = loss_fn(pred, Y)
                total_loss += loss_val.item()

            y_true = Y * stds + means
            y_pred = pred * stds + means

            err = y_true - y_pred
            mae_val = torch.abs(err).mean().item()
            total_mae += mae_val
            total_mse += (err ** 2).mean().item()

            all_y_true.append(y_true.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())

            count_batches += 1
            pbar.set_postfix(mae=f"{mae_val:.2f}")

    if count_batches == 0:
        return {'mae': 9999.0, 'mse': 9999.0, 'rmse': 9999.0, 'loss': 9999.0}

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    metrics = compute_metrics(all_y_true, all_y_pred)
    metrics['loss'] = total_loss / count_batches
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train TA-STGCN Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to traffic count CSV file")
    parser.add_argument("--adj_path", type=str, default=None, help="Path to graph adjacency Excel file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    # Load YAML config
    cfg_dict = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

    # Merge CLI arguments over YAML config
    csv_path = args.csv_path or cfg_dict.get("data", {}).get("csv_path", "traffic_volume_timeseries_1min_608nodes.csv")
    adj_path = args.adj_path or cfg_dict.get("data", {}).get("adj_path", "road_network_distance_608nodes.xlsx")

    if not os.path.exists(csv_path) and os.path.exists(csv_path + ".gz"):
        csv_path = csv_path + ".gz"

    epochs = args.epochs or cfg_dict.get("training", {}).get("epochs", 500)
    batch_size = args.batch_size or cfg_dict.get("training", {}).get("batch_size", 64)
    learning_rate = args.learning_rate or cfg_dict.get("training", {}).get("learning_rate", 0.0005)
    seed = args.seed or cfg_dict.get("training", {}).get("seed", 42)
    device_str = args.device or cfg_dict.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    save_dir = args.save_dir or cfg_dict.get("output", {}).get("save_dir", "checkpoints")
    log_dir = cfg_dict.get("output", {}).get("log_dir", "logs")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_filepath = os.path.join(log_dir, f"train_ta_stgcn_seed{seed}.log")
    sys.stdout = TeeLogger(log_filepath)

    set_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    print("\n" + "=" * 60)
    print(" 🚀 TA-STGCN TRAINING PIPELINE")
    print("=" * 60)
    print(f" Device          : {device}")
    print(f" Seed            : {seed}")
    print(f" Data CSV        : {csv_path}")
    print(f" Graph Matrix    : {adj_path}")
    print(f" Epochs          : {epochs}")
    print(f" Batch Size      : {batch_size}")
    print(f" Learning Rate   : {learning_rate}")
    print("=" * 60 + "\n")

    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"Adjacency matrix file not found: '{adj_path}'")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Traffic count CSV file not found: '{csv_path}'")

    # Build Graph & Laplacian
    A_raw, nodes = load_adj_from_excel(adj_path)
    L_tilde = compute_scaled_laplacian(A_raw)
    print(f"   Scaled Laplacian (L_tilde) computed for N={len(nodes)} nodes. Shape: {L_tilde.shape}")

    # Load Time Series Data
    T_in = cfg_dict.get("data", {}).get("history_minutes", 120) // cfg_dict.get("data", {}).get("time_step_minutes", 5)
    horizon = cfg_dict.get("data", {}).get("horizon", 6)

    df_all = load_timeseries_double_rolling(
        csv_path=csv_path,
        node_list=nodes,
        window1=cfg_dict.get("data", {}).get("window1", 3),
        window2=cfg_dict.get("data", {}).get("window2", 5),
        step_minutes=cfg_dict.get("data", {}).get("time_step_minutes", 5)
    )

    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    df_train = df_all.iloc[:n_train]
    df_val = df_all.iloc[n_train:n_train + n_val]
    df_test = df_all.iloc[n_train + n_val:]

    print(f"   Dataset Split -> Total={n_total} | Train={len(df_train)} (80%) | Val={len(df_val)} (10%) | Test={len(df_test)} (10%)")

    train_ds = MultiStepDataset(df_train, nodes, T_in, horizon)
    scaler = {'mean': train_ds.means, 'std': train_ds.stds}
    val_ds = MultiStepDataset(df_val, nodes, T_in, horizon, scaler)
    test_ds = MultiStepDataset(df_test, nodes, T_in, horizon, scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Initialize TA-STGCN Model
    model = TASTGCNModel(
        num_nodes=len(nodes),
        in_feat=5,
        block_hidden=cfg_dict.get("model", {}).get("block_hidden", 80),
        num_blocks=cfg_dict.get("model", {}).get("num_blocks", 2),
        T_in=T_in,
        cheb_K=cfg_dict.get("model", {}).get("cheb_K", 3),
        horizon=horizon,
        output_feat=2,
        L_tilde=L_tilde,
        dropout=cfg_dict.get("model", {}).get("dropout", 0.25),
        use_temporal_attention=cfg_dict.get("model", {}).get("use_temporal_attention", True),
        attn_num_heads=cfg_dict.get("model", {}).get("attn_num_heads", 4),
        attn_dropout=cfg_dict.get("model", {}).get("attn_dropout", 0.1),
        attn_position=cfg_dict.get("model", {}).get("attn_position", "middle")
    ).to(device)

    # Optimizer, Loss & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = HuberSmoothLoss(
        delta=cfg_dict.get("training", {}).get("loss_delta", 1.0),
        smooth_weight=cfg_dict.get("training", {}).get("smooth_loss_weight", 0.0)
    )
    grad_scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    use_lr_sched = cfg_dict.get("training", {}).get("use_lr_scheduler", True)
    lr_scheduler = None
    if use_lr_sched:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=cfg_dict.get("training", {}).get("lr_sched_factor", 0.5),
            patience=cfg_dict.get("training", {}).get("lr_sched_patience", 10),
            min_lr=cfg_dict.get("training", {}).get("lr_sched_min_lr", 1e-5)
        )

    use_ema = cfg_dict.get("training", {}).get("use_ema", True)
    ema_decay = cfg_dict.get("training", {}).get("ema_decay", 0.995)
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None

    best_mae = float('inf')
    patience = cfg_dict.get("training", {}).get("patience", 60)
    patience_cnt = 0
    best_model_path = os.path.join(save_dir, "ta_stgcn_best.pth")

    print(f"\n⚡ Starting Training for {epochs} Epochs...\n")

    for ep in range(epochs):
        train_loss, train_mae, train_mse, train_rmse = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_scaler=grad_scaler,
            scaler_stats=scaler,
            ema=ema,
            grad_clip_norm=cfg_dict.get("training", {}).get("grad_clip_norm", 5.0)
        )

        if ema is not None:
            raw_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema.shadow)
            val_metrics = evaluate(model, val_loader, device, scaler, loss_fn=loss_fn)
            model.load_state_dict(raw_state)
        else:
            val_metrics = evaluate(model, val_loader, device, scaler, loss_fn=loss_fn)

        val_mae = val_metrics['mae']
        val_loss = val_metrics['loss']

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Ep {ep+1:03d} | Loss: {train_loss:.4f} / {val_loss:.4f} | MAE: {train_mae:.2f} / {val_mae:.2f} | LR: {current_lr:.6f}", end="")

        if val_mae < best_mae:
            best_mae = val_mae
            patience_cnt = 0
            save_state = ema.shadow if ema is not None else model.state_dict()
            torch.save({
                'model_state_dict': save_state,
                'scaler': scaler,
                'nodes': nodes,
                'config': cfg_dict
            }, best_model_path)
            print(" -> Saved Best")
        else:
            patience_cnt += 1
            print(f" | Patience: {patience_cnt}/{patience}")
            if patience_cnt >= patience:
                print(f"\n⏹️ Early Stopping Triggered at Epoch {ep+1}")
                break

    print("\n" + "=" * 50)
    print(" 🏁 TRAINING COMPLETED - FINAL TEST EVALUATION")
    print("=" * 50)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device, scaler, loss_fn=loss_fn)

    print_metrics_table(test_metrics, horizon=horizon, step_minutes=cfg_dict.get("data", {}).get("time_step_minutes", 5))

    # Save benchmark text summary
    summary_file = os.path.join(save_dir, "benchmark_ta_stgcn_results.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(" 🏆 FINAL TEST BENCHMARK RESULTS 🏆\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model          : TA-STGCN\n")
        f.write(f"Seed           : {seed}\n")
        f.write(f"Horizon        : {horizon} steps (30 mins)\n")
        f.write(f"FINAL TEST LOSS: {test_metrics['loss']:.4f}\n")
        f.write(f"FINAL TEST MAE : {test_metrics['mae']:.4f}\n")
        f.write(f"FINAL TEST RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"FINAL TEST MAPE: {test_metrics['mape']:.2f}%\n")
        f.write(f"FINAL TEST WAPE: {test_metrics['wape']:.2f}%\n")
        for h in range(horizon):
            f.write(f"  └─ MAE t+{h+1} ({(h+1)*5}m): {test_metrics.get(f'mae_h{h+1}', 0.0):.4f}\n")
        f.write("=" * 50 + "\n")

    print(f"💾 Checkpoint saved to : {best_model_path}")
    print(f"💾 Benchmark summary saved to: {summary_file}\n")


if __name__ == "__main__":
    main()
