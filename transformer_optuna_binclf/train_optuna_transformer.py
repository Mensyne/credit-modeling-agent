import argparse
import json
import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.token_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Embedding(num_features + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        bsz, feats = x.shape
        x = x.unsqueeze(-1)
        x = self.token_proj(x)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        positions = torch.arange(0, feats + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)
        x = self.encoder(x)
        x = self.norm(x)
        cls_out = x[:, 0, :]
        logits = self.head(cls_out).squeeze(-1)
        return logits


@dataclass
class TrainConfig:
    data_path: str
    label_col: str
    test_size: float
    val_size: float
    seed: int
    n_trials: int
    max_epochs: int
    patience: int
    min_delta: float


def load_and_preprocess(data_path: str, label_col: str, test_size: float, val_size: float, seed: int):
    df = pd.read_csv(data_path)
    if label_col in df.columns:
        y = df[label_col].astype(int)
        x_df = df.drop(columns=[label_col])
    else:
        y = df.iloc[:, -1].astype(int)
        x_df = df.iloc[:, :-1]

    obj_cols = x_df.select_dtypes(include=["object", "category"]).columns
    for col in obj_cols:
        x_df[col] = x_df[col].fillna("NA")
    x_df = pd.get_dummies(x_df, columns=obj_cols, drop_first=True)

    num_cols = x_df.columns
    x_df[num_cols] = x_df[num_cols].replace([np.inf, -np.inf], np.nan)
    x_df[num_cols] = x_df[num_cols].fillna(x_df[num_cols].median())

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_df.values,
        y.values,
        test_size=test_size,
        random_state=seed,
        stratify=y.values,
    )

    val_ratio = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), scaler


def evaluate(model, loader, device):
    model.eval()
    y_true_all = []
    y_score_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            y_true_all.append(y.detach().cpu().numpy())
            y_score_all.append(probs.detach().cpu().numpy())
    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)
    auc = compute_auc(y_true, y_score)
    ks = compute_ks(y_true, y_score)
    return auc, ks


def evaluate_with_preds(model, loader, device):
    model.eval()
    y_true_all = []
    y_score_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            y_true_all.append(y.detach().cpu().numpy())
            y_score_all.append(probs.detach().cpu().numpy())
    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)
    return y_true, y_score


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def train_one_trial(trial, cfg: TrainConfig, data_splits, device, logger: logging.Logger):
    (x_train, y_train), (x_val, y_val), (x_test, y_test), _ = data_splits

    d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dim_ff = trial.suggest_categorical("dim_ff", [64, 128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    if d_model % n_heads != 0:
        raise optuna.TrialPruned()

    train_ds = TabularDataset(x_train, y_train)
    val_ds = TabularDataset(x_val, y_val)
    test_ds = TabularDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = TabularTransformer(
        num_features=x_train.shape[1],
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_true = y.detach().cpu().numpy()
            batch_auc = compute_auc(y_true, probs)
            batch_ks = compute_ks(y_true, probs)

            logger.info(
                f"[trial {trial.number}] epoch {epoch} batch {batch_idx}/{len(train_loader)} "
                f"bs={x.size(0)} loss={loss.item():.4f} auc={batch_auc:.4f} ks={batch_ks:.4f}"
            )

        val_auc, val_ks = evaluate(model, val_loader, device)
        test_auc, test_ks = evaluate(model, test_loader, device)
        logger.info(
            f"[trial {trial.number}] epoch {epoch} val_auc={val_auc:.4f} val_ks={val_ks:.4f} "
            f"test_auc={test_auc:.4f} test_ks={test_ks:.4f}"
        )

        trial.report(test_auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if test_auc > best_auc + cfg.min_delta:
            best_auc = test_auc
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                logger.info(f"[trial {trial.number}] early stopping at epoch {epoch}")
                break

    return best_auc, best_state


def build_model_from_params(num_features: int, params: dict, device):
    model = TabularTransformer(
        num_features=num_features,
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        num_layers=params["num_layers"],
        dim_ff=params["dim_ff"],
        dropout=params["dropout"],
    ).to(device)
    return model


def search_threshold(y_true: np.ndarray, y_score: np.ndarray, method: str = "f1", num_thresholds: int = 201):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    best_thr = 0.5
    best_metric = -1.0

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        if method == "f1":
            metric = f1_score(y_true, y_pred, zero_division=0)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            metric = float(np.max(tpr - fpr))
        if metric > best_metric:
            best_metric = metric
            best_thr = float(thr)

    return best_thr, float(best_metric)


def bucket_ks_stats(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10):
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    total_pos = float(np.sum(y_true_sorted))
    total_neg = float(len(y_true_sorted) - total_pos)
    total_pos = max(total_pos, 1.0)
    total_neg = max(total_neg, 1.0)

    bins = np.array_split(np.arange(len(y_true_sorted)), n_bins)
    rows = []
    cum_pos = 0.0
    cum_neg = 0.0

    for i, idx in enumerate(bins, start=1):
        ys = y_true_sorted[idx]
        scores = y_score_sorted[idx]
        bin_pos = float(np.sum(ys))
        bin_neg = float(len(ys) - bin_pos)
        cum_pos += bin_pos
        cum_neg += bin_neg
        tpr = cum_pos / total_pos
        fpr = cum_neg / total_neg
        ks = tpr - fpr
        rows.append(
            {
                "bin": i,
                "count": int(len(ys)),
                "pos": int(bin_pos),
                "neg": int(bin_neg),
                "pos_rate": float(bin_pos / max(len(ys), 1)),
                "score_min": float(np.min(scores)) if len(scores) > 0 else 0.0,
                "score_max": float(np.max(scores)) if len(scores) > 0 else 0.0,
                "cum_tpr": float(tpr),
                "cum_fpr": float(fpr),
                "ks": float(ks),
            }
        )

    best_ks = max(row["ks"] for row in rows) if rows else float("nan")
    return rows, float(best_ks)


def make_test_report(y_true: np.ndarray, y_score: np.ndarray):
    auc = compute_auc(y_true, y_score)
    ks = compute_ks(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    y_pred = (y_score >= best_threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    clf_report = classification_report(y_true, y_pred, digits=4)

    best_thr_f1, best_f1 = search_threshold(y_true, y_score, method="f1")
    best_thr_j, best_j = search_threshold(y_true, y_score, method="j")
    bins, bins_best_ks = bucket_ks_stats(y_true, y_score, n_bins=10)

    return {
        "auc": auc,
        "ks": ks,
        "best_threshold_j": best_threshold,
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "threshold_search": {
            "best_threshold_f1": best_thr_f1,
            "best_f1": best_f1,
            "best_threshold_j": best_thr_j,
            "best_j": best_j,
        },
        "bucket_ks": {
            "n_bins": 10,
            "best_ks": bins_best_ks,
            "bins": bins,
        },
    }


def write_text_report(report: dict, path: str) -> None:
    lines = []
    lines.append("Test Report")
    lines.append("=" * 40)
    for key in ["auc", "ks", "best_threshold_j", "accuracy", "precision", "recall", "f1"]:
        lines.append(f"{key}: {report[key]}")
    lines.append("")

    lines.append("Threshold Search:")
    ts = report["threshold_search"]
    lines.append(f"best_threshold_f1: {ts['best_threshold_f1']}")
    lines.append(f"best_f1: {ts['best_f1']}")
    lines.append(f"best_threshold_j: {ts['best_threshold_j']}")
    lines.append(f"best_j: {ts['best_j']}")
    lines.append("")

    lines.append("Confusion Matrix (rows=true, cols=pred):")
    cm = report["confusion_matrix"]
    lines.append(f"{cm[0]}")
    lines.append(f"{cm[1]}")
    lines.append("")

    lines.append("Classification Report:")
    lines.append(report["classification_report"])
    lines.append("")

    lines.append("Bucket KS (n_bins=10):")
    lines.append("bin,count,pos,neg,pos_rate,score_min,score_max,cum_tpr,cum_fpr,ks")
    for row in report["bucket_ks"]["bins"]:
        lines.append(
            f"{row['bin']},{row['count']},{row['pos']},{row['neg']},{row['pos_rate']:.6f},"
            f"{row['score_min']:.6f},{row['score_max']:.6f},{row['cum_tpr']:.6f},"
            f"{row['cum_fpr']:.6f},{row['ks']:.6f}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_curves(y_true: np.ndarray, y_score: np.ndarray, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = compute_auc(y_true, y_score)
    ks = np.max(tpr - fpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr - fpr, label=f"KS={ks:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("TPR - FPR")
    plt.title("KS Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ks_curve.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv", help="Path to CSV data file")
    parser.add_argument("--label", default="label", help="Label column name")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        label_col=args.label,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        n_trials=args.trials,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = os.path.join(os.path.dirname(cfg.data_path), "training.log")
    logger = setup_logger(log_path)

    data_splits = load_and_preprocess(
        cfg.data_path,
        cfg.label_col,
        cfg.test_size,
        cfg.val_size,
        cfg.seed,
    )

    study = optuna.create_study(direction="maximize")

    best_overall_auc = -1.0
    best_overall_state = None
    best_overall_params = None

    def objective(trial):
        nonlocal best_overall_auc, best_overall_state, best_overall_params
        best_auc, best_state = train_one_trial(trial, cfg, data_splits, device, logger)
        if best_auc > best_overall_auc and best_state is not None:
            best_overall_auc = best_auc
            best_overall_state = best_state
            best_overall_params = trial.params
        return best_auc

    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    if best_overall_state is not None:
        model_path = os.path.join(os.path.dirname(cfg.data_path), "best_model.pt")
        torch.save({"state_dict": best_overall_state, "params": best_overall_params}, model_path)

        (x_train, y_train), (x_val, y_val), (x_test, y_test), _ = data_splits
        test_ds = TabularDataset(x_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)
        best_model = build_model_from_params(x_train.shape[1], best_overall_params, device)
        best_model.load_state_dict(best_overall_state)
        y_true, y_score = evaluate_with_preds(best_model, test_loader, device)
        report = make_test_report(y_true, y_score)

        out_dir = os.path.dirname(cfg.data_path)
        report_path = os.path.join(out_dir, "test_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        text_report_path = os.path.join(out_dir, "test_report.txt")
        write_text_report(report, text_report_path)
        save_curves(y_true, y_score, out_dir)

        logger.info("Test report summary:")
        logger.info(json.dumps(report, indent=2))

    summary = {
        "best_auc": best_overall_auc,
        "best_params": best_overall_params,
        "study_best_params": study.best_params,
        "study_best_value": study.best_value,
    }
    logger.info("Best summary:")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
