"""
Classifier training loop for the GR vs massive-graviton task.

Wraps the CNN+Transformer model defined in `model.py` and drives it on
the (whitened strain, label) pairs produced by `data.build_injection_dataset`.
The task is binary: label 0 = GR injection, label 1 = MG injection at
the configured lambda_g.

The model is sized for laptop CPU runs by default. For full-scale runs,
push n_per_class and n_epochs up, and move device to CUDA.

Returns dict {history, best_val_acc, best_val_auc, model_state,
              X_val, y_val, val_scores, meta_val}.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from loguru import logger
from tqdm import tqdm

from .data import build_injection_dataset
from .model import build_model, build_compact_model
from .mf_baseline import roc_from_scores


def _seq_to_tensor(X: np.ndarray) -> torch.Tensor:
    """(N, T) float32 -> (N, 1, T) tensor."""
    return torch.tensor(X, dtype=torch.float32).unsqueeze(1)


def _train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return loss_sum / total, correct / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    scores_list = []
    y_list = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
        # Probability of class 1 (MG) for ROC
        prob_mg = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        scores_list.append(prob_mg)
        y_list.append(yb.cpu().numpy())
    scores = np.concatenate(scores_list)
    y = np.concatenate(y_list)
    return loss_sum / total, correct / total, scores, y


def train_classifier(
    n_per_class: int = 200,
    target_snr: float = 20.0,
    lambda_g_km: float = 1e15,
    n_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    train_frac: float = 0.8,
    seed: int = 42,
    reservoir_seconds: int = 1024,
    reservoir_event: str = "GW150914",
    compact: bool = False,
    noise_scale: float = 1.0,
    device: str | None = None,
) -> dict:
    """
    Full training pipeline.

    n_per_class:  injections per class. 2 * n_per_class total.
    target_snr:   network optimal SNR of each injection.
    lambda_g_km:  graviton Compton wavelength for the MG class.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Dataset
    X, Xraw, y, meta, reservoir = build_injection_dataset(
        n_per_class=n_per_class,
        target_snr=target_snr,
        lambda_g_km=lambda_g_km,
        reservoir_seconds=reservoir_seconds,
        reservoir_event=reservoir_event,
        seed=seed,
        noise_scale=noise_scale,
    )

    X_t = _seq_to_tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    n_total = len(ds)
    n_tr = int(n_total * train_frac)
    n_val = n_total - n_tr
    tr_ds, val_ds = random_split(
        ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Model
    seq_len = X.shape[1]
    builder = build_compact_model if compact else build_model
    model = builder(seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    # 3. Train
    history = {"tr_loss": [], "tr_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_val_acc, best_val_auc, best_state = 0.0, 0.0, None
    for epoch in tqdm(range(1, n_epochs + 1), desc="train"):
        tl, ta = _train_epoch(model, tr_loader, optimiser, criterion, device)
        vl, va, vscores, vy = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        # AUC on val
        auc = roc_from_scores(vy, vscores)["auc"] if len(set(vy)) > 1 else float("nan")
        history["tr_loss"].append(tl); history["tr_acc"].append(ta)
        history["val_loss"].append(vl); history["val_acc"].append(va)
        history["val_auc"].append(auc)
        if va > best_val_acc:
            best_val_acc = va
            best_val_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logger.info(
            f"epoch {epoch:3d} | tr_loss={tl:.4f} tr_acc={ta:.3f}"
            f" | val_loss={vl:.4f} val_acc={va:.3f} val_auc={auc:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval scores on val
    _, _, vscores, vy = _eval_epoch(model, val_loader, criterion, device)

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_val_auc": best_val_auc,
        "model_state": best_state,
        "val_scores": vscores,
        "val_labels": vy,
        "n_params": n_params,
        "lambda_g_km": lambda_g_km,
        "target_snr": target_snr,
    }
