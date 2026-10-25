"""ResNet-50 classifier on DeepInsight-style images (source-only training).

Trains on source images with source labels. Epoch / checkpoint selection uses a stratified
validation split of the SOURCE data; target images are only scored, never used for selection.
"""
import copy
from typing import Dict, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) float images -> ImageNet-normalised (N, 3, H, W) tensor."""
    x = torch.as_tensor(np.asarray(images), dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
    return (x - _MEAN) / _STD


def _predict(net: nn.Module, x: torch.Tensor, device, batch_size: int = 32) -> np.ndarray:
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            preds.append(net(x[i:i + batch_size].to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def train_eval_resnet50(
    images_train: np.ndarray,
    y_train: np.ndarray,
    eval_sets: Dict[str, "tuple[np.ndarray, np.ndarray]"],
    *,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_size: float = 0.2,
    pretrained: bool = True,
    seed: int = 0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train ResNet-50 on (images_train, y_train); score every (images, labels) in eval_sets.

    Returns a dict with per-set accuracy / balanced accuracy / confusion matrix, the label encoder,
    the training history, and the best epoch (chosen on the source validation split).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    le = LabelEncoder().fit(np.concatenate([y_train] + [y for _, y in eval_sets.values()]))
    y_enc = le.transform(y_train)
    idx_tr, idx_val = train_test_split(np.arange(len(y_enc)), test_size=val_size, stratify=y_enc, random_state=seed)

    X = _to_tensor(images_train)
    Xtr, ytr = X[idx_tr], torch.as_tensor(y_enc[idx_tr])
    Xval, yval = X[idx_val], y_enc[idx_val]

    net = timm.create_model("resnet50", pretrained=pretrained, num_classes=len(le.classes_)).to(device)
    counts = np.bincount(y_enc[idx_tr], minlength=len(le.classes_))
    weights = torch.tensor(len(idx_tr) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    best_state, best_val, best_epoch, history = None, -1.0, -1, []
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            if len(b) < 2:                      # BatchNorm needs more than one sample
                continue
            optimizer.zero_grad()
            loss = criterion(net(Xtr[b].to(device)), ytr[b].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        val_acc = balanced_accuracy_score(yval, _predict(net, Xval, device))
        scheduler.step(val_acc)                 # source validation drives the schedule, not target data
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_bal_acc": val_acc})
        if verbose:
            print(f"epoch {epoch:2d}  loss {total / len(Xtr):.4f}  source-val balanced acc {val_acc:.3f}")
        if val_acc > best_val:
            best_val, best_epoch, best_state = val_acc, epoch, copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    results = {}
    for name, (imgs, y) in eval_sets.items():
        pred = _predict(net, _to_tensor(imgs), device)
        yt = le.transform(y)
        results[name] = {
            "accuracy": accuracy_score(yt, pred),
            "balanced_accuracy": balanced_accuracy_score(yt, pred),
            "confusion_matrix": confusion_matrix(yt, pred, labels=range(len(le.classes_))),
        }
    return {"results": results, "classes": le.classes_, "best_epoch": best_epoch,
            "best_source_val_bal_acc": best_val, "history": history, "model": net}
