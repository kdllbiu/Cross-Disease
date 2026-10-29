import os
import csv
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)

warnings.filterwarnings("ignore")


# =========================
# Basic utils
# =========================
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def upper_tri_features(X):
    n = X.shape[1]
    iu = np.triu_indices(n, k=1)
    return X[:, iu[0], iu[1]], iu


def load_scaler(scaler_npz):
    s = np.load(scaler_npz)
    mean = s["mean"].astype(np.float32)
    scale = s["scale"].astype(np.float32)
    return mean, scale


def apply_scaler(X, mean, scale):
    return (X - mean) / (scale + 1e-8)


# =========================
# Models
# =========================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim=256, hidden=(1024, 512), dropout=0.3):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.to_emb = nn.Linear(d, emb_dim)

    def forward(self, x):
        h = self.net(x)
        emb = torch.relu(self.to_emb(h))
        return emb


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, emb_dim=256, hidden=(1024, 512), dropout=0.3):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, emb_dim=emb_dim, hidden=hidden, dropout=dropout)
        self.head = nn.Linear(emb_dim, 2)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.head(emb)
        return logits, emb


# =========================
# Metrics
# =========================
def compute_threshold_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "precision": float(prec),
        "sensitivity": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "mcc": float(mcc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred
    }


def summarize_predictions(y, probs, threshold=0.5):
    roc = roc_auc_score(y, probs) if len(np.unique(y)) == 2 else float("nan")
    pr = average_precision_score(y, probs) if len(np.unique(y)) == 2 else float("nan")
    thr = compute_threshold_metrics(y, probs, threshold=threshold)
    out = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "acc": float(thr["acc"]),
        "bal_acc": float(thr["bal_acc"]),
        "precision": float(thr["precision"]),
        "sensitivity": float(thr["sensitivity"]),
        "specificity": float(thr["specificity"]),
        "f1": float(thr["f1"]),
        "mcc": float(thr["mcc"]),
        "tn": int(thr["tn"]),
        "fp": int(thr["fp"]),
        "fn": int(thr["fn"]),
        "tp": int(thr["tp"]),
        "y_pred": thr["y_pred"],
        "probs": probs.astype(np.float64)
    }
    return out


# =========================
# Publication style
# =========================
def setup_publication_style():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })


def finalize_axes(ax):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(direction="out", length=4, width=1.0)
    ax.grid(False)


def save_figure(fig, out_path_base):
    png_path = out_path_base + ".png"
    pdf_path = out_path_base + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


# =========================
# Plotting
# =========================
def plot_roc_curve_pub(y_true, y_prob, out_path_base, title=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot(fpr, tpr, color="#1f77b4", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.4, label="Chance")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)

    finalize_axes(ax)
    return save_figure(fig, out_path_base)


def plot_pr_curve_pub(y_true, y_prob, out_path_base, title=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    pos_rate = float(np.mean(y_true == 1))

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot(recall, precision, color="#1f77b4", label=f"AP = {ap:.3f}")
    ax.axhline(pos_rate, linestyle="--", color="gray", linewidth=1.4,
               label=f"Chance = {pos_rate:.3f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower left", frameon=True)

    finalize_axes(ax)
    return save_figure(fig, out_path_base)


def plot_model_auc_comparison(results, out_path_base, sort_desc=True):
    names = list(results.keys())
    aucs = [results[k]["metrics"]["roc_auc"] for k in names]

    if sort_desc:
        order = np.argsort(-np.array(aucs))
        names = [names[i] for i in order]
        aucs = [aucs[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.4, max(4.5, 0.45 * len(names))))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, aucs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("LOOCV ROC-AUC")
    ax.set_title("Model comparison on FC dataset")
    ax.set_xlim(0, 1.0)

    for i, v in enumerate(aucs):
        ax.text(min(v + 0.01, 0.98), i, f"{v:.3f}", va="center")

    finalize_axes(ax)
    return save_figure(fig, out_path_base)


def plot_auc_distribution(models_auc_dict, out_path_base, title=None):
    """
    Plot per‑model AUC distribution across seeds.
    models_auc_dict: dict {model_name: list_of_auc_values}
    Six selected models shown in fixed order.
    Each column: scatter points of 10 runs, mean as red diamond, error bar for 95% CI.
    """
    # Fixed order of models to display
    selected = [
        "Transfer Learning (Frozen Encoder)",
        "PCA + Logistic Regression",
        "Transfer Learning (Full Fine-Tuning)",
        "Random Encoder + Linear Probe",
        "Direct MLP (FC Only)",
        "Logistic Regression (Raw Connectivity)"
    ]

    # Keep only the requested models (if not all present, raise)
    for m in selected:
        if m not in models_auc_dict:
            raise KeyError(f"Model '{m}' not found in provided data.")

    n_models = len(selected)
    fig, axes = plt.subplots(1, n_models, figsize=(3.0 * n_models, 4.0), sharey=True)
    if n_models == 1:
        axes = [axes]

    # Use a fixed color palette or default
    color_points = "#4C78A8"
    color_mean = "#D62728"

    for ax, name in zip(axes, selected):
        vals = np.array(models_auc_dict[name])
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        ci = 1.96 * std_val / np.sqrt(len(vals))

        # jitter x slightly for visibility
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(vals))
        ax.scatter(jitter, vals, s=30, color=color_points, alpha=0.8, edgecolors='white', linewidth=0.5)

        # Mean point and error bar
        ax.errorbar(0, mean_val, yerr=ci, fmt='D', color=color_mean, capsize=5,
                    markersize=8, markeredgewidth=1, markeredgecolor='white', label='Mean ± 95% CI')

        # Aesthetics
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(name.replace(' ', '\n', 2), fontsize=9, pad=8)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        finalize_axes(ax)

    axes[0].set_ylabel("ROC-AUC")
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()

    return save_figure(fig, out_path_base)


# =========================
# Generic sklearn LOOCV
# =========================
def loocv_sklearn_proba(X, y, estimator):
    loo = LeaveOneOut()
    probs = np.zeros(len(y), dtype=float)

    for tr, te in loo.split(X):
        est = clone(estimator)
        est.fit(X[tr], y[tr])

        if hasattr(est, "predict_proba"):
            probs[te[0]] = est.predict_proba(X[te])[0, 1]
        elif hasattr(est, "decision_function"):
            s = est.decision_function(X[te])
            probs[te[0]] = 1.0 / (1.0 + np.exp(-float(s[0])))
        else:
            pred = est.predict(X[te])[0]
            probs[te[0]] = float(pred)

    return probs


# =========================
# Torch training utils
# =========================
def build_optimizer(model, lr=1e-3, weight_decay=1e-4):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def class_weight_tensor(y, device):
    n0 = max((y == 0).sum(), 1)
    n1 = max((y == 1).sum(), 1)
    total = n0 + n1
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def train_torch_classifier(
    model,
    X_train,
    y_train,
    device,
    epochs=120,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=None,
    freeze_encoder=False,
    seed=0
):
    set_seed(seed)
    model = model.to(device)
    model.train()

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.encoder.parameters():
            p.requires_grad = True

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(y_train, device))

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

    n = len(y_train)
    if batch_size is None or batch_size <= 0:
        batch_size = n

    indices = np.arange(n)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model


def predict_torch_proba(model, X, device):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        logits, _ = model(xt)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


def loocv_direct_mlp(
    X, y, in_dim, device,
    emb_dim=256, hidden=(1024, 512), dropout=0.3,
    epochs=120, lr=1e-3, weight_decay=1e-4, batch_size=None,
    seed=0
):
    loo = LeaveOneOut()
    probs = np.zeros(len(y), dtype=float)

    for fold, (tr, te) in enumerate(loo.split(X)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        model = MLPClassifier(
            in_dim=in_dim,
            emb_dim=emb_dim,
            hidden=hidden,
            dropout=dropout
        )

        model = train_torch_classifier(
            model=model,
            X_train=Xtr,
            y_train=y[tr],
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            freeze_encoder=False,
            seed=seed + fold + 1
        )
        probs[te[0]] = predict_torch_proba(model, Xte, device)[0]

    return probs


def loocv_transfer_finetune(
    X_scaled_by_asd, y, pretrained_state_dict, in_dim, device,
    mode="head_only",
    emb_dim=256, hidden=(1024, 512), dropout=0.3,
    epochs=80, lr=5e-4, weight_decay=1e-4, batch_size=None,
    seed=0
):
    loo = LeaveOneOut()
    probs = np.zeros(len(y), dtype=float)

    for fold, (tr, te) in enumerate(loo.split(X_scaled_by_asd)):
        model = MLPClassifier(
            in_dim=in_dim,
            emb_dim=emb_dim,
            hidden=hidden,
            dropout=dropout
        )
        model.load_state_dict(pretrained_state_dict, strict=True)

        if mode == "head_only":
            freeze_encoder = True
        elif mode == "full":
            freeze_encoder = False
        else:
            raise ValueError("mode must be 'head_only' or 'full'")

        model = train_torch_classifier(
            model=model,
            X_train=X_scaled_by_asd[tr],
            y_train=y[tr],
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            freeze_encoder=freeze_encoder,
            seed=seed + fold + 1
        )

        probs[te[0]] = predict_torch_proba(model, X_scaled_by_asd[te], device)[0]

    return probs


# =========================
# Transfer / random embeddings
# =========================
def extract_embeddings_with_model(model, X_scaled, device, batch_size=256):
    model.eval()
    emb = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            xb = torch.tensor(X_scaled[i:i + batch_size], dtype=torch.float32, device=device)
            _, e = model(xb)
            emb.append(e.detach().cpu().numpy())
    return np.vstack(emb)


def loocv_probe(X, y, seed=0):
    loo = LeaveOneOut()
    probs = np.zeros(len(y), dtype=float)

    for tr, te in loo.split(X):
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed
        )
        clf.fit(X[tr], y[tr])
        probs[te[0]] = clf.predict_proba(X[te])[0, 1]

    return probs


# =========================
# Integrated Gradients
# =========================
def integrated_gradients(model, x, target_class=1, steps=64):
    model.eval()
    baseline = torch.zeros_like(x)
    scaled = [baseline + (float(i) / steps) * (x - baseline) for i in range(1, steps + 1)]
    grads = []

    for s in scaled:
        s = s.clone().detach().requires_grad_(True)
        logits, _ = model(s.unsqueeze(0))
        score = torch.log_softmax(logits, dim=1)[0, target_class]
        score.backward()
        grads.append(s.grad.detach().clone())

    avg_grad = torch.mean(torch.stack(grads, dim=0), dim=0)
    attr = (x - baseline) * avg_grad
    return attr


# =========================
# Single-seed evaluation of all models
# =========================
def evaluate_all_baselines_single_seed(
    F_raw,
    F_asd_scaled,
    y,
    pretrained_model,
    pretrained_state_dict,
    device,
    args,
    seed
):
    results = {}

    # 1. Raw + Logistic Regression
    lr_est = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed
        ))
    ])
    probs = loocv_sklearn_proba(F_raw, y, lr_est)
    results["Logistic Regression (Raw Connectivity)"] = {
        "metrics": summarize_predictions(y, probs)
    }

    # 2. Raw + Linear SVM
    lsvm_est = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=seed
        ))
    ])
    probs = loocv_sklearn_proba(F_raw, y, lsvm_est)
    results["Linear SVM (Raw Connectivity)"] = {
        "metrics": summarize_predictions(y, probs)
    }

    # 3. Raw + RBF SVM
    rbfsvm_est = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            gamma="scale",
            C=1.0,
            random_state=seed
        ))
    ])
    probs = loocv_sklearn_proba(F_raw, y, rbfsvm_est)
    results["RBF-SVM (Raw Connectivity)"] = {
        "metrics": summarize_predictions(y, probs)
    }

    # 4. Raw + Random Forest
    rf_est = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed
    )
    probs = loocv_sklearn_proba(F_raw, y, rf_est)
    results["Random Forest (Raw Connectivity)"] = {
        "metrics": summarize_predictions(y, probs)
    }

    # 5. PCA + LR
    pca_dim = min(args.pca_dim, len(y) - 2, F_raw.shape[1])
    pca_lr_est = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_dim, random_state=seed)),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed
        ))
    ])
    probs = loocv_sklearn_proba(F_raw, y, pca_lr_est)
    results["PCA + Logistic Regression"] = {
        "metrics": summarize_predictions(y, probs)
    }

    # 6. Random encoder + LR probe
    random_model = MLPClassifier(
        in_dim=F_raw.shape[1],
        emb_dim=args.emb,
        hidden=(args.h1, args.h2),
        dropout=args.dropout
    ).to(device)
    random_model.eval()
    emb_random = extract_embeddings_with_model(random_model, F_asd_scaled, device)
    probs = loocv_probe(emb_random, y, seed=seed)
    results["Random Encoder + Linear Probe"] = {
        "metrics": summarize_predictions(y, probs),
        "embeddings": emb_random
    }

    # 7. Transfer frozen encoder + LR probe
    emb_transfer = extract_embeddings_with_model(pretrained_model, F_asd_scaled, device)
    probs = loocv_probe(emb_transfer, y, seed=seed)
    results["Transfer Learning (Frozen Encoder)"] = {
        "metrics": summarize_predictions(y, probs),
        "embeddings": emb_transfer
    }

    # 8. Direct MLP on FC
    probs = loocv_direct_mlp(
        F_raw, y,
        in_dim=F_raw.shape[1],
        device=device,
        emb_dim=args.emb,
        hidden=(args.h1, args.h2),
        dropout=args.dropout,
        epochs=args.direct_mlp_epochs,
        lr=args.direct_mlp_lr,
        weight_decay=args.direct_mlp_wd,
        batch_size=args.direct_mlp_batch,
        seed=seed
    )
    results["Direct MLP (FC Only)"] = {
        "metrics": summarize_predictions(y, probs)
    }

    # 9. Transfer fine-tune head only
    probs = loocv_transfer_finetune(
        F_asd_scaled, y,
        pretrained_state_dict=pretrained_state_dict,
        in_dim=F_raw.shape[1],
        device=device,
        mode="head_only",
        emb_dim=args.emb,
        hidden=(args.h1, args.h2),
        dropout=args.dropout,
        epochs=args.ft_head_epochs,
        lr=args.ft_head_lr,
        weight_decay=args.ft_head_wd,
        batch_size=args.ft_head_batch,
        seed=seed
    )
    results["Transfer Learning (Head-Only Fine-Tuning)"] = {
        "metrics": summarize_predictions(y, probs)
    }

    # 10. Transfer fine-tune full model
    probs = loocv_transfer_finetune(
        F_asd_scaled, y,
        pretrained_state_dict=pretrained_state_dict,
        in_dim=F_raw.shape[1],
        device=device,
        mode="full",
        emb_dim=args.emb,
        hidden=(args.h1, args.h2),
        dropout=args.dropout,
        epochs=args.ft_full_epochs,
        lr=args.ft_full_lr,
        weight_decay=args.ft_full_wd,
        batch_size=args.ft_full_batch,
        seed=seed
    )
    results["Transfer Learning (Full Fine-Tuning)"] = {
        "metrics": summarize_predictions(y, probs)
    }

    return results


# =========================
# Aggregating multi-seed results
# =========================
def aggregate_results(all_seed_results, y, num_seeds):
    """all_seed_results: list of dicts, each dict as returned by evaluate_all_baselines_single_seed"""
    model_names = list(all_seed_results[0].keys())
    aggregated = {}

    for name in model_names:
        # Collect metrics across seeds
        metrics_list = []
        probs_list = []
        for seed_res in all_seed_results:
            m = seed_res[name]["metrics"]
            metrics_list.append({
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "acc": m["acc"],
                "bal_acc": m["bal_acc"],
                "precision": m["precision"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "mcc": m["mcc"],
            })
            probs_list.append(m["probs"])

        # Compute mean and std for each metric
        metric_keys = list(metrics_list[0].keys())
        agg_metrics = {}
        for key in metric_keys:
            vals = np.array([m[key] for m in metrics_list])
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            ci_low = mean - 1.96 * std / np.sqrt(num_seeds)
            ci_high = mean + 1.96 * std / np.sqrt(num_seeds)
            agg_metrics[key] = {
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high
            }

        # Average probs over seeds (per sample)
        avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
        # Recompute metrics on averaged probs (representative)
        final_m = summarize_predictions(y, avg_probs)
        final_m.pop("probs")
        final_m.pop("y_pred")

        aggregated[name] = {
            "metrics": final_m,
            "avg_probs": avg_probs,
            "cross_seed_stats": agg_metrics,
        }

    return aggregated


# =========================
# Saving comparison results
# =========================
def save_model_summary_csv(aggregated, out_csv):
    fieldnames = [
        "model",
        "roc_auc_mean", "roc_auc_std", "roc_auc_ci_low", "roc_auc_ci_high",
        "pr_auc_mean", "pr_auc_std", "pr_auc_ci_low", "pr_auc_ci_high",
        "acc_mean", "acc_std", "acc_ci_low", "acc_ci_high",
        "bal_acc_mean", "bal_acc_std", "bal_acc_ci_low", "bal_acc_ci_high",
        "precision_mean", "precision_std", "precision_ci_low", "precision_ci_high",
        "sensitivity_mean", "sensitivity_std", "sensitivity_ci_low", "sensitivity_ci_high",
        "specificity_mean", "specificity_std", "specificity_ci_low", "specificity_ci_high",
        "f1_mean", "f1_std", "f1_ci_low", "f1_ci_high",
        "mcc_mean", "mcc_std", "mcc_ci_low", "mcc_ci_high",
        "avg_prob_roc_auc", "avg_prob_pr_auc"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_name, obj in aggregated.items():
            stats = obj["cross_seed_stats"]
            avg_metrics = obj["metrics"]
            row = {
                "model": model_name,
                "roc_auc_mean": stats["roc_auc"]["mean"],
                "roc_auc_std": stats["roc_auc"]["std"],
                "roc_auc_ci_low": stats["roc_auc"]["ci_low"],
                "roc_auc_ci_high": stats["roc_auc"]["ci_high"],
                "pr_auc_mean": stats["pr_auc"]["mean"],
                "pr_auc_std": stats["pr_auc"]["std"],
                "pr_auc_ci_low": stats["pr_auc"]["ci_low"],
                "pr_auc_ci_high": stats["pr_auc"]["ci_high"],
                "acc_mean": stats["acc"]["mean"],
                "acc_std": stats["acc"]["std"],
                "acc_ci_low": stats["acc"]["ci_low"],
                "acc_ci_high": stats["acc"]["ci_high"],
                "bal_acc_mean": stats["bal_acc"]["mean"],
                "bal_acc_std": stats["bal_acc"]["std"],
                "bal_acc_ci_low": stats["bal_acc"]["ci_low"],
                "bal_acc_ci_high": stats["bal_acc"]["ci_high"],
                "precision_mean": stats["precision"]["mean"],
                "precision_std": stats["precision"]["std"],
                "precision_ci_low": stats["precision"]["ci_low"],
                "precision_ci_high": stats["precision"]["ci_high"],
                "sensitivity_mean": stats["sensitivity"]["mean"],
                "sensitivity_std": stats["sensitivity"]["std"],
                "sensitivity_ci_low": stats["sensitivity"]["ci_low"],
                "sensitivity_ci_high": stats["sensitivity"]["ci_high"],
                "specificity_mean": stats["specificity"]["mean"],
                "specificity_std": stats["specificity"]["std"],
                "specificity_ci_low": stats["specificity"]["ci_low"],
                "specificity_ci_high": stats["specificity"]["ci_high"],
                "f1_mean": stats["f1"]["mean"],
                "f1_std": stats["f1"]["std"],
                "f1_ci_low": stats["f1"]["ci_low"],
                "f1_ci_high": stats["f1"]["ci_high"],
                "mcc_mean": stats["mcc"]["mean"],
                "mcc_std": stats["mcc"]["std"],
                "mcc_ci_low": stats["mcc"]["ci_low"],
                "mcc_ci_high": stats["mcc"]["ci_high"],
                "avg_prob_roc_auc": avg_metrics["roc_auc"],
                "avg_prob_pr_auc": avg_metrics["pr_auc"]
            }
            writer.writerow(row)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # data / pretrained
    parser.add_argument("--fc_npz", type=str, default="fc_data.npz")
    parser.add_argument("--best_fold_dir", type=str, required=True,
                        help="Directory containing best_model.pt and scaler.npz")
    parser.add_argument("--out_dir", type=str, default="runs_fc_transfer_with_baselines")

    # architecture
    parser.add_argument("--emb", type=int, default=256)
    parser.add_argument("--h1", type=int, default=1024)
    parser.add_argument("--h2", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)

    # baseline params
    parser.add_argument("--pca_dim", type=int, default=20)

    # direct mlp
    parser.add_argument("--direct_mlp_epochs", type=int, default=150)
    parser.add_argument("--direct_mlp_lr", type=float, default=1e-3)
    parser.add_argument("--direct_mlp_wd", type=float, default=1e-4)
    parser.add_argument("--direct_mlp_batch", type=int, default=0)

    # transfer finetune head
    parser.add_argument("--ft_head_epochs", type=int, default=100)
    parser.add_argument("--ft_head_lr", type=float, default=5e-4)
    parser.add_argument("--ft_head_wd", type=float, default=1e-4)
    parser.add_argument("--ft_head_batch", type=int, default=0)

    # transfer finetune full
    parser.add_argument("--ft_full_epochs", type=int, default=80)
    parser.add_argument("--ft_full_lr", type=float, default=1e-4)
    parser.add_argument("--ft_full_wd", type=float, default=1e-4)
    parser.add_argument("--ft_full_batch", type=int, default=0)

    # evaluation
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ig_steps", type=int, default=64)
    parser.add_argument("--topk_edges", type=int, default=80)
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of random seeds for LOOCV repetitions")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_publication_style()

    # -------------------------
    # Load FC
    # -------------------------
    d = np.load(args.fc_npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    print("FC loaded:", X.shape, "labels:", {0: int((y == 0).sum()), 1: int((y == 1).sum())})

    F, iu = upper_tri_features(X)
    in_dim = F.shape[1]

    # -------------------------
    # Load pretrained ASD model + ASD scaler
    # -------------------------
    model_path = os.path.join(args.best_fold_dir, "best_model.pt")
    scaler_path = os.path.join(args.best_fold_dir, "scaler.npz")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("best_fold_dir must contain best_model.pt and scaler.npz")

    mean, scale = load_scaler(scaler_path)
    F_asd_scaled = apply_scaler(F, mean, scale)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model = MLPClassifier(
        in_dim=in_dim,
        emb_dim=args.emb,
        hidden=(args.h1, args.h2),
        dropout=args.dropout
    ).to(device)
    pretrained_state_dict = torch.load(model_path, map_location=device)
    pretrained_model.load_state_dict(pretrained_state_dict, strict=True)
    pretrained_model.eval()

    # -------------------------
    # Multi-seed LOOCV evaluation
    # -------------------------
    print(f"\nRunning {args.num_seeds}-seed LOOCV for all models ...")
    all_seed_results = []
    base_seed = args.seed

    for i in range(args.num_seeds):
        seed = base_seed + i
        print(f"  Seed {seed} ({i+1}/{args.num_seeds})")
        set_seed(seed)
        res = evaluate_all_baselines_single_seed(
            F_raw=F,
            F_asd_scaled=F_asd_scaled,
            y=y,
            pretrained_model=pretrained_model,
            pretrained_state_dict=pretrained_state_dict,
            device=device,
            args=args,
            seed=seed
        )
        all_seed_results.append(res)

    # Aggregate
    aggregated = aggregate_results(all_seed_results, y, args.num_seeds)

    # -------------------------
    # Save summary CSV
    # -------------------------
    summary_csv = os.path.join(args.out_dir, "model_comparison_summary_multiseed.csv")
    save_model_summary_csv(aggregated, summary_csv)

    # Save average probs for all models
    all_avg_probs_npz = os.path.join(args.out_dir, "all_model_avg_probs_multiseed.npz")
    npz_dict = {"y": y.astype(np.int64)}
    for name, obj in aggregated.items():
        npz_dict[f"{name}_avg_probs"] = obj["avg_probs"].astype(np.float64)
    np.savez_compressed(all_avg_probs_npz, **npz_dict)

    # Save aggregated metrics as JSON
    summary_json = os.path.join(args.out_dir, "model_comparison_summary_multiseed.json")
    json_out = {}
    for name, obj in aggregated.items():
        json_out[name] = {
            "cross_seed_stats": obj["cross_seed_stats"],
            "avg_prob_metrics": {k: v for k, v in obj["metrics"].items()}
        }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Console summary
    # -------------------------
    print("\n=== Model comparison summary (mean ± SD over seeds) ===")
    for name in sorted(aggregated, key=lambda n: aggregated[n]["cross_seed_stats"]["roc_auc"]["mean"], reverse=True):
        stats = aggregated[name]["cross_seed_stats"]["roc_auc"]
        print(f"{name:40s} | AUC = {stats['mean']:.3f} ± {stats['std']:.3f}  "
              f"95% CI = [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]")

    # -------------------------
    # AUC distribution plot (six models)
    # -------------------------
    # Extract per-seed AUC values for the six models
    auc_dist_data = {}
    for name in [
        "Transfer Learning (Frozen Encoder)",
        "PCA + Logistic Regression",
        "Transfer Learning (Full Fine-Tuning)",
        "Random Encoder + Linear Probe",
        "Direct MLP (FC Only)",
        "Logistic Regression (Raw Connectivity)"
    ]:
        auc_list = [seed_res[name]["metrics"]["roc_auc"] for seed_res in all_seed_results]
        auc_dist_data[name] = auc_list

    plot_auc_distribution(
        auc_dist_data,
        os.path.join(args.out_dir, "fig_six_models_auc_distribution"),
        title=None
    )

    # -------------------------
    # Main transfer method figures (using averaged probs)
    # -------------------------
    saved_figs = []
    main_avg_probs = aggregated["Transfer Learning (Frozen Encoder)"]["avg_probs"]

    saved_figs.extend(plot_roc_curve_pub(
        y, main_avg_probs,
        os.path.join(args.out_dir, "fig_transfer_frozen_probe_avg_roc"),
        title=None
    ))

    saved_figs.extend(plot_pr_curve_pub(
        y, main_avg_probs,
        os.path.join(args.out_dir, "fig_transfer_frozen_probe_avg_pr"),
        title=None
    ))

    # Bar plot of mean AUC for all models
    mean_aucs = {name: obj["cross_seed_stats"]["roc_auc"]["mean"] for name, obj in aggregated.items()}
    fig, ax = plt.subplots(figsize=(8.4, max(4.5, 0.45 * len(mean_aucs))))
    names = sorted(mean_aucs, key=mean_aucs.get, reverse=True)
    auc_vals = [mean_aucs[n] for n in names]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, auc_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean LOOCV ROC-AUC across seeds")
    ax.set_title("Model comparison on FC dataset (multi-seed)")
    ax.set_xlim(0, 1.0)
    for i, v in enumerate(auc_vals):
        ax.text(min(v + 0.01, 0.98), i, f"{v:.3f}", va="center")
    finalize_axes(ax)
    fig.tight_layout()
    saved_figs.extend(save_figure(fig, os.path.join(args.out_dir, "fig_model_auc_comparison_multiseed")))

    # -------------------------
    # Save embeddings for transfer (using first seed, as they are deterministic after pre-training)
    # -------------------------
    first_res = all_seed_results[0]
    if "embeddings" in first_res["Transfer Learning (Frozen Encoder)"]:
        np.savez_compressed(
            os.path.join(args.out_dir, "transfer_frozen_embeddings.npz"),
            emb=first_res["Transfer Learning (Frozen Encoder)"]["embeddings"],
            y=y
        )
    if "embeddings" in first_res["Random Encoder + Linear Probe"]:
        np.savez_compressed(
            os.path.join(args.out_dir, "random_encoder_embeddings.npz"),
            emb=first_res["Random Encoder + Linear Probe"]["embeddings"],
            y=y
        )

    # -------------------------
    # IG only for frozen transfer model (based on pretrained model, no seed needed)
    # -------------------------
    print("\nRunning IG for transfer_frozen_probe model ...")
    edge_freq_case = {}
    edge_freq_ctrl = {}
    per_subject_edges = np.zeros((len(F_asd_scaled), args.topk_edges, 2), dtype=np.int16)

    for idx in range(len(F_asd_scaled)):
        xb = torch.tensor(F_asd_scaled[idx], dtype=torch.float32, device=device)
        attr = integrated_gradients(
            pretrained_model, xb, target_class=1, steps=args.ig_steps
        ).detach().cpu().numpy()
        imp = np.abs(attr)
        topk = np.argsort(-imp)[:args.topk_edges]

        for t, k in enumerate(topk):
            i = int(iu[0][k])
            j = int(iu[1][k])
            if i > j:
                i, j = j, i
            per_subject_edges[idx, t, 0] = i
            per_subject_edges[idx, t, 1] = j

            key = (i, j)
            if y[idx] == 1:
                edge_freq_case[key] = edge_freq_case.get(key, 0) + 1
            else:
                edge_freq_ctrl[key] = edge_freq_ctrl.get(key, 0) + 1

    np.savez_compressed(
        os.path.join(args.out_dir, "per_subject_topk_edges_transfer_frozen_ig.npz"),
        edges=per_subject_edges,
        y=y.astype(np.int64)
    )

    def save_freq_map(freq_map, path):
        if len(freq_map) == 0:
            np.savez_compressed(
                path,
                edges=np.zeros((0, 2), dtype=np.int64),
                freqs=np.zeros((0,), dtype=np.int64)
            )
            return
        edges = np.array(list(freq_map.keys()), dtype=np.int64)
        freqs = np.array([freq_map[tuple(e)] for e in edges], dtype=np.int64)
        np.savez_compressed(path, edges=edges, freqs=freqs)

    save_freq_map(edge_freq_case, os.path.join(args.out_dir, "edge_freq_case_transfer_frozen_ig.npz"))
    save_freq_map(edge_freq_ctrl, os.path.join(args.out_dir, "edge_freq_ctrl_transfer_frozen_ig.npz"))

    # -------------------------
    # Full text report
    # -------------------------
    report_path = os.path.join(args.out_dir, "full_model_comparison_report_multiseed.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== FC model comparison report (multi-seed) ===\n\n")
        for name in sorted(aggregated, key=lambda n: aggregated[n]["cross_seed_stats"]["roc_auc"]["mean"], reverse=True):
            stats = aggregated[name]["cross_seed_stats"]
            avg_m = aggregated[name]["metrics"]
            f.write(f"[{name}]\n")
            f.write("Cross-seed mean ± SD and 95% CI:\n")
            for metric in ["roc_auc", "pr_auc", "acc", "bal_acc", "precision",
                           "sensitivity", "specificity", "f1", "mcc"]:
                ms = stats[metric]
                f.write(f"  {metric:12s} = {ms['mean']:.4f} ± {ms['std']:.4f}  "
                        f"CI = [{ms['ci_low']:.4f}, {ms['ci_high']:.4f}]\n")
            f.write("Metrics based on averaged probabilities:\n")
            f.write(f"  ROC-AUC = {avg_m['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC  = {avg_m['pr_auc']:.4f}\n")
            f.write(f"  Balanced Acc = {avg_m['bal_acc']:.4f}\n\n")

    print("\nSaved summary files:")
    print(" -", summary_csv)
    print(" -", summary_json)
    print(" -", all_avg_probs_npz)
    print(" -", report_path)

    print("\nSaved figures (including AUC distribution):")
    print(" -", os.path.join(args.out_dir, "fig_six_models_auc_distribution.png"))
    print(" -", os.path.join(args.out_dir, "fig_six_models_auc_distribution.pdf"))

    print("\nSaved IG outputs:")
    print(" -", os.path.join(args.out_dir, "per_subject_topk_edges_transfer_frozen_ig.npz"))
    print(" -", os.path.join(args.out_dir, "edge_freq_case_transfer_frozen_ig.npz"))
    print(" -", os.path.join(args.out_dir, "edge_freq_ctrl_transfer_frozen_ig.npz"))


if __name__ == "__main__":
    main()
