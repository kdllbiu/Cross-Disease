import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler


def upper_tri_features(X):
    """
    X: (N,116,116) symmetric matrices
    Return: (N,6670) upper-triangle (i<j) features
    """
    n = X.shape[1]
    iu = np.triu_indices(n, k=1)
    return X[:, iu[0], iu[1]]


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


@torch.no_grad()
def eval_split(model, X, y, device, batch_size=128):
    """
    Return: acc, auc
    """
    model.eval()
    probs = []
    preds = []
    ys = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        logits, _ = model(xb)
        pr = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pd = logits.argmax(dim=1).detach().cpu().numpy()
        probs.append(pr)
        preds.append(pd)
        ys.append(y[i:i + batch_size])
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    auc = roc_auc_score(ys, probs) if len(np.unique(ys)) == 2 else float("nan")
    acc = accuracy_score(ys, preds)
    return acc, auc


@torch.no_grad()
def predict_probs(model, X, device, batch_size=128):
    """
    Return: probs (N,)
    """
    model.eval()
    probs = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        logits, _ = model(xb)
        pr = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(pr)
    return np.concatenate(probs)


def class_weights(y, device):
    counts = np.bincount(y, minlength=2).astype(np.float32)
    w = (counts.sum() / (2.0 * counts + 1e-8)).astype(np.float32)
    return torch.tensor(w, device=device)


def aggregate_cv_predictions(roc_items):
    """
    将所有fold测试集预测拼接起来
    Return:
        y_true_all: (N,)
        y_prob_all: (N,)
    """
    y_true_all = np.concatenate([it["y_true"] for it in roc_items])
    y_prob_all = np.concatenate([it["y_prob"] for it in roc_items])
    return y_true_all, y_prob_all


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=2000, seed=42):
    """
    Bootstrap ROC-AUC distribution and 95% CI
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = y_prob[idx]

        if len(np.unique(y_b)) < 2:
            continue

        aucs.append(roc_auc_score(y_b, p_b))

    aucs = np.asarray(aucs, dtype=float)
    mean_auc = float(np.mean(aucs))
    ci_lower = float(np.percentile(aucs, 2.5))
    ci_upper = float(np.percentile(aucs, 97.5))
    return aucs, mean_auc, ci_lower, ci_upper


def run_mlp_cv_once(
    F,
    y,
    folds=5,
    seed=46,
    emb=256,
    h1=1024,
    h2=512,
    dropout=0.3,
    epochs=120,
    batch=128,
    lr=1e-3,
    weight_decay=1e-4,
    log_every=10,
    device=None,
    verbose=False,
):
    """
    对给定标签 y 跑一次完整 MLP CV，并返回聚合测试集预测的 ROC-AUC
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    all_probs = []
    all_preds = []
    all_true = []

    fold_aucs = []
    fold_accs = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        if verbose:
            print(f"\n[run_mlp_cv_once] Fold {fold_id}/{folds}")

        Xtr, Xte = F[train_idx], F[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        model = MLPClassifier(
            in_dim=F.shape[1],
            emb_dim=emb,
            hidden=(h1, h2),
            dropout=dropout
        ).to(device)

        crit = nn.CrossEntropyLoss(weight=class_weights(ytr, device))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        rng = np.random.default_rng(seed + fold_id)

        tr0 = train_idx[y[train_idx] == 0]
        tr1 = train_idx[y[train_idx] == 1]

        rng.shuffle(tr0)
        rng.shuffle(tr1)

        n0v = max(1, int(0.10 * len(tr0)))
        n1v = max(1, int(0.10 * len(tr1)))

        val_idx = np.concatenate([tr0[:n0v], tr1[:n1v]])
        sub_train_idx = np.setdiff1d(train_idx, val_idx)

        Xsub = scaler.transform(F[sub_train_idx])
        ysub = y[sub_train_idx]
        Xval = scaler.transform(F[val_idx])
        yval = y[val_idx]

        best_epoch_auc = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            perm = rng.permutation(len(ysub))
            losses = []

            for i in range(0, len(ysub), batch):
                idx = perm[i:i + batch]
                xb = torch.tensor(Xsub[idx], dtype=torch.float32, device=device)
                yb = torch.tensor(ysub[idx], dtype=torch.long, device=device)

                opt.zero_grad()
                logits, _ = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()

                losses.append(loss.item())

            if epoch % log_every == 0 or epoch == 1:
                val_acc, val_auc = eval_split(model, Xval, yval, device, batch_size=batch)

                if val_auc > best_epoch_auc:
                    best_epoch_auc = val_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                if verbose:
                    print(
                        f"  epoch {epoch:03d} "
                        f"loss={np.mean(losses):.4f} "
                        f"val_acc={val_acc:.3f} "
                        f"val_auc={val_auc:.3f}"
                    )

        if best_state is not None:
            model.load_state_dict(best_state)

        test_acc, test_auc = eval_split(model, Xte_s, yte, device, batch_size=batch)
        y_prob = predict_probs(model, Xte_s, device, batch_size=batch)
        y_pred = (y_prob >= 0.5).astype(np.int64)

        all_probs.append(y_prob)
        all_preds.append(y_pred)
        all_true.append(yte)

        fold_aucs.append(test_auc)
        fold_accs.append(test_acc)

        if verbose:
            print(f"  [Fold {fold_id}] test_acc={test_acc:.3f} test_auc={test_auc:.3f}")

    y_true_all = np.concatenate(all_true)
    y_prob_all = np.concatenate(all_probs)
    y_pred_all = np.concatenate(all_preds)

    auc_all = roc_auc_score(y_true_all, y_prob_all)
    acc_all = accuracy_score(y_true_all, y_pred_all)

    return {
        "auc": float(auc_all),
        "acc": float(acc_all),
        "y_true_all": y_true_all,
        "y_prob_all": y_prob_all,
        "fold_aucs": np.asarray(fold_aucs, dtype=float),
        "fold_accs": np.asarray(fold_accs, dtype=float),
    }


def permutation_test_auc_mlp_from_observed(
    F,
    y,
    observed_auc,
    folds=5,
    n_perm=100,
    seed=42,
    emb=256,
    h1=1024,
    h2=512,
    dropout=0.3,
    epochs=120,
    batch=128,
    lr=1e-3,
    weight_decay=1e-4,
    log_every=10,
    device=None,
    verbose=True,
):
    """
    基于已经算好的 observed_auc，只对 permutation 部分重跑 MLP
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(seed)
    perm_aucs = []

    if verbose:
        print(f"\n[Permutation Test - MLP] Observed aggregated CV AUC = {observed_auc:.4f}")
        print(f"[Permutation Test - MLP] Running {n_perm} permutations ...")

    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_seed = seed + 1000 + i

        if verbose:
            print(f"[Permutation Test - MLP] Permutation {i + 1}/{n_perm}")

        perm_result = run_mlp_cv_once(
            F=F,
            y=y_perm,
            folds=folds,
            seed=perm_seed,
            emb=emb,
            h1=h1,
            h2=h2,
            dropout=dropout,
            epochs=epochs,
            batch=batch,
            lr=lr,
            weight_decay=weight_decay,
            log_every=log_every,
            device=device,
            verbose=False,
        )

        auc_perm = perm_result["auc"]
        perm_aucs.append(auc_perm)

        if verbose:
            print(f"  perm_auc = {auc_perm:.4f}")

    perm_aucs = np.asarray(perm_aucs, dtype=float)
    p_value = (np.sum(perm_aucs >= observed_auc) + 1.0) / (len(perm_aucs) + 1.0)

    if verbose:
        print(f"[Permutation Test - MLP] Done. p-value = {p_value:.6f}")

    return observed_auc, perm_aucs, float(p_value)


def plot_cv_roc(roc_items, out_path, title="ASD 5-fold ROC (Test)"):
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.8,
    })

    fig, ax = plt.subplots(figsize=(4.6, 4.1))

    mean_fpr = np.linspace(0.0, 1.0, 200)
    tprs = []
    aucs = []

    for item in roc_items:
        y_true = item["y_true"]
        y_prob = item["y_prob"]
        auc = item["auc"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        ax.plot(fpr, tpr, color="#bdbdbd", linewidth=1.0, alpha=0.7)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(auc)

    mean_tpr = np.mean(np.vstack(tprs), axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0

    ax.plot(
        mean_fpr, mean_tpr,
        color="#1f77b4",
        linewidth=2.0,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Chance")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, pad=6)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.5, width=0.9)
    ax.grid(False)

    plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close()


def plot_auc_boxplot(auc_values, out_path, title="ASD CV AUC Distribution"):
    plt.figure()
    plt.boxplot([auc_values], tick_labels=["AUC"])
    plt.ylabel("AUC")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_acc_boxplot(acc_values, out_path, title="ASD CV Accuracy Distribution"):
    plt.figure()
    plt.boxplot([acc_values], tick_labels=["Accuracy"])
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_combined_figure(roc_items, auc_values, acc_values, out_path, title_prefix="ASD CV"):
    """
    三联图：ROC + AUC箱线图 + Accuracy箱线图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax_roc = axes[0]
    mean_fpr = np.linspace(0.0, 1.0, 200)
    tprs = []
    aucs = []
    for item in roc_items:
        y_true = item["y_true"]
        y_prob = item["y_prob"]
        auc = item["auc"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_roc.plot(fpr, tpr, color='gray', linewidth=1, alpha=0.5)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(auc)

    mean_tpr = np.mean(np.vstack(tprs), axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    ax_roc.plot(mean_fpr, mean_tpr, linewidth=2.5, label=f'Mean ROC (AUC={mean_auc:.3f}±{std_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', linewidth=1.0, label='Chance')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right', fontsize=8)
    ax_roc.text(0.02, 0.98, '(A)', transform=ax_roc.transAxes, fontsize=12, fontweight='bold', va='top')

    ax_auc = axes[1]
    ax_auc.boxplot([auc_values], tick_labels=['AUC'])
    ax_auc.set_ylabel('AUC')
    ax_auc.set_title(f'Mean AUC = {np.mean(auc_values):.3f} ± {np.std(auc_values, ddof=1):.3f}', fontsize=9)
    ax_auc.text(0.02, 0.98, '(B)', transform=ax_auc.transAxes, fontsize=12, fontweight='bold', va='top')

    ax_acc = axes[2]
    ax_acc.boxplot([acc_values], tick_labels=['Accuracy'])
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title(f'Mean Acc = {np.mean(acc_values):.3f} ± {np.std(acc_values, ddof=1):.3f}', fontsize=9)
    ax_acc.text(0.02, 0.98, '(C)', transform=ax_acc.transAxes, fontsize=12, fontweight='bold', va='top')

    fig.suptitle(f"{title_prefix} Performance", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_four_panel_paper_figure(
    roc_items,
    out_path,
    title_prefix="ASD",
    perm_auc_results=None,
    bootstrap_results=None,
    dpi=600
):
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ✅ 四边框版本（关键修改点）
    def finalize_axes(ax):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

        ax.tick_params(direction="out", length=3.5, width=0.9)
        ax.grid(False)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(9.2, 7.2),
        constrained_layout=True
    )
    ax_roc, ax_pr, ax_perm, ax_boot = axes.ravel()

    y_true_all, y_prob_all = aggregate_cv_predictions(roc_items)

    # =========================================================
    # A. ROC（标题里加 A.）
    # =========================================================
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    auc_val = roc_auc_score(y_true_all, y_prob_all)

    ax_roc.plot(
        fpr, tpr,
        color="#1f77b4",
        linewidth=2.0,
        label=f"AUC = {auc_val:.3f}"
    )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)


    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")

    ax_roc.set_title("ASD ROC curve", pad=6)
    ax_roc.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.0,
        label="Chance"
    )

    leg = ax_roc.legend(
        loc="lower right",
        frameon=True,
        facecolor="#eeeeee",
        edgecolor="none",
        framealpha=1.0
    )

    # 让legend更紧凑一点（可选）
    for text in leg.get_texts():
        text.set_fontsize(9)

    finalize_axes(ax_roc)

    # =========================================================
    # B. PR
    # =========================================================
    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    ap_val = average_precision_score(y_true_all, y_prob_all)
    prevalence = np.mean(y_true_all)

    ax_pr.plot(
        recall, precision,
        color="#1f77b4",
        linewidth=2.0,
        label=f"AP = {ap_val:.3f}"
    )
    ax_pr.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        linewidth=1.0,
        label=f"Chance = {prevalence:.3f}"
    )

    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")

    ax_pr.set_title("B. ASD Precision-Recall curve", pad=6)
    leg = ax_pr.legend(
        loc="lower left",
        frameon=True,
        facecolor="#eeeeee",
        edgecolor="none",
        framealpha=1.0
    )

    for text in leg.get_texts():
        text.set_fontsize(9)

    finalize_axes(ax_pr)

    # =========================================================
    # C. Permutation
    # =========================================================
    if perm_auc_results is not None:
        observed_auc, perm_aucs, p_value = perm_auc_results

        bins = min(30, max(8, len(perm_aucs)))
        ax_perm.hist(
            perm_aucs,
            bins=bins,
            color="#c7c7c7",
            edgecolor="white",
            linewidth=0.5
        )

        ax_perm.axvline(observed_auc, color="#d62728", linewidth=2.0)

        ax_perm.set_xlabel("ROC-AUC under label permutations")
        ax_perm.set_ylabel("Frequency")
        ax_perm.set_title("C. Permutation test of ROC-AUC", pad=6)

        ax_perm.text(
            0.04, 0.95,
            f"Observed = {observed_auc:.3f}\nP = {p_value:.3f}",
            transform=ax_perm.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#eeeeee",
                edgecolor="none",
                alpha=1.0
            )
        )

    finalize_axes(ax_perm)

    # =========================================================
    # D. Bootstrap
    # =========================================================
    if bootstrap_results is not None:
        boot_aucs, mean_auc, ci_lower, ci_upper = bootstrap_results
        observed_auc_boot = roc_auc_score(y_true_all, y_prob_all)

        ax_boot.hist(
            boot_aucs,
            bins=30,
            color="#4C78A8",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5
        )

        ax_boot.axvline(mean_auc, color="black", linewidth=1.6)
        ax_boot.axvline(ci_lower, color="#2E6FDC", linestyle="--", linewidth=1.5)
        ax_boot.axvline(ci_upper, color="#2E6FDC", linestyle="--", linewidth=1.5)
        ax_boot.axvline(observed_auc_boot, color="#d62728", linewidth=2.0)

        ax_boot.set_xlabel("Bootstrap ROC-AUC")
        ax_boot.set_ylabel("Frequency")
        ax_boot.set_title("D. Bootstrap distribution of ROC-AUC", pad=6)

        ax_boot.text(
            0.04, 0.95,
            f"Mean = {mean_auc:.3f}\n95% CI [{ci_lower:.3f}, {ci_upper:.3f}]\nObserved = {observed_auc_boot:.3f}",
            transform=ax_boot.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#eeeeee",
                edgecolor="none",
                alpha=1.0
            )
        )

    finalize_axes(ax_boot)

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asd_npz", type=str, default="asd_data.npz")
    parser.add_argument("--out_dir", type=str, default="runs_asd_mlp_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--emb", type=int, default=256)
    parser.add_argument("--h1", type=int, default=1024)
    parser.add_argument("--h2", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--n_perm", type=int, default=100)
    parser.add_argument("--n_bootstrap", type=int, default=2000)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.asd_npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    print("ASD loaded:", X.shape, "labels:", {0: int((y == 0).sum()), 1: int((y == 1).sum())})

    F = upper_tri_features(X)
    in_dim = F.shape[1]
    print("Feature dim:", in_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_test_auc = []
    fold_test_acc = []
    fold_models = []
    roc_items = []

    best_overall = {"test_auc": -1.0, "path": None}

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"\n=== Fold {fold_id}/{args.folds} ===")

        Xtr, Xte = F[train_idx], F[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        model = MLPClassifier(
            in_dim=in_dim,
            emb_dim=args.emb,
            hidden=(args.h1, args.h2),
            dropout=args.dropout
        ).to(device)

        crit = nn.CrossEntropyLoss(weight=class_weights(ytr, device))
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_epoch_auc = -1.0
        best_state = None

        rng = np.random.default_rng(args.seed + fold_id)

        tr0 = train_idx[y[train_idx] == 0]
        tr1 = train_idx[y[train_idx] == 1]
        rng.shuffle(tr0)
        rng.shuffle(tr1)
        n0v = max(1, int(0.10 * len(tr0)))
        n1v = max(1, int(0.10 * len(tr1)))
        val_idx = np.concatenate([tr0[:n0v], tr1[:n1v]])
        sub_train_idx = np.setdiff1d(train_idx, val_idx)

        Xsub = scaler.transform(F[sub_train_idx])
        ysub = y[sub_train_idx]
        Xval = scaler.transform(F[val_idx])
        yval = y[val_idx]

        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = rng.permutation(len(ysub))
            losses = []
            for i in range(0, len(ysub), args.batch):
                idx = perm[i:i + args.batch]
                xb = torch.tensor(Xsub[idx], dtype=torch.float32, device=device)
                yb = torch.tensor(ysub[idx], dtype=torch.long, device=device)
                opt.zero_grad()
                logits, _ = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())

            if epoch % args.log_every == 0 or epoch == 1:
                val_acc, val_auc = eval_split(model, Xval, yval, device, batch_size=args.batch)
                if val_auc > best_epoch_auc:
                    best_epoch_auc = val_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"epoch {epoch:03d} loss={np.mean(losses):.4f} val_acc={val_acc:.3f} val_auc={val_auc:.3f}")

        model.load_state_dict(best_state)
        test_acc, test_auc = eval_split(model, Xte_s, yte, device, batch_size=args.batch)

        y_prob = predict_probs(model, Xte_s, device, batch_size=args.batch)
        roc_items.append({
            "fold": fold_id,
            "y_true": yte.copy(),
            "y_prob": y_prob.copy(),
            "auc": float(test_auc),
        })

        fold_test_auc.append(test_auc)
        fold_test_acc.append(test_acc)

        fold_dir = os.path.join(args.out_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        np.savez(
            os.path.join(fold_dir, "split_indices.npz"),
            train_idx=train_idx,
            test_idx=test_idx
        )

        model_path = os.path.join(fold_dir, "best_model.pt")
        torch.save(model.state_dict(), model_path)

        scaler_path = os.path.join(fold_dir, "scaler.npz")
        np.savez_compressed(
            scaler_path,
            mean=scaler.mean_.astype(np.float32),
            scale=scaler.scale_.astype(np.float32)
        )

        fold_models.append((fold_id, model_path, scaler_path, test_auc, test_acc))
        print(f"[Fold {fold_id}] test_auc={test_auc:.3f} test_acc={test_acc:.3f} saved={fold_dir}")

        if test_auc > best_overall["test_auc"]:
            best_overall["test_auc"] = test_auc
            best_overall["path"] = fold_dir

    fold_test_auc = np.array(fold_test_auc, dtype=float)
    fold_test_acc = np.array(fold_test_acc, dtype=float)

    # basic plotting
    roc_fig_path = os.path.join(args.out_dir, "fig_asd_cv_roc_5fold.png")
    plot_cv_roc(roc_items, roc_fig_path, title="ASD 5-fold ROC (Test)")

    auc_fig_path = os.path.join(args.out_dir, "fig_asd_cv_auc_boxplot.png")
    acc_fig_path = os.path.join(args.out_dir, "fig_asd_cv_acc_boxplot.png")
    plot_auc_boxplot(fold_test_auc, auc_fig_path)
    plot_acc_boxplot(fold_test_acc, acc_fig_path)

    roc_npz_path = os.path.join(args.out_dir, "cv_test_roc_items.npz")
    y_true_obj = np.array([it["y_true"] for it in roc_items], dtype=object)
    y_prob_obj = np.array([it["y_prob"] for it in roc_items], dtype=object)

    np.savez_compressed(
        roc_npz_path,
        folds=np.array([it["fold"] for it in roc_items], dtype=int),
        y_true=y_true_obj,
        y_prob=y_prob_obj,
        auc=np.array([it["auc"] for it in roc_items], dtype=float),
    )

    combined_fig_path = os.path.join(args.out_dir, "fig_combined_triptych.png")
    plot_combined_figure(
        roc_items,
        fold_test_auc,
        fold_test_acc,
        combined_fig_path,
        title_prefix="ASD CV"
    )

    # aggregated CV predictions
    y_true_all, y_prob_all = aggregate_cv_predictions(roc_items)
    cv_auc_observed = roc_auc_score(y_true_all, y_prob_all)
    cv_ap_observed = average_precision_score(y_true_all, y_prob_all)

    # bootstrap
    boot_aucs, boot_mean_auc, boot_ci_lower, boot_ci_upper = bootstrap_auc_ci(
        y_true_all,
        y_prob_all,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed
    )
    bootstrap_results = (boot_aucs, boot_mean_auc, boot_ci_lower, boot_ci_upper)

    # full MLP permutation test
    observed_auc_perm, perm_aucs, perm_p = permutation_test_auc_mlp_from_observed(
        F=F,
        y=y,
        observed_auc=cv_auc_observed,
        folds=args.folds,
        n_perm=args.n_perm,
        seed=args.seed,
        emb=args.emb,
        h1=args.h1,
        h2=args.h2,
        dropout=args.dropout,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        device=device,
        verbose=True,
    )
    perm_results = (observed_auc_perm, perm_aucs, perm_p)

    perm_npz_path = os.path.join(args.out_dir, "permutation_test_mlp_auc.npz")
    np.savez_compressed(
        perm_npz_path,
        observed_auc=np.array([observed_auc_perm], dtype=float),
        perm_aucs=perm_aucs.astype(float),
        p_value=np.array([perm_p], dtype=float),
        n_perm=np.array([args.n_perm], dtype=int),
        seed=np.array([args.seed], dtype=int),
    )

    # four-panel paper figure
    four_panel_fig_path = os.path.join(args.out_dir, "fig_four_panel_paper.png")
    plot_four_panel_paper_figure(
        roc_items=roc_items,
        out_path=four_panel_fig_path,
        title_prefix="ASD",
        perm_auc_results=perm_results,
        bootstrap_results=bootstrap_results,
        dpi=600
    )

    summary = os.path.join(args.out_dir, "cv_summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        for fold_id, model_path, scaler_path, tauc, tacc in fold_models:
            f.write(f"fold={fold_id} test_auc={tauc:.4f} test_acc={tacc:.4f} model={model_path} scaler={scaler_path}\n")
        f.write("\n")
        f.write(f"test_auc_mean={fold_test_auc.mean():.4f} test_auc_std={fold_test_auc.std(ddof=1):.4f}\n")
        f.write(f"test_acc_mean={fold_test_acc.mean():.4f} test_acc_std={fold_test_acc.std(ddof=1):.4f}\n")
        f.write(f"best_fold_dir={best_overall['path']}\n")
        f.write(f"\nfig_roc={roc_fig_path}\n")
        f.write(f"fig_auc_boxplot={auc_fig_path}\n")
        f.write(f"fig_acc_boxplot={acc_fig_path}\n")
        f.write(f"roc_items_npz={roc_npz_path}\n")
        f.write(f"combined_figure={combined_fig_path}\n")
        f.write(f"four_panel_figure={four_panel_fig_path}\n")
        f.write(f"permutation_npz={perm_npz_path}\n")
        f.write(f"\ncv_auc_aggregated={cv_auc_observed:.4f}\n")
        f.write(f"cv_ap_aggregated={cv_ap_observed:.4f}\n")
        f.write(f"bootstrap_auc_mean={boot_mean_auc:.4f}\n")
        f.write(f"bootstrap_auc_ci95=[{boot_ci_lower:.4f}, {boot_ci_upper:.4f}]\n")
        f.write(f"permutation_auc_observed={observed_auc_perm:.4f}\n")
        f.write(f"permutation_p={perm_p:.4f}\n")

    print("\n=== CV Summary ===")
    print(f"test_auc_mean={fold_test_auc.mean():.3f} ± {fold_test_auc.std(ddof=1):.3f}")
    print(f"test_acc_mean={fold_test_acc.mean():.3f} ± {fold_test_acc.std(ddof=1):.3f}")
    print(f"Aggregated CV AUC = {cv_auc_observed:.3f}")
    print(f"Aggregated CV AP  = {cv_ap_observed:.3f}")
    print(f"Bootstrap AUC mean = {boot_mean_auc:.3f}, 95% CI [{boot_ci_lower:.3f}, {boot_ci_upper:.3f}]")
    print(f"Permutation observed AUC = {observed_auc_perm:.3f}, p = {perm_p:.3f}")
    print("Best fold dir:", best_overall["path"])
    print("Saved:", summary)
    print("Saved ROC figure:", roc_fig_path)
    print("Saved four-panel figure:", four_panel_fig_path)


if __name__ == "__main__":
    main()
