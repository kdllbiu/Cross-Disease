import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

# ---------------- AAL116 labels (0-based, standard order) ----------------
AAL116_NAMES = [
    # 1-90 (cortex/subcortex)
    "Precentral_L","Precentral_R","Frontal_Sup_L","Frontal_Sup_R","Frontal_Sup_Orb_L","Frontal_Sup_Orb_R",
    "Frontal_Mid_L","Frontal_Mid_R","Frontal_Mid_Orb_L","Frontal_Mid_Orb_R","Frontal_Inf_Oper_L","Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L","Frontal_Inf_Tri_R","Frontal_Inf_Orb_L","Frontal_Inf_Orb_R","Rolandic_Oper_L","Rolandic_Oper_R",
    "Supp_Motor_Area_L","Supp_Motor_Area_R","Olfactory_L","Olfactory_R","Frontal_Sup_Medial_L","Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L","Frontal_Med_Orb_R","Rectus_L","Rectus_R","Insula_L","Insula_R",
    "Cingulum_Ant_L","Cingulum_Ant_R","Cingulum_Mid_L","Cingulum_Mid_R","Cingulum_Post_L","Cingulum_Post_R",
    "Hippocampus_L","Hippocampus_R","ParaHippocampal_L","ParaHippocampal_R","Amygdala_L","Amygdala_R",
    "Calcarine_L","Calcarine_R","Cuneus_L","Cuneus_R","Lingual_L","Lingual_R",
    "Occipital_Sup_L","Occipital_Sup_R","Occipital_Mid_L","Occipital_Mid_R","Occipital_Inf_L","Occipital_Inf_R",
    "Fusiform_L","Fusiform_R","Postcentral_L","Postcentral_R","Parietal_Sup_L","Parietal_Sup_R",
    "Parietal_Inf_L","Parietal_Inf_R","SupraMarginal_L","SupraMarginal_R","Angular_L","Angular_R",
    "Precuneus_L","Precuneus_R","Paracentral_Lobule_L","Paracentral_Lobule_R","Caudate_L","Caudate_R",
    "Putamen_L","Putamen_R","Pallidum_L","Pallidum_R","Thalamus_L","Thalamus_R",
    "Heschl_L","Heschl_R","Temporal_Sup_L","Temporal_Sup_R","Temporal_Pole_Sup_L","Temporal_Pole_Sup_R",
    "Temporal_Mid_L","Temporal_Mid_R","Temporal_Pole_Mid_L","Temporal_Pole_Mid_R","Temporal_Inf_L","Temporal_Inf_R",
    # 91-116 (cerebellum + vermis)
    "Cerebelum_Crus1_L","Cerebelum_Crus1_R","Cerebelum_Crus2_L","Cerebelum_Crus2_R",
    "Cerebelum_3_L","Cerebelum_3_R","Cerebelum_4_5_L","Cerebelum_4_5_R","Cerebelum_6_L","Cerebelum_6_R",
    "Cerebelum_7b_L","Cerebelum_7b_R","Cerebelum_8_L","Cerebelum_8_R","Cerebelum_9_L","Cerebelum_9_R",
    "Cerebelum_10_L","Cerebelum_10_R","Vermis_1_2","Vermis_3","Vermis_4_5","Vermis_6","Vermis_7","Vermis_8","Vermis_9","Vermis_10"
]

# ---------------- Network mapping (coarse anatomical) ----------------
def aal_to_network(roi_name: str) -> str:
    n = roi_name

    # Cerebellum / vermis
    if n.startswith("Cerebelum") or n.startswith("Vermis"):
        return "Cerebellar"

    # Subcortical nuclei
    if any(x in n for x in ["Caudate", "Putamen", "Pallidum", "Thalamus"]):
        return "Subcortical"

    # Limbic-ish (coarse)
    if any(x in n for x in ["Hippocampus", "ParaHippocampal", "Amygdala", "Cingulum", "Olfactory", "Insula", "Rectus"]):
        return "Limbic"

    # Occipital
    if any(x in n for x in ["Calcarine","Cuneus","Lingual","Occipital"]):
        return "Occipital"

    # Temporal
    if any(x in n for x in ["Temporal","Heschl","Fusiform"]):
        return "Temporal"

    # Parietal
    if any(x in n for x in ["Postcentral","Parietal","SupraMarginal","Angular","Precuneus","Paracentral_Lobule"]):
        return "Parietal"

    # Frontal (default for remaining cortex)
    return "Frontal"

def pair_key(a: str, b: str) -> str:
    return " - ".join(sorted([a, b]))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

def compute_network_size_null(
    roi2net,
    networks,
    n_samples,
    k_edges,
    n_iter=5000,
    seed=0
):
    """
    Generate network-size-matched null distribution.

    Randomly samples k_edges from all possible AAL116 edges,
    preserving the anatomical network size distribution.
    """

    rng = np.random.default_rng(seed)

    # all possible undirected edges
    all_edges = []

    for i in range(116):
        for j in range(i+1,116):
            all_edges.append((i,j))

    all_edges = np.array(all_edges)


    # network pair index
    net_pairs = [
        pair_key(a,b)
        for a,b in combinations_with_replacement(networks,2)
    ]

    net2idx = {
        p:i for i,p in enumerate(net_pairs)
    }

    P=len(net_pairs)


    null_rates=np.zeros((n_iter,P))


    for it in range(n_iter):

        idx=rng.choice(
            len(all_edges),
            size=k_edges,
            replace=False
        )

        sampled_edges=all_edges[idx]


        counts=np.zeros(P)


        for i,j in sampled_edges:

            ni=roi2net[i]
            nj=roi2net[j]

            pk=pair_key(ni,nj)

            counts[net2idx[pk]]+=1


        null_rates[it]=counts/k_edges


    return net_pairs, null_rates

# =========================
# NEW: plotting helpers
# =========================
def plot_enrichment_bar(net_pairs, diff_rate, qvals, alpha, out_png, title="Network enrichment (case - control)"):
    """
    Bar plot over all network pairs. Significant bars (q<=alpha) are hatched.
    """
    x = np.arange(len(net_pairs))

    plt.figure(figsize=(max(10, 0.45 * len(net_pairs)), 4.5))
    sig = (qvals <= alpha)
    colors = []

    for d, s in zip(diff_rate, sig):

        if d > 0:
            colors.append("#053061" if s else "#4393c3")

        else:
            colors.append("#c11b1b" if s else "#d6604d")

    bars = plt.bar(
        x,
        diff_rate,
        color=colors
    )

    sig = (qvals <= alpha)
    for i, b in enumerate(bars):
        if sig[i]:
            b.set_hatch("//")  # hatch highlight without specifying color
            b.set_linewidth(1.5)

    plt.axhline(
        0,
        color="black",
        linewidth=1,
        zorder=10
    )
    plt.xticks(x, net_pairs, rotation=60, ha="right")
    plt.ylabel("diff_rate (rate_FC - rate_HC)")
    plt.title(title + f"   (hatched: q<= {alpha})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_perm_distribution(perm_vals, obs_val, out_png, title="Permutation distribution"):
    """
    Histogram of permuted differences for the frontal-cerebellar network pair.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.0, 4.5))
    plt.hist(perm_vals, bins=40, edgecolor="black", linewidth=0.6)

    plt.axvline(obs_val, linewidth=2.0, label=f"Observed = {obs_val:.4f}")
    plt.axvline(0.0, linestyle="--", linewidth=1.5, label="Null center = 0")

    plt.xlabel("Permuted diff_rate")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_sig_matrix(networks, net_pairs, diff_rate, qvals, alpha, out_png, title="Significant network-pairs (upper triangle)"):
    """
    7x7 matrix heatmap (upper triangle) using diff_rate; mark significant pairs with square markers.
    No manual colormap/color specified.
    """
    n = len(networks)
    net2i = {name: i for i, name in enumerate(networks)}

    M = np.full((n, n), np.nan, dtype=np.float32)
    Sig = np.zeros((n, n), dtype=bool)

    for pk, d, q in zip(net_pairs, diff_rate, qvals):
        a, b = pk.split(" - ")
        i = net2i[a]
        j = net2i[b]
        # fill symmetric then mask later; but we'll only show upper triangle
        M[i, j] = float(d)
        M[j, i] = float(d)
        if q <= alpha:
            Sig[i, j] = True
            Sig[j, i] = True

    # mask lower triangle + diagonal
    for i in range(n):
        M[i, :i+1] = np.nan

    plt.figure(figsize=(6.2, 5.2))
    im = plt.imshow(M, aspect="equal")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(n), networks, rotation=45, ha="right")
    plt.yticks(np.arange(n), networks)
    plt.title(title + f"   (q<= {alpha})")
    plt.tight_layout()

    # overlay significant markers (upper triangle only)
    ys, xs = np.where(Sig)
    mask_ut = (xs > ys)  # upper triangle positions
    xs = xs[mask_ut]
    ys = ys[mask_ut]
    if len(xs) > 0:
        plt.scatter(xs, ys, marker="s", facecolors="none")  # default edge color

    plt.savefig(out_png, dpi=300)
    plt.close()

HC_COLOR = "#E96552"
FC_COLOR = "#67C5DB"

def plot_fc_hc_violin_box_jitter(values_fc, values_hc, out_png, q_value=None,
                                 title="Frontal–cerebellar enrichment",
                                 ylabel="Proportion of key connections"):
    import numpy as np
    import matplotlib.pyplot as plt

    fc = np.asarray(values_fc, dtype=float)
    hc = np.asarray(values_hc, dtype=float)

    fig, ax = plt.subplots(figsize=(4.2, 5.0))

    data = [hc, fc]
    positions = [0, 1]

    # violin
    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for body, color in zip(vp["bodies"], [HC_COLOR, FC_COLOR]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.3)

    # box
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2)
    )
    for patch, color in zip(bp["boxes"], [HC_COLOR, FC_COLOR]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.75)

    # jitter points
    rng = np.random.default_rng(0)
    jitter_hc = rng.normal(positions[0], 0.045, size=len(hc))
    jitter_fc = rng.normal(positions[1], 0.045, size=len(fc))

    ax.scatter(
        jitter_hc,
        hc,
        s=26,
        alpha=0.85,
        color=HC_COLOR
    )

    ax.scatter(
        jitter_fc,
        fc,
        s=26,
        alpha=0.85,
        color=FC_COLOR
    )

    # mean marker
    ax.scatter(
        [positions[0]],
        [hc.mean()],
        marker="D",
        s=42,
        color=HC_COLOR,
        edgecolor="black",
        zorder=5
    )

    ax.scatter(
        [positions[1]],
        [fc.mean()],
        marker="D",
        s=42,
        color=FC_COLOR,
        edgecolor="black",
        zorder=5
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(["HC", "FC"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # significance line
    y_max = max(np.max(hc), np.max(fc))
    y_min = min(np.min(hc), np.min(fc))
    h = (y_max - y_min) * 0.08 if y_max > y_min else 0.02
    y_line = y_max + h

    ax.plot([0, 0, 1, 1], [y_line, y_line + h * 0.35, y_line + h * 0.35, y_line],
            color="black",
            linewidth=1.5)

    # ===== 修改这里：用 * 代替 q 值 =====
    if q_value is not None:
        # 根据 q 值显示星号
        if q_value <= 0.001:
            star = '***'
        elif q_value <= 0.01:
            star = '**'
        elif q_value <= 0.05:
            star = '*'
        else:
            star = ''  # 不显著就不显示星号

        if star:  # 只有显著时才显示
            ax.text(0.5, y_line + h * 0.42, star, ha="center", va="bottom",
                    fontsize=14, fontweight='bold')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def bootstrap_group_diff(values_fc, values_hc, n_boot=200, seed=0):
    """
    Bootstrap mean and 95% CI for:
    - FC group mean
    - HC group mean
    - FC - HC group difference
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    fc = np.asarray(values_fc, dtype=float)
    hc = np.asarray(values_hc, dtype=float)

    n_fc = len(fc)
    n_hc = len(hc)

    boot_fc = np.zeros(n_boot, dtype=float)
    boot_hc = np.zeros(n_boot, dtype=float)
    boot_diff = np.zeros(n_boot, dtype=float)

    for b in range(n_boot):
        fc_idx = rng.integers(0, n_fc, size=n_fc)
        hc_idx = rng.integers(0, n_hc, size=n_hc)

        fc_mean = float(np.mean(fc[fc_idx]))
        hc_mean = float(np.mean(hc[hc_idx]))
        diff_mean = fc_mean - hc_mean

        boot_fc[b] = fc_mean
        boot_hc[b] = hc_mean
        boot_diff[b] = diff_mean

    def summarize(arr):
        mean = float(np.mean(arr))
        ci_low = float(np.quantile(arr, 0.025))
        ci_high = float(np.quantile(arr, 0.975))
        return mean, ci_low, ci_high

    fc_mean, fc_low, fc_high = summarize(boot_fc)
    hc_mean, hc_low, hc_high = summarize(boot_hc)
    diff_mean, diff_low, diff_high = summarize(boot_diff)

    return {
        "boot_fc": boot_fc,
        "boot_hc": boot_hc,
        "boot_diff": boot_diff,
        "fc_mean": fc_mean,
        "fc_ci": (fc_low, fc_high),
        "hc_mean": hc_mean,
        "hc_ci": (hc_low, hc_high),
        "diff_mean": diff_mean,
        "diff_ci": (diff_low, diff_high),
    }

def plot_bootstrap_ci_panel(fc_mean, fc_ci, hc_mean, hc_ci, diff_mean, diff_ci,
                            out_png, title="Bootstrap estimates across groups"):
    """
    Horizontal error-bar plot for HC, FC, and FC-HC difference.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["FC", "HC", "FC - HC"]
    means = np.array([fc_mean, hc_mean, diff_mean], dtype=float)
    ci_lows = np.array([fc_ci[0], hc_ci[0], diff_ci[0]], dtype=float)
    ci_highs = np.array([fc_ci[1], hc_ci[1], diff_ci[1]], dtype=float)

    y = np.array([2, 1, 0])

    xerr = np.vstack([means - ci_lows, ci_highs - means])

    plt.figure(figsize=(6.2, 4.2))
    plt.errorbar(
        means, y, xerr=xerr,
        fmt='o',
        capsize=4,
        linewidth=1.5,
        markersize=6
    )

    plt.axvline(0.0, linestyle="--", linewidth=1.2)

    plt.yticks(y, labels)
    plt.xlabel("Bootstrap estimate (mean and 95% CI)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_bootstrap_distribution(boot_diff, diff_mean, diff_ci, out_png,
                                title="Bootstrap distribution of frontal-cerebellar difference"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.0, 4.5))
    plt.hist(boot_diff, bins=30, edgecolor="black", linewidth=0.6)

    plt.axvline(diff_mean, linewidth=2.0, label=f"Mean = {diff_mean:.4f}")
    plt.axvline(diff_ci[0], linestyle="--", linewidth=1.5, label=f"95% CI low = {diff_ci[0]:.4f}")
    plt.axvline(diff_ci[1], linestyle="--", linewidth=1.5, label=f"95% CI high = {diff_ci[1]:.4f}")

    plt.xlabel("Difference in enrichment rate (FC - HC)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_subject_npz", required=True,
                    help="runs_fc_mlp_transfer/per_subject_topk_edges.npz")
    ap.add_argument("--out_dir", default="runs_fc_network_enrichment")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument(
        "--n_null",
        type=int,
        default=5000
    )

    # NEW: plotting options
    ap.add_argument("--bar_sort", choices=["none", "abs"], default="none",
                    help="Sort bars by abs(diff_rate) for plotting (does not affect stats).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = np.load(args.per_subject_npz, allow_pickle=True)
    edges = d["edges"].astype(np.int16)  # (N, K, 2) with 0-based indices
    y = d["y"].astype(np.int64)
    N, K, _ = edges.shape

    if len(AAL116_NAMES) != 116:
        raise RuntimeError("AAL116_NAMES length is not 116. Please check the list.")

    print(f"Loaded: N={N}, topk={K}, cases={(y==1).sum()}, controls={(y==0).sum()}")

    # Build ROI->network
    roi2net = [aal_to_network(name) for name in AAL116_NAMES]
    networks = ["Frontal","Parietal","Temporal","Occipital","Limbic","Subcortical","Cerebellar"]

    # Define all network-pairs (unordered, with self-pairs)
    net_pairs = [pair_key(a, b) for a, b in combinations_with_replacement(networks, 2)]
    net2idx = {p:i for i,p in enumerate(net_pairs)}
    P = len(net_pairs)

    # For each subject, count how many of its topk edges fall into each network-pair
    subj_counts = np.zeros((N, P), dtype=np.float32)

    for s in range(N):
        for t in range(K):
            i = int(edges[s, t, 0]); j = int(edges[s, t, 1])
            if i == j:
                continue
            ni = roi2net[i]
            nj = roi2net[j]
            pk = pair_key(ni, nj)
            subj_counts[s, net2idx[pk]] += 1.0

    # Convert to rates (per subject, divide by K)
    subj_rates = subj_counts / float(K)

    case = (y == 1)
    ctrl = (y == 0)

    obs_case_raw = subj_rates[case].mean(axis=0)
    obs_ctrl_raw = subj_rates[ctrl].mean(axis=0)

    # ==============================
    # Network-size corrected null
    # ==============================

    null_pairs, null_rates = compute_network_size_null(
        roi2net=roi2net,
        networks=networks,
        n_samples=N,
        k_edges=K,
        n_iter=5000,
        seed=args.seed
    )

    null_mean = null_rates.mean(axis=0)

    # avoid division by zero
    null_mean[null_mean == 0] = 1e-12

    # normalized enrichment score
    obs_case = obs_case_raw / null_mean
    obs_ctrl = obs_ctrl_raw / null_mean

    diff_obs = obs_case - obs_ctrl
    abs_obs = np.abs(diff_obs)

    # Choose one "representative" pair for permutation distribution plotting:
    # If any significant exists after testing, pick smallest p (later), but we need p after permutation.
    # For collecting perm distribution, use the maximum |diff| pair as stable choice.
    # Fixed target pair for Figure 6B
    target_pair = "Cerebellar - Frontal"
    rep_idx = net2idx[target_pair]
    rep_pair = net_pairs[rep_idx]
    rep_obs = float(diff_obs[rep_idx])

    # Permutation test (two-sided on |diff|)
    exceed = np.zeros(P, dtype=np.int32)
    y_perm = y.copy()

    # NEW: record permutation distribution for representative pair
    rep_perm_vals = np.zeros(args.n_perm, dtype=np.float32)

    for b in range(args.n_perm):
        rng.shuffle(y_perm)
        cm = (y_perm == 1)
        tm = (y_perm == 0)
        case_perm_raw = subj_rates[cm].mean(axis=0)
        ctrl_perm_raw = subj_rates[tm].mean(axis=0)

        dp = (
                case_perm_raw / null_mean
                -
                ctrl_perm_raw / null_mean
        )

        exceed += (np.abs(dp) >= abs_obs).astype(np.int32)
        rep_perm_vals[b] = float(dp[rep_idx])

        if (b + 1) % 1000 == 0:
            print(f"perm {b+1}/{args.n_perm}")

    pvals = (exceed + 1) / (args.n_perm + 1)
    qvals = bh_fdr(pvals)

    out = pd.DataFrame({

        "network_pair": net_pairs,

        "raw_rate_case": obs_case_raw,

        "raw_rate_ctrl": obs_ctrl_raw,

        "null_expected_rate": null_mean,

        "NES_case": obs_case,

        "NES_ctrl": obs_ctrl,

        "NES_difference": diff_obs,

        "abs_difference": np.abs(diff_obs),

        "p_perm": pvals,

        "q_bh": qvals
    }).sort_values(["q_bh", "abs_difference"], ascending=[True, False])

    out_all = os.path.join(args.out_dir, "network_enrichment.csv")
    out.to_csv(out_all, index=False, encoding="utf-8-sig")

    sig = out[out["q_bh"] <= args.alpha]
    out_sig = os.path.join(args.out_dir, f"network_enrichment_sig_q{args.alpha:.2f}.csv")
    sig.to_csv(out_sig, index=False, encoding="utf-8-sig")

    # NEW: save permutation distribution for the representative pair
    perm_npz = os.path.join(args.out_dir, "perm_distribution_representative_pair.npz")
    np.savez_compressed(
        perm_npz,
        rep_pair=np.array([rep_pair], dtype=object),
        rep_obs=np.array([rep_obs], dtype=np.float32),
        rep_perm_vals=rep_perm_vals.astype(np.float32)
    )

    # =========================
    # NEW: A/B/C plots
    # =========================
    # For plotting, optionally sort bars by abs(diff) to look nicer
    plot_pairs = np.array(net_pairs, dtype=object)
    plot_diff = diff_obs.copy()
    plot_q = qvals.copy()

    if args.bar_sort == "abs":
        ordp = np.argsort(-np.abs(plot_diff))
        plot_pairs = plot_pairs[ordp]
        plot_diff = plot_diff[ordp]
        plot_q = plot_q[ordp]

    figA = os.path.join(args.out_dir, "fig6_A.png")
    plot_enrichment_bar(
        net_pairs=list(plot_pairs),
        diff_rate=plot_diff,
        qvals=plot_q,
        alpha=args.alpha,
        out_png=figA,
        title="A. Network enrichment (FC - HC)"
    )

    figB = os.path.join(args.out_dir, "fig6_C.png")
    plot_perm_distribution(
        perm_vals=rep_perm_vals,
        obs_val=rep_obs,
        out_png=figB,
        title="C. Permutation distribution: Frontal - Cerebellar"
    )

    figC = os.path.join(args.out_dir, "fig_significant_pair_highlight_matrix.png")
    plot_sig_matrix(
        networks=networks,
        net_pairs=net_pairs,
        diff_rate=diff_obs,
        qvals=qvals,
        alpha=args.alpha,
        out_png=figC,
        title="Network-pair enrichment matrix (upper triangle)"
    )

    print("\nSaved:")
    print(" -", out_all)
    print(" -", out_sig)
    print(" -", perm_npz)
    print(" -", figA)
    print(" -", figB)
    print(" -", figC)
    print(f"Significant network pairs (q <= {args.alpha}): {len(sig)}")
    print(f"Representative pair for permutation plot: {rep_pair} (obs diff_rate={rep_obs:.4f})")

    target_pair = "Cerebellar - Frontal"
    target_idx = net2idx[target_pair]

    fc_vals = subj_rates[case, target_idx] / null_mean[target_idx]

    hc_vals = subj_rates[ctrl, target_idx] / null_mean[target_idx]

    # 1) 给 Prism 的宽表
    max_len = max(len(hc_vals), len(fc_vals))
    hc_pad = np.full(max_len, np.nan)
    fc_pad = np.full(max_len, np.nan)
    hc_pad[:len(hc_vals)] = hc_vals
    fc_pad[:len(fc_vals)] = fc_vals

    df_prism = pd.DataFrame({
        "HC": hc_pad,
        "FC": fc_pad
    })
    df_prism.to_csv(os.path.join(args.out_dir, "prism_fig6_panelA_frontal_cerebellar.csv"), index=False)

    # 2) 长表备份
    df_long = pd.DataFrame({
        "Group": ["HC"] * len(hc_vals) + ["FC"] * len(fc_vals),
        "Value": np.concatenate([hc_vals, fc_vals])
    })
    df_long.to_csv(os.path.join(args.out_dir, "prism_fig6_panelA_frontal_cerebellar_long.csv"), index=False)

    # 从多重比较结果里取 q 值
    q_target = qvals[target_idx]
    p_target = pvals[target_idx]

    # 修改为：
    fig_panelA = os.path.join(args.out_dir, "fig6_B.png")
    plot_fc_hc_violin_box_jitter(
        values_fc=fc_vals,
        values_hc=hc_vals,
        out_png=fig_panelA,
        q_value=q_target,  # 直接传 q 值，不用字符串
        title="Frontal - Cerebellar enrichment",
        ylabel="Proportion of key connections"
    )

    # =========================
    # Figure 6C: bootstrap CI plot for frontal-cerebellar pair
    # =========================
    target_pair = "Cerebellar - Frontal"
    target_idx = net2idx[target_pair]

    fc_vals = subj_rates[case, target_idx]
    hc_vals = subj_rates[ctrl, target_idx]

    boot_res = bootstrap_group_diff(
        values_fc=fc_vals,
        values_hc=hc_vals,
        n_boot=args.n_boot,
        seed=args.seed + 200
    )

    # save bootstrap arrays/results
    np.savez_compressed(
        os.path.join(args.out_dir, "bootstrap_frontal_cerebellar_panelC.npz"),
        boot_fc=boot_res["boot_fc"],
        boot_hc=boot_res["boot_hc"],
        boot_diff=boot_res["boot_diff"],
        fc_mean=np.array([boot_res["fc_mean"]], dtype=float),
        fc_ci=np.array(boot_res["fc_ci"], dtype=float),
        hc_mean=np.array([boot_res["hc_mean"]], dtype=float),
        hc_ci=np.array(boot_res["hc_ci"], dtype=float),
        diff_mean=np.array([boot_res["diff_mean"]], dtype=float),
        diff_ci=np.array(boot_res["diff_ci"], dtype=float),
    )

    # save summary csv
    df_boot = pd.DataFrame({
        "Group": ["FC", "HC", "FC-HC"],
        "Mean": [boot_res["fc_mean"], boot_res["hc_mean"], boot_res["diff_mean"]],
        "CI_low": [boot_res["fc_ci"][0], boot_res["hc_ci"][0], boot_res["diff_ci"][0]],
        "CI_high": [boot_res["fc_ci"][1], boot_res["hc_ci"][1], boot_res["diff_ci"][1]],
    })
    df_boot.to_csv(
        os.path.join(args.out_dir, "bootstrap_frontal_cerebellar_panelC_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # main panel C plot
    figC_boot = os.path.join(args.out_dir, "fig6_D.png")
    plot_bootstrap_ci_panel(
        fc_mean=boot_res["fc_mean"],
        fc_ci=boot_res["fc_ci"],
        hc_mean=boot_res["hc_mean"],
        hc_ci=boot_res["hc_ci"],
        diff_mean=boot_res["diff_mean"],
        diff_ci=boot_res["diff_ci"],
        out_png=figC_boot,
        title="D. Bootstrap estimates across groups"
    )

    # optional: bootstrap histogram for supplementary or old Fig.9A style
    figC_hist = os.path.join(args.out_dir, "fig6_panelC_bootstrap_hist_frontal_cerebellar.png")
    plot_bootstrap_distribution(
        boot_diff=boot_res["boot_diff"],
        diff_mean=boot_res["diff_mean"],
        diff_ci=boot_res["diff_ci"],
        out_png=figC_hist,
        title="Bootstrap distribution of frontal-cerebellar difference"
    )

if __name__ == "__main__":
    main()
