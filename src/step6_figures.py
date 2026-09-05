"""그림 생성.

  그림 1  전처리 및 간섭 주입 (원본 / 주입 후 / 주입 성분)
  그림 2  모델 구조
  그림 3  분류 성능 (주 비교)
  그림 4  파괴까지 남은 시간 회귀, 스틱슬립 사이클별 색상
  그림 5  전조 상태 특징 ablation
  그림 6  손실 구성요소 ablation
  그림 7  교차 어텐션 해석 (step7에서 생성)
  그림 8  회귀 진단 (step8에서 생성)
  그림 9  동일 오경보율에서의 검출 성능

[검토 반영 v2]
  C-9   그림 1 을 백색 배경 계열 컬러맵 + 백분위 스케일링으로 교체.
        기존에는 vmin=0, vmax=최댓값 이라 화소 대부분이 컬러맵 최하단(검정)에
        눌려 음향 방출 신호가 보이지 않았다. 3패널(원본/주입후/주입성분)로
        만들어 "잡음 없는 그림과 비교가 안 된다"는 지적도 함께 해소한다.
  C-7   회귀 그림이 동일 관측치를 시드 수만큼 겹쳐 그리던 문제. 대표 시드를
        그리고 지표는 시드 평균±표준편차로 표기한다.
  C-8   혼동행렬을 마지막 시드 하나가 아니라 전 시드 합산으로 만든다.
  C-10  그림 간 모델 포함 규칙을 통일한다(융합 ablation 포함, 캡션에 명시).
  A-6   동일 FAR 그림의 격자가 촘촘해졌다.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_curve, auc, confusion_matrix)

import config
from figstyle import apply_style, style_of, panel_label, save, spectrogram_norm
from models_meta import PROPOSED_MODEL, NO_ATTN, SINGLE_AE, display, color

apply_style()

# 스틱슬립 사이클 구분용 색맹 안전 정성형 팔레트
CYCLE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                "#D55E00", "#F0E442", "#000000"]


def _mean_curve(seed_preds, kind="pr", n_points=200):
    grid = np.linspace(0, 1, n_points)
    ys, aucs = [], []
    for _, d in seed_preds.items():
        yt, ysc = d["y_true_cls"], d["y_score_cls"]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        if kind == "pr":
            prec, rec, _ = precision_recall_curve(yt, ysc)
            f = interp1d(rec[::-1], prec[::-1], bounds_error=False,
                         fill_value=(prec[-1], prec[0]))
            aucs.append(average_precision_score(yt, ysc))
        else:
            fpr, tpr, _ = roc_curve(yt, ysc)
            f = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
            aucs.append(auc(fpr, tpr))
        ys.append(f(grid))
    if not ys:
        return None, None, None, None
    Y = np.vstack(ys)
    return grid, Y.mean(0), Y.std(0), (float(np.mean(aucs)), float(np.std(aucs)))


def _model_order(preds, include_ablation=True):
    """[C-10] 그림 전반에 동일한 모델 순서·포함 규칙을 쓴다."""
    names = list(preds.keys())
    order = [m for m in [PROPOSED_MODEL, NO_ATTN, SINGLE_AE] if m in names]
    order += [m for m in names if m not in order]
    if not include_ablation:
        order = [m for m in order if m != NO_ATTN]
    return order


def figure_classification(preds, include_ablation=True):
    order = _model_order(preds, include_ablation)
    print("  그림 3  분류 성능")

    fig, axes = plt.subplots(1, 3, figsize=(config.FIG_DOUBLE_COL_IN, 2.4))
    any_seed = next(iter(next(iter(preds.values())).values()))
    chance = float(any_seed["y_true_cls"].mean())

    for name in order:
        g, m, s, ap = _mean_curve(preds[name], "pr")
        if g is None:
            continue
        axes[0].plot(g, m, color=color(name), linestyle=style_of(name),
                     label=f"{display(name)} ({ap[0]:.3f})")
        axes[0].fill_between(g, m - s, m + s, color=color(name), alpha=0.13, lw=0)
    axes[0].axhline(chance, color="0.5", lw=0.7, ls=":",
                    label=f"Chance ({chance:.3f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Precision–recall", pad=4)
    axes[0].legend(loc="lower left", handlelength=1.8)
    panel_label(axes[0], "a")

    for name in order:
        g, m, s, a = _mean_curve(preds[name], "roc")
        if g is None:
            continue
        axes[1].plot(g, m, color=color(name), linestyle=style_of(name),
                     label=f"{display(name)} ({a[0]:.3f})")
        axes[1].fill_between(g, m - s, m + s, color=color(name), alpha=0.13, lw=0)
    axes[1].plot([0, 1], [0, 1], color="0.5", lw=0.7, ls=":")
    axes[1].axvline(config.TARGET_FAR_PCT / 100, color=config.PALETTE["highlight"],
                    lw=0.7, ls="-.")
    axes[1].text(config.TARGET_FAR_PCT / 100 + 0.035, 1.005,
                 f"target FAR {config.TARGET_FAR_PCT:.0f}%",
                 fontsize=config.FIG_BASE_FONTSIZE - 2, va="bottom",
                 color=config.PALETTE["highlight"])
    axes[1].set_xlabel("False alarm rate")
    axes[1].set_ylabel("Detection rate")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_title("ROC", pad=4)
    axes[1].legend(loc="lower right", handlelength=1.8,
                   bbox_to_anchor=(1.02, -0.02))
    panel_label(axes[1], "b")

    # [C-8] 혼동행렬은 전 시드 합산으로 만든다 (기존: 마지막 시드 1개)
    if PROPOSED_MODEL in preds:
        sp = preds[PROPOSED_MODEL]
        cm = np.zeros((2, 2), dtype=np.int64)
        for d in sp.values():
            thr = d.get("threshold", 0.5)
            cm += confusion_matrix(d["y_true_cls"],
                                   (d["y_score_cls"] >= thr).astype(int),
                                   labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        far = 100 * fp / (fp + tn) if (fp + tn) else 0
        rec = 100 * tp / (tp + fn) if (tp + fn) else 0
        prec = 100 * tp / (tp + fp) if (tp + fp) else 0
        axes[2].imshow(cm, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                axes[2].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                             fontsize=config.FIG_BASE_FONTSIZE,
                             color="white" if cm[i, j] > cm.max() * 0.55 else "black")
        axes[2].set_xticks([0, 1], ["Normal", "Warning"])
        axes[2].set_yticks([0, 1], ["Normal", "Warning"])
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("Observed")
        axes[2].set_title(f"FAR {far:.1f}%, recall {rec:.1f}%\n"
                          f"precision {prec:.1f}% ({len(sp)} seeds pooled)", pad=4)
        for sp_ in axes[2].spines.values():
            sp_.set_visible(True)
            sp_.set_linewidth(0.6)
        panel_label(axes[2], "c")

    fig.tight_layout(w_pad=1.6)
    path = save(fig, "fig3_classification.pdf")
    print(f"      캡션: 곡선은 시드 {len(next(iter(preds.values())))}개의 평균, "
          f"음영은 ±1 표준편차. (c)는 전 시드 합산 혼동행렬")
    return path


def figure_regression(preds):
    from metrics import regression_metrics
    print("  그림 4  파괴까지 남은 시간 회귀")
    if PROPOSED_MODEL not in preds:
        return None
    sp = preds[PROPOSED_MODEL]
    seeds = sorted(sp.keys())

    # [C-7] 지표는 시드별로 계산해 평균±표준편차로 보고하고,
    # 산점도는 대표 시드 하나만 그린다 (같은 관측치를 15번 겹쳐 그리지 않음)
    per_seed = [regression_metrics(sp[s]["y_true_ttf"], sp[s]["y_pred_ttf"])
                for s in seeds]
    mstat = {k: (float(np.mean([m[k] for m in per_seed])),
                 float(np.std([m[k] for m in per_seed])))
             for k in ("rmse", "mae", "r2", "bias")}

    rep = seeds[-1]
    d = sp[rep]
    yt, yp = d["y_true_ttf"], d["y_pred_ttf"]
    has_cyc = "cycle_id" in d

    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE_COL_IN, 3.1))
    if has_cyc:
        cyc = d["cycle_id"]
        for i, c in enumerate(np.unique(cyc)):
            sel = cyc == c
            ax.scatter(yt[sel], yp[sel], s=1.8, alpha=0.4, lw=0,
                       color=CYCLE_COLORS[i % len(CYCLE_COLORS)],
                       label=f"Cycle {int(c)}", rasterized=True)
    else:
        ax.scatter(yt, yp, s=1.8, alpha=0.4, lw=0, color=color(PROPOSED_MODEL),
                   rasterized=True)

    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.9, ls="--", label="1:1 line")
    ax.axvspan(0, config.WARNING_TTF_THRESHOLD, color=config.PALETTE["highlight"],
               alpha=0.08, lw=0)
    ax.text(config.WARNING_TTF_THRESHOLD * 0.5, lo + (hi - lo) * 0.02,
            "warning\nzone", ha="center", va="bottom",
            fontsize=config.FIG_BASE_FONTSIZE - 2,
            color=config.PALETTE["highlight"])

    txt = (f"RMSE {mstat['rmse'][0]:.2f} ± {mstat['rmse'][1]:.2f} s\n"
           f"MAE {mstat['mae'][0]:.2f} ± {mstat['mae'][1]:.2f} s\n"
           f"$R^2$ {mstat['r2'][0]:.3f} ± {mstat['r2'][1]:.3f}\n"
           f"bias {mstat['bias'][0]:+.2f} s")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes,
            fontsize=config.FIG_BASE_FONTSIZE - 1, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.7", lw=0.5))
    ax.set_xlabel("Observed time to failure (s)")
    ax.set_ylabel("Predicted time to failure (s)")
    ax.grid(True, ls=":", lw=0.4)
    leg = ax.legend(loc="upper left", markerscale=4, handlelength=1.6)
    for h in leg.legend_handles:
        try:
            h.set_alpha(1.0)
        except Exception:
            pass
    fig.tight_layout()
    path = save(fig, "fig4_ttf_regression.pdf")
    print(f"      캡션: 산점도는 대표 시드({rep}), 지표는 시드 {len(seeds)}개의 "
          f"평균 ± 표준편차")
    return path


def figure_architecture():
    print("  그림 2  모델 구조")
    fig, ax = plt.subplots(figsize=(config.FIG_DOUBLE_COL_IN, 3.0))
    ax.set_xlim(0, 13.4); ax.set_ylim(0, 6.0); ax.axis("off")

    slow = dict(boxstyle="round,pad=0.34", facecolor="#E8F1F8",
                edgecolor=config.PALETTE["proposed"], lw=0.9)
    fast = dict(boxstyle="round,pad=0.34", facecolor="#FBEEE6",
                edgecolor=config.PALETTE["spectrogram_only"], lw=0.9)
    core = dict(boxstyle="round,pad=0.40", facecolor="#F2E9F4",
                edgecolor="#7B4A9C", lw=1.3)
    head = dict(boxstyle="round,pad=0.34", facecolor="#E9F4EE",
                edgecolor=config.PALETTE["no_attention"], lw=0.9)
    fs = config.FIG_BASE_FONTSIZE - 1

    ax.text(1.45, 4.75, "Slow branch\nPrecursor state vector\n"
            f"$\\mathbb{{R}}^{{{config.STATE_DIM}}}$  "
            f"({config.TREND_WINDOW} segments $\\approx$ "
            f"{config.TREND_WINDOW_SEC:.2f} s)",
            ha="center", va="center", bbox=slow, fontsize=fs)
    ax.text(1.45, 1.50, "Fast branch\nSTFT magnitude\n"
            f"{config.FREQ_BINS} $\\times$ {config.TIME_STEPS}  "
            f"(+ synthetic interference)",
            ha="center", va="center", bbox=fast, fontsize=fs)
    ax.text(4.75, 4.75, f"MLP encoder\n{config.D_MODEL}-d",
            ha="center", va="center", bbox=slow, fontsize=fs)
    ax.text(4.75, 2.25, f"2-D CNN\n$\\to$ {config.TIME_STEPS} $\\times$ "
            f"{64*config.CNN_FREQ_BANDS}", ha="center", va="center",
            bbox=fast, fontsize=fs)
    ax.text(4.75, 0.75, f"LSTM\n$\\to$ {config.TIME_STEPS} $\\times$ "
            f"{config.D_MODEL}", ha="center", va="center", bbox=fast, fontsize=fs)
    res = "$+\\,Q$ residual" if config.ATTENTION_FUSION == "add" else ""
    ax.text(8.30, 2.80, "Asymmetric cross-attention\n"
            "$\\mathrm{Attn}(Q_{\\mathrm{slow}},K_{\\mathrm{fast}},"
            "V_{\\mathrm{fast}})$" + (f"\n{res}" if res else ""),
            ha="center", va="center", bbox=core, fontsize=fs + 0.5)
    ax.text(11.75, 4.30, "Classification head\nWarning probability",
            ha="center", va="center", bbox=head, fontsize=fs)
    ax.text(11.75, 1.30, "Regression head\nTime to failure",
            ha="center", va="center", bbox=head, fontsize=fs)

    a = dict(arrowstyle="-|>", lw=0.8, color="0.35", mutation_scale=8)
    ax.annotate("", xy=(3.55, 4.75), xytext=(2.85, 4.75), arrowprops=a)
    ax.annotate("", xy=(3.60, 2.30), xytext=(2.85, 1.75), arrowprops=a)
    ax.annotate("", xy=(4.75, 1.25), xytext=(4.75, 1.75), arrowprops=a)
    ax.annotate("", xy=(6.85, 3.25), xytext=(5.90, 4.55),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, mutation_scale=8,
                                color=config.PALETTE["proposed"]))
    ax.annotate("", xy=(6.85, 2.55), xytext=(5.90, 0.95),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, mutation_scale=8,
                                color=config.PALETTE["spectrogram_only"]))
    ax.text(6.25, 4.10, "$Q$", fontsize=fs, color=config.PALETTE["proposed"])
    ax.text(6.25, 1.35, "$K,V$", fontsize=fs,
            color=config.PALETTE["spectrogram_only"])
    ax.annotate("", xy=(10.20, 4.10), xytext=(9.75, 3.20), arrowprops=a)
    ax.annotate("", xy=(10.20, 1.50), xytext=(9.75, 2.45), arrowprops=a)

    fig.tight_layout()
    return save(fig, "fig2_architecture.pdf")


# ----------------------------------------------------------------------
# 그림 1 — 백색 배경 스펙트로그램 [검토 C-9 / 심사 지적 06]
# ----------------------------------------------------------------------
def figure_preprocessing(h5_path=None, seg_idx=None, cmap=None, fname=None):
    """전처리 및 간섭 주입.

    수정 내용
      1. 컬러맵을 백색 배경 계열로 교체 (기본 Greys: 흰 배경 · 검은 신호).
         기존 magma 는 최하단이 검정이라 배경이 새까맣게 나왔다.
      2. vmin/vmax 를 최댓값이 아니라 백분위로 잡아 동적 범위 압축을 없앤다.
      3. 원본 · 주입 후 · 주입 성분 3패널로 만들어 비교가 가능하게 한다.
      4. 주입 대역 상한 경계를 실제 bin 계산값으로 표시한다.
    """
    import h5py
    h5_path = h5_path or config.H5_PATH
    cmap = cmap or config.FIG_SPEC_CMAP
    print(f"  그림 1  전처리 (cmap={cmap})")

    with h5py.File(h5_path, "r") as f:
        n = int(f.attrs.get("n_segments", f["X_ae"].shape[0]))
        has_clean = bool(f.attrs.get("has_clean", False))
        if seg_idx is None:
            # 경보 구간에서 신호가 잘 보이는 세그먼트를 고른다
            y = f["y_cls"][:n, 0]
            pos = np.where(y == 1.0)[0]
            seg_idx = int(pos[len(pos) // 2]) if len(pos) else 0
        noisy = f["X_ae"][seg_idx]
        clean = f["X_ae_clean"][seg_idx] if has_clean else noisy
        ttf = float(f["y_ttf"][seg_idx, 0])
        nb = int(f.attrs.get("noise_band_bins", config.NOISE_BAND_BINS))
        band_edge = float(f.attrs.get("noise_band_top_edge_hz",
                                      config.noise_band_bounds_hz()[1]))

    fb = config.FIG_SPEC_FREQ_BINS_SHOWN
    df_hz = config.FREQ_RESOLUTION_HZ
    freq_khz_max = fb * df_hz / 1000.0
    extent = [0, config.TIME_STEPS, 0.0, freq_khz_max]

    A = np.log1p(clean[:fb])
    B = np.log1p(noisy[:fb])

    # [C-9] 표시 범위는 '깨끗한 신호'(A) 기준으로 잡는다. 주입 대역(평균 12)이
    # 척도를 지배하면 실제 AE 성분이 다시 눌리기 때문이다. 주입 대역은 (b)에서
    # 상단 포화로 표현되며, 컬러바에 화살표(extend)로 표시한다.
    v_lo, v_hi = spectrogram_norm(A)

    fig, axes = plt.subplots(1, 3, figsize=(config.FIG_DOUBLE_COL_IN, 2.25))

    for ax, dat, ttl, lab in [
            (axes[0], A, "Laboratory AE (clean)", "a"),
            (axes[1], B, "With injected interference", "b")]:
        im = ax.imshow(dat, aspect="auto", cmap=cmap, origin="lower",
                       extent=extent, vmin=v_lo, vmax=v_hi, rasterized=True,
                       interpolation="nearest")
        ax.set_title(ttl, pad=4)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_facecolor("white")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                          extend="max" if lab == "b" else "neither")
        cb.set_label("log(1 + |STFT|)", fontsize=config.FIG_BASE_FONTSIZE - 2)
        cb.ax.tick_params(labelsize=config.FIG_BASE_FONTSIZE - 2, width=0.5)
        cb.outline.set_linewidth(0.4)
        panel_label(ax, lab)

    # 주입 대역 표시 (b 패널)
    edge_khz = band_edge / 1000.0
    axes[1].axhline(edge_khz, color=config.PALETTE["highlight"], lw=0.8, ls="--")
    axes[1].text(config.TIME_STEPS * 0.98, edge_khz + freq_khz_max * 0.03,
                 f"injected band\nDC – {edge_khz:.1f} kHz ({nb} bins)",
                 ha="right", va="bottom",
                 fontsize=config.FIG_BASE_FONTSIZE - 2.5,
                 color=config.PALETTE["highlight"])

    diff = B - A
    lim = float(np.percentile(np.abs(diff), 99.5)) or 1.0
    im = axes[2].imshow(diff, aspect="auto", cmap=config.FIG_SPEC_DIFF_CMAP,
                        origin="lower", extent=extent, vmin=-lim, vmax=lim,
                        rasterized=True, interpolation="nearest")
    axes[2].set_title("Injected component (b − a)", pad=4)
    axes[2].set_xlabel("Time frame")
    axes[2].set_ylabel("Frequency (kHz)")
    axes[2].set_facecolor("white")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    cb.set_label("difference", fontsize=config.FIG_BASE_FONTSIZE - 2)
    cb.ax.tick_params(labelsize=config.FIG_BASE_FONTSIZE - 2, width=0.5)
    cb.outline.set_linewidth(0.4)
    panel_label(axes[2], "c")

    fig.suptitle(f"Segment #{seg_idx}, time to failure = {ttf:.2f} s   "
                 f"(0 – {freq_khz_max:.0f} kHz shown of "
                 f"{config.NYQUIST_HZ/1000:.0f} kHz)",
                 fontsize=config.FIG_BASE_FONTSIZE, y=1.06)
    fig.tight_layout(w_pad=1.4)
    path = save(fig, fname or "fig1_preprocessing.pdf")
    print(f"      캡션에 넣을 것: 주파수 축은 하위 {fb}개 bin "
          f"(0–{freq_khz_max:.0f} kHz)만 표시한 확대 구간, "
          f"표시 범위는 {config.FIG_SPEC_PCT_LO}–{config.FIG_SPEC_PCT_HI} 백분위")
    return path


def figure_preprocessing_cmap_gallery(h5_path=None, seg_idx=None,
                                      cmaps=("Greys", "magma_r", "YlGnBu",
                                             "cividis_r", "magma")):
    """백색 배경 후보들을 나란히 렌더링해 고를 수 있게 한다."""
    for c in cmaps:
        fname = f"fig1_preprocessing__{c}.pdf"
        try:
            figure_preprocessing(h5_path, seg_idx, cmap=c, fname=fname)
        except Exception as e:
            print(f"    {c} 실패: {e}")
    print("  figures/fig1_preprocessing__*.png 를 비교해 config.FIG_SPEC_CMAP 를 "
          "정하세요.")


def _ablation_label(setting):
    """표기용 라벨. 약어의 대문자는 유지한다."""
    if setting in config.ABLATION_LABELS:
        return config.ABLATION_LABELS[setting]
    if setting.startswith("drop_"):
        feat = setting[5:]
        name = config.STATE_FEATURE_LABELS.get(feat, feat.replace("_", " "))
        head = name.split(" ")[0]
        if not head.isupper():          # RMS 같은 약어는 그대로 둔다
            name = name[0].lower() + name[1:]
        return "w/o " + name
    return setting


def figure_feature_ablation(csv=None):
    csv = csv or os.path.join(config.RESULT_DIR, "ablation_summary.csv")
    if not os.path.exists(csv):
        print("  그림 5 생략 (ablation 결과 없음)")
        return None
    print("  그림 5  특징 ablation")
    df = pd.read_csv(csv)
    n_seeds = int(df["n_seeds"].iloc[0]) if "n_seeds" in df.columns else None
    df = df[df.setting != "full"].sort_values("delta_pr_auc")

    sig_csv = os.path.join(config.RESULT_DIR, "ablation_feature_significance.csv")
    sig = {}
    if os.path.exists(sig_csv):
        s = pd.read_csv(sig_csv)
        s = s[s.metric == "pr_auc"]
        sig = dict(zip(s["setting"], s["significant_holm"]))

    group = {"drop_inter_segment", "drop_intra_segment", "drop_all"}
    colors = [config.PALETTE["highlight"] if x in group
              else config.PALETTE["proposed"] for x in df.setting]
    labels = [_ablation_label(x) for x in df.setting]

    fig, ax = plt.subplots(figsize=(config.FIG_DOUBLE_COL_IN * 0.72, 2.7))
    ax.barh(range(len(df)), df.delta_pr_auc, color=colors,
            edgecolor="black", lw=0.4, height=0.68)
    ax.set_yticks(range(len(df)), labels)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("$\\Delta$ PR-AUC relative to full state vector")
    ax.grid(True, axis="x", ls=":", lw=0.4)

    span = max(abs(df.delta_pr_auc.min()), abs(df.delta_pr_auc.max()), 1e-6)
    ax.set_xlim(df.delta_pr_auc.min() - span * 0.42,
                df.delta_pr_auc.max() + span * 0.30)
    for i, (v, s_) in enumerate(zip(df.delta_pr_auc, df.setting)):
        mark = "*" if sig.get(s_, False) else ""
        ax.text(v + (span * 0.04 if v >= 0 else -span * 0.04), i,
                f"{v:+.3f}{mark}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=config.FIG_BASE_FONTSIZE - 1.5)
    ax.set_ylim(-0.7, len(df) - 0.3)
    fig.tight_layout()
    path = save(fig, "fig5_feature_ablation.pdf")
    print("      캡션 주석: 별표는 Holm-Bonferroni 보정 후에도 유의한 설정"
          + (f", 시드 {n_seeds}개" if n_seeds else ""))
    return path


def figure_loss_ablation(csv=None):
    csv = csv or os.path.join(config.RESULT_DIR, "ablation_loss.csv")
    if not os.path.exists(csv):
        print("  그림 6 생략 (손실 ablation 결과 없음)")
        return None
    print("  그림 6  손실 ablation")
    df = pd.read_csv(csv)
    order = [s for s in ["full", "no_physics", "no_focusing", "plain_bce", "neither"]
             if s in df.loss_setting.unique()]
    labels = {"full": "Full loss", "no_physics": "w/o energy weighting",
              "no_focusing": "w/o focusing ($\\gamma$=0)",
              "plain_bce": "plain BCE", "neither": "Neither"}

    fig, axes = plt.subplots(1, 2, figsize=(config.FIG_DOUBLE_COL_IN * 0.82, 2.4))
    for ax, metric, ylab, lab in [(axes[0], "pr_auc", "PR-AUC", "a"),
                                  (axes[1], "rmse", "RMSE (s)", "b")]:
        data = [df.loc[df.loss_setting == s, metric].values for s in order]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True,
                        medianprops=dict(color="black", lw=0.9),
                        flierprops=dict(marker="o", ms=2, mfc="0.4", mec="none"),
                        boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5),
                        capprops=dict(lw=0.5))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(config.PALETTE["proposed"] if order[i] == "full"
                                else "0.85")
            patch.set_edgecolor("black")
        ax.set_xticks(range(1, len(order) + 1),
                      [labels.get(s, s) for s in order], rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", ls=":", lw=0.4)
        panel_label(ax, lab)
    fig.tight_layout(w_pad=1.6)
    return save(fig, "fig6_loss_ablation.pdf")


def figure_iso_far(csv=None):
    csv = csv or os.path.join(config.RESULT_DIR, "iso_far_recall_raw.csv")
    if not os.path.exists(csv):
        print("  그림 9 생략 (동일 FAR 분석 결과 없음)")
        return None
    print("  그림 9  동일 오경보율 검출 성능")
    df = pd.read_csv(csv)
    agg = df.groupby(["model", "target_far"])["recall"].agg(
        ["mean", "std"]).reset_index()

    sig_csv = os.path.join(config.RESULT_DIR, "iso_far_significance.csv")
    sig = pd.read_csv(sig_csv) if os.path.exists(sig_csv) else None

    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE_COL_IN * 1.15, 2.7))
    order = [PROPOSED_MODEL] + [m for m in agg.model.unique() if m != PROPOSED_MODEL]
    for name in order:
        sub = agg[agg.model == name].sort_values("target_far")
        if sub.empty:
            continue
        ax.errorbar(sub.target_far, sub["mean"], yerr=sub["std"].fillna(0),
                    color=color(name), linestyle=style_of(name), marker="o",
                    ms=3, capsize=2, lw=1.1, label=display(name))

    # 유의하지 않은 구간을 음영으로 표시 (심사 지적 18)
    if sig is not None and not sig.empty:
        ns = sig.groupby("target_far")["significant_holm"].apply(lambda s: not s.any())
        fars = sorted(ns.index)
        for i, f in enumerate(fars):
            if ns[f]:
                w = (fars[1] - fars[0]) / 2 if len(fars) > 1 else 1.0
                ax.axvspan(f - w * 0.5, f + w * 0.5, color="0.5", alpha=0.08, lw=0)

    ax.set_xlabel("Matched false alarm rate (%)")
    ax.set_ylabel("Recall (%)")
    ax.grid(True, ls=":", lw=0.4)
    handles, labels = ax.get_legend_handles_labels()
    if sig is not None and not sig.empty:
        handles.append(Line2D([], [], color="0.5", alpha=0.25, lw=6))
        labels.append("no significant difference")
    ax.legend(handles, labels, loc="lower right", handlelength=2.0,
              fontsize=config.FIG_BASE_FONTSIZE - 1.5)
    fig.tight_layout()
    path = save(fig, "fig9_iso_far.pdf")
    print("      캡션: 재현율은 평가 곡선에서 동일 FAR 로 정렬한 상한이며 "
          "배포 성능이 아님 (검토 A-7)")
    return path


def run(skip_data_figures=False, cmap_gallery=False):
    print("\n" + "=" * 72)
    print("그림 생성")
    print("=" * 72)
    figure_architecture()

    cache = os.path.join(config.ARTIFACT_DIR, "prediction_cache.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            preds = pickle.load(f)["predictions"]
        clas = os.path.join(config.ARTIFACT_DIR, "classical_prediction_cache.pkl")
        if os.path.exists(clas):
            with open(clas, "rb") as f:
                preds.update(pickle.load(f)["predictions"])
        # [C-1] 캐시 무결성 확인
        # [수정 v2.1] 예전에는 config.EVAL_SEEDS(15개)와 비교해서,
        #   --smoke / --fast 로 돌리면 모든 모델에 경고가 떴다. 잡을 대상은
        #   '모델마다 시드 수가 다른' 들쭉날쭉한 캐시이므로 최대값과 비교한다.
        if preds:
            counts = {m: len(sp) for m, sp in preds.items()}
            n_max = max(counts.values())
            ragged = {m: c for m, c in counts.items() if c != n_max}
            for m, c in ragged.items():
                print(f"  경고: {display(m)} 의 시드가 {c}개로 다른 모델"
                      f"({n_max}개)보다 적습니다. "
                      f"tools/check_integrity.py 로 확인하세요.")
            if not ragged and n_max != len(config.EVAL_SEEDS):
                print(f"  참고: 이 그림은 시드 {n_max}개로 그렸습니다 "
                      f"(config.EVAL_SEEDS 는 {len(config.EVAL_SEEDS)}개). "
                      f"캡션의 시드 수를 맞추세요.")
        figure_classification(preds)
        figure_regression(preds)
    else:
        print("  그림 3, 4 생략 (예측 캐시 없음)")

    figure_feature_ablation()
    figure_loss_ablation()
    figure_iso_far()

    if not skip_data_figures and os.path.exists(config.H5_PATH):
        try:
            if cmap_gallery:
                figure_preprocessing_cmap_gallery()
            else:
                figure_preprocessing()
        except Exception as e:
            print(f"  그림 1 생성 실패: {e}")

    print(f"\n  그림 저장 위치: {config.FIGURE_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data-figures", action="store_true")
    ap.add_argument("--cmap-gallery", action="store_true",
                    help="그림 1을 여러 컬러맵으로 렌더링해 비교")
    ap.add_argument("--only-fig1", action="store_true")
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--seg", type=int, default=None)
    args = ap.parse_args()
    if args.only_fig1:
        if args.cmap_gallery:
            figure_preprocessing_cmap_gallery(seg_idx=args.seg)
        else:
            figure_preprocessing(seg_idx=args.seg, cmap=args.cmap)
    else:
        run(skip_data_figures=args.skip_data_figures,
            cmap_gallery=args.cmap_gallery)
