"""그림 1 컬러맵 비교 미리보기.

데이터셋이 있으면 실제 세그먼트로, 없으면 통계를 모사한 합성 데이터로
'기존 렌더링'과 '백색 배경 후보들'을 나란히 그려 준다.

    python tools/preview_fig1_colormaps.py            # h5 있으면 실제 데이터
    python tools/preview_fig1_colormaps.py --synthetic # 항상 합성 데이터
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# (컬러맵, 영문 설명)  — 설명은 그림에 그대로 찍히므로 ASCII 로 둔다
CANDIDATES = [
    ("Greys", "white bg, black signal (print-safe, default)"),
    ("magma_r", "cream bg, dark signal (perceptually uniform)"),
    ("YlGnBu", "near-white bg, blue signal"),
    ("cividis_r", "colour-vision-deficiency safe"),
    ("bone_r", "white bg, cool grey"),
]


def synthetic_pair(fb=120, T=None, seed=0):
    """실제 스펙트로그램의 통계를 흉내 낸 (clean, noisy) 쌍."""
    T = T or config.TIME_STEPS
    rng = np.random.default_rng(seed)

    # 배경: 주파수가 올라갈수록 감쇠하는 저수준 잡음
    f = np.arange(fb)[:, None]
    base = rng.exponential(0.18, size=(fb, T)) * np.exp(-f / 70.0)

    # 산발적 AE 이벤트: 좁은 시간 폭의 수직 줄무늬
    clean = base.copy()
    for t in rng.choice(T, size=9, replace=False):
        amp = rng.uniform(1.5, 6.0)
        prof = np.exp(-f / rng.uniform(25, 60)).ravel()
        w = rng.integers(1, 3)
        for dt in range(-w, w + 1):
            tt = int(np.clip(t + dt, 0, T - 1))
            clean[:, tt] += amp * prof * rng.uniform(0.5, 1.0)

    # 합성 저주파 간섭 (config 와 동일 규칙)
    noisy = clean.copy()
    nb = config.NOISE_BAND_BINS
    noise = rng.normal(config.NOISE_MEAN, config.NOISE_STD, size=(nb, T))
    noisy[:nb, :] = np.clip(clean[:nb, :] + noise, 0, None)
    return clean, noisy


def real_pair(fb=120, seg_idx=None):
    import h5py
    with h5py.File(config.H5_PATH, "r") as f:
        n = int(f.attrs.get("n_segments", f["X_ae"].shape[0]))
        if seg_idx is None:
            y = f["y_cls"][:n, 0]
            pos = np.where(y == 1.0)[0]
            seg_idx = int(pos[len(pos) // 2]) if len(pos) else 0
        noisy = f["X_ae"][seg_idx][:fb]
        clean = (f["X_ae_clean"][seg_idx][:fb]
                 if bool(f.attrs.get("has_clean", False)) else noisy)
    return clean, noisy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seg", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.FIGURE_DIR,
                                                  "fig1_colormap_comparison.png"))
    args = ap.parse_args()

    fb = config.FIG_SPEC_FREQ_BINS_SHOWN
    if args.synthetic or not os.path.exists(config.H5_PATH):
        clean, noisy = synthetic_pair(fb)
        src = "synthetic data (matched to real statistics)"
    else:
        clean, noisy = real_pair(fb, args.seg)
        src = f"real data ({os.path.basename(config.H5_PATH)})"

    A, B = np.log1p(clean), np.log1p(noisy)
    T = B.shape[1]
    khz = fb * config.FREQ_RESOLUTION_HZ / 1000.0
    extent = [0, T, 0.0, khz]

    n_rows = 1 + len(CANDIDATES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9.2, 2.05 * n_rows))
    plt.rcParams.update({"font.size": 8})

    # --- 기존 렌더링 -------------------------------------------------
    vmax_old = float(A.max())
    for j, (dat, ttl) in enumerate([(A, "clean"), (B, "with interference")]):
        ax = axes[0, j]
        im = ax.imshow(dat, aspect="auto", cmap="magma", origin="lower",
                       extent=extent, vmin=0, vmax=vmax_old,
                       interpolation="nearest")
        ax.set_title(f"BEFORE  magma, vmin=0, vmax=max — {ttl}",
                     fontsize=8.5, color="#9B2F27", pad=3)
        ax.set_ylabel("kHz", fontsize=7.5)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6.5)

    # --- 수정 렌더링 -------------------------------------------------
    # 표시 범위는 '깨끗한 신호'(A) 기준. 주입 대역은 상단에서 포화시킨다.
    v_lo = float(np.percentile(A, config.FIG_SPEC_PCT_LO))
    v_hi = float(np.percentile(A, config.FIG_SPEC_PCT_HI))

    for i, (cm, desc) in enumerate(CANDIDATES, start=1):
        for j, (dat, ttl) in enumerate([(A, "clean"), (B, "with interference")]):
            ax = axes[i, j]
            im = ax.imshow(dat, aspect="auto", cmap=cm, origin="lower",
                           extent=extent, vmin=v_lo, vmax=v_hi,
                           interpolation="nearest")
            ax.set_facecolor("white")
            if j == 0:
                ax.set_title(f"AFTER  {cm} + {config.FIG_SPEC_PCT_LO}–"
                             f"{config.FIG_SPEC_PCT_HI} pct — {desc}",
                             fontsize=8.5, color="#14655A", pad=3, loc="left")
            else:
                ax.set_title(ttl, fontsize=8, pad=3)
            ax.set_ylabel("kHz", fontsize=7.5)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                         extend="max" if j == 1 else "neither"
                         ).ax.tick_params(labelsize=6.5)

    for ax in axes[-1]:
        ax.set_xlabel("Time frame", fontsize=7.5)

    fig.suptitle(f"Figure 1 background fix — colormap comparison  ({src})",
                 fontsize=10.5, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out, dpi=170, facecolor="white")
    print(f"저장: {args.out}")
    print(f"표시 범위: {v_lo:.3f} ~ {v_hi:.3f}  (기존: 0 ~ {vmax_old:.3f})")
    print("마음에 드는 컬러맵을 config.FIG_SPEC_CMAP 에 지정하세요.")


if __name__ == "__main__":
    main()
