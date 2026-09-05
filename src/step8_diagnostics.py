"""회귀 밴드 구조와 운영 임계값 전이에 대한 진단.

예측-관측 산점도가 대각선 주변에 흩어지지 않고 여러 갈래의 밴드를 형성한다.
이 모듈은 그 밴드가 스틱슬립 사이클과 대응하는지, 그리고 학습 사이클이 다루는
파괴까지 남은 시간 범위를 벗어난 외삽에서 비롯되는지를 검증한다.

두 번째 진단은 검증에서 선택한 운영 임계값이 평가 데이터에 적용될 때 얼마나
어긋나는지를 정량화한다.

[검토 반영 v2]
  C-14 계통 편향 판정 기준을 metrics.per_cycle_regression 하나로 통일.
       (기존에는 여기와 metrics 가 서로 다른 식을 써서 같은 질문에 다른
        답을 낼 수 있었다.)
  B-5  임계값 전이를 명목 목표와 선택 기준 양쪽 대비로 보고.
  C-2  사이클 경계 세그먼트 제외 여부를 진단 출력에 명시.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import config
from figstyle import apply_style, panel_label, save
from metrics import per_cycle_regression, selection_far_target_pct
from models_meta import PROPOSED_MODEL, display

apply_style()

CYCLE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                "#D55E00", "#F0E442", "#000000"]


def _load_cache():
    p = os.path.join(config.ARTIFACT_DIR, "prediction_cache.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found; run step3 first.")
    with open(p, "rb") as f:
        return pickle.load(f)["predictions"]


def check_extrapolation(h5_path=None):
    """파티션별 TTF 범위를 비교하여 외삽 여부를 확인."""
    import h5py
    from splitting import cycle_aware_split
    from data import load_arrays

    h5_path = h5_path or config.H5_PATH
    if not os.path.exists(h5_path):
        print("  외삽 확인 생략 (데이터셋 없음)")
        return None

    arrays = load_arrays(h5_path, verbose=False)
    ttf = arrays["y_ttf"].reshape(-1)
    cyc = arrays["cycle_id"].reshape(-1)

    split = cycle_aware_split(cyc, config.TRAIN_RATIO, config.VAL_RATIO)
    part_of = {}
    for part in ("train", "val", "test"):
        for c in split["cycles"][part]:
            part_of[c] = part

    rows = []
    for c in np.unique(cyc):
        m = cyc == c
        rows.append({"cycle": int(c), "partition": part_of.get(int(c), "?"),
                     "n": int(m.sum()), "ttf_min": float(ttf[m].min()),
                     "ttf_max": float(ttf[m].max())})
    df = pd.DataFrame(rows)

    train_max = df.loc[df.partition == "train", "ttf_max"].max()
    df["exceeds_train_range"] = df.ttf_max > train_max + 1e-9

    print("\n  사이클별 파괴까지 남은 시간 범위")
    print("-" * 76)
    print(df.to_string(index=False))
    print("-" * 76)
    print(f"  학습 사이클의 최대 TTF: {train_max:.2f}초 "
          f"(= TTF 정규화 계수)")

    offenders = df[(df.partition == "test") & df.exceeds_train_range]
    verdict = {"train_ttf_max": float(train_max),
               "test_cycles_exceeding_train_range":
                   offenders["cycle"].astype(int).tolist(),
               "max_excess_s": float((offenders.ttf_max - train_max).max())
                   if not offenders.empty else 0.0}
    if not offenders.empty:
        print(f"  평가 사이클 {verdict['test_cycles_exceeding_train_range']}이 최대 "
              f"{offenders.ttf_max.max():.2f}초까지 이어져 학습 범위를 "
              f"{verdict['max_excess_s']:.2f}초 초과합니다.")
        print("  해당 구간의 예측은 외삽이며 계통 편향이 예상되므로, 한계에")
        print("  명시해야 합니다.")
    else:
        print("  학습 TTF 범위를 넘는 평가 사이클이 없으므로, 밴드 구조의 원인은 "
              "외삽이 아닙니다.")

    df.to_csv(os.path.join(config.RESULT_DIR, "diagnostics_ttf_range.csv"),
              index=False)
    return df, verdict


def diagnose_regression_bands(model_name=None, save_fig=True):
    model_name = model_name or PROPOSED_MODEL
    preds = _load_cache()
    if model_name not in preds:
        print(f"  캐시에 {display(model_name)}이 없습니다")
        return None

    sp = preds[model_name]
    seeds = sorted(sp.keys())

    print("\n" + "=" * 76)
    print(f"회귀 밴드 진단 ({display(model_name)}, 시드 {len(seeds)}개)")
    print("=" * 76)
    print(f"  사이클 경계 세그먼트 제외: {config.DROP_CYCLE_BOUNDARY_SEGMENT}")
    print(f"  inter 워밍업 정책        : {config.INTER_WARMUP_POLICY}")

    all_rows, overalls = [], []
    for seed in seeds:
        d = sp[seed]
        if "cycle_id" not in d:
            print("  캐시에 사이클 식별자가 없습니다. step3를 다시 실행하세요.")
            return None
        rows, overall = per_cycle_regression(d["y_true_ttf"], d["y_pred_ttf"],
                                             d["cycle_id"])
        for r in rows:
            all_rows.append({"seed": seed, **r})
        if overall:
            overalls.append(overall)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(config.RESULT_DIR, "diagnostics_per_cycle.csv"),
              index=False)

    agg = df.groupby("cycle").agg(
        n=("n", "mean"), bias_mean=("bias", "mean"), bias_std=("bias", "std"),
        rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean")).reset_index()

    print("\n  사이클별 회귀 성능 (시드 평균)")
    print("-" * 76)
    show = agg.copy()
    for c in ["bias_mean", "bias_std", "rmse_mean", "r2_mean"]:
        show[c] = show[c].map("{:+.3f}".format)
    print(show.to_string(index=False))
    print("-" * 76)

    biases = agg["bias_mean"].values
    spread = float(biases.max() - biases.min())
    pooled_rmse = float(np.mean([o["pooled_rmse"] for o in overalls])) if overalls else np.nan
    bias_sd = float(np.mean([o["bias_std_across_cycles"] for o in overalls])) \
        if overalls else np.nan
    detected = bool(np.mean([o["systematic_bias_detected"] for o in overalls]) > 0.5) \
        if overalls else False

    print(f"  사이클 간 편향 범위: {spread:.3f}초 "
          f"(최소 {biases.min():+.3f}, 최대 {biases.max():+.3f})")
    print(f"  사이클 간 편향 표준편차(시드 평균): {bias_sd:.3f}초")
    print(f"  통합 RMSE: {pooled_rmse:.3f}초")
    print(f"  판정 기준: 편향 표준편차 > {config.SYSTEMATIC_BIAS_RULE} x 통합 RMSE")

    verdict = {"bias_spread_s": spread,
               "bias_std_across_cycles_s": bias_sd,
               "pooled_rmse_s": pooled_rmse,
               "systematic_bias_rule": config.SYSTEMATIC_BIAS_RULE,
               "systematic_bias_detected": detected,
               "n_cycles": int(len(agg)), "n_seeds": len(seeds),
               "per_cycle_bias": {int(r["cycle"]): float(r["bias_mean"])
                                  for _, r in agg.iterrows()}}
    if detected:
        print("\n  사이클마다 편향이 계통적으로 다릅니다. 밴드 구조는 무작위 오차가")
        print("  아니라 사이클 간 분포 이동에서 비롯된 것이므로, 사이클을 넘어선")
        print("  일반화의 한계로 보고해야 합니다.")
        verdict["hypothesis"] = "inter_cycle_shift"
    else:
        print("\n  사이클 간 편향 차이가 통합 오차에 비해 작습니다. 밴드는 사이클")
        print("  내부 구조를 반영할 가능성이 큽니다.")
        verdict["hypothesis"] = "within_cycle_structure"

    try:
        ext = check_extrapolation()
        if ext is not None:
            verdict["extrapolation"] = ext[1]
    except Exception as e:
        print(f"  외삽 확인 불가: {e}")

    with open(os.path.join(config.RESULT_DIR, "diagnostics_regression.json"), "w") as f:
        json.dump(verdict, f, indent=2)

    if save_fig:
        _plot_cycle_diagnostics(sp, seeds, model_name)
    return df, agg, verdict


def _plot_cycle_diagnostics(sp, seeds, model_name):
    """사이클별로 분리한 예측값과 잔차."""
    print("\n  그림  회귀 진단")
    d = sp[seeds[-1]]
    yt, yp, cyc = d["y_true_ttf"], d["y_pred_ttf"], d["cycle_id"]
    uniq = np.unique(cyc)

    fig, axes = plt.subplots(2, 1, figsize=(config.FIG_SINGLE_COL_IN * 1.5, 4.6),
                             sharex=True)

    for i, c in enumerate(uniq):
        m = cyc == c
        axes[0].scatter(yt[m], yp[m], s=2.2, alpha=0.5, lw=0,
                        color=CYCLE_COLORS[i % len(CYCLE_COLORS)],
                        label=f"Cycle {int(c)} ($n$ = {int(m.sum())})",
                        rasterized=True)
    x_lo, x_hi = 0.0, float(yt.max())
    pad = 0.03 * (x_hi - x_lo)
    axes[0].plot([x_lo, x_hi], [x_lo, x_hi], color="black", lw=0.9, ls="--",
                 label="1:1 line")
    axes[0].axvspan(0, config.WARNING_TTF_THRESHOLD,
                    color=config.PALETTE["highlight"], alpha=0.08, lw=0)
    axes[0].set_ylabel("Predicted time to failure (s)")
    axes[0].set_title(f"Predictions by cycle (seed {seeds[-1]})", pad=4)
    axes[0].grid(True, ls=":", lw=0.4)
    leg = axes[0].legend(loc="upper left", markerscale=3, handlelength=1.4,
                         ncol=2, columnspacing=1.0)
    for h in leg.legend_handles:
        try:
            h.set_alpha(1.0)
        except Exception:
            pass
    panel_label(axes[0], "a", dx=-0.10, dy=1.06)

    for i, c in enumerate(uniq):
        m = cyc == c
        axes[1].scatter(yt[m], yp[m] - yt[m], s=2.2, alpha=0.5, lw=0,
                        color=CYCLE_COLORS[i % len(CYCLE_COLORS)],
                        rasterized=True)
    axes[1].axhline(0, color="black", lw=0.9, ls="--")
    axes[1].axvspan(0, config.WARNING_TTF_THRESHOLD,
                    color=config.PALETTE["highlight"], alpha=0.08, lw=0)
    axes[1].set_xlabel("Observed time to failure (s)")
    axes[1].set_ylabel("Residual (s)")
    axes[1].set_title("Residual structure", pad=4)
    axes[1].grid(True, ls=":", lw=0.4)
    panel_label(axes[1], "b", dx=-0.10, dy=1.06)

    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    for ax in axes:
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fig.tight_layout(h_pad=1.2)
    return save(fig, "fig8_regression_diagnostics.pdf")


def diagnose_threshold(model_name=None):
    model_name = model_name or PROPOSED_MODEL
    csv = os.path.join(config.RESULT_DIR, "seed_level_results.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    d = df[df.model == model_name]
    if d.empty:
        return None

    sel_target = selection_far_target_pct()

    print("\n" + "=" * 76)
    print(f"  운영 임계값 전이 ({display(model_name)})")
    print("=" * 76)
    print(f"    명목 목표 FAR         {config.TARGET_FAR_PCT:.1f}%")
    print(f"    안전계수              {config.FAR_SAFETY_FACTOR} "
          f"-> 검증 선택 기준 {sel_target:.1f}%")
    print(f"    선택된 임계값         {d['threshold_used'].mean():.4f} "
          f"(표준편차 {d['threshold_used'].std():.4f})")
    if "threshold_std" in d.columns and d["threshold_std"].notna().any():
        print(f"    검증 부트스트랩 편차  {d['threshold_std'].mean():.4f}")
        print(f"      -> 이 값이 안전계수가 필요한 이유의 정량적 근거입니다")
        print(f"         (심사 지적 14: kappa 의 근거)")
    print(f"    달성 평가 FAR         {d['far_fixed'].mean():.2f}% "
          f"(표준편차 {d['far_fixed'].std():.2f})")
    print(f"    명목 목표 대비        {d['far_overshoot'].mean():+.2f}%p")
    if "far_drift_vs_selection" in d.columns:
        print(f"    선택 기준 대비        "
              f"{d['far_drift_vs_selection'].mean():+.2f}%p  "
              f"<- val -> test 실제 이동폭")
    print(f"    임계값에서의 재현율   {d['recall_fixed'].mean():.2f}% "
          f"(표준편차 {d['recall_fixed'].std():.2f})")
    if "precision_fixed" in d.columns:
        print(f"    임계값에서의 정밀도   {d['precision_fixed'].mean():.2f}% "
              f"(경보 10번 중 약 {d['precision_fixed'].mean()/10:.1f}번이 진짜)")

    ov = d["far_overshoot"].mean()
    if abs(ov) > 2:
        suggested = (config.FAR_SAFETY_FACTOR * config.TARGET_FAR_PCT
                     / max(d["far_fixed"].mean(), 1e-6))
        print(f"\n    달성 FAR이 {ov:+.2f}%p 벗어났습니다. 참고로 FAR_SAFETY_FACTOR를")
        print(f"    약 {suggested:.2f}로 조정하면 수치상 보정되지만 재현율이 낮아집니다.")
        print("    주의: 이 값을 그대로 채택하면 평가 데이터 정보가 운영점 결정에")
        print("    유입됩니다. 논문에 kappa 를 검증만으로 정했다고 쓰려면 이 경로를")
        print("    사용하지 마세요 (검토 B-5).")
    print("\n    목표 오경보율과 달성 오경보율을 함께 보고하세요. 그 차이는 운영")
    print("    중 주기적인 재보정이 필요함을 의미합니다.")

    out = {"target_far_nominal": config.TARGET_FAR_PCT,
           "safety_factor": config.FAR_SAFETY_FACTOR,
           "target_far_selection": sel_target,
           "threshold_mean": float(d["threshold_used"].mean()),
           "threshold_std_across_seeds": float(d["threshold_used"].std()),
           "threshold_bootstrap_std": float(d["threshold_std"].mean())
               if "threshold_std" in d.columns else None,
           "achieved_far_mean": float(d["far_fixed"].mean()),
           "far_overshoot_vs_nominal": float(d["far_overshoot"].mean()),
           "far_drift_vs_selection": float(d["far_drift_vs_selection"].mean())
               if "far_drift_vs_selection" in d.columns else None,
           "recall_mean": float(d["recall_fixed"].mean()),
           "precision_mean": float(d["precision_fixed"].mean())
               if "precision_fixed" in d.columns else None}
    with open(os.path.join(config.RESULT_DIR, "diagnostics_threshold.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def run(model_name=None):
    diagnose_regression_bands(model_name)
    diagnose_threshold(model_name)
    print("\n  진단 결과 저장: results/diagnostics_*.json, "
          "figures/fig8_regression_diagnostics.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    run(args.model)
