"""평가 시드에 걸친 유의성 검정.

같은 시드로 학습한 심층 모델끼리는 초기화와 셔플링에서 오는 잡음을 공유하므로,
대응 검정으로 그 공통 분산 성분을 제거한다. Holm-Bonferroni 보정으로 다중
비교의 family-wise 오류율을 통제한다.

주 비교(제안 모델 대 스펙트로그램 전용 및 고전 베이스라인)와 융합
ablation(어텐션 대 단순 결합)을 분리해 보고하여, 구조에 관한 주장과 표현에
관한 주장이 뒤섞이지 않도록 한다.

[검토 반영 v2]
  B-2  고전 베이스라인의 시드는 '학습셋 부트스트랩' 시드이고 심층 모델의
       시드는 '초기화' 시드다. 두 계열 사이에 공유 잡음원이 없으므로 대응
       검정의 전제가 성립하지 않는다. 기본값은 비대응 Welch 검정.
  C-6  cohens_d_paired 는 실제로는 d_z(차이 점수의 표준화값)이다. Cohen's d
       기준(0.2/0.5/0.8)을 그대로 붙이면 효과크기가 과대 표시된다. 이름과
       라벨을 d_z 로 바꾸고 해석 기준도 그에 맞춘다.
  C-12 CLASSICAL_BOOTSTRAP 이 켜져 있으면 Logistic/Ridge 도 시드마다 예측이
       달라진다. '결정론적'이라는 표시는 관측된 분산으로만 판정한다.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import config
from models_meta import (get_proposed_name, CLASSICAL_MODEL_NAMES,
                         NO_ATTN, SINGLE_AE, deterministic_models,
                         is_classical, display)

METRICS = [
    ("pr_auc", True),
    ("roc_auc", True),
    ("recall_fixed", True),
    ("precision_fixed", True),
    ("far_fixed", False),
    ("rmse", False),
    ("mae", False),
    ("r2", True),
]


def cohens_dz(a, b):
    """대응 표본의 표준화 효과크기 d_z = mean(diff) / sd(diff).

    [C-6] 이는 Cohen's d 가 아니다. d_z 는 통상 d 보다 크게 나오므로
    0.2/0.5/0.8 기준을 그대로 적용하면 효과가 과대 표시된다.
    """
    d = np.asarray(a) - np.asarray(b)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() == 0 else float("inf")
    return float(d.mean() / sd)


def hedges_g_unpaired(a, b):
    """비대응 비교용 효과크기 (소표본 보정 포함)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return 0.0 if a.mean() == b.mean() else float("inf")
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    j = 1.0 - 3.0 / (4 * (na + nb) - 9)
    return float(d * j)


def interpret_dz(d):
    """d_z 용 해석 구간. Cohen's d 기준보다 보수적으로 잡는다."""
    ad = abs(d)
    if not np.isfinite(ad):
        return "undefined"
    if ad < 0.5:
        return "negligible"
    if ad < 1.0:
        return "small"
    if ad < 2.0:
        return "medium"
    return "large"


# 하위 호환
cohens_d_paired = cohens_dz
interpret_d = interpret_dz


def _p(v):
    """보정 p 표기. NaN 은 검정이 성립하지 않은 경우다."""
    return f"{v:.4f}" if v == v else "검정 불가"


def holm_bonferroni(pvals, alpha=0.05):
    """Holm-Bonferroni 하강 보정.

    [수정 v2.1 — 치명] NaN p값이 '유의함'으로 뒤집히던 결함.
      running = max(running, (n-rank) * nan) 에서 파이썬 max 는
      nan > running 비교가 False 라 running 을 그대로 둔다. 따라서
      모든 p 가 NaN 이면 running 이 0.0 에 머물러 보정 p 가 전부 0.0 이
      되고, adj <= alpha 가 전부 True 가 되었다. 시드가 1개여서
      대응 t검정이 NaN 을 돌려준 경우(표 6b) 12개 비교가 모두
      '유의함'으로 출력되고, 그 위에서 "에너지 가중항이 유의하게
      기여한다"는 결론 문장까지 찍혔다.

      더 고약한 점은 결과가 배열 구성에 따라 달라졌다는 것이다.
      비교 하나라도 p=1.0 이 섞여 있으면 running 이 1.0 이상으로
      올라가 나머지 NaN 이 전부 1.0(비유의)이 되었다. 같은 결함이
      표 5b 에서는 반대 방향으로 나타나 눈에 띄지 않았다.

    이제 NaN 은 검정 불가로 보아 보정 p = NaN, 기각 = False 로 둔다.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    adj = np.full(n, np.nan, dtype=float)
    reject = np.zeros(n, dtype=bool)

    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return adj, reject

    # 보정은 검정이 성립한 비교끼리만 수행한다.
    m = len(valid)
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(p[idx]))
        adj[idx] = min(running, 1.0)
        reject[idx] = adj[idx] <= alpha
    return adj, reject


def load_all_results(deep_csv=None, classical_csv=None):
    deep_csv = deep_csv or os.path.join(config.RESULT_DIR, "seed_level_results.csv")
    classical_csv = classical_csv or os.path.join(config.RESULT_DIR,
                                                  "classical_baseline_results.csv")
    if not os.path.exists(deep_csv):
        raise FileNotFoundError(f"{deep_csv}가 없습니다. step3를 먼저 실행하세요.")
    frames = [pd.read_csv(deep_csv)]
    if os.path.exists(classical_csv):
        c = pd.read_csv(classical_csv)
        frames.append(c)
        print(f"  고전 베이스라인 병합: "
              f"{[display(m) for m in c['model'].unique()]}")
    else:
        print("  고전 베이스라인 결과가 없습니다 (step2b 미실행).")
    return pd.concat(frames, ignore_index=True)


def compare_models(df, proposed=None, alpha=0.05, baselines=None):
    proposed = proposed or get_proposed_name()
    models = sorted(df["model"].unique())
    if proposed not in models:
        raise ValueError(f"제안 모델 '{proposed}'이 없습니다. 존재: {models}")
    if baselines is None:
        baselines = [m for m in models if m != proposed]

    seeds = sorted(df["seed"].unique())
    n = len(seeds)
    det_models = deterministic_models()
    rows, skipped = [], []

    for metric, higher_better in METRICS:
        if metric not in df.columns:
            skipped.append((metric, "컬럼 없음"))
            continue
        piv = df.pivot_table(index="seed", columns="model", values=metric,
                             dropna=False).reindex(seeds)
        if proposed not in piv.columns:
            skipped.append((metric, "제안 모델 데이터 없음"))
            continue
        prop_vals = piv[proposed].values
        if np.isnan(prop_vals).any():
            skipped.append((metric, "제안 모델 값에 NaN"))
            continue

        for base in baselines:
            if base not in piv.columns:
                continue
            base_vals = piv[base].values
            if np.isnan(base_vals).any():
                skipped.append((f"{metric} vs {base}", "베이스라인 값에 NaN"))
                continue

            diff = prop_vals - base_vals
            improvement = diff.mean() if higher_better else -diff.mean()
            base_std = float(base_vals.std(ddof=1)) if n > 1 else 0.0
            degenerate = base_std < 1e-12

            # [C-12] '결정론적' 판정은 관측된 분산으로만 한다.
            observed_deterministic = bool(degenerate)
            nominally_deterministic = base in det_models

            # [B-2] 고전 베이스라인은 시드의 의미가 달라 짝짓기가 성립하지 않는다.
            use_welch = (config.CLASSICAL_TEST == "welch"
                         and is_classical(base) and not degenerate)

            w_p = np.nan
            if np.allclose(diff, 0):
                t_stat, t_p, w_p = 0.0, 1.0, 1.0
                test_used = "none"
                eff, eff_name = 0.0, "d_z"
            elif degenerate and float(np.std(prop_vals)) < 1e-12:
                # [수정 v2.1] 양쪽 모두 분산이 0 인 경우.
                #   ttest_1samp 가 t = inf, p = 0 을 돌려주어 "p < 0.0001 로
                #   유의"라는 문장이 만들어졌지만, 시드마다 똑같은 값이
                #   나왔다는 것은 표본 변동에 대한 정보가 전혀 없다는 뜻이다.
                #   차이의 크기는 그대로 보고하되 유의성은 판정하지 않는다.
                #   (오경보율은 평가 음성 개수로 양자화되어 시드가 많아도
                #    상수가 되기 쉬우므로 본 실행에서도 발생한다.)
                t_stat, t_p, w_p = np.nan, np.nan, np.nan
                test_used = "분산 0 (검정 불가)"
                eff, eff_name = np.nan, "d_z"
            elif degenerate:
                # 베이스라인이 상수이므로 대응 검정이 일표본 검정으로 축약된다.
                t_stat, t_p = stats.ttest_1samp(prop_vals, popmean=base_vals[0])
                try:
                    _, w_p = stats.wilcoxon(prop_vals - base_vals[0])
                except Exception:
                    w_p = np.nan
                test_used = "one-sample"
                eff, eff_name = cohens_dz(prop_vals, base_vals), "d_z"
            elif use_welch:
                t_stat, t_p = stats.ttest_ind(prop_vals, base_vals, equal_var=False)
                try:
                    _, w_p = stats.mannwhitneyu(prop_vals, base_vals,
                                                alternative="two-sided")
                except Exception:
                    w_p = np.nan
                test_used = "welch"
                eff, eff_name = hedges_g_unpaired(prop_vals, base_vals), "hedges_g"
            else:
                t_stat, t_p = stats.ttest_rel(prop_vals, base_vals)
                try:
                    _, w_p = stats.wilcoxon(prop_vals, base_vals)
                except Exception:
                    w_p = np.nan
                test_used = "paired"
                eff, eff_name = cohens_dz(prop_vals, base_vals), "d_z"

            rows.append({
                "metric": metric, "higher_is_better": higher_better, "baseline": base,
                "baseline_label": display(base),
                "proposed_mean": float(prop_vals.mean()),
                "proposed_std": float(prop_vals.std(ddof=1)) if n > 1 else 0.0,
                "baseline_mean": float(base_vals.mean()),
                "baseline_std": base_std,
                "baseline_deterministic_observed": observed_deterministic,
                "baseline_deterministic_nominal": nominally_deterministic,
                "baseline_variance_source": ("train_bootstrap" if is_classical(base)
                                             else "init_and_shuffle"),
                "test_used": test_used,
                "mean_improvement": float(improvement),
                "pct_improvement": float(100 * improvement / abs(base_vals.mean()))
                                   if base_vals.mean() != 0 else np.nan,
                "t_stat": float(t_stat), "p_ttest": float(t_p),
                "p_rank": float(w_p) if w_p == w_p else np.nan,
                "effect_size_name": eff_name,
                "effect_size": float(eff),
                "effect_label": interpret_dz(eff) if eff_name == "d_z" else "",
                "n_seeds": n,
                "wins": int((diff > 0).sum() if higher_better else (diff < 0).sum())})

    res = pd.DataFrame(rows)
    if res.empty:
        return res, skipped

    adj, rej = holm_bonferroni(res["p_ttest"].values, alpha=alpha)
    res["p_ttest_holm"] = adj
    res["significant_holm"] = rej
    floor = 2.0 ** (-(n - 1)) if n > 1 else 1.0
    res["rank_test_p_floor"] = floor
    res["rank_test_underpowered"] = floor > alpha
    return res, skipped


def _print_table(res, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    show = res[["metric", "baseline_label", "proposed_mean", "baseline_mean",
                "pct_improvement", "p_ttest", "p_ttest_holm", "p_rank",
                "effect_size", "effect_size_name", "wins", "significant_holm",
                "test_used"]].copy()
    show.columns = ["Metric", "Baseline", "Proposed", "Baseline mean", "Delta %",
                    "p", "p (Holm)", "p (rank)", "Effect", "Type", "Wins",
                    "Sig.", "Test"]
    for c in ["Proposed", "Baseline mean", "Effect"]:
        show[c] = show[c].map("{:.4f}".format)
    for c in ["p", "p (Holm)", "p (rank)"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}" if v == v else "n/a")
    show["Delta %"] = show["Delta %"].map("{:+.2f}".format)
    n = int(res["n_seeds"].iloc[0])
    show["Wins"] = show["Wins"].astype(str) + f"/{n}"
    print(show.to_string(index=False))
    print("=" * 100)


def run(alpha=0.05):
    print("\n" + "=" * 100)
    print("4단계  통계적 유의성 검정")
    print("=" * 100)
    df = load_all_results()

    main_baselines = [b for b in [SINGLE_AE] + CLASSICAL_MODEL_NAMES
                      if b in df["model"].unique()]
    res_main, skipped = compare_models(df, alpha=alpha, baselines=main_baselines)

    if skipped:
        print("\n  검정에서 제외된 항목:")
        for name, reason in skipped:
            print(f"    {name}: {reason}")

    if res_main.empty:
        print("\n  검정 가능한 지표가 없습니다.")
        return res_main

    res_main.to_csv(os.path.join(config.RESULT_DIR, "significance_main.csv"),
                    index=False)
    n = int(res_main["n_seeds"].iloc[0])
    _print_table(res_main, f"표 3a  주 비교: 제안 모델 대 베이스라인 (시드 {n}개)")

    # [B-2] 검정 방식이 섞여 있으면 그 사실을 명시한다.
    used = sorted(res_main["test_used"].unique())
    if "welch" in used:
        print("\n  참고: 고전 베이스라인은 시드의 의미가 다릅니다.")
        print("  (심층 = 초기화 시드, 고전 = 학습셋 부트스트랩 시드)")
        print("  공유 잡음원이 없어 짝짓기가 성립하지 않으므로 비대응 Welch")
        print("  검정을 사용했습니다. 논문에도 그렇게 서술하세요.")

    det_obs = res_main[res_main.baseline_deterministic_observed]
    if not det_obs.empty:
        names = sorted(det_obs["baseline_label"].unique())
        print(f"\n  참고: {', '.join(names)}는 관측된 시드 간 분산이 0입니다.")
        print("  해당 상수에 대한 일표본 검정을 사용했습니다.")
    elif config.CLASSICAL_BOOTSTRAP:
        print("\n  참고: 부트스트랩이 켜져 있어 고전 베이스라인도 시드마다")
        print("  예측이 다릅니다. '결정론적 베이스라인'으로 서술하지 마세요.")

    res_abl = pd.DataFrame()
    if NO_ATTN in df["model"].unique():
        res_abl, _ = compare_models(df, alpha=alpha, baselines=[NO_ATTN])
        if not res_abl.empty:
            res_abl.to_csv(os.path.join(config.RESULT_DIR,
                                        "significance_fusion_ablation.csv"),
                           index=False)
            _print_table(res_abl,
                         f"표 4a  융합 ablation: 어텐션 대 단순 결합 (시드 {n}개)")

            key = res_abl[res_abl.metric == "pr_auc"]
            if not key.empty:
                r = key.iloc[0]
                print("\n  어텐션 기여 (PR-AUC 기준):")
                print(f"    제안 {r['proposed_mean']:.4f} 대 "
                      f"단순 결합 {r['baseline_mean']:.4f} "
                      f"({r['pct_improvement']:+.2f}%, "
                      f"p_Holm = {_p(r['p_ttest_holm'])}, "
                      f"{r['effect_size_name']} = {r['effect_size']:.2f}, "
                      f"wins {int(r['wins'])}/{n})")
                if r["significant_holm"] and r["mean_improvement"] > 0:
                    print("    어텐션이 유의한 성능 향상을 제공합니다.")
                else:
                    print("    어텐션의 유의한 성능 기여가 확인되지 않았습니다.")
                    print(f"    참고: 현재 융합 방식은 ATTENTION_FUSION="
                          f"'{config.ATTENTION_FUSION}' 입니다.")
                    if config.ATTENTION_FUSION == "none":
                        print("    'none' 은 어텐션 출력에 질의를 더하지 않으므로")
                        print("    상태 벡터가 헤드에 직접 도달하지 못합니다.")
                        print("    단순 결합과 동일 조건 비교가 아니니, 'add' 로")
                        print("    다시 돌려 비교하세요 (검토 B-14).")

    floor = float(res_main["rank_test_p_floor"].iloc[0])
    if bool(res_main["rank_test_underpowered"].iloc[0]):
        print(f"\n  순위 검정 p 하한 {floor:.4f}이 0.05를 초과합니다. 시드를 늘리세요.")
    else:
        print(f"\n  순위 검정 p 하한 {floor:.2e} (시드 {n}개에서 유효).")
    print(f"  Holm 보정 후 유의한 주 비교: "
          f"{int(res_main['significant_holm'].sum())} / {len(res_main)}")

    cls_rows = res_main[res_main.baseline.isin(CLASSICAL_MODEL_NAMES)]
    if not cls_rows.empty:
        print("\n" + "=" * 100)
        print("  고전 베이스라인 대비 심층 모델이 유리한 영역")
        print("=" * 100)
        for metric in ["pr_auc", "recall_fixed", "rmse", "r2"]:
            for _, r in cls_rows[cls_rows.metric == metric].iterrows():
                verdict = ("제안 모델 우세"
                           if r["significant_holm"] and r["mean_improvement"] > 0
                           else "베이스라인 우세" if r["significant_holm"]
                           else "검정 불가" if r["p_ttest_holm"] != r["p_ttest_holm"]
                           else "유의차 없음")
                print(f"    {metric:16s} vs {r['baseline_label']:18s} "
                      f"{r['pct_improvement']:+7.2f}%  "
                      f"(p_Holm = {_p(r['p_ttest_holm'])}, {r['test_used']})  {verdict}")

    with open(os.path.join(config.RESULT_DIR, "significance_summary.json"), "w") as f:
        json.dump({"n_seeds": n, "alpha": alpha,
                   "n_main_comparisons": len(res_main),
                   "n_significant_main": int(res_main["significant_holm"].sum()),
                   "rank_test_p_floor": floor,
                   "rank_test_underpowered": bool(
                       res_main["rank_test_underpowered"].iloc[0]),
                   "classical_test": config.CLASSICAL_TEST,
                   "attention_fusion": config.ATTENTION_FUSION,
                   "focal_alpha_mode": config.FOCAL_ALPHA_MODE,
                   "deterministic_observed": sorted(
                       res_main.loc[res_main.baseline_deterministic_observed,
                                    "baseline"].unique().tolist()),
                   "fusion_ablation_tested": not res_abl.empty,
                   "skipped": [list(s) for s in skipped]}, f, indent=2)

    print(f"\n  저장: {config.RESULT_DIR}/significance_main.csv")
    if not res_abl.empty:
        print(f"  저장: {config.RESULT_DIR}/significance_fusion_ablation.csv")
    return res_main


def _self_test():
    rng = np.random.default_rng(7)
    seeds = list(range(15))
    specs = {"Cross-Timescale Attn": (0.72, 0.91, 70, 8, 1.90, 1.5, 0.55),
             "Single AE": (0.60, 0.85, 55, 12, 2.30, 1.9, 0.40),
             "Ours w/o Attention": (0.70, 0.905, 68, 8.5, 1.95, 1.55, 0.53),
             "GradientBoosting": (0.69, 0.88, 66, 13, 2.85, 2.4, 0.41)}
    rows = []
    for s in seeds:
        for m, (pr, roc, rec, far, rmse, mae, r2) in specs.items():
            rows.append({"seed": s, "model": m,
                         "pr_auc": pr + rng.normal(0, 0.02),
                         "roc_auc": roc + rng.normal(0, 0.01),
                         "recall_fixed": rec + rng.normal(0, 3),
                         "precision_fixed": 60 + rng.normal(0, 2),
                         "far_fixed": far + rng.normal(0, 1),
                         "rmse": rmse + rng.normal(0, 0.05),
                         "mae": mae + rng.normal(0, 0.04),
                         "r2": r2 + rng.normal(0, 0.03)})
        rows.append({"seed": s, "model": "LogReg/Ridge",
                     "pr_auc": 0.7183, "roc_auc": 0.8769, "recall_fixed": 62.77,
                     "precision_fixed": 61.0,
                     "far_fixed": 13.77, "rmse": 2.8507, "mae": 2.31, "r2": 0.4169})
    df = pd.DataFrame(rows)

    res_main, _ = compare_models(df, baselines=["Single AE", "GradientBoosting",
                                                "LogReg/Ridge"])
    assert "Ours w/o Attention" not in res_main.baseline.values
    det = res_main[res_main.baseline == "LogReg/Ridge"]
    assert det["baseline_deterministic_observed"].all()
    assert (det["test_used"] == "one-sample").all()
    assert det["p_ttest"].notna().all()

    gbm = res_main[res_main.baseline == "GradientBoosting"]
    expect = "welch" if config.CLASSICAL_TEST == "welch" else "paired"
    assert (gbm["test_used"] == expect).all(), gbm["test_used"].unique()

    res_abl, _ = compare_models(df, baselines=["Ours w/o Attention"])
    assert set(res_abl.baseline.unique()) == {"Ours w/o Attention"}
    assert (res_abl["test_used"] == "paired").all()
    assert float(res_main["rank_test_p_floor"].iloc[0]) < 0.05

    print("  고전 베이스라인 검정 방식:")
    print(gbm[["metric", "test_used", "effect_size_name"]].head(3).to_string(index=False))
    print("  상수 베이스라인은 일표본 검정으로 처리:")
    print(det[["metric", "p_ttest", "test_used"]].head(3).to_string(index=False))

    # [수정 v2.1] 양쪽 모두 분산이 0 이면 유의 판정을 내면 안 된다.
    #   오경보율은 평가 음성 개수로 양자화되어 시드가 많아도 상수가 되기 쉽다.
    flat = df.copy()
    flat.loc[flat.model == "Cross-Timescale Attn", "far_fixed"] = 11.6505
    flat.loc[flat.model == "LogReg/Ridge", "far_fixed"] = 15.5340
    res_flat, _ = compare_models(flat, baselines=["LogReg/Ridge"])
    row = res_flat[(res_flat.metric == "far_fixed")].iloc[0]
    assert row["test_used"] == "분산 0 (검정 불가)", row["test_used"]
    assert row["p_ttest"] != row["p_ttest"], row["p_ttest"]
    assert not bool(row["significant_holm"])
    print(f"  양쪽 분산 0: test_used='{row['test_used']}', 유의 판정 없음")
    print("step4: 정상")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    _self_test() if args.self_test else run()
