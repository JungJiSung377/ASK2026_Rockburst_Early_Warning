"""동일 오경보율에서의 운영점 비교.

각 모델이 자신의 임계값에서 얻은 재현율을 그대로 비교하면 검출 능력과
운영점 선택이 뒤섞인다. 오경보율을 더 높게 잡은 모델은 그 이유만으로도
재현율이 높게 나오기 때문이다. 이 모듈은 모든 모델을 공통 오경보율에서
평가하여 검출 능력을 직접 비교하고, 그에 따른 검출-오경보 상충 관계를
보고한다.

[주의] 여기서 얻는 재현율은 평가 데이터의 곡선에서 읽은 상한이며 배포
성능이 아니다. 논문에서는 "동일 FAR 로 정렬했을 때의 검출 능력"으로
서술하고, 배포 가능한 수치는 표 2 의 recall_fixed 를 쓸 것 (검토 A-7).

[검토 반영 v2]
  A-6  FAR 격자를 촘촘히 하여 교차 구간을 실제로 측정한다. 재학습 없이
       캐시된 예측만으로 계산되므로 비용이 거의 없다.
  B-2  고전 베이스라인과의 비교는 비대응 Welch 검정을 사용한다.
  18   "차이가 없었다"를 눈으로 판단하지 않도록, 격자점마다 검정 결과를
       표로 남긴다.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats

import config
from metrics import recall_at_matched_far
from models_meta import display, PROPOSED_MODEL, is_classical
from step4_stats import holm_bonferroni, cohens_dz, hedges_g_unpaired, interpret_dz


def _load_caches():
    preds = {}
    deep = os.path.join(config.ARTIFACT_DIR, "prediction_cache.pkl")
    clas = os.path.join(config.ARTIFACT_DIR, "classical_prediction_cache.pkl")
    if os.path.exists(deep):
        with open(deep, "rb") as f:
            preds.update(pickle.load(f)["predictions"])
    else:
        raise FileNotFoundError(f"{deep}가 없습니다. step3를 먼저 실행하세요.")
    if os.path.exists(clas):
        with open(clas, "rb") as f:
            preds.update(pickle.load(f)["predictions"])
    else:
        print("  고전 모델 예측 캐시가 없어 심층 모델만 비교합니다.")
    return preds


def _crossover(agg, a, b):
    """두 모델의 재현율 곡선이 교차하는 FAR 구간을 격자 위에서 찾는다.

    [A-6] 격자에 없는 지점을 '교차점'으로 특정하면 안 되므로, 부호가 바뀌는
    '구간'을 반환한다.
    """
    fa = agg[agg.model == a].set_index("target_far")["mean"]
    fb = agg[agg.model == b].set_index("target_far")["mean"]
    common = sorted(set(fa.index) & set(fb.index))
    if len(common) < 2:
        return None
    d = [fa[c] - fb[c] for c in common]
    for i in range(len(common) - 1):
        if d[i] == 0:
            continue
        if np.sign(d[i]) != np.sign(d[i + 1]):
            return (common[i], common[i + 1])
    return None


def run(alpha=0.05):
    print("\n" + "=" * 84)
    print("동일 오경보율 운영점 비교")
    print("=" * 84)
    print(f"  FAR 격자: {config.MATCHED_FAR_GRID}")
    print("  주의: 이 재현율은 평가 곡선에서 읽은 상한이며 배포 성능이 아닙니다.")
    preds = _load_caches()

    rows = []
    for name, sp in preds.items():
        for seed, d in sp.items():
            for far in config.MATCHED_FAR_GRID:
                rows.append({"model": name, "seed": seed, "target_far": far,
                             "recall": recall_at_matched_far(
                                 d["y_true_cls"], d["y_score_cls"], far)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULT_DIR, "iso_far_recall_raw.csv"), index=False)

    # [수정 v2.1] config.EVAL_SEEDS 가 아니라 모델 간 일관성을 본다.
    #   축소 실행에서 모든 행에 '확인 필요'가 붙어 진짜 결함을 덮었다.
    n_seed = df.groupby("model")["seed"].nunique()
    n_max = int(n_seed.max()) if len(n_seed) else 0
    print("\n  모델별 시드 수:")
    for m_, v in n_seed.items():
        flag = "" if v == n_max else "   <- 확인 필요 (다른 모델보다 적음)"
        print(f"    {display(m_):24s} {v}개{flag}")
    if (n_seed != n_max).any():
        print("  모델마다 시드 수가 다릅니다. 동일 FAR 비교의 대응 관계가")
        print("  깨지므로 예측 캐시를 확인하세요 (tools/check_integrity.py, 검토 C-1).")
    elif n_max != len(config.EVAL_SEEDS):
        print(f"  참고: 시드 {n_max}개로 계산했습니다 "
              f"(config.EVAL_SEEDS 는 {len(config.EVAL_SEEDS)}개). "
              f"표 캡션의 시드 수를 맞추세요.")

    piv = df.pivot_table(index="target_far", columns="model", values="recall",
                         aggfunc="mean").round(2)
    piv_sd = df.pivot_table(index="target_far", columns="model", values="recall",
                            aggfunc="std").round(2)

    ordered = [m for m in [PROPOSED_MODEL] +
               [c for c in piv.columns if c != PROPOSED_MODEL] if m in piv.columns]
    piv, piv_sd = piv[ordered], piv_sd[ordered]

    table = pd.DataFrame(index=piv.index)
    for m in ordered:
        table[display(m)] = (piv[m].map("{:.1f}".format) + " ± " +
                             piv_sd[m].fillna(0).map("{:.1f}".format))
    table.index.name = "목표 FAR (%)"

    print("\n  표 7  동일 오경보율에서의 재현율(%), 시드 간 평균 ± 표준편차")
    print("-" * 84)
    print(table.to_string())
    print("-" * 84)
    table.to_csv(os.path.join(config.RESULT_DIR, "iso_far_recall_table.csv"))

    # 각 동일 오경보율에서의 검정
    agg = df.groupby(["model", "target_far"])["recall"].agg(
        ["mean", "std"]).reset_index()
    test_rows = []
    for far in config.MATCHED_FAR_GRID:
        sub = df[df.target_far == far]
        p = sub.pivot_table(index="seed", columns="model", values="recall")
        if PROPOSED_MODEL not in p.columns:
            continue
        prop = p[PROPOSED_MODEL].values
        for base in p.columns:
            if base == PROPOSED_MODEL:
                continue
            bv = p[base].values
            if np.isnan(prop).any() or np.isnan(bv).any():
                continue
            diff = prop - bv
            degenerate = bv.std(ddof=1) < 1e-12
            use_welch = (config.CLASSICAL_TEST == "welch"
                         and is_classical(base) and not degenerate)

            if np.allclose(diff, 0):
                t_p, used = 1.0, "none"
                eff, eff_name = 0.0, "d_z"
            elif degenerate:
                _, t_p = stats.ttest_1samp(prop, popmean=bv[0])
                used = "one-sample"
                eff, eff_name = cohens_dz(prop, bv), "d_z"
            elif use_welch:
                _, t_p = stats.ttest_ind(prop, bv, equal_var=False)
                used = "welch"
                eff, eff_name = hedges_g_unpaired(prop, bv), "hedges_g"
            else:
                _, t_p = stats.ttest_rel(prop, bv)
                used = "paired"
                eff, eff_name = cohens_dz(prop, bv), "d_z"

            test_rows.append({"target_far": far, "baseline": base,
                              "baseline_label": display(base),
                              "proposed_recall": float(prop.mean()),
                              "baseline_recall": float(bv.mean()),
                              "delta_pp": float(diff.mean()),
                              "p_ttest": float(t_p),
                              "effect_size": float(eff),
                              "effect_size_name": eff_name,
                              "effect_label": interpret_dz(eff) if eff_name == "d_z" else "",
                              "test_used": used,
                              "n_seeds": len(prop)})

    res = pd.DataFrame(test_rows)
    if not res.empty:
        adj, rej = holm_bonferroni(res["p_ttest"].values, alpha=alpha)
        res["p_ttest_holm"] = adj
        res["significant_holm"] = rej
        res.to_csv(os.path.join(config.RESULT_DIR, "iso_far_significance.csv"),
                   index=False)

        print("\n  동일 오경보율에서의 유의성 (제안 모델 대 베이스라인)")
        print("-" * 96)
        show = res[["target_far", "baseline_label", "proposed_recall",
                    "baseline_recall", "delta_pp", "p_ttest_holm", "effect_size",
                    "significant_holm", "test_used"]].copy()
        for c in ["proposed_recall", "baseline_recall", "delta_pp", "effect_size"]:
            show[c] = show[c].map("{:+.2f}".format)
        show["p_ttest_holm"] = show["p_ttest_holm"].map("{:.4f}".format)
        show.columns = ["FAR(%)", "베이스라인", "제안", "베이스라인", "차이(%p)",
                        "p (Holm)", "효과", "유의", "검정"]
        print(show.to_string(index=False))
        print("-" * 96)

        # [심사 지적 18] "차이가 없었다"를 검정 결과로 서술할 수 있게 요약
        print("\n  저FAR 구간 요약 (본문에 이 문장을 쓰세요)")
        low = res[res.target_far <= 10.0]
        if not low.empty:
            ns = low[~low.significant_holm]
            print(f"    FAR <= 10% 구간의 비교 {len(low)}건 중 "
                  f"{len(ns)}건이 Holm 보정 후 유의하지 않았습니다.")
            if len(ns):
                pmin = low.loc[~low.significant_holm, "p_ttest_holm"].min()
                print(f"    (유의하지 않은 비교의 최소 p_Holm = {pmin:.4f})")

    # [A-6] 교차 구간을 격자 위에서 실제로 찾는다
    print("\n  격차가 벌어지는 구간 (측정 격자 기준)")
    for base in [m for m in agg.model.unique() if m != PROPOSED_MODEL]:
        cx = _crossover(agg, PROPOSED_MODEL, base)
        if cx:
            print(f"    vs {display(base):24s} FAR {cx[0]:g}% ~ {cx[1]:g}% 사이에서 역전")
        else:
            fa = agg[agg.model == PROPOSED_MODEL].set_index("target_far")["mean"]
            fb = agg[agg.model == base].set_index("target_far")["mean"]
            common = sorted(set(fa.index) & set(fb.index))
            sign = "우세" if fa[common[0]] > fb[common[0]] else "열세"
            print(f"    vs {display(base):24s} 전 구간에서 제안 모델 {sign}")
    print("    측정하지 않은 지점을 교차점으로 특정하지 마세요 (검토 A-6).")

    print("\n  해석: 이 표는 검출 능력을 운영점 선택과 분리해 보여준다. 각 모델이")
    print("  실제 배포 임계값에서 얻는 재현율은 표 2에 별도로 보고된다.")
    print(f"\n  저장: {config.RESULT_DIR}/iso_far_recall_table.csv")
    return df, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    run()
