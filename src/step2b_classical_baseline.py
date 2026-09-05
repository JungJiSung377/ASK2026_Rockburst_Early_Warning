"""전조 상태 벡터만 사용하는 고전 기계학습 베이스라인.

스펙트로그램 분기와 심층 구조가 정당한지를 판단하는 기준이 된다. 수제 특징
6개를 그래디언트 부스팅에 넣은 결과가 제안 모델과 대등하다면, 추가된 복잡성은
정당화되지 않는다.

로지스틱 회귀와 릿지 회귀는 결정론적이어서 시드를 반복해도 예측이 동일하고
대응 검정의 분산이 0이 된다. 따라서 학습 집합을 부트스트랩 재표집하여 비교
가능한 분산을 부여한다. 분할, 특징 정규화, 임계값 선택 절차는 심층 모델과
완전히 동일하다.

[검토 반영 v2]
  B-4  HistGradientBoosting 의 early_stopping=True 는 학습셋을 **무작위로**
       잘라 내부 검증을 만든다. 시간·사이클상 인접한 세그먼트가 섞여
       조기종료가 낙관적으로 결정되고, "심층 모델과 동일한 분할" 이라는
       주장과 어긋난다. warm_start 로 반복 수를 늘려가며 **사이클 기반
       검증 파티션**에서 PR-AUC 를 보고 고르도록 바꾼다.
  C-5  Ridge(random_state=...) 는 기본 solver 에서 아무 효과가 없다.
       분산의 출처가 부트스트랩뿐임을 결과에 명시한다.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_squared_error

import config
from data import load_arrays
from splitting import cycle_aware_split, summarize_split
from metrics import full_evaluation, select_threshold_at_far, matched_far_columns
from models_meta import LOGREG, GBM, display

CACHE = os.path.join(config.ARTIFACT_DIR, "classical_prediction_cache.pkl")

# [B-4] warm_start 로 탐색할 반복 수 격자
GBM_ITER_GRID = [50, 100, 150, 200, 300]


def _fit_gbm_with_cycle_val(estimator_cls, X_tr, y_tr, X_va, y_va, seed,
                            score_fn, greater_is_better=True):
    """사이클 기반 검증 파티션에서 max_iter 를 고른다.

    sklearn 의 내부 early_stopping 은 무작위 분할을 쓰므로 쓰지 않고,
    warm_start 로 반복 수를 늘려가며 외부 검증 점수를 직접 본다.
    """
    est = estimator_cls(random_state=seed, warm_start=True,
                        early_stopping=False, max_iter=GBM_ITER_GRID[0])
    best = {"score": -np.inf, "iter": GBM_ITER_GRID[0], "model": None}
    for it in GBM_ITER_GRID:
        est.set_params(max_iter=it)
        est.fit(X_tr, y_tr)
        s = score_fn(est, X_va, y_va)
        s = s if greater_is_better else -s
        if s > best["score"]:
            import copy as _copy
            best = {"score": s, "iter": it, "model": _copy.deepcopy(est)}
    return best["model"], best["iter"]


def _clf_score(est, X, y):
    if len(np.unique(y)) < 2:
        return 0.0
    return float(average_precision_score(y, est.predict_proba(X)[:, 1]))


def _reg_score(est, X, y):
    return float(np.sqrt(mean_squared_error(y, est.predict(X))))


def build_linear_models(seed):
    # Ridge 의 random_state 는 기본 solver 에서 무효 -> 전달하지 않는다 (C-5)
    return (LogisticRegression(max_iter=3000, random_state=seed,
                               class_weight="balanced"),
            Ridge())


def run(h5_path=None, seeds=None, verbose=True, bootstrap=None):
    h5_path = h5_path or config.H5_PATH
    seeds = seeds or config.EVAL_SEEDS
    bootstrap = config.CLASSICAL_BOOTSTRAP if bootstrap is None else bootstrap

    print("\n" + "=" * 72)
    print("2b단계  고전 베이스라인 (상태 벡터만 사용, 스펙트로그램 미사용)")
    print("=" * 72)
    if bootstrap:
        print("  결정론적 추정기에 비교 가능한 분산을 부여하기 위해 시드마다")
        print("  학습 집합을 부트스트랩 재표집합니다.")
        print("  (이 분산은 초기화 잡음이 아니라 학습셋 재표집에서 옵니다.")
        print("   따라서 심층 모델과의 대응 검정은 성립하지 않습니다 — step4 참조)")
    print(f"  GBM 조기종료: 사이클 기반 검증에서 max_iter 선택 "
          f"{GBM_ITER_GRID}")

    arrays = load_arrays(h5_path, verbose=verbose)
    split = cycle_aware_split(arrays["cycle_id"], config.TRAIN_RATIO, config.VAL_RATIO)
    if verbose:
        summarize_split(split, arrays["y_cls"], config.MIN_TEST_POSITIVES,
                        verbose=True, label="(심층 모델과 동일)")

    tr, va, te = split["train"], split["val"], split["test"]
    X = arrays["x_state"]
    y_cls = arrays["y_cls"].reshape(-1)
    y_ttf = arrays["y_ttf"].reshape(-1)
    cyc = arrays["cycle_id"].reshape(-1)

    # 심층 모델(data._make_loaders)과 완전히 동일한 정규화 식
    g_min, g_max = X[tr].min(axis=0), X[tr].max(axis=0)
    Xn = np.clip((X - g_min) / (g_max - g_min + 1e-8), -5.0, 5.0)

    far_cols = matched_far_columns()
    rows, cache = [], {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(tr, size=len(tr), replace=True) if bootstrap else tr

        # --- 선형 계열 -------------------------------------------------
        clf, reg = build_linear_models(seed)
        clf.fit(Xn[tr_idx], y_cls[tr_idx])
        reg.fit(Xn[tr_idx], y_ttf[tr_idx])
        fitted = {LOGREG: (clf, reg, {"n_iter": int(getattr(clf, "n_iter_", [0])[0])})}

        # --- 부스팅 계열 (사이클 기반 검증으로 반복 수 선택) -----------
        gclf, g_it_c = _fit_gbm_with_cycle_val(
            HistGradientBoostingClassifier, Xn[tr_idx], y_cls[tr_idx],
            Xn[va], y_cls[va], seed, _clf_score, greater_is_better=True)
        greg, g_it_r = _fit_gbm_with_cycle_val(
            HistGradientBoostingRegressor, Xn[tr_idx], y_ttf[tr_idx],
            Xn[va], y_ttf[va], seed, _reg_score, greater_is_better=False)
        fitted[GBM] = (gclf, greg, {"max_iter_cls": g_it_c, "max_iter_reg": g_it_r})

        for name, (c_, r_, info) in fitted.items():
            score_va = c_.predict_proba(Xn[va])[:, 1]
            t_star = select_threshold_at_far(y_cls[va], score_va)

            score_te = c_.predict_proba(Xn[te])[:, 1]
            pred_ttf_te = r_.predict(Xn[te])

            m = full_evaluation(y_cls[te], score_te, y_ttf[te], pred_ttf_te,
                                operating_threshold=t_star, seed=seed,
                                cycle_id=cyc[te])

            extra = (f" | iter {info.get('max_iter_cls')}" if name == GBM else "")
            print(f"  seed {seed:4d}  {display(name):18s} "
                  f"PR-AUC {m['pr_auc']:.4f} | "
                  f"recall@fixed {m.get('recall_fixed', np.nan):5.1f}% "
                  f"(FAR {m.get('far_fixed', np.nan):5.1f}%) | "
                  f"RMSE {m['rmse']:.4f} | R2 {m['r2']:.4f}{extra}")

            row = {"seed": seed, "model": name, "n_params": int(Xn.shape[1]),
                   "bootstrap": bool(bootstrap),
                   "variance_source": "train_bootstrap" if bootstrap else "none"}
            row.update(info)
            row.update({k: m.get(k, np.nan) for k in
                        ["pr_auc", "roc_auc", "baseline_pr_auc", "pr_auc_headroom",
                         "far_fixed", "recall_fixed", "precision_fixed", "f1_fixed",
                         "threshold_used", "far_overshoot", "far_drift_vs_selection",
                         "far_05", "recall_05", "recall_at_far_oracle",
                         "rmse", "mae", "r2", "bias",
                         "cyc_bias_std_across_cycles", "cyc_bias_range",
                         "n_positive", "n_negative"]})
            row.update({k: m.get(k, np.nan) for k in far_cols})
            rows.append(row)

            cache.setdefault(name, {})[seed] = {
                "y_true_cls": y_cls[te], "y_score_cls": score_te,
                "y_true_ttf": y_ttf[te], "y_pred_ttf": pred_ttf_te,
                "cycle_id": cyc[te], "threshold": t_star}

    df = pd.DataFrame(rows)
    out = os.path.join(config.RESULT_DIR, "classical_baseline_results.csv")
    df.to_csv(out, index=False)
    with open(CACHE, "wb") as f:
        pickle.dump({"predictions": cache, "seeds": list(seeds)}, f)

    summary = df.groupby("model")[["pr_auc", "roc_auc", "recall_fixed",
                                   "far_fixed", "rmse", "r2"]].agg(
                                       ["mean", "std"]).round(4)
    summary.index = [display(i) for i in summary.index]
    print("\n" + "=" * 72)
    print("  고전 베이스라인 요약")
    print("=" * 72)
    print(summary.to_string())

    zero_var = df.groupby("model")["pr_auc"].std()
    for m_, s_ in zero_var.items():
        if s_ is not None and s_ < 1e-9:
            print(f"\n  참고: {display(m_)}의 시드 간 분산이 0입니다. "
                  f"CLASSICAL_BOOTSTRAP을 켜거나 결정론적 모델로 보고하세요.")

    print(f"\n  저장: {out}")
    print(f"  저장: {CACHE}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    run(seeds=args.seeds, bootstrap=not args.no_bootstrap)
