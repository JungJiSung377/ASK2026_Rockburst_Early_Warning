"""불균형 조기경보 분류와 TTF 회귀를 위한 평가 지표.

음성이 다수인 상황에서 ROC-AUC는 낙관적으로 나오는 반면 실제 관심사는 드문
양성을 놓치지 않는 능력이므로, 주 분류 지표로 average precision(PR-AUC)을
사용한다.

운영 임계값은 검증 파티션에서 결정하여 평가 데이터에 그대로 적용한다.
평가 데이터에서 임계값을 고르면 배포 시점에 알 수 없는 정보를 사용하는
누수가 된다. 검증에서 정한 임계값이 사이클을 넘어가면 그대로 옮겨지지
않으므로 안전계수를 적용한다.

[검토 반영 v2]
  B-5  far_overshoot 은 명목 목표(15%) 기준이라 안전계수가 겨냥한 검증
       목표(9%) 대비 실제 이동폭이 보이지 않았다. 두 값을 모두 낸다.
  A-7  recall_at_matched_far 는 평가 곡선에서 읽은 상한이다. 반환 키 이름에
       그 사실을 남기고, 배포 성능과 구분해 보고하도록 한다.
  C-14 사이클 간 계통 편향 판정 기준을 config.SYSTEMATIC_BIAS_RULE 하나로 통일.
  00-B bootstrap_ci 가 실제로 호출되도록 full_evaluation 에 연결.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
)

import config


# ----------------------------------------------------------------------
# 운영 임계값
# ----------------------------------------------------------------------
def selection_far_target_pct(target_far_pct=None, safety_factor=None):
    """안전계수를 적용한 '검증에서 겨냥하는' 오경보율(%)."""
    t = config.TARGET_FAR_PCT if target_far_pct is None else target_far_pct
    s = config.FAR_SAFETY_FACTOR if safety_factor is None else safety_factor
    return t * s


def select_threshold_at_far(y_true_val, y_score_val, target_far_pct=None,
                            safety_factor=None):
    """검증 데이터만으로 운영 임계값을 선택.

    t* = argmax_t { Recall_val(t) | FAR_val(t) <= kappa * FAR_target }
    fpr 은 단조 비감소이므로, 제약을 만족하는 마지막 인덱스가 곧 제약 하
    최대 재현율 지점이다.
    """
    target = selection_far_target_pct(target_far_pct, safety_factor) / 100.0

    y_true_val = np.asarray(y_true_val).reshape(-1)
    y_score_val = np.asarray(y_score_val).reshape(-1)
    if y_true_val.sum() == 0 or y_true_val.sum() == len(y_true_val):
        return 0.5

    fpr, _, thr = roc_curve(y_true_val, y_score_val)
    ok = np.where(fpr <= target)[0]
    if len(ok) == 0:
        return 1.0
    t = float(thr[ok[-1]])
    return t if np.isfinite(t) else 1.0


def threshold_stability(y_true_val, y_score_val, target_far_pct=None,
                        safety_factor=None, n_boot=None, seed=0):
    """검증 임계값의 부트스트랩 분포.

    안전계수의 필요성을 정량적으로 뒷받침하는 근거로 쓸 수 있다
    (심사 지적 14: kappa 의 근거).
    """
    n_boot = n_boot or config.THRESHOLD_BOOTSTRAP_N
    rng = np.random.default_rng(seed)
    y_true_val = np.asarray(y_true_val).reshape(-1)
    y_score_val = np.asarray(y_score_val).reshape(-1)
    n = len(y_true_val)

    ts = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true_val[idx], y_score_val[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        ts.append(select_threshold_at_far(yt, ys, target_far_pct, safety_factor))
    if not ts:
        return {"threshold_mean": np.nan, "threshold_std": np.nan,
                "threshold_lo": np.nan, "threshold_hi": np.nan}
    ts = np.array(ts)
    return {"threshold_mean": float(ts.mean()), "threshold_std": float(ts.std()),
            "threshold_lo": float(np.quantile(ts, 0.025)),
            "threshold_hi": float(np.quantile(ts, 0.975))}


def evaluate_at_fixed_threshold(y_true, y_score, threshold,
                                target_far_pct=None, safety_factor=None):
    """미리 정해진 임계값을 평가 데이터에 적용."""
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    far = 100.0 * fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    nominal = config.TARGET_FAR_PCT if target_far_pct is None else target_far_pct
    selection = selection_far_target_pct(target_far_pct, safety_factor)

    return {"far_fixed": float(far), "recall_fixed": float(recall),
            "precision_fixed": float(precision), "f1_fixed": float(f1),
            "threshold_used": float(threshold),
            # 명목 목표(15%) 대비
            "far_overshoot": float(far - nominal),
            # [B-5] 임계값을 실제로 고른 기준(kappa x 15% = 9%) 대비 이동폭.
            # 안전계수가 필요했다는 직접 증거가 된다.
            "far_drift_vs_selection": float(far - selection),
            "far_selection_target": float(selection),
            "tp_fixed": int(tp), "fp_fixed": int(fp),
            "tn_fixed": int(tn), "fn_fixed": int(fn)}


def recall_at_matched_far(y_true, y_score, target_far_pct):
    """동일 오경보율에서 달성되는 재현율.

    운영점이 다른 모델들을 재현율만으로 비교할 수 없으므로, 모든 모델을
    공통 운영점에서 평가한다.

    [주의] 임계값을 평가 데이터의 같은 곡선에서 읽으므로 이 값은 상한이며,
    배포 성능이 아니다. 논문에서는 "동일 FAR 로 정렬했을 때의 검출 능력"
    으로 서술하고, 배포 가능한 수치는 evaluate_at_fixed_threshold 의
    recall_fixed 를 쓸 것.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = np.where(fpr <= target_far_pct / 100.0)[0]
    return float(tpr[ok[-1]] * 100) if len(ok) else 0.0


def recall_at_far_oracle(y_true, y_score, target_far_pct=None):
    t = config.TARGET_FAR_PCT if target_far_pct is None else target_far_pct
    r = recall_at_matched_far(y_true, y_score, t)
    return {"recall_at_far_oracle": r,
            "achieved_far_oracle": t if r == r else np.nan}


# ----------------------------------------------------------------------
def classification_metrics(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0 or n_neg == 0:
        out = {"pr_auc": np.nan, "roc_auc": np.nan, "baseline_pr_auc": np.nan,
               "pr_auc_headroom": np.nan,
               "far_05": np.nan, "recall_05": np.nan, "precision_05": np.nan,
               "n_positive": n_pos, "n_negative": n_neg,
               "recall_at_far_oracle": np.nan, "achieved_far_oracle": np.nan}
        for far in config.MATCHED_FAR_GRID:
            out[f"recall_at_far{_fkey(far)}"] = np.nan
        return out

    ap = float(average_precision_score(y_true, y_score))
    chance = float(n_pos / len(y_true))
    out = {"pr_auc": ap,
           "roc_auc": float(roc_auc_score(y_true, y_score)),
           "baseline_pr_auc": chance,
           # 무작위(=양성 비율)에서 천장(1.0)까지의 여유폭 중 회수한 비율.
           # PR-AUC 는 기준선이 0.5 가 아니므로 이 값을 함께 보고하면
           # 모델 간 격차가 정직하게 드러난다.
           "pr_auc_headroom": float((ap - chance) / (1.0 - chance))
                              if chance < 1.0 else np.nan,
           "n_positive": n_pos, "n_negative": n_neg}

    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["far_05"] = float(100.0 * fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    out["recall_05"] = float(100.0 * tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    out["precision_05"] = float(100.0 * tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    out.update(recall_at_far_oracle(y_true, y_score))

    for far in config.MATCHED_FAR_GRID:
        out[f"recall_at_far{_fkey(far)}"] = recall_at_matched_far(y_true, y_score, far)
    return out


def _fkey(far):
    """FAR 격자값을 컬럼 이름에 안전하게 쓰기 위한 키 (2.5 -> '2p5')."""
    return f"{far:g}".replace(".", "p")


def matched_far_columns():
    return [f"recall_at_far{_fkey(f)}" for f in config.MATCHED_FAR_GRID]


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            # 양수 편향 = 실제보다 시간이 더 남았다고 예측 = 위험한 방향
            "bias": float(np.mean(y_pred - y_true))}


def per_cycle_regression(y_true, y_pred, cycle_id):
    """사이클별로 회귀 오차를 분해하여 계통 편향을 확인.

    [C-14] 판정 기준은 config.SYSTEMATIC_BIAS_RULE 하나만 쓴다.
    (기존에는 여기와 step8 이 서로 다른 기준을 써서 같은 질문에 다른 답을
    낼 수 있었다.)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    cycle_id = np.asarray(cycle_id).reshape(-1)

    rows = []
    for c in np.unique(cycle_id):
        m = cycle_id == c
        if m.sum() < 2:
            continue
        rows.append({"cycle": int(c), "n": int(m.sum()),
                     **regression_metrics(y_true[m], y_pred[m])})
    if not rows:
        return rows, {}

    biases = np.array([r["bias"] for r in rows])
    ns = np.array([r["n"] for r in rows], dtype=float)
    rmses = np.array([r["rmse"] for r in rows])
    pooled_rmse = float(np.sqrt(np.sum(rmses ** 2 * ns) / ns.sum()))
    bias_sd = float(biases.std(ddof=1)) if len(biases) > 1 else 0.0

    overall = {
        "n_cycles": len(rows),
        "bias_mean": float(biases.mean()),
        "bias_std_across_cycles": bias_sd,
        "bias_range": float(biases.max() - biases.min()),
        "pooled_rmse": pooled_rmse,
        "systematic_bias_rule": config.SYSTEMATIC_BIAS_RULE,
        "systematic_bias_detected": bool(
            len(biases) > 1 and bias_sd > config.SYSTEMATIC_BIAS_RULE * pooled_rmse),
    }
    return rows, overall


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=None, alpha=0.05, seed=0):
    n_boot = n_boot or config.BOOTSTRAP_N
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            vals.append(metric_fn(yt, ys))
        except Exception:
            continue
    if not vals:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan}
    vals = np.array(vals)
    return {"mean": float(vals.mean()), "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2))}


def full_evaluation(y_true_cls, y_score_cls, y_true_ttf, y_pred_ttf,
                    operating_threshold=None, with_ci=None, seed=0, cycle_id=None):
    with_ci = config.REPORT_BOOTSTRAP_CI if with_ci is None else with_ci

    res = classification_metrics(y_true_cls, y_score_cls)
    res.update(regression_metrics(y_true_ttf, y_pred_ttf))

    if operating_threshold is not None and not np.isnan(res["pr_auc"]):
        res.update(evaluate_at_fixed_threshold(y_true_cls, y_score_cls,
                                               operating_threshold))

    if cycle_id is not None:
        _, overall = per_cycle_regression(y_true_ttf, y_pred_ttf, cycle_id)
        res.update({f"cyc_{k}": v for k, v in overall.items()})

    if with_ci and not np.isnan(res["pr_auc"]):
        ci = bootstrap_ci(y_true_cls, y_score_cls, average_precision_score, seed=seed)
        res["pr_auc_ci_lo"] = ci["lo"]
        res["pr_auc_ci_hi"] = ci["hi"]
    return res


def _self_test():
    rng = np.random.default_rng(42)
    n = 2000
    y = (rng.random(n) < 0.05).astype(float)
    score = np.clip(0.35 * y + rng.normal(0.3, 0.2, n), 0, 1)

    m = classification_metrics(y, score)
    assert m["roc_auc"] > m["pr_auc"]
    assert m["pr_auc"] > m["baseline_pr_auc"]
    print(f"  불균형(양성 5%): ROC-AUC {m['roc_auc']:.4f} vs "
          f"PR-AUC {m['pr_auc']:.4f} (무작위 {m['baseline_pr_auc']:.4f}, "
          f"여유폭 회수 {100 * m['pr_auc_headroom']:.1f}%)")

    n_val = 800
    y_val, s_val, y_te, s_te = y[:n_val], score[:n_val], y[n_val:], score[n_val:]
    t_plain = select_threshold_at_far(y_val, s_val, 10.0, safety_factor=1.0)
    t_safe = select_threshold_at_far(y_val, s_val, 10.0, safety_factor=0.6)
    f_plain = evaluate_at_fixed_threshold(y_te, s_te, t_plain, 10.0, 1.0)
    f_safe = evaluate_at_fixed_threshold(y_te, s_te, t_safe, 10.0, 0.6)
    assert t_safe >= t_plain and f_safe["far_fixed"] <= f_plain["far_fixed"] + 1e-9
    print(f"  안전계수 1.0 -> FAR {f_plain['far_fixed']:.2f}%, "
          f"재현율 {f_plain['recall_fixed']:.1f}%")
    print(f"  안전계수 0.6 -> FAR {f_safe['far_fixed']:.2f}%, "
          f"재현율 {f_safe['recall_fixed']:.1f}%, "
          f"선택기준 대비 이동 {f_safe['far_drift_vs_selection']:+.2f}%p")

    r10 = recall_at_matched_far(y_te, s_te, 10.0)
    r20 = recall_at_matched_far(y_te, s_te, 20.0)
    assert r20 >= r10
    print(f"  동일 FAR 재현율(상한): {r10:.1f}% @10%  ->  {r20:.1f}% @20%")

    cyc = np.repeat([0, 1, 2], n // 3 + 1)[:n]
    yt = rng.uniform(0, 15, n)
    yp = yt + np.where(cyc == 0, 2.0, np.where(cyc == 1, -2.0, 0.0)) + rng.normal(0, 0.3, n)
    _, overall = per_cycle_regression(yt, yp, cyc)
    assert overall["systematic_bias_detected"]
    print(f"  사이클별 편향 범위 {overall['bias_range']:.2f}초, "
          f"통합 RMSE {overall['pooled_rmse']:.2f}초 -> 계통 편향 검출됨")

    keys = matched_far_columns()
    assert all(k in m for k in keys)
    print(f"  동일 FAR 격자 {len(keys)}개: {[k.split('far')[-1] for k in keys]}")
    print("metrics: 정상")


if __name__ == "__main__":
    _self_test()
