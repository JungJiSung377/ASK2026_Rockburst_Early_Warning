"""다중 시드 학습 및 평가.

시드를 반복하여 초기화와 배치 순서에서 오는 확률적 변동을 정량화한다. 운영
임계값은 검증 데이터에서 선택하여 평가 파티션에 그대로 적용한다.

각 실행이 끝날 때마다 진행 상황을 저장하므로, 세션이 중단되어도 완료된
작업을 반복하지 않고 이어서 실행할 수 있다.

[검토 반영 v2]
  C-1  재개 시 예측 캐시가 부분 데이터로 덮어써지던 문제. 기존 pkl 을 먼저
       읽어 병합한다. 이 캐시는 그림 3·4, step8 진단, step9 동일 FAR 표의
       입력이므로, 불완전하면 논문 그림이 조용히 일부 시드로만 만들어진다.
  A-8  모델마다 배치 순서를 시드로 되돌린다. 되돌리지 않으면 (1) 같은 시드의
       세 모델이 서로 다른 배치 순서를 보아 대응 검정의 전제가 깨지고,
       (2) 앞 모델의 조기 종료 시점이 뒤 모델 결과를 바꾸며,
       (3) 재개 실행이 처음부터 돌린 실행과 다른 숫자를 낸다.
  A-1  --ae-key 로 무잡음 조건(X_ae_clean) 대조를 한 줄로 돌릴 수 있게 한다.
  B-10 --label-threshold 로 라벨 임계값 민감도 분석을 실행할 수 있게 한다.
  B-5  달성 FAR 을 명목 목표와 '선택 기준(kappa x 목표)' 양쪽 대비로 보고한다.
"""

import argparse
import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch

import config
import env_report
from data import build_dataloaders, build_cv_dataloaders, n_cv_folds, reset_loader_order
from metrics import (full_evaluation, select_threshold_at_far, threshold_stability,
                     matched_far_columns, selection_far_target_pct)
from models import MODEL_REGISTRY, checkpoint_path, count_parameters, parameter_report
from models_meta import MAIN_TABLE_MODELS, ABLATION_TABLE_MODELS, display
from step2_tune import load_best_hp
from trainer import get_device, train_model, predict
from utils import save_json, load_json, restore_nan

METRIC_KEYS = ["pr_auc", "roc_auc", "baseline_pr_auc", "pr_auc_headroom",
               "far_fixed", "recall_fixed", "precision_fixed", "f1_fixed",
               "threshold_used", "far_overshoot", "far_drift_vs_selection",
               "far_selection_target",
               "far_05", "recall_05", "recall_at_far_oracle",
               "rmse", "mae", "r2", "bias",
               "cyc_bias_std_across_cycles", "cyc_bias_range", "cyc_pooled_rmse",
               "threshold_std", "n_positive", "n_negative",
               "pr_auc_ci_lo", "pr_auc_ci_hi"] + matched_far_columns()


def _suffix(ae_key, label_threshold):
    """조건이 기본값과 다르면 산출물 파일명을 분리한다."""
    s = ""
    if ae_key != "X_ae":
        s += "__" + ae_key.replace("X_ae_", "")
    if label_threshold is not None and abs(
            float(label_threshold) - config.WARNING_TTF_THRESHOLD) > 1e-9:
        s += f"__thr{float(label_threshold):g}".replace(".", "p")
    return s


def _paths(ae_key="X_ae", label_threshold=None):
    sfx = _suffix(ae_key, label_threshold)
    return {
        "results": os.path.join(config.RESULT_DIR, f"seed_level_results{sfx}.csv"),
        "cache": os.path.join(config.ARTIFACT_DIR, f"prediction_cache{sfx}.pkl"),
        "progress": os.path.join(config.RESULT_DIR, f"step3_progress{sfx}.json"),
        "summary": os.path.join(config.RESULT_DIR, f"summary_mean_std{sfx}.csv"),
        "suffix": sfx,
    }


# 기본 조건의 경로 (다른 모듈이 참조)
SEED_RESULTS_CSV = _paths()["results"]
PRED_CACHE = _paths()["cache"]
PROGRESS = _paths()["progress"]


def _load_existing_cache(path):
    """[C-1] 재개 시 기존 예측 캐시를 읽어 병합 기반으로 삼는다."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            old = pickle.load(f)
        preds = old.get("predictions", {})
        n = sum(len(v) for v in preds.values())
        print(f"  기존 예측 캐시 병합: 모델 {len(preds)}개 / 실행 {n}건")
        return {k: dict(v) for k, v in preds.items()}
    except Exception as e:
        print(f"  경고: 기존 캐시를 읽지 못했습니다 ({e}). 새로 만듭니다.")
        return {}


def run(seeds=None, epochs=None, verbose=False, use_cv=None, resume=True,
        ae_key="X_ae", label_threshold=None):
    seeds = seeds or config.EVAL_SEEDS
    epochs = epochs or config.EPOCHS
    use_cv = config.USE_CROSS_VALIDATION if use_cv is None else use_cv
    device = get_device()
    best_hp = load_best_hp()
    P = _paths(ae_key, label_threshold)

    print("\n" + "=" * 72)
    print("3단계  다중 시드 학습 및 평가")
    print("=" * 72)
    print(f"  device : {device}")
    print(f"  seeds  : n={len(seeds)}  {seeds}")
    print(f"  모델   : {[display(m) for m in MODEL_REGISTRY]}")
    print(f"           주 비교 {[display(m) for m in MAIN_TABLE_MODELS]}, "
          f"융합 ablation {[display(m) for m in ABLATION_TABLE_MODELS]}")
    print(f"  스펙트로그램 : {ae_key}"
          + ("   <- 무잡음 대조 실행" if ae_key != "X_ae" else ""))
    if label_threshold is not None:
        print(f"  라벨 임계값 : {label_threshold}초 (민감도 분석)")
    print(f"  조기 종료   : {config.EARLY_STOP_METRIC}")
    print(f"  배치 순서 리셋 : {config.FIX_LOADER_SEED_PER_MODEL}")
    print(f"  운영점      : 목표 FAR {config.TARGET_FAR_PCT:.0f}%, "
          f"안전계수 {config.FAR_SAFETY_FACTOR} "
          f"-> 검증 선택 기준 {selection_far_target_pct():.1f}%")
    floor = 2.0 ** (-(len(seeds) - 1)) if len(seeds) > 1 else 1.0
    print(f"  Wilcoxon 양측 p 하한 = {floor:.2e} "
          f"({'충분' if floor < 0.05 else '부족'})")
    if not use_cv:
        print("  참고: 분할이 고정되어 있으므로 시드 간 표준편차는 최적화 잡음이며")
        print("        일반화 오차가 아닙니다. --cv 로 분할 변동까지 확인하세요.")

    folds = list(range(n_cv_folds())) if use_cv else [None]
    done = load_json(P["progress"], {}) if resume else {}
    if done:
        print(f"  재개: 이미 완료된 실행 {len(done)}건")
    rows = list(done.values())

    # [C-1] 재개면 기존 캐시를 먼저 읽어 둔다.
    pred_cache = _load_existing_cache(P["cache"]) if (resume and not use_cv) else {}
    total_sec = 0.0

    for fold in folds:
        for seed in seeds:
            tag = f"fold{fold}_" if fold is not None else ""
            head = (f"{'FOLD ' + str(fold) + ' | ' if fold is not None else ''}"
                    f"SEED {seed}")
            print(f"\n{'='*72}\n{head}\n{'='*72}")
            env_report.set_deterministic(seed)

            show = (seed == seeds[0])
            if use_cv:
                train_loader, val_loader, test_loader, meta = build_cv_dataloaders(
                    fold, seed=seed, verbose=show, ae_key=ae_key,
                    label_threshold=label_threshold)
            else:
                train_loader, val_loader, test_loader, meta = build_dataloaders(
                    seed=seed, verbose=show, ae_key=ae_key,
                    label_threshold=label_threshold)
            ttf_scale = meta["ttf_scale"]

            for name, cls in MODEL_REGISTRY.items():
                key = f"{tag}{name}__seed{seed}"
                if key in done and (use_cv or (name in pred_cache
                                               and seed in pred_cache.get(name, {}))):
                    print(f"  {display(name)} 완료됨 (지표·예측 모두 보유)")
                    continue
                if key in done:
                    # 지표는 있는데 예측이 없다 -> 캐시 무결성을 위해 다시 계산
                    print(f"  {display(name)} 지표는 있으나 예측 캐시가 없어 "
                          f"재실행합니다 (C-1)")

                hp = best_hp.get(name, {"lr": 1e-3, "ttf_weight": 1.0})
                lr, tw = hp["lr"], hp["ttf_weight"]
                print(f"\n[{name}] lr={lr:.0e}, ttf_weight={tw}")

                env_report.set_deterministic(seed)
                reset_loader_order(meta)                    # [A-8]
                model = cls().to(device)
                n_params = count_parameters(model)
                ckpt = checkpoint_path(name, seed, tag=f"main{tag}{P['suffix']}")

                res = train_model(model, name, train_loader, val_loader, device,
                                  lr=lr, ttf_weight=tw, epochs=epochs,
                                  verbose=verbose, save_path=ckpt,
                                  ttf_scale=ttf_scale, loader_meta=meta)
                total_sec += res["train_seconds"]

                # 검증 데이터에서 임계값을 선택하고 부트스트랩으로 안정성을 추정
                v_true, v_score, _, _, _, _ = predict(res["model"], name, val_loader,
                                                      device, ttf_scale=ttf_scale)
                t_star = select_threshold_at_far(v_true, v_score)
                stab = threshold_stability(v_true, v_score, seed=seed)

                # 고정 임계값으로 평가 파티션 검증
                y_true_c, y_score_c, y_true_t, y_pred_t, cyc, _ = predict(
                    res["model"], name, test_loader, device, ttf_scale=ttf_scale)
                m = full_evaluation(y_true_c, y_score_c, y_true_t, y_pred_t,
                                    operating_threshold=t_star,
                                    seed=seed, cycle_id=cyc)
                m["threshold_std"] = stab["threshold_std"]

                print(f"   -> PR-AUC {m['pr_auc']:.4f} "
                      f"(무작위 {m['baseline_pr_auc']:.3f}, "
                      f"여유폭 {100*m['pr_auc_headroom']:.1f}%) | "
                      f"ROC-AUC {m['roc_auc']:.4f} | "
                      f"Recall@fixed {m.get('recall_fixed', np.nan):5.1f}% "
                      f"(FAR {m.get('far_fixed', np.nan):5.1f}%, "
                      f"선택기준 대비 {m.get('far_drift_vs_selection', np.nan):+.1f}%p) | "
                      f"RMSE {m['rmse']:.4f} | R2 {m['r2']:.4f} "
                      f"({res['train_seconds']:.0f}s)")

                row = {"seed": seed, "fold": fold if fold is not None else 0,
                       "model": name, "n_params": n_params, "lr": lr, "ttf_weight": tw,
                       "best_epoch": res["best_epoch"],
                       "best_val_loss": res["best_val_loss"],
                       "best_select_score": res["best_select_score"],
                       "early_stop_metric": res["early_stop_metric"],
                       "ae_key": ae_key,
                       "label_threshold": meta["label_threshold"],
                       "ttf_scale": ttf_scale,
                       "train_seconds": round(res["train_seconds"], 1),
                       **{k: m.get(k, np.nan) for k in METRIC_KEYS}}
                rows = [r for r in rows
                        if not (r.get("model") == name and r.get("seed") == seed
                                and r.get("fold", 0) == (fold or 0))]
                rows.append(row)
                done[key] = row
                save_json(P["progress"], done)

                if not use_cv:
                    pred_cache.setdefault(name, {})[seed] = {
                        "y_true_cls": y_true_c, "y_score_cls": y_score_c,
                        "y_true_ttf": y_true_t, "y_pred_ttf": y_pred_t,
                        "cycle_id": cyc, "threshold": t_star}
                    # 매 실행마다 저장해 중단에도 캐시가 살아남게 한다
                    with open(P["cache"], "wb") as f:
                        pickle.dump({"predictions": pred_cache,
                                     "seeds": sorted({s for v in pred_cache.values()
                                                      for s in v}),
                                     "ae_key": ae_key,
                                     "label_threshold": meta["label_threshold"]}, f)

                del model, res
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(restore_nan(rows, METRIC_KEYS))
    df.to_csv(P["results"], index=False)

    metric_cols = ["pr_auc", "roc_auc", "recall_fixed", "far_fixed",
                   "precision_fixed", "rmse", "mae", "r2"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(P["summary"])

    print("\n" + "=" * 72)
    print(f"표 2  성능 요약, 시드 {len(seeds)}개의 평균 +/- 표준편차")
    print("=" * 72)
    pretty = pd.DataFrame(index=[display(i) for i in summary.index])
    for c in metric_cols:
        pretty[c] = (summary[(c, "mean")].map("{:.4f}".format) + " +/- " +
                     summary[(c, "std")].fillna(0).map("{:.4f}".format)).values
    # 논문 표에 넣을 파라미터 수 열
    npar = df.groupby("model")["n_params"].first()
    pretty["n_params"] = [f"{int(npar[i]):,}" for i in summary.index]
    print(pretty.to_string())
    print("=" * 72)
    if df["baseline_pr_auc"].notna().any():
        print(f"  무작위 수준 PR-AUC: {df['baseline_pr_auc'].dropna().iloc[0]:.4f}")

    # 달성된 오경보율이 목표에서 벗어난 정도 (두 기준 모두)
    if "far_overshoot" in df.columns and df["far_overshoot"].notna().any():
        ov = df.groupby("model")[["far_overshoot", "far_drift_vs_selection"]].mean()
        print(f"\n  오경보율 이동 (명목 목표 {config.TARGET_FAR_PCT:.0f}% / "
              f"선택 기준 {selection_far_target_pct():.1f}%)")
        for m_, r in ov.iterrows():
            flag = " " if abs(r["far_overshoot"]) <= 3 else "*"
            print(f"    {flag} {display(m_):24s}: "
                  f"명목 대비 {r['far_overshoot']:+.2f}%p, "
                  f"선택 기준 대비 {r['far_drift_vs_selection']:+.2f}%p")
        print("    선택 기준 대비 이동폭이 안전계수가 필요했던 근거입니다.")
        print("    모델마다 이동폭이 다르면 단일 안전계수로 운영점이 정렬되지")
        print("    않는다는 뜻이므로, 동일 FAR 재평가(step9)가 필요합니다.")

    # 사이클 간 계통적 회귀 편향
    if ("cyc_bias_std_across_cycles" in df.columns
            and df["cyc_bias_std_across_cycles"].notna().any()):
        cb = df.groupby("model")["cyc_bias_std_across_cycles"].mean()
        print("\n  사이클 간 회귀 편향의 표준편차:")
        for m_, v in cb.items():
            print(f"      {display(m_):24s}: {v:.3f} s")
        print("    사이클별 분해는 step8_diagnostics.py를 참조하세요.")

    rep = parameter_report()
    if "_reduction_vs_proposed_pct" in rep:
        print(f"\n  파라미터: 단순 결합이 제안 모델보다 "
              f"{rep['_reduction_vs_proposed_pct']:.2f}% 적게 사용 "
              f"(역방향 표기는 {rep['_increase_vs_ablation_pct']:.2f}%)")

    print(f"\n  총 학습 시간: {total_sec/60:.1f}분")
    env_report.save_environment(extra={"step": "step3_train_eval", "seeds": list(seeds),
                                       "epochs": epochs, "use_cv": use_cv,
                                       "ae_key": ae_key,
                                       "label_threshold": label_threshold,
                                       "total_train_seconds": round(total_sec, 1),
                                       "best_hyperparams": best_hp})
    print(f"\n  저장: {P['results']}")
    print(f"  저장: {P['cache']}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ae-key", default="X_ae",
                    choices=["X_ae", "X_ae_clean"],
                    help="X_ae_clean 이면 간섭 없는 조건으로 대조 실행 (검토 A-1)")
    ap.add_argument("--label-threshold", type=float, default=None,
                    help="라벨 임계값(초) 민감도 분석 (검토 B-10)")
    args = ap.parse_args()
    run(seeds=args.seeds, epochs=args.epochs, verbose=args.verbose,
        use_cv=args.cv or None, resume=not args.no_resume,
        ae_key=args.ae_key, label_threshold=args.label_threshold)
