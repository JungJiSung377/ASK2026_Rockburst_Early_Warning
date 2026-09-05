"""Ablation 실험.

특징 ablation은 전조 상태 벡터의 개별 성분과 세그먼트 내/세그먼트 간 그룹을
각각 제거하여, 성능 향상이 스펙트로그램 인코더가 접근할 수 없는 정보에서
비롯되는지를 검증한다. 제거된 성분은 0이 아니라 학습 평균으로 대체하는데,
0은 관측 분포를 벗어난 값이기 때문이다.

손실 ablation은 에너지 가중항과 focal 항을 제거하여 각각의 기여가 측정
가능한 수준인지 확인한다.

각 설정은 전체 구성과 대응 검정으로 비교하고 Holm-Bonferroni 보정을 적용한다.

[검토 반영 v2]
  B-18 drop_all 은 '상태 입력 전체 제외'인데, stress(=정규화 RMS)가 손실
       가중치로 남아 상태 정보가 완전히 제거되지 않았다. lambda_p 도 함께 끈다.
  C-4  no_focal 설정은 gamma=0 만 주므로 focal 이 아니라 'focusing 항 제거'다.
       라벨을 정확히 바꾸고, alpha 까지 끄는 설정을 별도로 추가한다.
  B-11 ablation 시드 수를 산출물에 명시해 표 2(15시드)와 섞이지 않게 한다.
  B-12 개별 제거 효과와 동시 제거 효과를 함께 출력해 상호작용을 명시한다.
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats

import config
import env_report
from data import build_dataloaders
from metrics import full_evaluation, select_threshold_at_far
from models import MODEL_REGISTRY
from models_meta import PROPOSED_MODEL, display
from step2_tune import load_best_hp
from step4_stats import holm_bonferroni, cohens_dz, interpret_dz
from trainer import get_device, train_model, predict, make_criterion
from utils import save_json, load_json, restore_nan

FEATURE_PROGRESS = os.path.join(config.RESULT_DIR, "ablation_feature_progress.json")
LOSS_PROGRESS = os.path.join(config.RESULT_DIR, "ablation_loss_progress.json")
METRIC_KEYS = ["pr_auc", "roc_auc", "recall_fixed", "precision_fixed",
               "far_fixed", "rmse", "mae", "r2"]


def build_feature_settings():
    s = [("full", [], "전체 상태 벡터 (기준)")]
    for i, fname in enumerate(config.STATE_FEATURE_NAMES):
        s.append((f"drop_{fname}", [i], f"{fname} 제외"))
    s.append(("drop_inter_segment", list(config.INTER_SEGMENT_IDX),
              "세그먼트 간 이력 제외 (스펙트로그램에 없는 정보)"))
    s.append(("drop_intra_segment", list(config.INTRA_SEGMENT_IDX),
              "세그먼트 내 통계 제외 (스펙트로그램에서 복원 가능)"))
    s.append(("drop_all", list(range(config.STATE_DIM)), "상태 입력 전체 제외"))
    return s


def _train_eval_once(cls, model_name, mask, hp, seed, epochs, device,
                     criterion=None, verbose=False, setting=None):
    env_report.set_deterministic(seed)
    tr, va, te, meta = build_dataloaders(mask_features=mask if mask else None,
                                         seed=seed, verbose=False)
    ttf_scale = meta["ttf_scale"]

    # [B-18] 상태 입력을 전부 제거하는 설정에서는 손실의 에너지 가중항도 꺼야
    # 상태 정보가 실제로 완전히 빠진다.
    if criterion is None and setting == "drop_all" and config.DROP_ALL_ALSO_DISABLES_PHYSICS:
        criterion = make_criterion(model_name, hp["ttf_weight"], lambda_p=0.0)

    env_report.set_deterministic(seed)
    model = cls().to(device)
    res = train_model(model, model_name, tr, va, device, lr=hp["lr"],
                      ttf_weight=hp["ttf_weight"], epochs=epochs,
                      verbose=verbose, criterion=criterion,
                      ttf_scale=ttf_scale, loader_meta=meta)

    v_true, v_score, _, _, _, _ = predict(res["model"], model_name, va, device,
                                          ttf_scale=ttf_scale)
    t_star = select_threshold_at_far(v_true, v_score)
    yc, ys, yt, yp, cyc, _ = predict(res["model"], model_name, te, device,
                                     ttf_scale=ttf_scale)
    m = full_evaluation(yc, ys, yt, yp, operating_threshold=t_star,
                        seed=seed, cycle_id=cyc)

    del model, res
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return m


def _test_vs_full(df, group_col, metrics=("pr_auc", "recall_fixed", "rmse"),
                  alpha=0.05):
    """각 ablation 설정을 전체 구성과 대응 비교.

    같은 시드로 같은 구조를 학습한 것끼리의 비교이므로 대응 검정이 타당하다.
    """
    seeds = sorted(df["seed"].unique())
    n = len(seeds)
    rows = []
    # [수정 v2.1] 시드가 1개면 대응 t검정의 자유도가 0이라 p 가 NaN 이다.
    #   예전에는 그 NaN 이 holm_bonferroni 에서 0.0 으로 뒤집혀
    #   모든 비교가 '유의함'으로 출력되었다. 이제 검정 불가로 표기한다.
    testable = n >= 2
    for metric in metrics:
        if metric not in df.columns:
            continue
        piv = df.pivot_table(index="seed", columns=group_col, values=metric,
                             dropna=False).reindex(seeds)
        if "full" not in piv.columns:
            continue
        full_v = piv["full"].values
        if np.isnan(full_v).any():
            continue
        for setting in piv.columns:
            if setting == "full":
                continue
            v = piv[setting].values
            if np.isnan(v).any():
                continue
            diff = v - full_v
            if not testable:
                t_p = w_p = np.nan          # 시드 1개: 검정 불가
            elif np.allclose(diff, 0):
                t_p, w_p = 1.0, 1.0
            else:
                _, t_p = stats.ttest_rel(v, full_v)
                try:
                    _, w_p = stats.wilcoxon(v, full_v)
                except Exception:
                    w_p = np.nan
            d = cohens_dz(v, full_v) if testable else np.nan
            rows.append({"metric": metric, "setting": setting,
                         "full_mean": float(full_v.mean()),
                         "setting_mean": float(v.mean()),
                         "delta": float(diff.mean()),
                         "p_ttest": float(t_p),
                         "p_wilcoxon": float(w_p) if w_p == w_p else np.nan,
                         "effect_dz": d,
                         "effect_label": interpret_dz(d) if testable else "검정 불가",
                         "n_seeds": n, "testable": bool(testable)})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    adj, rej = holm_bonferroni(res["p_ttest"].values, alpha=alpha)
    res["p_ttest_holm"] = adj
    res["significant_holm"] = rej
    return res


def run_feature_ablation(seeds=None, epochs=None, model_name=None,
                         verbose=False, resume=True):
    model_name = model_name or PROPOSED_MODEL
    seeds = seeds or config.ABLATION_SEEDS
    epochs = epochs or config.EPOCHS
    device = get_device()
    hp = load_best_hp().get(model_name, {"lr": 1e-3, "ttf_weight": 1.0})
    settings = build_feature_settings()

    print("\n" + "=" * 76)
    print(f"5A단계  특징 ablation ({display(model_name)})")
    print("=" * 76)
    floor = 2.0 ** (-(len(seeds) - 1)) if len(seeds) > 1 else 1.0
    print(f"  시드 {len(seeds)}개 | 에폭 {epochs} | 순위 검정 하한 "
          f"{floor:.4f} ({'충분' if floor < 0.05 else '부족'})")
    print(f"  설정 {len(settings)}개 x 시드 {len(seeds)}개 = "
          f"학습 {len(settings)*len(seeds)}회")
    # [B-11] 표 2 는 EVAL_SEEDS 로 만들어지므로 기준값이 다르다.
    print(f"  주의: 이 결과의 기준선(full)은 시드 {len(seeds)}개 평균입니다.")
    print(f"        표 2 의 제안 모델 값은 시드 {len(config.EVAL_SEEDS)}개 평균이라")
    print(f"        두 숫자가 미세하게 다릅니다. 논문에 시드 수를 명시하세요.\n")

    done = load_json(FEATURE_PROGRESS, {}) if resume else {}
    if done:
        print(f"  재개: 이미 완료된 실행 {len(done)}건")
    rows = list(done.values())
    cls = MODEL_REGISTRY[model_name]

    for sname, mask, desc in settings:
        print(f"\n{'-'*76}\n[{sname}] {desc}\n{'-'*76}")
        for seed in seeds:
            key = f"{sname}__seed{seed}"
            if key in done:
                print(f"   seed {seed} (완료)")
                continue
            m = _train_eval_once(cls, model_name, mask, hp, seed, epochs, device,
                                 verbose=verbose, setting=sname)
            print(f"   seed {seed}: PR-AUC {m['pr_auc']:.4f} | "
                  f"Recall@fixed {m.get('recall_fixed', np.nan):5.1f}% | "
                  f"RMSE {m['rmse']:.4f}")
            row = {"setting": sname, "description": desc, "seed": seed,
                   "n_seeds_total": len(seeds),
                   "masked": ",".join(config.STATE_FEATURE_NAMES[i] for i in mask)
                             or "none",
                   **{k: m.get(k, np.nan) for k in METRIC_KEYS}}
            rows.append(row)
            done[key] = row
            save_json(FEATURE_PROGRESS, done)

    df = pd.DataFrame(restore_nan(rows, METRIC_KEYS))
    df.to_csv(os.path.join(config.RESULT_DIR, "ablation_feature_raw.csv"), index=False)

    agg = df.groupby(["setting", "description"]).agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        recall_mean=("recall_fixed", "mean"), rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean")).reset_index()
    base = agg[agg.setting == "full"].iloc[0]
    agg["delta_pr_auc"] = agg["pr_auc_mean"] - base["pr_auc_mean"]
    agg["delta_pr_auc_pct"] = 100 * agg["delta_pr_auc"] / base["pr_auc_mean"]
    agg["delta_recall"] = agg["recall_mean"] - base["recall_mean"]
    agg["delta_rmse"] = agg["rmse_mean"] - base["rmse_mean"]
    agg["n_seeds"] = len(seeds)
    agg = agg.sort_values("delta_pr_auc")
    agg.to_csv(os.path.join(config.RESULT_DIR, "ablation_summary.csv"), index=False)

    print("\n" + "=" * 88)
    print(f"표 5  전체 상태 벡터 대비 특징 ablation (시드 {len(seeds)}개)")
    print("=" * 88)
    show = agg[["setting", "pr_auc_mean", "pr_auc_std", "delta_pr_auc",
                "delta_pr_auc_pct", "delta_recall", "delta_rmse"]].copy()
    show.columns = ["설정", "PR-AUC", "표준편차", "PR-AUC 변화",
                    "PR-AUC 변화(%)", "재현율 변화", "RMSE 변화"]
    for c in ["PR-AUC", "표준편차", "PR-AUC 변화", "RMSE 변화"]:
        show[c] = show[c].map("{:+.4f}".format)
    show["PR-AUC 변화(%)"] = show["PR-AUC 변화(%)"].map("{:+.2f}".format)
    show["재현율 변화"] = show["재현율 변화"].map("{:+.2f}".format)
    print(show.to_string(index=False))

    sig = _test_vs_full(df, "setting")
    if not sig.empty:
        sig.to_csv(os.path.join(config.RESULT_DIR,
                                "ablation_feature_significance.csv"), index=False)
        print("\n" + "=" * 88)
        print("표 5b  각 ablation의 유의성 (대응 검정, Holm 보정)")
        print("=" * 88)
        s = sig[sig.metric == "pr_auc"][["setting", "delta", "p_ttest", "p_ttest_holm",
                                         "p_wilcoxon", "effect_dz", "effect_label",
                                         "significant_holm"]].copy()
        s = s.sort_values("delta")
        for c in ["delta", "effect_dz"]:
            s[c] = s[c].map("{:+.4f}".format)
        for c in ["p_ttest", "p_ttest_holm", "p_wilcoxon"]:
            s[c] = s[c].map(lambda v: f"{v:.4f}" if v == v else "n/a")
        print(s.to_string(index=False))
        print("=" * 88)

    _report_core_claim(agg, sig, len(seeds))
    return df, agg


def _report_core_claim(agg, sig, n_seeds):
    """핵심 주장 검증 + [B-12] 개별/동시 제거 상호작용 명시."""
    try:
        def _d(name):
            r = agg[agg.setting == name]
            return float(r["delta_pr_auc"].iloc[0]) if not r.empty else np.nan

        def _p(name):
            if sig is None or sig.empty:
                return np.nan, False
            r = sig[(sig.metric == "pr_auc") & (sig.setting == name)]
            if r.empty:
                return np.nan, False
            return float(r["p_ttest_holm"].iloc[0]), bool(r["significant_holm"].iloc[0])

        inter, intra = _d("drop_inter_segment"), _d("drop_intra_segment")
        allx = _d("drop_all")
        p_inter, sig_ok = _p("drop_inter_segment")

        print("\n" + "=" * 76)
        print("  핵심 주장: 어느 시간 스케일이 정보를 담고 있는가")
        print("=" * 76)
        ptxt = f"p_holm={p_inter:.4f}" if p_inter == p_inter else "검정 불가"
        print(f"  세그먼트 간 특징 제거: {inter:+.4f}  {ptxt}")
        print(f"  세그먼트 내 특징 제거: {intra:+.4f}")
        print(f"  상태 벡터 전체 제거  : {allx:+.4f}")

        # [B-12] 개별 제거의 합 대 동시 제거
        singles = [(config.STATE_FEATURE_NAMES[i], _d(f"drop_{config.STATE_FEATURE_NAMES[i]}"))
                   for i in config.INTER_SEGMENT_IDX]
        ssum = sum(v for _, v in singles if v == v)
        print("\n  inter 성분 개별 제거:")
        for nm, v in singles:
            print(f"    {nm:26s} {v:+.4f}")
        print(f"    개별 효과의 합            {ssum:+.4f}")
        if inter == inter and abs(ssum) > 1e-9:
            print(f"    동시 제거                 {inter:+.4f}  "
                  f"({inter/ssum:.1f}배)")
            print("    -> 한쪽이 남아 있으면 다른 쪽 정보를 상당 부분 보완할 수")
            print("       있다는 뜻입니다. '동일 물리량'이라는 해석은 별도 근거가")
            print("       필요하므로, 결과를 그대로 서술하세요 (심사 지적 22).")
        pos = [nm for nm, v in singles if v == v and v > 0]
        if pos:
            print(f"    주의: {pos} 는 단독 제거 시 오히려 성능이 올랐습니다.")
            print("          그림에 숫자가 그대로 보이므로 본문에 명시하세요.")

        # [B-11] '이득의 몇 %' 는 같은 실험 안의 drop_all 기준으로 계산
        if allx == allx and abs(allx) > 1e-9:
            share = 100.0 * inter / allx
            print(f"\n  상태 벡터 기여 대비 inter 비중: {share:.1f}%")
            print("  (같은 ablation 실험 안의 drop_all 을 기준으로 계산.")
            print("   표 2 의 스펙트로그램 전용 값과 섞지 마세요 — 시드 수가 다릅니다.)")

        if n_seeds < 2:
            # [수정 v2.1] 시드 1개에서는 유의성 문장을 아예 내지 않는다.
            if inter < intra:
                print("\n  방향은 예상과 같습니다. 다만 시드가 1개라 유의성을")
                print("  판정할 수 없습니다. 논문에 쓸 결론은 시드를 늘려")
                print("  다시 실행한 뒤에 내리세요.")
            else:
                print("\n  세그먼트 내 특징 제거의 영향이 더 큽니다 (시드 1개,")
                print("  유의성 판정 불가).")
        elif inter < intra and sig_ok:
            print("\n  세그먼트 간 특징을 제거할 때 성능 저하가 세그먼트 내 특징")
            print("  제거보다 유의하게 큽니다. 따라서 성능 향상은 세그먼트 단위")
            print("  스펙트로그램 인코더가 접근하기 어려운 정보에서 비롯됩니다.")
        elif inter < intra:
            print("\n  방향은 예상과 같으나 Holm 보정 후 유의하지 않습니다.")
        else:
            print("\n  세그먼트 내 특징 제거의 영향이 더 큽니다. 관측된 대로")
            print("  보고하고 추세 창 길이 민감도 분석을 추가하세요.")
    except (IndexError, KeyError):
        pass


def run_loss_ablation(seeds=None, epochs=None, model_name=None,
                      verbose=False, resume=True):
    model_name = model_name or PROPOSED_MODEL
    seeds = seeds or config.ABLATION_SEEDS
    epochs = epochs or config.EPOCHS
    device = get_device()
    hp = load_best_hp().get(model_name, {"lr": 1e-3, "ttf_weight": 1.0})

    # [C-4] gamma=0 은 focal 을 없애는 것이 아니라 focusing 항만 없앤다.
    # alpha 까지 없애는 설정을 별도로 둔다.
    loss_settings = [
        ("full", {"lambda_p": config.LAMBDA_PHYSICS, "gamma": config.FOCAL_GAMMA,
                  "alpha_mode": config.FOCAL_ALPHA_MODE},
         "에너지 가중 + focal (기준)"),
        ("no_physics", {"lambda_p": 0.0, "gamma": config.FOCAL_GAMMA,
                        "alpha_mode": config.FOCAL_ALPHA_MODE},
         "에너지 가중항 제외"),
        ("no_focusing", {"lambda_p": config.LAMBDA_PHYSICS, "gamma": 0.0,
                         "alpha_mode": config.FOCAL_ALPHA_MODE},
         "focusing 항 제거 (gamma=0, 클래스 가중은 유지)"),
        ("plain_bce", {"lambda_p": config.LAMBDA_PHYSICS, "gamma": 0.0,
                       "alpha_mode": "none"},
         "순수 BCE (gamma=0, alpha 없음)"),
        ("neither", {"lambda_p": 0.0, "gamma": 0.0, "alpha_mode": "none"},
         "에너지 가중·focal 모두 제외"),
    ]

    print("\n" + "=" * 76)
    print(f"5B단계  손실 ablation ({display(model_name)})")
    print("=" * 76)
    print(f"  focal alpha 모드: {config.FOCAL_ALPHA_MODE}")
    print(f"  설정 {len(loss_settings)}개 x 시드 {len(seeds)}개 = "
          f"학습 {len(loss_settings)*len(seeds)}회\n")

    done = load_json(LOSS_PROGRESS, {}) if resume else {}
    if done:
        print(f"  재개: 이미 완료된 실행 {len(done)}건")
    rows = list(done.values())
    cls = MODEL_REGISTRY[model_name]

    for sname, params, desc in loss_settings:
        print(f"\n{'-'*76}\n[{sname}] {desc}\n{'-'*76}")
        for seed in seeds:
            key = f"{sname}__seed{seed}"
            if key in done:
                print(f"   seed {seed} (완료)")
                continue
            criterion = make_criterion(model_name, hp["ttf_weight"],
                                       lambda_p=params["lambda_p"],
                                       gamma=params["gamma"],
                                       alpha_mode=params["alpha_mode"])
            m = _train_eval_once(cls, model_name, [], hp, seed, epochs, device,
                                 criterion=criterion, verbose=verbose,
                                 setting=sname)
            print(f"   seed {seed}: PR-AUC {m['pr_auc']:.4f} | "
                  f"RMSE {m['rmse']:.4f} | R2 {m['r2']:.4f}")
            row = {"loss_setting": sname, "description": desc, "seed": seed, **params,
                   **{k: m.get(k, np.nan) for k in METRIC_KEYS}}
            rows.append(row)
            done[key] = row
            save_json(LOSS_PROGRESS, done)

    df = pd.DataFrame(restore_nan(rows, METRIC_KEYS))
    df.to_csv(os.path.join(config.RESULT_DIR, "ablation_loss.csv"), index=False)

    agg = df.groupby("loss_setting")[["pr_auc", "recall_fixed", "rmse", "r2"]].agg(
        ["mean", "std"]).round(4)
    print("\n" + "=" * 76)
    print("표 6  손실 ablation")
    print("=" * 76)
    print(agg.to_string())

    sig = _test_vs_full(df, "loss_setting", metrics=("pr_auc", "rmse", "r2"))
    if not sig.empty:
        sig.to_csv(os.path.join(config.RESULT_DIR,
                                "ablation_loss_significance.csv"), index=False)
        print("\n표 6b  손실 ablation 유의성 (전체 손실 대비, Holm 보정)")
        s = sig[["metric", "setting", "delta", "p_ttest_holm", "effect_dz",
                 "significant_holm"]].copy()
        for c in ["delta", "effect_dz"]:
            s[c] = s[c].map("{:+.4f}".format)
        s["p_ttest_holm"] = s["p_ttest_holm"].map(
            lambda v: f"{v:.4f}" if v == v else "검정 불가")
        print(s.to_string(index=False))
        if not bool(sig["testable"].iloc[0]):
            print("  주의: 시드가 1개라 유의성 검정이 성립하지 않습니다.")
            print("        차이(delta)의 방향만 참고하고, 논문에는 시드를 늘려")
            print("        다시 실행한 결과를 쓰세요.")

        nop = sig[(sig.setting == "no_physics") & (sig.metric == "rmse")]
        print("\n" + "=" * 76)
        print("  에너지 가중항의 기여 (RMSE 기준)")
        print("=" * 76)
        if not nop.empty:
            r = nop.iloc[0]
            ph = r["p_ttest_holm"]
            ptxt = f"p_Holm = {ph:.4f}" if ph == ph else "검정 불가"
            print(f"   full RMSE {r['full_mean']:.4f} vs "
                  f"no_physics {r['setting_mean']:.4f} "
                  f"(차이 {r['delta']:+.4f}, {ptxt})")
            if not bool(r.get("testable", True)):
                print("  시드가 1개라 유의성을 판정할 수 없습니다. 이 문장은")
                print("  시드를 늘려 다시 실행한 뒤에 쓰세요.")
            elif r["significant_holm"] and r["delta"] > 0:
                print("  해당 항을 제거하면 성능이 유의하게 저하되므로, 물리 정보를")
                print("  반영했다는 서술이 뒷받침됩니다.")
            else:
                print("  유의한 기여가 확인되지 않았습니다. 손실을 physics-informed가")
                print("  아니라 도메인 지식 기반 가중을 적용한 다중 과제 목적함수로")
                print("  서술하고, 이 ablation 결과를 함께 보고하세요.")
                print("  학습 로그의 loss 성분 비중(step3 --verbose)을 함께 확인하면")
                print("  이 항이 스케일상 애초에 작았는지 알 수 있습니다 (검토 B-17).")
    return df, agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--only", choices=["feature", "loss"], default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    resume = not args.no_resume
    if args.only != "loss":
        run_feature_ablation(args.seeds, args.epochs, args.model, args.verbose, resume)
    if args.only != "feature":
        run_loss_ablation(args.seeds, args.epochs, args.model, args.verbose, resume)
