"""모든 모델에 동일하게 적용하는 하이퍼파라미터 탐색.

선택 기준은 검증 PR-AUC와 R2의 복합 지표이다. 손실값은 탐색 대상인 회귀
가중치에 비례해 커지므로, 손실로 선택하면 예측 성능과 무관하게 가장 작은
가중치가 뽑히는 편향이 생긴다.

탐색 시드는 평가 시드와 분리하여, 보고되는 성능이 탐색 과정 때문에 낙관적으로
치우치지 않도록 한다.

[검토 반영 v2]
  A-8  조합·모델마다 배치 순서를 시드로 되돌린다. 기존에는 하나의 로더를
       계속 재사용해 조합마다 데이터 순서가 달라졌고, 그 차이가 탐색 결과에
       잡음으로 섞였다.
"""

import argparse
import itertools
import json
import os

import pandas as pd
import torch

import config
import env_report
from data import build_dataloaders, reset_loader_order
from models import MODEL_REGISTRY, count_parameters, parameter_report
from models_meta import display
from trainer import get_device, train_model, hp_selection_score

BEST_HP_PATH = os.path.join(config.RESULT_DIR, "best_hyperparams.json")


def run_tuning(epochs=None, quick=False, verbose=False):
    epochs = epochs or config.TUNE_EPOCHS
    device = get_device()
    print("\n" + "=" * 72)
    print("2단계  하이퍼파라미터 탐색")
    print("=" * 72)
    print(f"  device : {device}")
    print(f"  탐색 시드   : {config.TUNE_SEED} (평가 시드와 분리)")
    criterion = ("검증 PR-AUC + R2"
                 if config.HP_SELECTION_METRIC == "composite" else "검증 손실")
    print(f"  선택 기준   : {criterion}")
    print(f"  조기 종료   : {config.EARLY_STOP_METRIC}")
    print(f"  배치 순서 리셋: {config.FIX_LOADER_SEED_PER_MODEL}")

    keys = list(config.HP_GRID.keys())
    combos = [dict(zip(keys, v))
              for v in itertools.product(*[config.HP_GRID[k] for k in keys])]
    if quick:
        combos = combos[:2]
    print(f"  탐색 격자   : 조합 {len(combos)}개 x 모델 {len(MODEL_REGISTRY)}개 "
          f"= 학습 {len(combos)*len(MODEL_REGISTRY)}회\n")

    env_report.set_deterministic(config.TUNE_SEED)
    train_loader, val_loader, _, meta = build_dataloaders(seed=config.TUNE_SEED,
                                                          verbose=True)
    ttf_scale = meta["ttf_scale"]

    rows, best_hp = [], {}

    for name, cls in MODEL_REGISTRY.items():
        print(f"\n{'-'*72}\n  {display(name)}\n{'-'*72}")
        best = {"score": -float("inf")}

        for ci, hp in enumerate(combos, 1):
            env_report.set_deterministic(config.TUNE_SEED)
            reset_loader_order(meta)          # [A-8]
            model = cls().to(device)
            n_params = count_parameters(model)

            print(f"  [{ci}/{len(combos)}] lr={hp['lr']:.0e}, "
                  f"ttf_weight={hp['ttf_weight']}")
            res = train_model(model, name, train_loader, val_loader, device,
                              lr=hp["lr"], ttf_weight=hp["ttf_weight"],
                              epochs=epochs, verbose=verbose, patience=4,
                              ttf_scale=ttf_scale, loader_meta=meta)

            sel = hp_selection_score(res["model"], name, val_loader, device,
                                     ttf_scale=ttf_scale,
                                     val_loss=res["best_val_loss"])
            print(f"       -> composite {sel['score']:.4f} "
                  f"(PR-AUC {sel['val_pr_auc']:.4f} + R2 {sel['val_r2']:.4f}) "
                  f"| val loss {res['best_val_loss']:.4f} "
                  f"| ep {res['best_epoch']} | {res['train_seconds']:.0f}s")

            rows.append({"model": name, "n_params": n_params, **hp,
                         "composite_score": sel["score"],
                         "val_pr_auc": sel["val_pr_auc"], "val_r2": sel["val_r2"],
                         "val_loss": res["best_val_loss"],
                         "best_epoch": res["best_epoch"],
                         "train_seconds": round(res["train_seconds"], 1)})

            if sel["score"] > best["score"]:
                best = {"score": sel["score"], **hp}

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        best_hp[name] = {k: v for k, v in best.items() if k != "score"}
        best_hp[name]["tuned_composite"] = best["score"]
        print(f"  {display(name)} 선택: lr={best['lr']:.0e}, "
              f"회귀 가중치={best['ttf_weight']} "
              f"(composite {best['score']:.4f})")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.RESULT_DIR, "hyperparameter_search.csv")
    df.to_csv(csv_path, index=False)

    with open(BEST_HP_PATH, "w") as f:
        json.dump({"grid": config.HP_GRID, "tune_seed": config.TUNE_SEED,
                   "tune_epochs": epochs,
                   "selection_metric": config.HP_SELECTION_METRIC,
                   "early_stop_metric": config.EARLY_STOP_METRIC,
                   "normalize_ttf": config.NORMALIZE_TTF, "ttf_scale": ttf_scale,
                   "best_per_model": best_hp}, f, indent=2)

    print("\n" + "=" * 72)
    print("  탐색 전체 결과 (부록 표)")
    print("=" * 72)
    print(df.to_string(index=False))

    # 모든 모델이 같은 회귀 가중치를 고르면, 선택 기준이 여전히 예측 성능이
    # 아니라 손실 스케일에 지배될 가능성이 있다.
    chosen = [best_hp[m]["ttf_weight"] for m in best_hp]
    if len(set(chosen)) == 1 and len(chosen) > 1:
        print(f"\n  모든 모델이 동일한 회귀 가중치({chosen[0]})를 선택했습니다.")
        print("  복합 점수가 해당 가중치에 단조적으로 반응하는지 확인하세요.")
        print("  그렇다면 스케일 왜곡이 남아 있다는 신호입니다.")
    else:
        print(f"\n  모델별로 서로 다른 회귀 가중치가 선택되었습니다: {chosen}")
        print("  탐색이 손실 스케일이 아니라 과제 균형을 반영하고 있습니다.")

    # [검토 D-6] 논문에 쓸 파라미터 비율을 여기서 미리 확정해 둔다.
    rep = parameter_report()
    if "_reduction_vs_proposed_pct" in rep:
        print("\n  파라미터 수 (논문 표 2에 열로 추가할 값)")
        for k, v in rep.items():
            if not k.startswith("_"):
                print(f"    {display(k):24s} {v:>10,}")
        print(f"    단순 결합이 적게 쓰는 비율 : "
              f"{rep['_reduction_vs_proposed_pct']:.2f}%  <- 논문 표기용")
        print(f"    제안 모델이 더 쓰는 비율   : "
              f"{rep['_increase_vs_ablation_pct']:.2f}%  (기존 원고의 15.5%)")

    print(f"\n  저장: {csv_path}")
    print(f"  저장: {BEST_HP_PATH}")
    return best_hp


def load_best_hp():
    if not os.path.exists(BEST_HP_PATH):
        print(f"  {BEST_HP_PATH}가 없어 기본값을 사용합니다")
        return {n: {"lr": 1e-3, "ttf_weight": 1.0} for n in MODEL_REGISTRY}
    with open(BEST_HP_PATH) as f:
        return json.load(f)["best_per_model"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.TUNE_EPOCHS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run_tuning(epochs=args.epochs, quick=args.quick, verbose=args.verbose)
