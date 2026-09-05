"""파이프라인 실행기.

사용법
    python run_all.py --smoke      축소 데이터로 전체 단계를 빠르게 점검
    python run_all.py              전체 실행 (평가 시드 15개, ablation 시드 8개)
    python run_all.py --fast       시드 수를 줄여 실행
    python run_all.py --from 3     지정한 단계부터 재개
    python run_all.py --cv         rolling-origin 교차검증으로 3단계 실행
    python run_all.py --ae-key X_ae_clean   무잡음 조건 대조 실행
    python run_all.py --force-rebuild

[검토 반영 v2]
  A-9  기존 run_all 은 존재하지 않는 모듈명(step10_operating_analysis,
       step9_diagnostics)을 import 해서 6·9단계에서 죽었다. 실제 파일명으로
       고치고, 파일명 / 내부 출력 / 실행기 단계 번호가 서로 달랐던 문제도
       하나로 통일했다.
"""

import argparse
import os
import sys
import time

import config

# 단계 번호 -> (모듈, 설명). 파일명과 단계 번호를 한곳에서 관리한다.
STAGES = [
    (1, "step1_build_dataset", "데이터셋 구축"),
    (2, "step2_tune", "하이퍼파라미터 탐색 + 고전 베이스라인"),
    (3, "step3_train_eval", "다중 시드 학습 및 평가"),
    (4, "step4_stats", "유의성 검정"),
    (5, "step5_ablation", "Ablation 실험"),
    (6, "step9_operating_analysis", "동일 오경보율 운영점 분석"),
    (7, "step6_figures", "그림 생성"),
    (8, "step7_xai_latency", "해석 및 지연시간"),
    (9, "step8_diagnostics", "진단"),
]


def _stage(num, msg):
    print("\n\n" + "#" * 76)
    print(f"#  {num}단계  {msg}")
    print("#" * 76)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=9)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ae-key", default="X_ae", choices=["X_ae", "X_ae_clean"],
                    help="X_ae_clean 이면 간섭 없는 조건 대조 실행 (검토 A-1)")
    ap.add_argument("--label-threshold", type=float, default=None,
                    help="라벨 임계값 민감도 분석 (검토 B-10)")
    ap.add_argument("--cmap-gallery", action="store_true",
                    help="그림 1을 여러 컬러맵으로 렌더링해 비교")
    args = ap.parse_args()

    if args.smoke:
        # [수정] 400개는 LANL 기준 사이클 2개뿐이라 학습/검증/평가 3분할이
        #        불가능했다(사이클 1개 약 250 세그먼트). 최소 5개 사이클을
        #        확보하도록 1,800개로 올린다. 점검 실행 약 20분.
        seeds, abl_seeds = [11, 22], [11]
        epochs, tune_epochs, max_segments, quick = 3, 2, config.SMOKE_SEGMENTS, True
    elif args.fast:
        seeds, abl_seeds = config.EVAL_SEEDS[:8], config.ABLATION_SEEDS[:5]
        epochs, tune_epochs = config.EPOCHS, config.TUNE_EPOCHS
        max_segments, quick = config.MAX_SEGMENTS, False
    else:
        seeds, abl_seeds = config.EVAL_SEEDS, config.ABLATION_SEEDS
        epochs, tune_epochs = config.EPOCHS, config.TUNE_EPOCHS
        max_segments, quick = config.MAX_SEGMENTS, False

    h5_path = config.h5_path_for(max_segments)
    config.H5_PATH = h5_path

    from models_meta import DEEP_MODEL_NAMES, display
    n_models = len(DEEP_MODEL_NAMES)
    n_combo = 1
    for v in config.HP_GRID.values():
        n_combo *= len(v)
    n_feat = 1 + config.STATE_DIM + 3
    n_loss = 5
    est = (n_combo * n_models) + (n_models * len(seeds)) \
        + (n_feat * len(abl_seeds)) + (n_loss * len(abl_seeds))

    print("=" * 76)
    print("  교차 시간 스케일 전조 상태 융합: 실험 파이프라인 [검토 반영 v2]")
    print("=" * 76)
    print(f"  모드              {'점검' if args.smoke else '단축' if args.fast else '전체'}")
    print(f"  모델              {[display(m) for m in DEEP_MODEL_NAMES]}")
    print(f"  평가 시드         {len(seeds)}개")
    print(f"  ablation 시드     {len(abl_seeds)}개")
    print(f"  에폭              {epochs} (탐색 {tune_epochs})")
    print(f"  세그먼트          {max_segments}개")
    print(f"  데이터셋          {h5_path}")
    print(f"  스펙트로그램      {args.ae_key}")
    print(f"  타깃 정규화       {config.NORMALIZE_TTF}, "
          f"선택 기준 {config.HP_SELECTION_METRIC}")
    print(f"  조기 종료 기준    {config.EARLY_STOP_METRIC}")
    print(f"  focal alpha       {config.FOCAL_ALPHA_MODE}")
    print(f"  어텐션 융합       {config.ATTENTION_FUSION}")
    print(f"  배치 순서 리셋    {config.FIX_LOADER_SEED_PER_MODEL}")
    print(f"  목표 FAR          {config.TARGET_FAR_PCT:.0f}% "
          f"(안전계수 {config.FAR_SAFETY_FACTOR})")
    print(f"  동일 FAR 격자     {config.MATCHED_FAR_GRID}")
    print(f"  예상 학습 횟수    {est}회 "
          f"(1회 100초 가정 시 약 {est * 100 / 3600:.1f}시간)")
    print(f"  실행 단계         {args.start} ~ {args.end}")
    print("=" * 76)

    t0 = time.perf_counter()

    def _run(num):
        return args.start <= num <= args.end

    if _run(1):
        _stage(1, "데이터셋 구축")
        from step1_build_dataset import (build_dataset, check_h5_validity,
                                          report_class_balance, report_segment_timing)
        ok, reason = check_h5_validity(h5_path, max_segments)
        if ok and not args.force_rebuild:
            print(f"  기존 데이터셋 재사용 ({reason})")
            report_segment_timing(h5_path)
            report_class_balance(h5_path)
        else:
            if os.path.exists(h5_path) and not args.force_rebuild:
                print(f"  기존 데이터셋 부적합 ({reason}), 재생성합니다")
            build_dataset(output_h5=h5_path, max_segments=max_segments)

    if _run(2):
        _stage(2, "하이퍼파라미터 탐색")
        from step2_tune import run_tuning
        run_tuning(epochs=tune_epochs, quick=quick, verbose=args.verbose)

        _stage(2, "고전 베이스라인 (2b)")
        import step2b_classical_baseline
        step2b_classical_baseline.run(h5_path=h5_path, seeds=seeds)

    if _run(3):
        _stage(3, f"다중 시드 학습 및 평가 (시드 {len(seeds)}개)")
        import step3_train_eval
        step3_train_eval.run(seeds=seeds, epochs=epochs, verbose=args.verbose,
                             use_cv=args.cv or None, resume=not args.no_resume,
                             ae_key=args.ae_key,
                             label_threshold=args.label_threshold)

    if _run(4):
        _stage(4, "유의성 검정")
        import step4_stats
        step4_stats.run()

    if _run(5):
        _stage(5, f"Ablation 실험 (시드 {len(abl_seeds)}개)")
        import step5_ablation
        step5_ablation.run_feature_ablation(abl_seeds, epochs, verbose=args.verbose,
                                            resume=not args.no_resume)
        step5_ablation.run_loss_ablation(abl_seeds, epochs, verbose=args.verbose,
                                         resume=not args.no_resume)

    if _run(6):
        _stage(6, "동일 오경보율 운영점 분석")
        import step9_operating_analysis
        step9_operating_analysis.run()

    if _run(7):
        _stage(7, "그림 생성")
        import step6_figures
        step6_figures.run(cmap_gallery=args.cmap_gallery)

    if _run(8):
        _stage(8, "해석 및 지연시간")
        import step7_xai_latency
        step7_xai_latency.run(seed=seeds[-1])

    if _run(9):
        _stage(9, "진단")
        import step8_diagnostics
        step8_diagnostics.run()

    el = time.perf_counter() - t0
    print("\n\n" + "#" * 76)
    print(f"#  파이프라인 완료, 총 {el/60:.1f}분 소요")
    print("#" * 76)
    print("  표 및 통계")
    print("    results/class_balance.json                클래스 균형 + 세그먼트 경과시간")
    print("    results/split_summary.json                데이터 분할")
    print("    results/hyperparameter_search.csv         탐색 기록 (부록)")
    print("    results/classical_baseline_results.csv    고전 베이스라인")
    print("    results/summary_mean_std.csv              표 2")
    print("    results/significance_main.csv             표 3")
    print("    results/significance_fusion_ablation.csv  표 4")
    print("    results/ablation_summary.csv              표 5")
    print("    results/ablation_feature_significance.csv 표 5b")
    print("    results/ablation_loss.csv                 표 6")
    print("    results/iso_far_recall_table.csv          표 7")
    print("    results/iso_far_significance.csv          표 7 유의성")
    print("    results/diagnostics_per_cycle.csv         사이클별 오차")
    print("    results/diagnostics_ttf_range.csv         외삽 확인")
    print("    results/diagnostics_threshold.json        임계값 전이")
    print("    results/xai_latency.json                  어텐션·삭제실험·지연시간")
    print("    results/environment.json                  재현성 기록")
    print("  그림")
    print("    figures/fig1_preprocessing.pdf            그림 1 (백색 배경 3패널)")
    print("    figures/fig2_architecture.pdf             그림 2")
    print("    figures/fig3_classification.pdf           그림 3")
    print("    figures/fig4_ttf_regression.pdf           그림 4")
    print("    figures/fig5_feature_ablation.pdf         그림 5")
    print("    figures/fig6_loss_ablation.pdf            그림 6")
    print("    figures/fig7_attention.pdf                그림 7")
    print("    figures/fig8_regression_diagnostics.pdf   그림 8")
    print("    figures/fig9_iso_far.pdf                  그림 9")
    print("    figures/fig10_attention_deletion.pdf      그림 10 (삭제 실험)")
    print("\n  다음 확인을 권합니다:")
    print("    python tools/check_integrity.py")


if __name__ == "__main__":
    sys.exit(main())
