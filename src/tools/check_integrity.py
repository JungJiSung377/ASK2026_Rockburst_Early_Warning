"""산출물 무결성 점검.

검토 리포트 F절의 1~3번(캐시 무결성 · 생성기 리셋 · 세그먼트 실경과시간)을
한 번에 확인한다. 각각 몇 초면 끝나고, "지금 가진 숫자를 그대로 써도 되는가"를
결정한다.

    python tools/check_integrity.py
"""

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import config

OK, WARN, BAD = "  [정상]", "  [주의]", "  [문제]"


def _hdr(t):
    print("\n" + "=" * 72)
    print(f"  {t}")
    print("=" * 72)


def check_prediction_cache():
    """[C-1] 예측 캐시가 모든 모델 x 모든 시드를 담고 있는가."""
    _hdr("1. 예측 캐시 무결성 (검토 C-1)")
    problems = 0
    for label, path in [
        ("심층", os.path.join(config.ARTIFACT_DIR, "prediction_cache.pkl")),
        ("고전", os.path.join(config.ARTIFACT_DIR, "classical_prediction_cache.pkl")),
    ]:
        if not os.path.exists(path):
            print(f"{WARN} {label} 캐시 없음: {path}")
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        preds = d.get("predictions", {})
        print(f"\n  {label} 캐시: {os.path.basename(path)}")
        for m, sp in preds.items():
            got = sorted(sp.keys())
            missing = [s for s in config.EVAL_SEEDS if s not in got]
            if missing:
                print(f"{BAD} {m:26s} 시드 {len(got)}/{len(config.EVAL_SEEDS)}개 "
                      f"— 누락 {missing}")
                problems += 1
            else:
                print(f"{OK} {m:26s} 시드 {len(got)}개")
        # 캐시가 어떤 조건으로 만들어졌는지
        if "ae_key" in d:
            print(f"       ae_key={d['ae_key']}, "
                  f"label_threshold={d.get('label_threshold')}")
    if problems:
        print(f"\n{BAD} 캐시가 불완전합니다. 그림 3·4, 진단, 동일 FAR 표가")
        print("       일부 시드로만 만들어졌을 수 있습니다.")
        print("       해결: python step3_train_eval.py --no-resume 로 전체 재실행")
    return problems == 0


def check_loader_reset():
    """[A-8] 배치 순서 생성기 리셋이 실제로 동작하는가."""
    _hdr("2. 배치 순서 생성기 리셋 (검토 A-8)")
    print(f"  config.FIX_LOADER_SEED_PER_MODEL = "
          f"{config.FIX_LOADER_SEED_PER_MODEL}")
    if not config.FIX_LOADER_SEED_PER_MODEL:
        print(f"{BAD} 꺼져 있습니다. 같은 시드의 모델들이 서로 다른 배치 순서를")
        print("       보게 되어 대응 t-검정의 전제가 깨집니다.")
        return False
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        print(f"{WARN} torch 없음 ({e}). 실동작 확인을 건너뜁니다.")
        return True

    ds = TensorDataset(torch.arange(20).float().unsqueeze(1))
    gen = torch.Generator()
    gen.manual_seed(1234)
    dl = DataLoader(ds, batch_size=4, shuffle=True, generator=gen, num_workers=0)

    first = [b[0].flatten().tolist() for b in dl]
    second = [b[0].flatten().tolist() for b in dl]           # 리셋 없음
    gen.manual_seed(1234)
    third = [b[0].flatten().tolist() for b in dl]            # 리셋 후

    if first == second:
        print(f"{WARN} 이 torch 버전에서는 생성기 상태가 전진하지 않는 것으로")
        print("       보입니다. 그래도 리셋을 유지하는 편이 안전합니다.")
        return True
    if first == third:
        print(f"{OK} 리셋 없이는 순서가 달라지고, manual_seed 로 되돌리면 동일해집니다.")
        print("       -> step3 / step2_tune 의 리셋이 의도대로 동작합니다.")
        return True
    print(f"{BAD} 리셋 후에도 순서가 복원되지 않습니다. 확인이 필요합니다.")
    return False


def check_segment_timing():
    """[D-2] 세그먼트당 실제 경과시간이 37.5 ms 인가."""
    _hdr("3. 세그먼트 실경과시간 (검토 D-2 / 심사 지적)")
    if not os.path.exists(config.H5_PATH):
        print(f"{WARN} 데이터셋 없음: {config.H5_PATH}")
        return True
    import h5py
    with h5py.File(config.H5_PATH, "r") as f:
        n = int(f.attrs["n_segments"])
        nominal = float(f.attrs.get("segment_seconds_nominal",
                                    config.SEGMENT_SIZE / config.SAMPLING_RATE))
        measured = float(f.attrs.get("segment_seconds_measured", float("nan")))
        n_cyc = int(f.attrs.get("n_complete_cycles", 0))
        if measured != measured:      # 구버전 데이터셋이면 직접 계산
            ttf = f["y_ttf"][:n, 0].astype(float)
            d = ttf[:-1] - ttf[1:]
            d = d[(d > 0) & (d < config.CYCLE_JUMP_THRESHOLD)]
            measured = float(np.median(d)) if len(d) else float("nan")

    print(f"  공칭   : {nominal*1000:.3f} ms  (150,000 / 4 MHz)")
    if measured != measured:
        print(f"{WARN} 측정 불가")
        return True
    ratio = measured / nominal
    print(f"  실측   : {measured*1000:.3f} ms  (TTF 감소량 중앙값)")
    print(f"  비율   : {ratio:.4f}")
    if abs(ratio - 1.0) <= 0.02:
        print(f"{OK} 공칭값과 일치합니다. 논문의 37.5 ms / "
              f"{config.TREND_WINDOW_SEC:.2f}초 표기를 그대로 쓰면 됩니다.")
        return True
    print(f"{BAD} 2% 이상 다릅니다. 다음 표기를 모두 정정해야 합니다:")
    print(f"       세그먼트 길이  : {measured*1000:.2f} ms")
    print(f"       추세 창        : {config.TREND_WINDOW * measured:.3f} 초")
    if n_cyc:
        print(f"       사이클 평균    : {n * measured / n_cyc:.2f} 초")
    return False


def check_paper_numbers():
    """논문에 그대로 쓸 파생 수치를 한 번에 뽑는다."""
    _hdr("4. 논문 표기용 수치")
    # 주입 대역
    top_c, top_e = config.noise_band_bounds_hz()
    print(f"  주파수 해상도 : {config.FREQ_RESOLUTION_HZ:.2f} Hz")
    print(f"  주입 대역     : {config.NOISE_BAND_BINS}개 bin "
          f"(중심 최대 {top_c/1000:.2f} kHz, 상한 경계 {top_e/1000:.2f} kHz)")
    print(f"                  -> 논문 표기: '{config.NOISE_BAND_LABEL}' "
          f"({config.NOISE_BAND_BINS}개 bin)")
    print(f"       [심사 지적 05] 8개 bin(0..7)의 상한은 10.0 kHz 입니다.")
    print(f"       10.7 kHz 는 8 x df = bin 8 의 중심으로, 9개 bin 이어야 나옵니다.")

    # 파라미터 수
    try:
        from models import parameter_report
        from models_meta import display as _d
        rep = parameter_report()
        print("\n  파라미터 수")
        for k, v in rep.items():
            if not k.startswith("_"):
                print(f"    {_d(k):24s} {v:>10,}")
        if "_reduction_vs_proposed_pct" in rep:
            print(f"    단순 결합이 적게 쓰는 비율 : "
                  f"{rep['_reduction_vs_proposed_pct']:.2f}%  <- 논문에 쓸 값")
            print(f"    제안 모델이 더 쓰는 비율   : "
                  f"{rep['_increase_vs_ablation_pct']:.2f}%  "
                  f"(기존 원고의 15.5%, 분모가 반대)")
    except Exception as e:
        print(f"\n{WARN} 파라미터 계산 불가 ({e})")

    # 지연시간
    p = os.path.join(config.RESULT_DIR, "xai_latency.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        e2e = d.get("latency_end_to_end", {})
        if e2e:
            print("\n  지연시간 (논문 3.3절)")
            for k, lab in [("stft_mean_ms", "STFT"),
                           ("state_feature_mean_ms", "상태 벡터"),
                           ("transfer_mean_ms", "텐서 전송"),
                           ("inference_mean_ms", "모델 추론")]:
                if k in e2e:
                    print(f"    {lab:12s} {e2e[k]:.3f} ms")
            print(f"    {'합계':12s} {e2e.get('end_to_end_mean_ms', 0):.3f} ms")
            print(f"    실시간 배수  {e2e.get('realtime_factor', 0):.2f}배")
            print(f"    측정 환경    {e2e.get('device', '?')}")
            print("    -> 네 항목의 합이 본문 숫자와 맞는지 확인하세요.")
    return True


def check_seed_results():
    """seed_level_results.csv 의 행 수가 기대와 맞는가."""
    _hdr("5. 시드별 결과 파일")
    p = os.path.join(config.RESULT_DIR, "seed_level_results.csv")
    if not os.path.exists(p):
        print(f"{WARN} 없음: {p}")
        return True
    import pandas as pd
    df = pd.read_csv(p)
    from models_meta import DEEP_MODEL_NAMES, display
    ok = True
    for m in DEEP_MODEL_NAMES:
        sub = df[df.model == m]
        if len(sub) != len(config.EVAL_SEEDS):
            print(f"{BAD} {display(m):24s} {len(sub)}행 "
                  f"(기대 {len(config.EVAL_SEEDS)}행)")
            ok = False
        else:
            print(f"{OK} {display(m):24s} {len(sub)}행")
    if "far_drift_vs_selection" in df.columns:
        print("\n  운영점 이동 (선택 기준 대비):")
        for m, v in df.groupby("model")["far_drift_vs_selection"].mean().items():
            print(f"    {display(m):24s} {v:+.2f}%p")
    return ok


def main():
    print("\n" + "#" * 72)
    print("#  CTPF 산출물 무결성 점검")
    print("#" * 72)
    results = [
        ("예측 캐시", check_prediction_cache()),
        ("생성기 리셋", check_loader_reset()),
        ("세그먼트 경과시간", check_segment_timing()),
        ("시드별 결과", check_seed_results()),
    ]
    check_paper_numbers()

    _hdr("종합")
    bad = [n for n, ok in results if not ok]
    if bad:
        print(f"  확인이 필요한 항목: {', '.join(bad)}")
        return 1
    print("  모든 점검을 통과했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
