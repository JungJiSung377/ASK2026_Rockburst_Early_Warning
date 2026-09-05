"""원시 음향방출 기록으로부터 데이터셋을 구축한다.

각 37.5 ms 세그먼트를 크기 스펙트로그램으로 변환하고, 합성 저주파 간섭을
더한 뒤, 전조 상태 벡터를 계산한다. 추세 성분은 이전 세그먼트만 사용하므로
미래 정보가 특징에 들어가지 않는다.

파괴까지 남은 시간의 불연속으로 스틱슬립 사이클 경계를 감지하여 저장하고,
파괴에 도달하지 못한 사이클에는 별도 플래그를 남긴다. 간섭을 더한 것과
더하지 않은 스펙트로그램을 모두 저장하여, 재처리 없이 간섭에 대한 강건성을
평가할 수 있도록 한다.

[검토 반영 v2]
  C-2  파괴가 세그먼트 내부에서 일어나면 그 세그먼트의 마지막 샘플 TTF 는
       이미 다음 사이클의 큰 값이다. 결과적으로 가장 에너지가 큰 세그먼트가
       "정상 / 큰 TTF" 로 잘못 라벨된다. cycle_boundary 플래그로 표시.
  C-3  사이클 초반에는 이력이 TREND_WINDOW 보다 짧아 cumulative 가
       기계적으로 작아지고, 사이클 초반 = TTF 가 큰 구간이므로 라벨과
       정렬된 인공 상관이 생긴다. 부분합을 전체 창으로 환산(scaled)하고,
       워밍업 구간을 inter_warmup 플래그로 표시.
  B-6  사이클 시작부터의 진짜 누적합을 7번째 특징으로 추가하는 옵션.
  09   RMS 는 진폭 차원이므로 "에너지"가 아니다. 실제 에너지 비례량
       (mean(x^2))을 쓰는 옵션을 둔다.
  05   주입 대역 표기를 bin 인덱스 기준으로 정정 (8개 bin -> DC~10.0 kHz).
  D-2  세그먼트당 실제 경과시간을 TTF 컬럼에서 직접 측정해 보고한다.
       37.5 ms 가정이 맞는지 데이터로 확인할 수 있다.
"""

import argparse
import json
import os
from collections import deque

import h5py
import numpy as np
import pandas as pd
from scipy.signal import stft

import config


def compute_state_features(ae_f64, rms_history, cycle_running_sum=0.0):
    """전조 상태 벡터.

    인덱스 0-3은 현재 세그먼트를 기술하고, 인덱스 4 이상은 이전 세그먼트들을
    요약한다. 미래 정보가 사용되지 않도록, 현재 값을 이력에 추가하기 전에
    계산해야 한다.

    반환: (특징 벡터, 갱신된 사이클 누적합)
    """
    rms = float(np.sqrt(np.mean(ae_f64 ** 2)))
    # [심사 지적 09] level 이 특징 0번에 들어간다. USE_TRUE_ENERGY_FEATURE 를
    # 켜면 진폭(RMS)이 아니라 에너지 비례량(mean(x^2))을 쓴다.
    level = float(np.mean(ae_f64 ** 2)) if config.USE_TRUE_ENERGY_FEATURE else rms

    peak_amp = float(np.max(np.abs(ae_f64)))
    kurtosis = float(pd.Series(ae_f64).kurt())
    crest_factor = float(peak_amp / (rms + 1e-8))

    hist = np.asarray(rms_history, dtype=np.float64)
    n_hist = len(hist)

    if n_hist >= 2:
        slope = float(np.polyfit(np.arange(n_hist), hist, 1)[0])
    else:
        slope = 0.0

    if n_hist == 0:
        cumulative = 0.0
    elif config.INTER_WARMUP_POLICY == "scaled":
        # [C-3] 부분합을 전체 창 길이로 환산해 항수 의존성을 제거한다.
        # 창이 가득 찬 뒤에는 단순 합과 동일하다.
        cumulative = float(hist.sum() * config.TREND_WINDOW / n_hist)
    else:
        cumulative = float(hist.sum())

    feats = [level, kurtosis, peak_amp, crest_factor, slope, cumulative]
    if config.ADD_CYCLE_CUMULATIVE_FEATURE:
        # [B-6] 사이클 시작부터 직전 세그먼트까지의 진짜 누적합.
        feats.append(float(cycle_running_sum))

    return np.array(feats, dtype=np.float32), cycle_running_sum + level


def build_dataset(csv_path=None, output_h5=None, max_segments=None,
                  store_clean=True, warning_threshold=None):
    csv_path = csv_path or config.RAW_CSV
    max_segments = max_segments or config.MAX_SEGMENTS
    output_h5 = output_h5 or config.h5_path_for(max_segments)
    warning_threshold = warning_threshold or config.WARNING_TTF_THRESHOLD

    print("1단계  데이터셋 구축")
    print(f"  출력: {output_h5} (최대 {max_segments}개 세그먼트)")
    print(f"  상태 벡터: {config.STATE_FEATURE_NAMES}")
    top_c, top_e = config.noise_band_bounds_hz()
    print(f"  간섭 주입: 최저 {config.NOISE_BAND_BINS}개 bin "
          f"(중심 최대 {top_c/1000:.2f} kHz, 상한 경계 {top_e/1000:.2f} kHz), "
          f"N({config.NOISE_MEAN}, {config.NOISE_STD})")
    print(f"  워밍업 정책: {config.INTER_WARMUP_POLICY}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n원천 파일 '{csv_path}'을 찾을 수 없습니다.\n"
            "step0_download.py를 먼저 실행하세요."
        )

    _, _, dummy = stft(np.zeros(config.SEGMENT_SIZE), fs=config.SAMPLING_RATE,
                       nperseg=config.NPERSEG, noverlap=config.NOVERLAP)
    freq_bins, time_steps = np.abs(dummy).shape
    print(f"  STFT 격자: 주파수 bin {freq_bins}개 x 시간 프레임 {time_steps}개")
    assert (freq_bins, time_steps) == (config.FREQ_BINS, config.TIME_STEPS), (
        f"설정된 격자 ({config.FREQ_BINS}, {config.TIME_STEPS})가 실제 계산된 "
        f"격자 ({freq_bins}, {time_steps})와 다릅니다")

    with h5py.File(output_h5, "w") as h5:
        h5.create_dataset("X_ae", shape=(max_segments, freq_bins, time_steps),
                          dtype=np.float32, chunks=(1, freq_bins, time_steps),
                          compression="lzf")
        if store_clean:
            h5.create_dataset("X_ae_clean", shape=(max_segments, freq_bins, time_steps),
                              dtype=np.float32, chunks=(1, freq_bins, time_steps),
                              compression="lzf")
        h5.create_dataset("X_state", shape=(max_segments, config.STATE_DIM),
                          dtype=np.float32)
        h5.create_dataset("y_cls", shape=(max_segments, 1), dtype=np.float32)
        h5.create_dataset("y_ttf", shape=(max_segments, 1), dtype=np.float32)
        h5.create_dataset("cycle_id", shape=(max_segments,), dtype=np.int32)
        # [C-2] 파괴 순간을 담은 세그먼트
        h5.create_dataset("cycle_boundary", shape=(max_segments,), dtype=np.int8)
        # [C-3] inter 특징 이력이 아직 다 차지 않은 구간
        h5.create_dataset("inter_warmup", shape=(max_segments,), dtype=np.int8)

    reader = pd.read_csv(csv_path, chunksize=config.SEGMENT_SIZE, header=0,
                         names=["acoustic_data", "time_to_failure"],
                         dtype={"acoustic_data": np.int16,
                                "time_to_failure": np.float32})

    noise_rng = np.random.default_rng(config.NOISE_SEED)
    rms_history = deque(maxlen=config.TREND_WINDOW)
    prev_ttf = None
    cycle_counter = 0
    cycle_running_sum = 0.0
    processed = 0
    n_boundary = 0
    ttf_first = []      # [D-2] 세그먼트 시작 TTF (경과시간 측정용)

    with h5py.File(output_h5, "r+") as h5:
        for idx in range(max_segments):
            try:
                chunk = next(reader)
            except StopIteration:
                print("  원천 파일 읽기 완료")
                break

            ae_raw = chunk["acoustic_data"].values
            if len(ae_raw) < config.SEGMENT_SIZE:
                print("  마지막 불완전 청크 폐기")
                break

            ttf_vals = chunk["time_to_failure"].values
            ttf_val = float(ttf_vals[-1])
            ttf_first.append(float(ttf_vals[0]))

            # 스틱슬립 사이클 경계 감지
            is_boundary = 0
            if prev_ttf is not None and ttf_val > prev_ttf + config.CYCLE_JUMP_THRESHOLD:
                cycle_counter += 1
                rms_history.clear()          # 사이클 간 이력 이월 방지
                cycle_running_sum = 0.0
                # [C-2] 이 세그먼트 안에서 파괴가 일어났다. 마지막 샘플의 TTF는
                # 이미 다음 사이클 값이므로 라벨이 실제와 어긋난다.
                is_boundary = 1
                n_boundary += 1
            prev_ttf = ttf_val

            # STFT
            _, _, Zxx = stft(ae_raw, fs=config.SAMPLING_RATE,
                             nperseg=config.NPERSEG, noverlap=config.NOVERLAP)
            clean_spec = np.abs(Zxx)[:, :time_steps].astype(np.float32)

            # 합성 저주파 간섭 주입
            noisy_spec = clean_spec.copy()
            nb = config.NOISE_BAND_BINS
            noise = noise_rng.normal(config.NOISE_MEAN, config.NOISE_STD,
                                     size=(nb, time_steps))
            noisy_spec[:nb, :] = np.clip(clean_spec[:nb, :] + noise,
                                         a_min=0.0, a_max=None)

            # 과거 데이터만 사용하도록 이력 갱신 전에 계산한다
            warmup = 1 if len(rms_history) < config.TREND_WINDOW else 0
            ae_f64 = ae_raw.astype(np.float64)
            state_vec, cycle_running_sum = compute_state_features(
                ae_f64, rms_history, cycle_running_sum)
            rms_history.append(float(np.sqrt(np.mean(ae_f64 ** 2))))

            h5["X_ae"][idx] = noisy_spec
            if store_clean:
                h5["X_ae_clean"][idx] = clean_spec
            h5["X_state"][idx] = state_vec
            h5["y_cls"][idx, 0] = 1.0 if ttf_val <= warning_threshold else 0.0
            h5["y_ttf"][idx, 0] = ttf_val
            h5["cycle_id"][idx] = cycle_counter
            h5["cycle_boundary"][idx] = is_boundary
            h5["inter_warmup"][idx] = warmup

            processed += 1
            if processed % 250 == 0:
                print(f"   - {processed}/{max_segments} (cycle {cycle_counter})")

        # 파괴에 도달한 사이클 표시
        cyc = h5["cycle_id"][:processed]
        ttf_all = h5["y_ttf"][:processed, 0]
        is_complete = np.zeros(processed, dtype=np.int8)
        complete_cycles, incomplete_cycles = [], []
        for c in np.unique(cyc):
            mask = cyc == c
            if ttf_all[mask].min() <= warning_threshold:
                is_complete[mask] = 1
                complete_cycles.append(int(c))
            else:
                incomplete_cycles.append(int(c))

        if "cycle_complete" in h5:
            del h5["cycle_complete"]
        h5.create_dataset("cycle_complete", data=is_complete, dtype=np.int8)

        # [D-2] 세그먼트당 실제 경과시간. LANL 원자료의 time_to_failure 는
        # 블록 단위로 기록되어 연속이 아닐 수 있다. 37.5 ms 가정이 맞는지
        # 데이터에서 직접 확인한다.
        tf = np.asarray(ttf_first[:processed], dtype=np.float64)
        elapsed = tf[:-1] - tf[1:]                       # 다음 세그먼트까지의 감소량
        elapsed = elapsed[(elapsed > 0) & (elapsed < config.CYCLE_JUMP_THRESHOLD)]
        seg_sec_measured = float(np.median(elapsed)) if len(elapsed) else float("nan")

        # 데이터셋 재사용 검증용 메타데이터
        h5.attrs["n_segments"] = processed
        h5.attrs["requested_max_segments"] = max_segments
        h5.attrs["n_cycles"] = int(cyc.max()) + 1
        h5.attrs["n_complete_cycles"] = len(complete_cycles)
        h5.attrs["incomplete_cycles"] = json.dumps(incomplete_cycles)
        h5.attrs["has_clean"] = bool(store_clean)
        h5.attrs["state_feature_names"] = json.dumps(config.STATE_FEATURE_NAMES)
        h5.attrs["warning_ttf_threshold"] = warning_threshold
        h5.attrs["noise_seed"] = config.NOISE_SEED
        h5.attrs["noise_band_bins"] = config.NOISE_BAND_BINS
        h5.attrs["noise_band_top_edge_hz"] = config.noise_band_bounds_hz()[1]
        h5.attrs["trend_window"] = config.TREND_WINDOW
        h5.attrs["inter_warmup_policy"] = config.INTER_WARMUP_POLICY
        h5.attrs["use_true_energy_feature"] = bool(config.USE_TRUE_ENERGY_FEATURE)
        h5.attrs["segment_seconds_nominal"] = config.SEGMENT_SIZE / config.SAMPLING_RATE
        h5.attrs["segment_seconds_measured"] = seg_sec_measured
        h5.attrs["n_cycle_boundary_segments"] = int(n_boundary)

    print(f"\n  데이터셋 저장: {output_h5}")
    print(f"  세그먼트 {processed}개, 사이클 {int(cyc.max())+1}개 "
          f"(파괴 도달 {len(complete_cycles)}개, 미도달 {len(incomplete_cycles)}개)")
    if incomplete_cycles:
        print(f"  미도달 사이클 {incomplete_cycles}은 EXCLUDE_INCOMPLETE_CYCLES가 "
              f"활성화된 경우 제외됩니다.")
    print(f"  사이클 경계 세그먼트 {n_boundary}개 "
          f"(DROP_CYCLE_BOUNDARY_SEGMENT={config.DROP_CYCLE_BOUNDARY_SEGMENT})")

    report_segment_timing(output_h5)
    report_class_balance(output_h5)
    return output_h5


def report_segment_timing(h5_path=None):
    """[D-2] 세그먼트당 실제 경과시간을 확인한다.

    논문의 37.5 ms / 0.75초 / 사이클 평균 길이가 모두 이 값에 의존한다.
    """
    h5_path = h5_path or config.H5_PATH
    with h5py.File(h5_path, "r") as f:
        nominal = float(f.attrs.get("segment_seconds_nominal",
                                    config.SEGMENT_SIZE / config.SAMPLING_RATE))
        measured = float(f.attrs.get("segment_seconds_measured", float("nan")))
        n = int(f.attrs["n_segments"])
        n_cyc = int(f.attrs.get("n_complete_cycles", 0))

    print("\n" + "=" * 64)
    print("  세그먼트 경과시간 검증")
    print("=" * 64)
    print(f"  공칭 (150,000 / 4 MHz)      : {nominal*1000:.3f} ms")
    if measured == measured:
        print(f"  TTF 컬럼에서 측정한 중앙값  : {measured*1000:.3f} ms")
        ratio = measured / nominal if nominal else float("nan")
        print(f"  비율                        : {ratio:.4f}")
        if abs(ratio - 1.0) > 0.02:
            print("\n  경고: 실제 경과시간이 공칭값과 2% 이상 다릅니다.")
            print("  LANL 원자료의 time_to_failure 는 블록 단위로 기록되어")
            print("  연속이 아닐 수 있습니다. 이 경우 논문의 '37.5 ms',")
            print(f"  '{config.TREND_WINDOW_SEC:.2f}초 추세 창', 사이클 평균 길이를")
            print(f"  모두 측정값 기준으로 정정해야 합니다.")
            print(f"    추세 창 실측 = {config.TREND_WINDOW * measured:.3f} 초")
            # [수정 v2.1] 여기서 찍던 `n * measured / n_cyc` 는 분자에 전체
            #   세그먼트(미완결 사이클 포함), 분모에 완결 사이클 수를 써서
            #   기준이 어긋나 있었다. 같은 실행의 클래스 균형 블록이
            #   9.162 초를 낼 때 이 줄은 10.105 초를 냈다. 논문에 쓸 값은
            #   아래 클래스 균형 블록에서 사이클별 TTF 구간으로 계산한다.
            print("    사이클 평균 길이는 아래 '클래스 균형' 블록을 보세요.")
        else:
            print("  -> 공칭값과 일치합니다. 논문 표기를 그대로 쓰면 됩니다.")
    else:
        print("  측정 실패 (유효한 TTF 감소 구간 없음)")
    print("=" * 64)


def check_h5_validity(h5_path, required_segments):
    """기존 데이터셋이 요청한 설정과 일치하는지 확인."""
    if not os.path.exists(h5_path):
        return False, "파일 없음"
    try:
        with h5py.File(h5_path, "r") as f:
            n = int(f.attrs.get("n_segments", 0))
            thr = float(f.attrs.get("warning_ttf_threshold", -1))
            names = f.attrs.get("state_feature_names", "[]")
            policy = f.attrs.get("inter_warmup_policy", "raw")
            has_flags = ("cycle_boundary" in f) and ("inter_warmup" in f)
    except Exception as e:
        return False, f"읽기 실패: {e}"

    if n < required_segments * 0.95:      # 파일이 일찍 끝난 경우를 감안한 여유
        return False, f"세그먼트 부족 ({n} < {required_segments})"
    if abs(thr - config.WARNING_TTF_THRESHOLD) > 1e-6:
        return False, f"라벨 임계값 불일치 (파일 {thr}, 설정 {config.WARNING_TTF_THRESHOLD})"
    try:
        if list(json.loads(names)) != list(config.STATE_FEATURE_NAMES):
            return False, "상태 벡터 구성 불일치"
    except Exception:
        return False, "상태 벡터 메타데이터 손상"
    if str(policy) != config.INTER_WARMUP_POLICY:
        return False, f"워밍업 정책 불일치 (파일 {policy}, 설정 {config.INTER_WARMUP_POLICY})"
    if not has_flags:
        return False, "cycle_boundary / inter_warmup 플래그 없음 (구버전 데이터셋)"
    return True, f"유효 (세그먼트 {n}개)"


def report_class_balance(h5_path=None):
    """클래스 균형을 보고한다. 분류 지표를 해석하려면 반드시 확인해야 한다."""
    h5_path = h5_path or config.H5_PATH
    with h5py.File(h5_path, "r") as h5:
        n = int(h5.attrs["n_segments"])
        n_complete_cycles = int(h5.attrs.get("n_complete_cycles", -1))
        y = h5["y_cls"][:n, 0]
        cyc = h5["cycle_id"][:n]
        ttf = h5["y_ttf"][:n, 0]
        complete = h5["cycle_complete"][:n] if "cycle_complete" in h5 else np.ones(n, np.int8)
        boundary = h5["cycle_boundary"][:n] if "cycle_boundary" in h5 else np.zeros(n, np.int8)
        warmup = h5["inter_warmup"][:n] if "inter_warmup" in h5 else np.zeros(n, np.int8)
        seg_sec = float(h5.attrs.get("segment_seconds_measured",
                                     config.SEGMENT_SIZE / config.SAMPLING_RATE))

    # 실제 학습에 쓰이는 마스크
    keep = np.ones(n, dtype=bool)
    if config.EXCLUDE_INCOMPLETE_CYCLES:
        keep &= (complete == 1)
    if config.DROP_CYCLE_BOUNDARY_SEGMENT:
        keep &= (boundary != 1)
    if config.INTER_WARMUP_POLICY == "exclude":
        keep &= (warmup != 1)

    n_pos = int(y.sum())
    y_used = y[keep]
    n_used = int(keep.sum())
    n_pos_used = int(y_used.sum())

    stats = {
        "n_segments_raw": n,
        "n_segments_used": n_used,
        "n_cycles": int(cyc.max()) + 1,
        "n_complete_cycles": n_complete_cycles,
        "n_positive_raw": n_pos,
        "n_positive_used": n_pos_used,
        "positive_rate_pct_raw": round(100.0 * n_pos / n, 3),
        "positive_rate_pct_used": round(100.0 * n_pos_used / max(n_used, 1), 3),
        "imbalance_ratio_neg_per_pos_used":
            round((n_used - n_pos_used) / max(n_pos_used, 1), 2),
        "n_cycle_boundary": int((boundary == 1).sum()),
        "n_inter_warmup": int((warmup == 1).sum()),
        "ttf_min": float(ttf.min()),
        "ttf_max": float(ttf.max()),
        "segment_seconds_measured": seg_sec,
        "segments_per_cycle_mean": round(float(n / (cyc.max() + 1)), 1),
        "cycle_seconds_mean_complete":
            round(float(n_used * seg_sec / max(n_complete_cycles, 1)), 3),
    }

    # [수정 v2.1] 논문에 쓸 사이클 평균 길이는 세그먼트를 세는 대신
    #   사이클별 TTF 최대값(= 사이클 시작부터 파괴까지)으로 잰다.
    #   세그먼트를 세면 녹음 중간부터 시작한 첫 사이클이 그대로 섞여
    #   평균을 끌어내린다. 아래에서 그런 사이클을 명시적으로 뺀다.
    spans, truncated = [], []
    for c in np.unique(cyc):
        m = cyc == c
        if not bool((complete[m] == 1).all()):
            continue                                   # 파괴 미도달 사이클
        spans.append((int(c), float(ttf[m].max())))
    if spans:
        med = float(np.median([v for _, v in spans]))
        keep = [(c, v) for c, v in spans if v >= 0.5 * med]
        truncated = [c for c, v in spans if v < 0.5 * med]
        stats["cycle_seconds_by_ttf_span_mean"] = round(
            float(np.mean([v for _, v in keep])), 3)
        stats["cycle_seconds_by_ttf_span_std"] = round(
            float(np.std([v for _, v in keep])), 3)
        stats["n_cycles_in_span_mean"] = len(keep)
        stats["truncated_cycles_excluded"] = truncated

    print("\n" + "=" * 64)
    print("  클래스 균형")
    print("=" * 64)
    for k, v in stats.items():
        print(f"  {k:34s}: {v}")
    print("-" * 64)
    if stats["positive_rate_pct_used"] < 10:
        print("  양성 비율이 낮아 ROC-AUC가 낙관적으로 나옵니다. PR-AUC와 고정")
        print("  오경보율에서의 재현율을 주 지표로 사용합니다.")
    print(f"  무작위 수준 PR-AUC = {stats['positive_rate_pct_used']/100:.4f}")
    if "cycle_seconds_by_ttf_span_mean" in stats:
        print(f"  논문에 쓸 사이클 평균 길이 = "
              f"{stats['cycle_seconds_by_ttf_span_mean']:.2f} +/- "
              f"{stats['cycle_seconds_by_ttf_span_std']:.2f}초 "
              f"(사이클별 TTF 구간, {stats['n_cycles_in_span_mean']}개 기준)")
        if stats["truncated_cycles_excluded"]:
            print(f"    앞이 잘린 사이클 제외: {stats['truncated_cycles_excluded']} "
                  f"(녹음 중간부터 시작해 전체 길이를 알 수 없음)")
        print(f"    참고 - 세그먼트 수로 나눈 값 "
              f"{stats['cycle_seconds_mean_complete']:.2f}초 "
              f"(잘린 사이클이 섞여 과소평가됩니다)")
    print("=" * 64)

    out = os.path.join(config.RESULT_DIR, "class_balance.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  저장: {out}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=config.RAW_CSV)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-segments", type=int, default=config.MAX_SEGMENTS)
    ap.add_argument("--threshold", type=float, default=None,
                    help="라벨 임계값(초). 기본값은 설정 파일의 값")
    ap.add_argument("--no-clean", action="store_true",
                    help="디스크 절약을 위해 간섭 없는 사본을 저장하지 않음")
    args = ap.parse_args()

    build_dataset(args.csv, args.out, args.max_segments,
                  store_clean=not args.no_clean, warning_threshold=args.threshold)
