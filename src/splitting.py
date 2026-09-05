"""스틱슬립 사이클 단위 데이터 분할.

고정 인덱스로 자르면 하나의 지진 사이클이 여러 파티션에 걸치게 되는데,
같은 사이클 내 세그먼트는 동일한 하중 이력을 공유하므로 이는 누수에 해당한다.
따라서 사이클 전체를 하나의 파티션에 배정하되 시간 순서를 유지하여, 학습이
항상 평가보다 시간상 앞서도록 한다.

파괴에 도달하지 않은 사이클(녹음이 중간에 끊긴 경우)은 양성 라벨이 없어
test 파티션의 분류 지표를 정의할 수 없게 만들므로 제외한다.

[검토 반영 v2]
  심사 지적 03  10:3:3 이 사이클 개수 기준임을 요약 출력에 명시하고,
                세그먼트 기준 실현 비율을 함께 보고한다.
"""

import numpy as np


class InsufficientCyclesError(ValueError):
    """3분할을 만들 만큼 사이클이 남지 않았을 때."""


def cycle_aware_split(cycle_id, train_ratio=0.70, val_ratio=0.15):
    """사이클 전체를 시간 순서대로 학습/검증/평가에 배정.

    [수정] 기존 구현은 fallback 조건이 `len(unique_cycles) >= 3` 이라,
    사이클이 2개면 둘 다 train 으로 가고 val·test 가 빈 채로 반환되었다.
    그 빈 로더가 predict() 까지 흘러가 `need at least one array to
    concatenate` 라는 엉뚱한 지점에서 죽었다. 이제
      - 사이클이 3개 미만이면 원인을 밝히며 즉시 실패하고,
      - 3개 이상이면 두 절단 지점을 1 <= i < j < n 으로 죄어 세 파티션이
        비지 않으면서 시간 순서도 반드시 유지되게 한다.
    """
    cycle_id = np.asarray(cycle_id)
    unique_cycles = list(np.unique(cycle_id))
    n_cycles = len(unique_cycles)
    n_total = len(cycle_id)

    if n_cycles < 3:
        raise InsufficientCyclesError(
            f"사이클이 {n_cycles}개뿐이라 학습/검증/평가 3분할을 만들 수 없습니다.\n"
            f"  (미도달 사이클·경계 세그먼트를 제외한 뒤 남은 수입니다)\n"
            f"  LANL 데이터는 사이클 하나가 약 250 세그먼트이므로, 최소 5개 사이클을\n"
            f"  확보하려면 1,500개 이상의 세그먼트가 필요합니다.\n"
            f"  config.MAX_SEGMENTS 를 늘리거나, 점검 실행이라면\n"
            f"  run_all.py --smoke 의 세그먼트 수를 늘리세요.")

    # 사이클은 시간순으로 정렬되어 있으므로, 분할은 두 개의 절단 지점
    # (i, j) 하나로 완전히 기술된다:  train = [:i], val = [i:j], test = [j:]
    # 먼저 세그먼트 수 목표에 맞춰 절단 지점을 고른 뒤,
    # 1 <= i < j < n 으로 죔으로써 세 파티션이 비지 않고 시간 순서도 유지되게 한다.
    target_train = n_total * train_ratio
    target_val = n_total * (train_ratio + val_ratio)
    counts = {c: int((cycle_id == c).sum()) for c in unique_cycles}

    i = j = None
    running = 0
    for k, c in enumerate(unique_cycles):
        if i is None and running >= target_train:
            i = k
        if j is None and running >= target_val:
            j = k
        running += counts[c]
    if i is None:
        i = n_cycles - 2
    if j is None:
        j = n_cycles - 1

    # 죄기: train >= 1, val >= 1, test >= 1 을 보장한다.
    i = min(max(i, 1), n_cycles - 2)
    j = min(max(j, i + 1), n_cycles - 1)

    train_cycles = unique_cycles[:i]
    val_cycles = unique_cycles[i:j]
    test_cycles = unique_cycles[j:]

    assert train_cycles and val_cycles and test_cycles, \
        f"3분할 보정 실패: {train_cycles} / {val_cycles} / {test_cycles}"

    def _idx(cycles):
        if not cycles:
            return np.array([], dtype=np.int64)
        return np.where(np.isin(cycle_id, cycles))[0]

    return {
        "train": _idx(train_cycles),
        "val": _idx(val_cycles),
        "test": _idx(test_cycles),
        "cycles": {
            "train": [int(c) for c in train_cycles],
            "val": [int(c) for c in val_cycles],
            "test": [int(c) for c in test_cycles],
        },
    }


def blocked_cv_splits(cycle_id, n_folds=4, val_cycles=2, test_cycles=2):
    """시간 순서를 유지하는 rolling-origin 교차검증.

    [검토 B-1] 본 실험은 단일 분할에서 시드만 바꾸므로, 시드 간 표준편차는
    최적화 잡음이지 일반화 오차가 아니다. 이 교차검증을 켜면(--cv) 분할
    변동까지 포함한 결과를 얻을 수 있다.
    """
    cycle_id = np.asarray(cycle_id)
    cycles = np.unique(cycle_id)
    splits = []

    for k in range(n_folds):
        test_end = len(cycles) - k * test_cycles
        test_start = test_end - test_cycles
        val_start = test_start - val_cycles
        if val_start <= 1:
            break

        tr_c = [int(c) for c in cycles[:val_start]]
        va_c = [int(c) for c in cycles[val_start:test_start]]
        te_c = [int(c) for c in cycles[test_start:test_end]]

        def _idx(cs):
            return np.where(np.isin(cycle_id, cs))[0]

        splits.append({
            "fold": k,
            "train": _idx(tr_c), "val": _idx(va_c), "test": _idx(te_c),
            "cycles": {"train": tr_c, "val": va_c, "test": te_c},
        })
    return splits


def filter_incomplete_cycles(cycle_id, y_ttf, threshold, verbose=True):
    """최소 TTF가 파괴에 도달한 사이클만 남기는 마스크를 반환."""
    cycle_id = np.asarray(cycle_id)
    y_ttf = np.asarray(y_ttf).reshape(-1)

    keep = np.zeros(len(cycle_id), dtype=bool)
    complete_cycles, incomplete_cycles = [], []

    for c in np.unique(cycle_id):
        mask = cycle_id == c
        if y_ttf[mask].min() <= threshold:
            keep |= mask
            complete_cycles.append(int(c))
        else:
            incomplete_cycles.append(int(c))

    if verbose and incomplete_cycles:
        print(f"  파괴 미도달 사이클 제외: {incomplete_cycles} "
              f"({int((~keep).sum())}개 세그먼트, 양성 라벨 없음)")

    return keep, complete_cycles, incomplete_cycles


def summarize_split(split, y_cls, min_test_positives=30, verbose=True, label=""):
    """파티션 크기와 클래스 균형을 보고하고 사이클 중복이 없음을 검증."""
    y_cls = np.asarray(y_cls).reshape(-1)
    summary = {}
    n_all = sum(len(split[k]) for k in ("train", "val", "test"))
    for name in ("train", "val", "test"):
        idx = split[name]
        y = y_cls[idx]
        n_pos = int(y.sum())
        summary[name] = {
            "n_segments": int(len(idx)),
            "n_cycles": len(split["cycles"][name]),
            "cycles": split["cycles"][name],
            "n_positive": n_pos,
            "n_negative": int(len(idx) - n_pos),
            "positive_rate_pct": round(100.0 * n_pos / max(len(idx), 1), 3),
            "segment_share_pct": round(100.0 * len(idx) / max(n_all, 1), 2),
        }

    tr, va, te = (set(split["cycles"][k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te), \
        "파티션 간 사이클 중복이 감지되었습니다"

    ratio_cycles = " : ".join(str(summary[k]["n_cycles"]) for k in ("train", "val", "test"))
    ratio_segs = " : ".join(f"{summary[k]['segment_share_pct']:.0f}"
                            for k in ("train", "val", "test"))
    summary["ratio_by_cycles"] = ratio_cycles
    summary["ratio_by_segments_pct"] = ratio_segs

    if verbose:
        print("-" * 68)
        print(f"  사이클 단위 분할 {label}".rstrip())
        for name in ("train", "val", "test"):
            s = summary[name]
            print(f"    {name:5s} | 세그먼트 {s['n_segments']:5d} ({s['segment_share_pct']:5.1f}%) "
                  f"| 사이클 {s['n_cycles']:3d} "
                  f"| 양성 {s['n_positive']:5d} ({s['positive_rate_pct']:6.2f}%)")
        # [심사 지적 03] 논문에 쓸 비율을 무엇 기준인지 명시해 출력
        print(f"    학습 : 검증 : 평가 = {ratio_cycles} (사이클 개수 기준), "
              f"{ratio_segs} % (세그먼트 기준)")
        if summary["test"]["n_positive"] < min_test_positives:
            print(f"    경고: 평가 양성이 {summary['test']['n_positive']}개뿐입니다 "
                  f"(권장 {min_test_positives}개 이상)")
        print("-" * 68)

    return summary


def _self_test():
    rng = np.random.default_rng(0)
    cycle_id, y_cls, ttf = [], [], []
    for c in range(16):
        L = int(rng.integers(200, 420))
        t = np.linspace(14.0, 6.0 if c == 15 else 0.0, L)
        cycle_id += [c] * L
        ttf += list(t)
        y_cls += list((t <= 3.0).astype(float))

    cycle_id, ttf, y_cls = np.array(cycle_id), np.array(ttf), np.array(y_cls)

    keep, complete, incomplete = filter_incomplete_cycles(cycle_id, ttf, 3.0)
    assert incomplete == [15]

    cid_f, ycls_f = cycle_id[keep], y_cls[keep]
    sp_before = cycle_aware_split(cycle_id, 0.60, 0.15)
    n_pos_before = int(y_cls[sp_before["test"]].sum())
    sp_after = cycle_aware_split(cid_f, 0.60, 0.15)
    summary = summarize_split(sp_after, ycls_f, verbose=True,
                              label="(미도달 사이클 제외 후)")
    assert summary["test"]["n_positive"] > n_pos_before

    all_idx = np.concatenate([sp_after["train"], sp_after["val"], sp_after["test"]])
    assert len(np.unique(all_idx)) == len(cid_f)
    assert sp_after["train"].max() < sp_after["test"].min()

    folds = blocked_cv_splits(cid_f, 4, 2, 2)
    for f in folds:
        assert f["train"].max() < f["test"].min()

    # [수정 검증] 사이클이 2개면 즉시, 원인을 밝히며 실패해야 한다.
    two = np.repeat([0, 1], 160)
    try:
        cycle_aware_split(two)
    except InsufficientCyclesError as e:
        assert "2개뿐" in str(e)
    else:
        raise AssertionError("사이클 2개인데도 예외가 발생하지 않았습니다")

    # [수정 검증] 사이클이 3개면 val·test 가 절대 비지 않아야 한다.
    three = np.repeat([0, 1, 2], 160)
    sp3 = cycle_aware_split(three, 0.70, 0.15)
    assert len(sp3["val"]) > 0 and len(sp3["test"]) > 0, sp3["cycles"]
    assert sp3["cycles"] == {"train": [0], "val": [1], "test": [2]}, sp3["cycles"]

    # 사이클 길이가 크게 치우쳐도(첫 사이클이 전체의 90%) 3분할이 성립해야 한다.
    skew = np.concatenate([np.zeros(900), np.ones(50), np.full(50, 2)]).astype(int)
    sp_s = cycle_aware_split(skew, 0.70, 0.15)
    assert len(sp_s["val"]) > 0 and len(sp_s["test"]) > 0, sp_s["cycles"]

    print(f"splitting: 정상 (평가 양성 {n_pos_before} -> "
          f"{summary['test']['n_positive']}, CV fold {len(folds)}개, "
          f"빈 파티션 가드 통과)")


if __name__ == "__main__":
    _self_test()
