"""지연 로딩 데이터셋과 사이클 단위 데이터로더.

스펙트로그램은 HDF5에서 세그먼트 단위로 그때그때 읽어들여, 전체 데이터
약 2.5 GB가 메모리에 상주하지 않도록 한다. 크기가 작은 특징·라벨 배열만
메모리에 유지한다.

원시 TTF 범위가 복합 손실을 지배하므로 학습 시에는 [0, 1]로 정규화하고,
보고 시에는 초 단위로 되돌린다. 사이클별 오차 분석을 위해 사이클 식별자를
배치마다 함께 전달한다.

배치 구성: (스펙트로그램, 상태 벡터, 분류 라벨, 정규화 TTF,
            에너지 가중치, 사이클 id)

[검토 반영 v2]
  C-2  사이클 경계 세그먼트(파괴 순간을 담았으나 다음 사이클의 큰 TTF가
       붙은 세그먼트) 제외.
  C-3  inter 특징 워밍업 구간 처리 정책 적용.
  B-10 라벨 임계값을 로딩 시점에 바꿀 수 있도록 노출. y_cls 는 y_ttf 에서
       다시 계산되므로 데이터셋 재생성이 필요 없다.
  A-8  로더의 generator 를 meta 로 돌려주어 모델마다 시드를 리셋할 수 있게 함.
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

import config
from splitting import (cycle_aware_split, blocked_cv_splits,
                       filter_incomplete_cycles, summarize_split)


class LazySpectrogramDataset(Dataset):
    """스펙트로그램을 요청 시점에 읽으며, h5_index가 원본 HDF5 행을 가리킨다."""

    def __init__(self, h5_path, h5_index, x_state, y_cls, y_ttf_norm,
                 stress, cycle_id, ae_key="X_ae"):
        self.h5_path = h5_path
        self.ae_key = ae_key
        self.h5_index = np.asarray(h5_index, dtype=np.int64)
        self.x_state = np.ascontiguousarray(x_state, dtype=np.float32)
        self.y_cls = np.ascontiguousarray(y_cls, dtype=np.float32)
        self.y_ttf = np.ascontiguousarray(y_ttf_norm, dtype=np.float32)
        self.stress = np.ascontiguousarray(stress, dtype=np.float32)
        self.cycle_id = np.ascontiguousarray(cycle_id, dtype=np.int64)
        self._h5 = None

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.y_ttf)

    def __getitem__(self, i):
        row = int(self.h5_index[i])
        spec = self._file()[self.ae_key][row]
        spec = np.log1p(np.abs(spec) + 1e-6).astype(np.float32)
        spec = np.expand_dims(spec, 0)
        return (
            torch.from_numpy(spec),
            torch.from_numpy(self.x_state[i]),
            torch.from_numpy(self.y_cls[i]),
            torch.from_numpy(self.y_ttf[i]),
            torch.from_numpy(self.stress[i]),
            torch.tensor(self.cycle_id[i], dtype=torch.long),
        )

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_h5"] = None
        return s


def _read_optional(f, key, n, default):
    if key in f:
        return f[key][:n]
    return np.full(n, default, dtype=np.int8)


def load_arrays(h5_path=None, exclude_incomplete=None, verbose=True,
                label_threshold=None, drop_boundary=None, drop_warmup=None):
    """HDF5에서 작은 배열들을 읽고, 제외 규칙을 적용한다.

    label_threshold 를 주면 y_cls 를 y_ttf 로부터 다시 계산한다(B-10).
    """
    h5_path = h5_path or config.H5_PATH
    exclude_incomplete = (config.EXCLUDE_INCOMPLETE_CYCLES
                          if exclude_incomplete is None else exclude_incomplete)
    drop_boundary = (config.DROP_CYCLE_BOUNDARY_SEGMENT
                     if drop_boundary is None else drop_boundary)
    drop_warmup = ((config.INTER_WARMUP_POLICY == "exclude")
                   if drop_warmup is None else drop_warmup)

    with h5py.File(h5_path, "r") as f:
        n = int(f.attrs.get("n_segments", f["X_ae"].shape[0]))
        x_state = f["X_state"][:n].astype(np.float64)
        y_cls = f["y_cls"][:n]
        y_ttf = f["y_ttf"][:n]
        cycle_id = f["cycle_id"][:n]
        file_thr = float(f.attrs.get("warning_ttf_threshold",
                                     config.WARNING_TTF_THRESHOLD))
        is_boundary = _read_optional(f, "cycle_boundary", n, 0)
        is_warmup = _read_optional(f, "inter_warmup", n, 0)
        feature_names = f.attrs.get("state_feature_names", None)

    if feature_names is not None:
        try:
            names = json.loads(feature_names)
            if list(names) != list(config.STATE_FEATURE_NAMES):
                print("  경고: 데이터셋의 특징 구성이 config 와 다릅니다.\n"
                      f"        파일 {names}\n        설정 {config.STATE_FEATURE_NAMES}\n"
                      "        step1 을 다시 실행하세요.")
        except Exception:
            pass

    # [B-10] 라벨 임계값 재적용
    thr = file_thr if label_threshold is None else float(label_threshold)
    if abs(thr - file_thr) > 1e-9:
        y_cls = (y_ttf <= thr).astype(np.float32)
        if verbose:
            print(f"  라벨 임계값 재적용: {file_thr:.2f}s -> {thr:.2f}s "
                  f"(양성 {float(y_cls.sum()):.0f}개, "
                  f"{100 * float(y_cls.mean()):.2f}%)")

    h5_index = np.arange(n)
    keep_all = np.ones(n, dtype=bool)

    if exclude_incomplete:
        keep, complete, incomplete = filter_incomplete_cycles(
            cycle_id, y_ttf, thr, verbose=verbose)
        keep_all &= keep

    if drop_boundary:
        n_b = int((is_boundary[keep_all] == 1).sum())
        keep_all &= (is_boundary != 1)
        if verbose and n_b:
            print(f"  사이클 경계 세그먼트 제외: {n_b}개 "
                  f"(파괴 순간을 담았으나 다음 사이클의 TTF가 붙은 구간)")

    if drop_warmup:
        n_w = int((is_warmup[keep_all] == 1).sum())
        keep_all &= (is_warmup != 1)
        if verbose and n_w:
            print(f"  inter 특징 워밍업 구간 제외: {n_w}개 "
                  f"(사이클 초반 {config.TREND_WINDOW}구간)")

    h5_index = h5_index[keep_all]
    x_state, y_cls, y_ttf, cycle_id = (x_state[keep_all], y_cls[keep_all],
                                       y_ttf[keep_all], cycle_id[keep_all])

    return {"h5_index": h5_index, "x_state": x_state, "y_cls": y_cls,
            "y_ttf": y_ttf, "cycle_id": cycle_id, "n_original": n,
            "threshold": thr, "n_used": int(len(y_cls))}


def _make_loaders(h5_path, arrays, split, batch_size, num_workers,
                  mask_features, ae_key, seed, verbose, label=""):
    # [수정] 빈 파티션은 여기서 잡는다. 예전에는 그대로 통과해서
    #   trainer.predict() 의 np.concatenate([]) 에서 엉뚱하게 죽었다.
    empty = [k for k in ("train", "val", "test") if len(split[k]) == 0]
    if empty:
        sizes = ", ".join(f"{k} {len(split[k])}개" for k in ("train", "val", "test"))
        raise ValueError(
            f"분할 결과 {', '.join(empty)} 파티션이 비어 있습니다 ({sizes}).\n"
            f"  배정된 사이클: {split.get('cycles')}\n"
            f"  완결된 사이클이 3개 이상 있어야 합니다. LANL 데이터는 사이클\n"
            f"  하나가 약 250 세그먼트이므로 최소 1,500개 이상을 쓰세요\n"
            f"  (config.MAX_SEGMENTS, 점검 실행이면 config.SMOKE_SEGMENTS).")

    tr_idx = split["train"]
    x_state = arrays["x_state"]
    y_ttf = arrays["y_ttf"]

    # 정규화 통계는 학습 파티션에서만 산출한다.
    g_min = x_state[tr_idx].min(axis=0)
    g_max = x_state[tr_idx].max(axis=0)
    x_norm = np.clip((x_state - g_min) / (g_max - g_min + 1e-8), -5.0, 5.0)

    # 마스킹 전에 복사하여 특징 ablation이 손실함수를 바꾸지 않도록 한다.
    stress = x_norm[:, 0:1].copy()

    if mask_features:
        train_mean = x_norm[tr_idx].mean(axis=0)
        for fi in mask_features:
            x_norm[:, fi] = train_mean[fi]
        if verbose:
            names = [config.STATE_FEATURE_NAMES[i] for i in mask_features]
            print(f"    Ablation: {names} 를 학습 평균으로 대체")

    # 학습 파티션의 최대값을 기준으로 타깃을 정규화한다.
    if config.NORMALIZE_TTF:
        ttf_scale = float(y_ttf[tr_idx].max())
        if ttf_scale <= 0:
            ttf_scale = 1.0
    else:
        ttf_scale = 1.0
    y_ttf_norm = y_ttf / ttf_scale

    full = LazySpectrogramDataset(h5_path, arrays["h5_index"], x_norm.astype(np.float32),
                                  arrays["y_cls"], y_ttf_norm, stress.astype(np.float32),
                                  arrays["cycle_id"], ae_key=ae_key)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(Subset(full, split["train"].tolist()), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, generator=gen,
                              pin_memory=pin)
    val_loader = DataLoader(Subset(full, split["val"].tolist()), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(Subset(full, split["test"].tolist()), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin)

    summary = summarize_split(split, arrays["y_cls"],
                              min_test_positives=config.MIN_TEST_POSITIVES,
                              verbose=verbose, label=label)

    meta = {"split_summary": summary, "g_min": g_min.tolist(), "g_max": g_max.tolist(),
            "ttf_scale": ttf_scale, "n_used": len(arrays["y_cls"]),
            "n_original": arrays["n_original"], "ae_key": ae_key,
            "label_threshold": arrays["threshold"],
            "mask_features": mask_features or [],
            # [A-8] 모델마다 배치 순서를 리셋할 수 있도록 생성기와 시드를 노출
            "loader_generator": gen, "loader_seed": seed}

    if verbose:
        print(f"    TTF 정규화 계수(학습 최대값): {ttf_scale:.3f}초")
        print(f"    스펙트로그램 소스: {ae_key}")
    return train_loader, val_loader, test_loader, meta


def reset_loader_order(meta):
    """[검토 A-8] 배치 순서 생성기를 시드로 되돌린다.

    step3 는 한 시드 안에서 세 모델이 같은 train_loader 객체를 재사용한다.
    RandomSampler 는 __iter__ 마다 generator 상태를 전진시키므로, 리셋하지
    않으면 (1) 모델마다 배치 순서가 달라져 대응 검정의 전제가 깨지고,
    (2) 앞 모델의 조기 종료 시점이 뒤 모델 결과를 바꾸며,
    (3) 재개 실행이 처음부터 돌린 실행과 다른 숫자를 낸다.
    """
    if not config.FIX_LOADER_SEED_PER_MODEL:
        return False
    gen = meta.get("loader_generator")
    seed = meta.get("loader_seed")
    if gen is None or seed is None:
        return False
    gen.manual_seed(int(seed))
    return True


def build_dataloaders(h5_path=None, batch_size=None, num_workers=None,
                      mask_features=None, ae_key="X_ae", verbose=True, seed=None,
                      exclude_incomplete=None, label_threshold=None):
    h5_path = h5_path or config.H5_PATH
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers

    arrays = load_arrays(h5_path, exclude_incomplete, verbose=verbose,
                         label_threshold=label_threshold)
    split = cycle_aware_split(arrays["cycle_id"], config.TRAIN_RATIO, config.VAL_RATIO)
    out = _make_loaders(h5_path, arrays, split, batch_size, num_workers,
                        mask_features, ae_key, seed, verbose)

    if verbose:
        p = os.path.join(config.RESULT_DIR, "split_summary.json")
        dump = {k: v for k, v in out[3].items() if k != "loader_generator"}
        with open(p, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"    분할 요약 저장: {p}")
    return out


def build_cv_dataloaders(fold, h5_path=None, batch_size=None, num_workers=None,
                         mask_features=None, ae_key="X_ae", verbose=True, seed=None,
                         exclude_incomplete=None, label_threshold=None):
    h5_path = h5_path or config.H5_PATH
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers

    arrays = load_arrays(h5_path, exclude_incomplete, verbose=False,
                         label_threshold=label_threshold)
    folds = blocked_cv_splits(arrays["cycle_id"], config.CV_N_FOLDS,
                              config.CV_VAL_CYCLES, config.CV_TEST_CYCLES)
    if fold >= len(folds):
        raise ValueError(f"fold {fold}가 없습니다 (총 {len(folds)}개)")
    return _make_loaders(h5_path, arrays, folds[fold], batch_size, num_workers,
                         mask_features, ae_key, seed, verbose, label=f"[CV fold {fold}]")


def n_cv_folds(h5_path=None, exclude_incomplete=None):
    arrays = load_arrays(h5_path or config.H5_PATH, exclude_incomplete, verbose=False)
    return len(blocked_cv_splits(arrays["cycle_id"], config.CV_N_FOLDS,
                                 config.CV_VAL_CYCLES, config.CV_TEST_CYCLES))


if __name__ == "__main__":
    tr, va, te, meta = build_dataloaders(batch_size=4, num_workers=0)
    x_ae, x_state, y_cls, y_ttf, stress, cyc = next(iter(tr))
    print("\n  배치 텐서 규격")
    print(f"  x_ae     : {tuple(x_ae.shape)}")
    print(f"  x_state  : {tuple(x_state.shape)}")
    print(f"  y_cls    : {tuple(y_cls.shape)}")
    print(f"  y_ttf (정규화): {tuple(y_ttf.shape)}  "
          f"범위 [{float(y_ttf.min()):.3f}, {float(y_ttf.max()):.3f}]")
    print(f"  stress   : {tuple(stress.shape)}")
    print(f"  cycle_id : {tuple(cyc.shape)}  값 {cyc.tolist()}")
    print(f"  ttf_scale: {meta['ttf_scale']:.3f}")
    print(f"  라벨 임계값: {meta['label_threshold']:.2f}s")
