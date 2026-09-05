"""공용 학습·추론 루틴.

모든 모델이 동일한 최적화 경로를 따르도록 하여, 학습 절차의 차이가 비교
결과를 교란하지 않게 한다. 검증 지표에 대한 조기 종료로 실질 학습량의
차이도 제한한다.

하이퍼파라미터는 손실값이 아니라 검증 PR-AUC와 R2의 복합 지표로 선택한다.
손실 크기는 탐색 대상인 회귀 가중치에 비례해 커지므로 선택 기준으로
부적합하기 때문이다.

지연시간 측정 시 CUDA 스트림을 동기화한다. 동기화하지 않으면 커널 실행
요청 시간만 측정된다.

[검토 반영 v2]
  B-16 조기 종료 기준을 검증 손실이 아니라 복합 지표로 둘 수 있게 한다.
       손실이 선택 기준으로 부적합한 이유는 탐색에서나 한 실행 안에서나
       동일하다. 기존 동작에서는 w_ttf 가 큰 설정일 때 주 지표(PR-AUC)가
       모델 선택에 전혀 개입하지 못했다.
  B-17 손실 세 항의 실제 비중을 기록한다. criterion 은 성분을 반환하는데
       기존 코드는 `loss, *_ =` 로 버리고 있었다.
  A-4  전체 파이프라인 지연시간에 상태 벡터 계산을 포함한다. 기존 측정은
       STFT + 전송 + forward 만 재서 "원신호 취득부터 추론까지"라는
       서술과 맞지 않았다. 하드웨어 정보도 함께 반환한다.
  A-8  모델마다 배치 순서를 되돌릴 수 있도록 data.reset_loader_order 를 사용.
"""

import copy
import os
import time

import numpy as np
import torch
import torch.optim as optim

import config
from models import forward_model, PhysicsInformedMultiTaskLoss, uses_physics_penalty


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_criterion(name, ttf_weight, lambda_p=None, gamma=None, alpha=None,
                   alpha_mode=None):
    """손실함수 생성. 스펙트로그램 전용 베이스라인은 에너지 가중항을 제외한다."""
    if lambda_p is None:
        lambda_p = config.LAMBDA_PHYSICS if uses_physics_penalty(name) else 0.0
    return PhysicsInformedMultiTaskLoss(alpha=alpha, gamma=gamma,
                                        lambda_p=lambda_p, ttf_weight=ttf_weight,
                                        alpha_mode=alpha_mode)


def _epoch_pass(model, name, loader, device, criterion, optimizer=None,
                collect=False):
    """한 에폭. optimizer 가 있으면 학습, 없으면 검증.

    [수정] collect=True 이면 예측값도 함께 모은다. 기존에는 검증 순전파를
    한 번 돌린 뒤 hp_selection_score() 가 predict() 로 같은 검증 집합을
    다시 돌려, 에폭마다 검증 순전파가 두 번 실행되고 있었다.
    """
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = {"loss": 0.0, "cls": 0.0, "ttf": 0.0, "phys": 0.0}
    n = 0
    buf = {"y_cls": [], "s_cls": [], "y_ttf": [], "p_ttf": []}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_ae, x_state, y_cls, y_ttf, stress, _ in loader:
            x_ae, x_state = x_ae.to(device), x_state.to(device)
            y_cls, y_ttf, stress = y_cls.to(device), y_ttf.to(device), stress.to(device)

            if train:
                optimizer.zero_grad()
            p_cls, p_ttf, _ = forward_model(name, model, x_ae, x_state)
            loss, l_cls, l_ttf, l_phys = criterion(p_cls, p_ttf, y_cls, y_ttf, stress)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

            b = x_ae.size(0)
            tot["loss"] += float(loss.item()) * b
            tot["cls"] += float(l_cls.item()) * b
            tot["ttf"] += float(l_ttf.item()) * b
            tot["phys"] += float(l_phys.item()) * b
            n += b

            if collect:
                buf["y_cls"].append(y_cls.detach().cpu().numpy().reshape(-1))
                buf["s_cls"].append(torch.sigmoid(p_cls).detach().cpu().numpy().reshape(-1))
                buf["y_ttf"].append(y_ttf.detach().cpu().numpy().reshape(-1))
                buf["p_ttf"].append(p_ttf.detach().cpu().numpy().reshape(-1))

    n = max(n, 1)
    out = {k: v / n for k, v in tot.items()}
    if collect:
        out["_pred"] = ({k: np.concatenate(v) for k, v in buf.items()}
                        if buf["y_cls"] else None)
    return out


def train_model(model, name, train_loader, val_loader, device,
                lr=1e-3, ttf_weight=1.0, epochs=None, weight_decay=1e-4,
                patience=None, verbose=True, save_path=None, criterion=None,
                ttf_scale=1.0, early_stop_metric=None, loader_meta=None):
    epochs = epochs or config.EPOCHS
    patience = config.PATIENCE if patience is None else patience
    criterion = criterion or make_criterion(name, ttf_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    metric_mode = config.EARLY_STOP_METRIC if early_stop_metric is None else early_stop_metric

    # [A-8] 이 모델의 학습을 시작하기 전에 배치 순서를 시드로 되돌린다.
    if loader_meta is not None:
        from data import reset_loader_order
        reset_loader_order(loader_meta)

    best_score, best_state, best_epoch, bad = -float("inf"), None, -1, 0
    best_val_loss = float("inf")
    history = []
    t0 = time.perf_counter()

    for ep in range(epochs):
        want_pred = (metric_mode == "composite")
        tr = _epoch_pass(model, name, train_loader, device, criterion, optimizer)
        va = _epoch_pass(model, name, val_loader, device, criterion, None,
                         collect=want_pred)

        if metric_mode == "composite":
            sel = hp_selection_score(model, name, val_loader, device,
                                     ttf_scale=ttf_scale, val_loss=va["loss"],
                                     force_metric="composite",
                                     precomputed=va.get("_pred"))
            score = sel["score"]
            extra = (f" | PR-AUC {sel['val_pr_auc']:.4f} R2 {sel['val_r2']:+.4f}"
                     f" -> {score:.4f}")
        else:
            score = -va["loss"]
            sel = {"score": score, "val_pr_auc": np.nan, "val_r2": np.nan}
            extra = ""

        rec = {"epoch": ep + 1, "train_loss": tr["loss"], "val_loss": va["loss"],
               "select_score": score}
        if config.LOG_LOSS_COMPONENTS:
            # [B-17] 세 항의 실제 크기. 에너지 가중항이 학습에 실제로
            # 관여했는지를 사후에 확인할 수 있다.
            rec.update({
                "train_cls": tr["cls"], "train_ttf_weighted": ttf_weight * tr["ttf"],
                "train_physics_weighted": criterion.lambda_p * tr["phys"],
                "val_cls": va["cls"], "val_ttf_weighted": ttf_weight * va["ttf"],
                "val_physics_weighted": criterion.lambda_p * va["phys"],
            })
        history.append(rec)

        if verbose:
            comp = ""
            if config.LOG_LOSS_COMPONENTS:
                den = max(abs(rec["train_cls"]) + abs(rec["train_ttf_weighted"])
                          + abs(rec["train_physics_weighted"]), 1e-12)
                comp = (f" | cls {100 * rec['train_cls'] / den:4.1f}%"
                        f" ttf {100 * rec['train_ttf_weighted'] / den:4.1f}%"
                        f" phy {100 * rec['train_physics_weighted'] / den:4.1f}%")
            print(f"      ep {ep+1:02d}/{epochs}  train {tr['loss']:.4f} | "
                  f"val {va['loss']:.4f}{extra}{comp}")

        if score > best_score + 1e-9:
            best_score, best_epoch, bad = score, ep + 1, 0
            best_val_loss = va["loss"]
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"      early stop @ epoch {ep+1}")
                break

    elapsed = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        torch.save({"state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_select_score": best_score,
                    "early_stop_metric": metric_mode,
                    "best_epoch": best_epoch, "lr": lr,
                    "ttf_weight": ttf_weight}, save_path)

    return {"model": model, "best_val_loss": best_val_loss,
            "best_select_score": best_score, "early_stop_metric": metric_mode,
            "best_epoch": best_epoch, "history": history, "train_seconds": elapsed}


@torch.no_grad()
def predict(model, name, loader, device, collect_attn=False, ttf_scale=1.0,
            per_head_attn=False):
    """추론을 수행하고 예측값을 초 단위로 되돌린다.

    관측·예측 분류 점수와 파괴까지 남은 시간, 사이클 식별자, 그리고 가능한
    경우 어텐션 가중치를 반환한다.

    per_head_attn=True 이면 헤드 평균 대신 헤드별 가중치를 모은다
    ([검토 B-15] 보고된 엔트로피는 헤드 평균이라 집중도의 상한이었다).
    """
    model.eval()
    yc, ys, yt, yp, cyc, attns = [], [], [], [], [], []

    for x_ae, x_state, y_cls, y_ttf, _, cycle_id in loader:
        x_ae, x_state = x_ae.to(device), x_state.to(device)
        kw = {"return_head_attn": True} if (collect_attn and per_head_attn) else {}
        p_cls, p_ttf, attn = forward_model(name, model, x_ae, x_state, **kw)

        yc.append(y_cls.cpu().numpy().reshape(-1))
        ys.append(torch.sigmoid(p_cls).cpu().numpy().reshape(-1))
        yt.append(y_ttf.cpu().numpy().reshape(-1) * ttf_scale)     # back to seconds
        yp.append(p_ttf.cpu().numpy().reshape(-1) * ttf_scale)     # back to seconds
        cyc.append(cycle_id.cpu().numpy().reshape(-1))
        if collect_attn and attn is not None:
            a = attn.cpu().numpy()
            # 평균 모드: (B,1,T) -> (B,T)   헤드 모드: (B,H,1,T) -> (B,H,T)
            a = a.squeeze(2) if a.ndim == 4 else a.squeeze(1)
            attns.append(a)

    # [수정] 빈 로더가 들어오면 np.concatenate([]) 가
    #   "need at least one array to concatenate" 로 죽어, 진짜 원인(분할이
    #   비었음)과 무관한 곳을 가리켰다. 원인을 밝히며 실패하도록 한다.
    if not yc:
        raise ValueError(
            "predict(): 데이터로더가 비어 있습니다. 해당 파티션에 세그먼트가 "
            "하나도 배정되지 않았습니다.\n"
            "  분할 요약의 '세그먼트 0' 항목을 확인하세요. 대개 사이클 수가 "
            "부족한 경우입니다(config.MAX_SEGMENTS / --smoke 세그먼트 수).")

    return (np.concatenate(yc), np.concatenate(ys), np.concatenate(yt),
            np.concatenate(yp), np.concatenate(cyc),
            np.concatenate(attns, axis=0) if attns else None)


def hp_selection_score(model, name, val_loader, device, ttf_scale=1.0, val_loss=None,
                       force_metric=None, precomputed=None):
    """스케일에 무관한 선택 기준.

    손실값은 탐색 대상인 회귀 가중치에 비례해 커지므로, 손실로 선택하면
    예측 성능과 무관하게 가장 작은 가중치가 뽑힌다. 검증 PR-AUC와 R2의 합은
    두 과제를 비슷한 척도에서 함께 반영한다.
    """
    from metrics import classification_metrics, regression_metrics

    metric = force_metric or config.HP_SELECTION_METRIC
    if metric == "val_loss":
        return {"score": -(val_loss if val_loss is not None else 0.0),
                "val_pr_auc": np.nan, "val_r2": np.nan}

    if precomputed is not None:
        # [수정] 검증 순전파를 두 번 돌리지 않도록 _epoch_pass 의 결과를 재사용.
        y_true_c = precomputed["y_cls"]
        y_score_c = precomputed["s_cls"]
        y_true_t = precomputed["y_ttf"] * ttf_scale
        y_pred_t = precomputed["p_ttf"] * ttf_scale
    else:
        y_true_c, y_score_c, y_true_t, y_pred_t, _, _ = predict(
            model, name, val_loader, device, ttf_scale=ttf_scale)
    cm = classification_metrics(y_true_c, y_score_c)
    rm = regression_metrics(y_true_t, y_pred_t)

    pr = cm["pr_auc"] if not np.isnan(cm["pr_auc"]) else 0.0
    r2 = rm["r2"]
    r2 = max(r2, -1.0)     # 발산한 적합이 선택을 지배하지 않도록 하한을 둔다
    return {"score": float(pr + r2), "val_pr_auc": float(pr), "val_r2": float(r2)}


@torch.no_grad()
def measure_latency(model, name, loader, device, warmup=10, max_batches=300):
    """CUDA 동기화를 포함한 forward pass 지연시간."""
    model.eval()
    times = []
    for i, batch in enumerate(loader):
        x_ae, x_state = batch[0].to(device), batch[1].to(device)
        if i < warmup:
            forward_model(name, model, x_ae, x_state)
            continue
        if i - warmup >= max_batches:
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_model(name, model, x_ae, x_state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times) if times else np.array([np.nan])
    return {"n_samples": int(len(times)), "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)), "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "fps": float(1000.0 / np.mean(times)) if np.mean(times) > 0 else float("nan")}


def _device_label(device):
    if device.type == "cuda":
        try:
            return f"GPU: {torch.cuda.get_device_name(0)}"
        except Exception:
            return "GPU (name unavailable)"
    import platform
    return f"CPU: {platform.processor() or platform.machine()}"


@torch.no_grad()
def measure_end_to_end_latency(model, name, device, n_trials=100, warmup=10):
    """원신호 한 세그먼트에서 추론까지의 전체 파이프라인 지연시간.

    [A-4] 상태 벡터 계산을 포함한다. 기존 측정은 STFT + 전송 + forward 만
    재고 있어서 "원신호 취득부터 추론까지"라는 서술과 맞지 않았다.
    실측 시 상태 벡터 계산(특히 pandas kurtosis)이 약 1.5 ms 로,
    무시할 수 없는 크기다.
    """
    from scipy.signal import stft as scipy_stft
    from step1_build_dataset import compute_state_features
    from collections import deque

    rng = np.random.default_rng(0)
    raw = rng.integers(-3000, 3000, size=config.SEGMENT_SIZE).astype(np.int16)
    hist = deque([1.0] * config.TREND_WINDOW, maxlen=config.TREND_WINDOW)

    stft_t, state_t, xfer_t, infer_t, total_t = [], [], [], [], []
    model.eval()

    for i in range(n_trials + warmup):
        t0 = time.perf_counter()
        _, _, Zxx = scipy_stft(raw, fs=config.SAMPLING_RATE,
                               nperseg=config.NPERSEG, noverlap=config.NOVERLAP)
        spec = np.abs(Zxx)[:, :config.TIME_STEPS]
        spec = np.log1p(spec + 1e-6).astype(np.float32)
        t1 = time.perf_counter()

        state_vec, _ = compute_state_features(raw.astype(np.float64), hist,
                                              cycle_running_sum=0.0)
        t2 = time.perf_counter()

        x_ae = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)
        x_state = torch.from_numpy(state_vec).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        forward_model(name, model, x_ae, x_state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        if i < warmup:
            continue
        stft_t.append((t1 - t0) * 1000)
        state_t.append((t2 - t1) * 1000)
        xfer_t.append((t3 - t2) * 1000)
        infer_t.append((t4 - t3) * 1000)
        total_t.append((t4 - t0) * 1000)

    # [수정 v2.1] 실시간 배수의 분자는 세그먼트의 '실제' 경과시간이어야 한다.
    #   공칭 37.5 ms 를 쓰면 LANL 실측(약 39.3 ms)과 4.8% 어긋난 값이 논문에
    #   실린다. step1 이 저장해 둔 실측값이 있으면 그것을 쓴다.
    nominal_ms = config.SEGMENT_SIZE / config.SAMPLING_RATE * 1000
    seg_ms, seg_src = nominal_ms, "공칭"
    try:
        import json as _json
        with open(os.path.join(config.RESULT_DIR, "class_balance.json")) as f:
            m = _json.load(f).get("segment_seconds_measured")
        if m and m == m and m > 0:
            seg_ms, seg_src = float(m) * 1000.0, "실측"
    except Exception:
        pass

    mean_total = float(np.mean(total_t))
    return {"stft_mean_ms": float(np.mean(stft_t)),
            "state_feature_mean_ms": float(np.mean(state_t)),
            "transfer_mean_ms": float(np.mean(xfer_t)),
            "inference_mean_ms": float(np.mean(infer_t)),
            "end_to_end_mean_ms": mean_total,
            "end_to_end_p95_ms": float(np.percentile(total_t, 95)),
            "segment_duration_ms": seg_ms,
            "segment_duration_source": seg_src,
            "segment_duration_nominal_ms": nominal_ms,
            "realtime_factor": seg_ms / mean_total if mean_total > 0 else float("nan"),
            "device": _device_label(device),
            "note": ("데이터 취득 시간은 포함하지 않음. 보고된 값은 위 device "
                     "기준이며, 다른 하드웨어에서는 달라진다.")}
