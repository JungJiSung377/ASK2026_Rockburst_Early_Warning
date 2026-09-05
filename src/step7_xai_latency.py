"""교차 어텐션 해석과 추론 지연시간.

어텐션 가중치는 정의상 음수가 될 수 없으므로, 평균 표준편차 밴드를 그리면
0 아래로 내려가 분포를 잘못 나타낼 수 있다. 대신 중앙값과 사분위 범위를
사용한다. 집중도는 어텐션 분포의 정규화 엔트로피로 정량화하여 경보 구간과
정상 구간을 비교한다.

지연시간은 신경망 forward pass만 측정한 값과 전처리·상태 벡터 계산을 포함한
전체 파이프라인 값을 함께 보고한다. 배포 시에는 전처리도 실시간 제약을 받기
때문이다.

[검토 반영 v2 / 심사 지적 24]
  가장 중요한 보강. 어텐션이 성능을 올리지 못했으므로 존재 이유가 해석
  가능성뿐인데, 기존에는 엔트로피 숫자 둘로만 제시되어 있었다. 다음을 추가한다.
    - 삭제 실험(deletion test): 어텐션 상위 k개 프레임을 가렸을 때 예측이
      실제로 바뀌는가를 무작위 k개와 비교. "가중치가 진짜 근거"라는 주장의
      유일한 실증이다.
    - 헤드별 엔트로피(B-15): 보고값이 4개 헤드 평균이라 집중도의 상한이었다.
    - 유효 프레임 수 환산: 0.8694 -> 101^0.8694 ≈ 55/101. "특정 구간으로
      모인다"보다 "유의하게 좁아지지만 여전히 넓다"가 데이터에 맞다.
  A-4  지연시간에 상태 벡터 계산과 하드웨어 정보를 포함한다.
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

import config
import env_report
from data import build_dataloaders
from figstyle import apply_style, panel_label, save
from models import MODEL_REGISTRY, checkpoint_path, forward_model, returns_attention
from models_meta import PROPOSED_MODEL, display
from trainer import get_device, predict, measure_latency, measure_end_to_end_latency

apply_style()


def attention_entropy(w):
    """정규화 섀넌 엔트로피. 0이면 완전 집중, 1이면 균일 분포."""
    w = np.clip(w, 1e-12, None)
    w = w / w.sum(axis=-1, keepdims=True)
    ent = -(w * np.log(w)).sum(axis=-1)
    return ent / np.log(w.shape[-1])


def effective_frames(H, n_frames):
    """엔트로피를 '실질적으로 보고 있는 프레임 수'로 환산."""
    return float(n_frames ** float(H))


# ----------------------------------------------------------------------
# 삭제 실험 [심사 지적 24]
# ----------------------------------------------------------------------
@torch.no_grad()
def deletion_test(model, model_name, loader, device, ttf_scale=1.0,
                  ks=(5, 10, 20), seed=0, max_batches=None):
    """어텐션 상위 k 프레임을 가렸을 때 예측이 얼마나 바뀌는가.

    같은 개수의 무작위 프레임을 가린 경우와 비교한다. 상위 프레임을 가릴 때
    예측 변화가 유의하게 크다면, 어텐션 가중치가 실제로 판단 근거를 가리키고
    있다는 직접 증거가 된다.

    구현: LSTM 출력 시퀀스(Key/Value)의 해당 시간 프레임을 0 으로 만든다.
    """
    if not returns_attention(model_name):
        return None
    model.eval()
    rng = np.random.default_rng(seed)

    from models import _encode_fast   # noqa: WPS437  (내부 헬퍼 재사용)

    out = {int(k): {"top": [], "rand": []} for k in ks}
    base_scores = []

    for bi, (x_ae, x_state, y_cls, y_ttf, _, _) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x_ae, x_state = x_ae.to(device), x_state.to(device)
        ae_seq = _encode_fast(model.ae_cnn, model.ae_lstm, x_ae)   # (B,T,d)
        query = model.state_mlp(x_state).unsqueeze(1)

        def _head(seq):
            a_out, a_w = model.cross_attn(query, seq, seq)
            q = query.squeeze(1)
            a = a_out.squeeze(1)
            if model.fusion_mode == "add":
                fus = a + q
            elif model.fusion_mode == "concat":
                fus = torch.cat([a, q], dim=1)
            else:
                fus = a
            return torch.sigmoid(model.cls_head(fus)).squeeze(-1), a_w

        p0, attn = _head(ae_seq)
        base_scores.append(p0.cpu().numpy())

        w = attn.squeeze(1)                       # (B, T)
        T = w.shape[1]
        order = torch.argsort(w, dim=1, descending=True)

        for k in ks:
            k = int(min(k, T))
            # 상위 k 프레임 마스킹
            mask_top = torch.ones_like(ae_seq[:, :, :1])
            idx = order[:, :k]
            mask_top.scatter_(1, idx.unsqueeze(-1), 0.0)
            p_top, _ = _head(ae_seq * mask_top)

            # 무작위 k 프레임 마스킹
            mask_rnd = torch.ones_like(mask_top)
            rid = torch.from_numpy(
                np.stack([rng.choice(T, size=k, replace=False)
                          for _ in range(ae_seq.shape[0])])).to(device)
            mask_rnd.scatter_(1, rid.unsqueeze(-1), 0.0)
            p_rnd, _ = _head(ae_seq * mask_rnd)

            out[int(k)]["top"].append((p_top - p0).abs().cpu().numpy())
            out[int(k)]["rand"].append((p_rnd - p0).abs().cpu().numpy())

    res = {}
    for k, v in out.items():
        top = np.concatenate(v["top"]) if v["top"] else np.array([])
        rnd = np.concatenate(v["rand"]) if v["rand"] else np.array([])
        if len(top) < 2:
            continue
        t, p = stats.ttest_rel(top, rnd)
        res[k] = {"k": k,
                  "mean_abs_change_top": float(top.mean()),
                  "mean_abs_change_random": float(rnd.mean()),
                  "ratio": float(top.mean() / (rnd.mean() + 1e-12)),
                  "paired_t": float(t), "p_value": float(p),
                  "n": int(len(top))}
    return res


def figure_deletion(dele, fname="fig10_attention_deletion.pdf"):
    if not dele:
        return None
    print("  그림 10  어텐션 삭제 실험")
    ks = sorted(dele.keys())
    top = [dele[k]["mean_abs_change_top"] for k in ks]
    rnd = [dele[k]["mean_abs_change_random"] for k in ks]

    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE_COL_IN, 2.3))
    x = np.arange(len(ks))
    ax.bar(x - 0.18, top, width=0.34, color=config.PALETTE["highlight"],
           edgecolor="black", lw=0.4, label="Top-$k$ attended frames")
    ax.bar(x + 0.18, rnd, width=0.34, color="0.75",
           edgecolor="black", lw=0.4, label="Random $k$ frames")
    for i, k in enumerate(ks):
        p = dele[k]["p_value"]
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
        ax.text(i, max(top[i], rnd[i]) * 1.05, star, ha="center",
                fontsize=config.FIG_BASE_FONTSIZE - 1)
    ax.set_xticks(x, [f"$k$={k}" for k in ks])
    ax.set_ylabel("|Δ warning probability|")
    ax.set_title("Deletion test", pad=4)
    ax.grid(True, axis="y", ls=":", lw=0.4)
    ax.legend(loc="upper left", fontsize=config.FIG_BASE_FONTSIZE - 1.5)
    fig.tight_layout()
    return save(fig, fname)


def figure_attention(attn, y_cls, entropy_stats=None):
    """어텐션 해석 그림 (3패널)."""
    if attn is None or len(attn) == 0:
        print("  그림 7 생략 (어텐션 가중치 없음)")
        return None
    print("  그림 7  교차 어텐션 해석")

    pos, neg = attn[y_cls == 1.0], attn[y_cls == 0.0]
    if len(pos) < 2 or len(neg) < 2:
        print("  그림 7 생략 (클래스별 표본 부족)")
        return None

    x = np.arange(attn.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(config.FIG_DOUBLE_COL_IN, 2.3),
                             gridspec_kw={"width_ratios": [1.7, 0.85, 1.05]})

    # (a) 부트스트랩 신뢰구간을 포함한 평균 어텐션 차이
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    rng = np.random.default_rng(0)
    boot = np.empty((400, attn.shape[1]))
    for b in range(400):
        ip = rng.integers(0, len(pos), len(pos))
        ineg = rng.integers(0, len(neg), len(neg))
        boot[b] = pos[ip].mean(axis=0) - neg[ineg].mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    axes[0].axhline(0, color="0.4", lw=0.7, ls="--")
    axes[0].fill_between(x, lo * 1e3, hi * 1e3, color=config.PALETTE["proposed"],
                         alpha=0.22, lw=0, label="95% CI")
    axes[0].plot(x, diff * 1e3, color=config.PALETTE["proposed"], lw=1.2,
                 label="Warning − normal")
    sig = (lo > 0) | (hi < 0)
    if sig.any():
        axes[0].plot(x[sig], diff[sig] * 1e3, ".", ms=2.4,
                     color=config.PALETTE["highlight"], label="CI excludes zero")
    axes[0].set_xlabel("Spectro-temporal frame")
    axes[0].set_ylabel("Attention difference ($\\times 10^{-3}$)")
    axes[0].set_title("Class-conditional reweighting", pad=4)
    axes[0].grid(True, ls=":", lw=0.4)
    axes[0].legend(loc="upper left", handlelength=1.6)
    panel_label(axes[0], "a", dx=-0.22)

    # (b) 엔트로피 분포
    ep, en = attention_entropy(pos), attention_entropy(neg)
    bp = axes[1].boxplot([ep, en], widths=0.55, patch_artist=True,
                         medianprops=dict(color="black", lw=0.9),
                         flierprops=dict(marker="o", ms=1.5, mfc="0.5",
                                         mec="none", alpha=0.5),
                         boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5),
                         capprops=dict(lw=0.5))
    for patch, col in zip(bp["boxes"], [config.PALETTE["highlight"],
                                        config.PALETTE["proposed"]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
    axes[1].set_xticks([1, 2], ["Warning", "Normal"])
    axes[1].set_ylabel("Normalised entropy")
    axes[1].set_title("Concentration", pad=4)
    axes[1].grid(True, axis="y", ls=":", lw=0.4)
    if entropy_stats is not None:
        p = entropy_stats.get("p_value", np.nan)
        txt = "$p < 0.001$" if p < 1e-3 else f"$p = {p:.3f}$"
        axes[1].text(0.5, 0.03, txt, transform=axes[1].transAxes, ha="center",
                     va="bottom", fontsize=config.FIG_BASE_FONTSIZE - 1)
    panel_label(axes[1], "b", dx=-0.46)

    # (c) 엔트로피의 경험적 누적분포
    for vals, lab, col in [(ep, "Warning", config.PALETTE["highlight"]),
                           (en, "Normal", config.PALETTE["proposed"])]:
        s = np.sort(vals)
        axes[2].step(s, np.arange(1, len(s) + 1) / len(s), where="post",
                     color=col, lw=1.2, label=lab)
    axes[2].set_xlabel("Normalised entropy")
    axes[2].set_ylabel("Cumulative fraction")
    axes[2].set_title("Entropy distribution", pad=4)
    axes[2].grid(True, ls=":", lw=0.4)
    axes[2].legend(loc="upper left", handlelength=1.6)
    panel_label(axes[2], "c", dx=-0.34)

    fig.tight_layout(w_pad=1.5)
    path = save(fig, "fig7_attention.pdf")
    T = attn.shape[1]
    print(f"      엔트로피: 경보 {ep.mean():.4f} "
          f"(유효 프레임 {effective_frames(ep.mean(), T):.0f}/{T}), "
          f"정상 {en.mean():.4f} "
          f"(유효 프레임 {effective_frames(en.mean(), T):.0f}/{T})")
    print(f"      차이의 95% 신뢰구간이 0을 포함하지 않는 프레임: "
          f"{T}개 중 {int(sig.sum())}개")
    return path


def run(seed=None, model_name=None, do_deletion=True, per_head=True):
    seed = seed if seed is not None else config.EVAL_SEEDS[-1]
    model_name = model_name or PROPOSED_MODEL
    device = get_device()

    print("\n" + "=" * 72)
    print("해석 및 추론 지연시간")
    print("=" * 72)
    print(f"  device {device} | model {display(model_name)} | seed {seed}")
    print(f"  융합 방식 {config.ATTENTION_FUSION}")

    env_report.set_deterministic(seed)
    _, _, test_loader, meta = build_dataloaders(batch_size=1, num_workers=0,
                                                seed=seed, verbose=False)
    ttf_scale = meta["ttf_scale"]

    model = MODEL_REGISTRY[model_name]().to(device)
    ckpt = checkpoint_path(model_name, seed, tag="main")
    if os.path.exists(ckpt):
        payload = torch.load(ckpt, map_location=device)
        model.load_state_dict(payload.get("state_dict", payload))
        print(f"  가중치 로드: {ckpt}")
    else:
        print(f"  경고: {ckpt}를 찾을 수 없습니다. 아래 결과는 학습되지 않은 "
              f"모델에서 나온 것이므로 해석하면 안 됩니다.")

    result = {"model": model_name, "seed": seed,
              "checkpoint_exists": os.path.exists(ckpt),
              "attention_fusion": config.ATTENTION_FUSION}

    y_true, y_score, _, _, _, attn = predict(model, model_name, test_loader, device,
                                             collect_attn=True, ttf_scale=ttf_scale)
    entropy_stats = None
    if attn is not None and len(attn) > 0:
        T = attn.shape[-1]
        pos, neg = attn[y_true == 1.0], attn[y_true == 0.0]
        print(f"\n  어텐션 수집: 경보 {len(pos)}개 / 정상 {len(neg)}개 세그먼트")
        if len(pos) > 1 and len(neg) > 1:
            ep, en = attention_entropy(pos), attention_entropy(neg)
            t, p = stats.ttest_ind(ep, en, equal_var=False)
            entropy_stats = {
                "warning_entropy_mean": float(ep.mean()),
                "warning_entropy_std": float(ep.std()),
                "normal_entropy_mean": float(en.mean()),
                "normal_entropy_std": float(en.std()),
                "warning_effective_frames": effective_frames(ep.mean(), T),
                "normal_effective_frames": effective_frames(en.mean(), T),
                "n_frames": int(T),
                "welch_t": float(t), "p_value": float(p),
                "warning_narrower": bool(ep.mean() < en.mean()),
                "significant_005": bool((p == p) and p < 0.05),
                "head_averaged": True}
            result["attention_entropy"] = entropy_stats
            print(f"  정규화 엔트로피(헤드 평균): 경보 {ep.mean():.4f} 대 "
                  f"정상 {en.mean():.4f} (Welch t검정 p = {p:.3g})")
            print(f"  유효 프레임 수: 경보 "
                  f"{entropy_stats['warning_effective_frames']:.0f}/{T}, "
                  f"정상 {entropy_stats['normal_effective_frames']:.0f}/{T}")
            # [수정 v2.1] 예전에는 이 세 줄이 무조건 출력되어, 검정 결과와
            #   무관하게 "유의하게 좁아진다"고 단정했다. 점검 실행에서는
            #   p = 0.7 이고 경보 쪽 엔트로피가 오히려 더 높았는데도
            #   같은 문장이 찍혔다. 이제 p 값과 방향을 함께 보고 서술한다.
            narrower = ep.mean() < en.mean()
            signif = (p == p) and (p < 0.05)
            frac = entropy_stats["warning_effective_frames"] / T
            if signif and narrower:
                print(f"  -> 경보 구간에서 어텐션이 유의하게 좁아집니다 (p = {p:.3g}).")
                print(f"     다만 유효 프레임이 여전히 전체의 {100*frac:.0f}% 이므로,")
                print("     '특정 구간으로 모인다'가 아니라 '유의하게 좁아지지만")
                print("     여전히 넓게 분포한다'가 데이터에 맞는 서술입니다.")
            elif signif and not narrower:
                print(f"  -> 경보 구간에서 어텐션이 오히려 유의하게 넓어집니다 "
                      f"(p = {p:.3g}).")
                print("     '경보 시 특정 구간에 집중한다'는 서술은 이 데이터로")
                print("     뒷받침되지 않습니다. 관측된 방향 그대로 쓰세요.")
            else:
                print(f"  -> 경보와 정상의 엔트로피 차이가 유의하지 않습니다 "
                      f"(p = {p:.3g}).")
                print(f"     유효 프레임이 전체의 {100*frac:.0f}% 로 넓게 퍼져 있습니다.")
                print("     집중도에 관한 주장은 하지 말고, 아래 삭제 실험을")
                print("     3.3절의 근거로 쓰세요. (엔트로피는 어텐션이 '무엇을'")
                print("     보는지가 아니라 '얼마나 퍼져 있는지'만 말해 줍니다.)")

            cp = float((pos * np.arange(T)).sum(-1).mean() / max(T - 1, 1))
            cn = float((neg * np.arange(T)).sum(-1).mean() / max(T - 1, 1))
            result["attention_centroid"] = {"warning": cp, "normal": cn}
            print(f"  어텐션 질량중심 (0=시작, 1=끝): "
                  f"경보 {cp:.3f}, 정상 {cn:.3f}")
        figure_attention(attn, y_true, entropy_stats)

        # [B-15] 헤드별 엔트로피 — 헤드 평균은 집중도의 상한이다
        if per_head and returns_attention(model_name):
            _, _, _, _, _, attn_h = predict(model, model_name, test_loader, device,
                                            collect_attn=True, ttf_scale=ttf_scale,
                                            per_head_attn=True)
            if attn_h is not None and attn_h.ndim == 3:
                eh = attention_entropy(attn_h)                # (N, H)
                ph = eh[y_true == 1.0].mean(axis=0)
                nh = eh[y_true == 0.0].mean(axis=0)
                result["attention_entropy_per_head"] = {
                    "warning": ph.tolist(), "normal": nh.tolist(),
                    "warning_min": float(ph.min()), "normal_min": float(nh.min())}
                print(f"\n  헤드별 엔트로피 (경보): "
                      f"{np.array2string(ph, precision=4)}")
                print(f"  헤드별 엔트로피 (정상): "
                      f"{np.array2string(nh, precision=4)}")
                print(f"  가장 집중된 헤드: 경보 {ph.min():.4f} "
                      f"(유효 프레임 {effective_frames(ph.min(), T):.0f}/{T})")
                print("  헤드 평균은 서로 다른 헤드를 섞어 평평해지므로,")
                print("  헤드별 값이 집중도를 더 정확히 보여줍니다.")
    else:
        print("  이 모델에는 어텐션 가중치가 없습니다.")

    # ------------------------------------------------------------------
    # 삭제 실험 [심사 지적 24]
    # ------------------------------------------------------------------
    if do_deletion and returns_attention(model_name):
        print("\n  어텐션 삭제 실험 (상위 k 프레임 vs 무작위 k 프레임)")
        dele = deletion_test(model, model_name, test_loader, device,
                             ttf_scale=ttf_scale, seed=seed)
        if dele:
            result["deletion_test"] = dele
            print(f"    {'k':>4s} {'상위 마스킹':>12s} {'무작위 마스킹':>14s} "
                  f"{'비율':>7s} {'p':>10s}")
            for k in sorted(dele):
                r = dele[k]
                print(f"    {k:>4d} {r['mean_abs_change_top']:>12.4f} "
                      f"{r['mean_abs_change_random']:>14.4f} "
                      f"{r['ratio']:>7.2f} {r['p_value']:>10.3g}")
            best = max(dele.values(), key=lambda r: r["ratio"])
            if best["p_value"] < 0.05 and best["ratio"] > 1.0:
                print("    -> 어텐션이 가리킨 프레임을 가릴 때 예측이 유의하게 더")
                print("       크게 바뀝니다. 가중치가 실제 판단 근거를 가리킨다는")
                print("       직접 증거이며, 3.3절의 핵심 논거로 쓸 수 있습니다.")
            else:
                print("    -> 상위 프레임과 무작위 프레임의 차이가 유의하지")
                print("       않습니다. '해석 가능성'을 기여로 내세우기 어려우니")
                print("       서술을 낮추거나 다른 해석 근거를 제시하세요.")
            figure_deletion(dele)

    # ------------------------------------------------------------------
    print("\n  forward pass 지연시간")
    lat = measure_latency(model, model_name, test_loader, device)
    result["latency_forward_only"] = lat
    print(f"    표본 {lat['n_samples']}개, 평균 {lat['mean_ms']:.3f} ms "
          f"(표준편차 {lat['std_ms']:.3f}); p50/p95/p99 "
          f"{lat['p50_ms']:.3f}/{lat['p95_ms']:.3f}/{lat['p99_ms']:.3f} ms")

    print("\n  전체 파이프라인 지연시간 (STFT + 상태 벡터 + 전송 + 추론)")
    e2e = measure_end_to_end_latency(model, model_name, device)
    result["latency_end_to_end"] = e2e
    print(f"    STFT          {e2e['stft_mean_ms']:.3f} ms")
    print(f"    상태 벡터     {e2e['state_feature_mean_ms']:.3f} ms   "
          f"<- 기존 측정에서 누락되어 있던 항목 (검토 A-4)")
    print(f"    텐서 전송     {e2e['transfer_mean_ms']:.3f} ms")
    print(f"    모델 추론     {e2e['inference_mean_ms']:.3f} ms")
    print(f"    ─────────────────────────")
    print(f"    합계          {e2e['end_to_end_mean_ms']:.3f} ms "
          f"(p95 {e2e['end_to_end_p95_ms']:.3f} ms)")
    print(f"    세그먼트 길이 {e2e['segment_duration_ms']:.2f} ms, "
          f"실시간 배수 {e2e['realtime_factor']:.2f}배 "
          f"[{e2e.get('segment_duration_source', '공칭')}]")
    if e2e.get("segment_duration_source") == "실측":
        print(f"    (공칭 {e2e['segment_duration_nominal_ms']:.2f} ms 가 아니라 "
              f"실측 경과시간을 분모로 썼습니다)")
    print(f"    측정 환경     {e2e['device']}")
    print(f"    {e2e['note']}")
    if e2e["realtime_factor"] <= 1:
        print("    파이프라인이 데이터 취득보다 느립니다. 실시간 처리가 가능하다고 "
              "주장하지 마세요.")
    print("    논문에는 네 항목의 합과 측정 하드웨어를 함께 적으세요.")

    out = os.path.join(config.RESULT_DIR, "xai_latency.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  저장: {out}")
    print("\n  참고: 본 결과는 실험실 화강암 마찰 데이터에서 얻은 것이며 현장 "
          "계측으로 검증되지 않았습니다. 이 문장은 논문 한계 절에 반드시 "
          "포함되어야 합니다.")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-deletion", action="store_true")
    ap.add_argument("--no-per-head", action="store_true")
    args = ap.parse_args()
    run(seed=args.seed, model_name=args.model,
        do_deletion=not args.no_deletion, per_head=not args.no_per_head)
