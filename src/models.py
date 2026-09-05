"""모델 정의와 다중 과제 손실함수.

제안 신경망은 두 시간 스케일을 융합한다. 빠른 분기는 단일 37.5 ms 세그먼트
내부의 스펙트로-템포럴 구조를 인코딩하고, 느린 분기는 직전 약 0.75초를
요약한 전조 상태를 인코딩한다. 빠른 분기는 한 번에 한 세그먼트만 관측하므로
세그먼트 간 이력에 접근할 수 없으며, 따라서 느린 분기는 스펙트로그램에 없는
정보를 제공한다. 교차 어텐션은 느린 상태를 질의로, 빠른 시퀀스를 키와 값으로
사용한다.

손실에 들어가는 에너지 가중치는 상태 벡터와 별도로 전달하여, 특징 ablation이
목적함수를 교란하지 않도록 한다.

[검토 반영 v2]
  B-13  focal 항의 alpha 를 Lin et al. 원식의 alpha_t 로 구현. 기존 상수
        alpha 는 클래스 재가중을 하지 않으면서 분류항 전체를 1/4 로
        축소시키는 부작용만 있었다.
  B-14  교차 어텐션 출력에 질의를 더하는 잔차 경로 추가. 잔차가 없으면
        상태 벡터가 어텐션 가중치를 통해서만 출력에 도달하여, 상태 벡터를
        헤드에 직접 넘기는 단순 결합과 동일 조건 비교가 되지 않는다.
  B-19  주파수 밴드 병목(4밴드)을 설정 가능하게 노출.
"""

import os

import torch
import torch.nn as nn

import config
from models_meta import SINGLE_AE, NO_ATTN, CROSS_ATTN, PROPOSED_MODEL  # noqa: F401

STATE_DIM = config.STATE_DIM
_CNN_CHANNELS = 64
_CNN_FLAT = _CNN_CHANNELS * config.CNN_FREQ_BANDS


def _ae_cnn_backbone():
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 1)),
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 1)),
        nn.Conv2d(32, _CNN_CHANNELS, kernel_size=3, padding=1),
        nn.BatchNorm2d(_CNN_CHANNELS), nn.ReLU(),
        nn.AdaptiveAvgPool2d((config.CNN_FREQ_BANDS, config.TIME_STEPS)),
    )


def _encode_fast(cnn, lstm, x_ae):
    """(B,1,F,T) -> (B, T, d_model) 프레임별 표현."""
    b = x_ae.size(0)
    feats = cnn(x_ae).reshape(b, _CNN_FLAT, config.TIME_STEPS).permute(0, 2, 1)
    seq, _ = lstm(feats)
    return seq


class CrossTimescaleAttentionNN(nn.Module):
    """제안 모델(교차 시간 스케일 융합)."""

    def __init__(self, d_model=None, nhead=None, num_lstm_layers=None,
                 state_dim=STATE_DIM, fusion=None):
        super().__init__()
        d_model = config.D_MODEL if d_model is None else d_model
        nhead = config.ATTENTION_HEADS if nhead is None else nhead
        num_lstm_layers = config.LSTM_LAYERS if num_lstm_layers is None else num_lstm_layers
        self.fusion_mode = config.ATTENTION_FUSION if fusion is None else fusion

        self.ae_cnn = _ae_cnn_backbone()
        self.ae_lstm = nn.LSTM(input_size=_CNN_FLAT, hidden_size=d_model,
                               num_layers=num_lstm_layers, batch_first=True)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, d_model), nn.ReLU())
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                                batch_first=True)
        head_in = d_model * 2 if self.fusion_mode == "concat" else d_model
        self.cls_head = nn.Sequential(nn.Linear(head_in, 32), nn.ReLU(), nn.Linear(32, 1))
        self.ttf_head = nn.Sequential(nn.Linear(head_in, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_ae, x_state, return_head_attn=False):
        ae_seq = _encode_fast(self.ae_cnn, self.ae_lstm, x_ae)
        query = self.state_mlp(x_state).unsqueeze(1)          # (B, 1, d)

        # average_attn_weights=False 로 받으면 (B, heads, 1, T) 가 나온다.
        # 보고용 가중치는 헤드 평균을 쓰되, 헤드별 분석이 필요하면 함께 반환.
        attn_out, attn_w = self.cross_attn(query, ae_seq, ae_seq,
                                           average_attn_weights=not return_head_attn)

        q = query.squeeze(1)
        a = attn_out.squeeze(1)
        if self.fusion_mode == "add":
            fusion = a + q                    # [B-14] 잔차 경로
        elif self.fusion_mode == "concat":
            fusion = torch.cat([a, q], dim=1)
        else:                                  # "none" — 기존 동작
            fusion = a

        return self.cls_head(fusion), self.ttf_head(fusion), attn_w


class SingleAENN(nn.Module):
    """스펙트로그램만 사용하는 베이스라인."""

    def __init__(self, d_model=None, num_lstm_layers=None):
        super().__init__()
        d_model = config.D_MODEL if d_model is None else d_model
        num_lstm_layers = config.LSTM_LAYERS if num_lstm_layers is None else num_lstm_layers
        self.ae_cnn = _ae_cnn_backbone()
        self.ae_lstm = nn.LSTM(input_size=_CNN_FLAT, hidden_size=d_model,
                               num_layers=num_lstm_layers, batch_first=True)
        self.cls_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.ttf_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_ae):
        ae_seq = _encode_fast(self.ae_cnn, self.ae_lstm, x_ae)
        last = ae_seq[:, -1, :]
        return self.cls_head(last), self.ttf_head(last)


class NoAttentionFusionNN(nn.Module):
    """융합 ablation: 교차 어텐션을 단순 결합으로 대체.

    제안 모델과의 유일한 차이가 융합 연산자이므로, 이 비교가 어텐션의 기여를
    분리해낸다. 경쟁 베이스라인이 아니라 제안 모델의 ablation이다.
    """

    def __init__(self, d_model=None, num_lstm_layers=None, state_dim=STATE_DIM):
        super().__init__()
        d_model = config.D_MODEL if d_model is None else d_model
        num_lstm_layers = config.LSTM_LAYERS if num_lstm_layers is None else num_lstm_layers
        self.ae_cnn = _ae_cnn_backbone()
        self.ae_lstm = nn.LSTM(input_size=_CNN_FLAT, hidden_size=d_model,
                               num_layers=num_lstm_layers, batch_first=True)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, d_model), nn.ReLU())
        self.cls_head = nn.Sequential(nn.Linear(d_model * 2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.ttf_head = nn.Sequential(nn.Linear(d_model * 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_ae, x_state):
        ae_seq = _encode_fast(self.ae_cnn, self.ae_lstm, x_ae)
        last = ae_seq[:, -1, :]
        ctx = self.state_mlp(x_state)
        fusion = torch.cat([last, ctx], dim=1)
        return self.cls_head(fusion), self.ttf_head(fusion)


_ALL_CLASSES = {
    SINGLE_AE: SingleAENN,
    NO_ATTN: NoAttentionFusionNN,
    CROSS_ATTN: CrossTimescaleAttentionNN,
}

from models_meta import DEEP_MODEL_NAMES  # noqa: E402

MODEL_REGISTRY = {n: _ALL_CLASSES[n] for n in DEEP_MODEL_NAMES}

_TAKES_STATE = {SINGLE_AE: False, NO_ATTN: True, CROSS_ATTN: True}
_RETURNS_ATTN = {SINGLE_AE: False, NO_ATTN: False, CROSS_ATTN: True}


def takes_state(name):
    return _TAKES_STATE[name]


def returns_attention(name):
    return _RETURNS_ATTN[name]


def forward_model(name, model, x_ae, x_state, **kw):
    """모델별 forward 차이를 흡수하여 (분류 로짓, TTF, 어텐션 또는 None)을 반환."""
    if not _TAKES_STATE[name]:
        c, t = model(x_ae)
        return c, t, None
    if _RETURNS_ATTN[name]:
        return model(x_ae, x_state, **kw)
    c, t = model(x_ae, x_state)
    return c, t, None


def uses_physics_penalty(name):
    """스펙트로그램 전용 베이스라인은 상태 가중항을 사용하지 않는다.

    stress 가 상태 벡터의 0번 특징이므로, 이 항을 넣으면 "스펙트로그램만
    사용" 이라는 베이스라인 정의가 손실을 통해 깨진다. 이 예외는 의도된
    것이며, 논문 본문에도 명시해야 한다.
    """
    if config.PURE_AE_BASELINE and name == SINGLE_AE:
        return False
    return True


def checkpoint_path(name, seed, tag="main", root=None):
    root = root or config.ARTIFACT_DIR
    safe = name.replace(" ", "_").replace("/", "_")
    return os.path.join(root, f"{tag}__{safe}__seed{seed}.pth")


class PhysicsInformedMultiTaskLoss(nn.Module):
    """Focal 분류항, 정규화된 회귀항, 그리고 음향 에너지가 클 때 회귀 오차의
    비용을 키우는 에너지 가중 페널티로 구성된다.

    지배방정식 제약이 아니라 도메인 지식 기반 가중이며, 그 기여는 5단계의
    손실 ablation으로 검증한다.

    [검토 B-13] alpha 처리 방식을 config.FOCAL_ALPHA_MODE 로 선택한다.
      balanced : alpha_t = alpha (y=1), 1-alpha (y=0)   — Lin et al. 원식
      constant : alpha 를 전체에 곱함                     — 기존 구현
      none     : alpha 를 쓰지 않음                       — focusing 항만
    """

    def __init__(self, alpha=None, gamma=None, lambda_p=None, ttf_weight=1.0,
                 alpha_mode=None):
        super().__init__()
        self.alpha_mode = config.FOCAL_ALPHA_MODE if alpha_mode is None else alpha_mode
        if alpha is None:
            alpha = (config.FOCAL_ALPHA_BALANCED if self.alpha_mode == "balanced"
                     else config.FOCAL_ALPHA)
        self.alpha = alpha
        self.gamma = config.FOCAL_GAMMA if gamma is None else gamma
        self.lambda_p = config.LAMBDA_PHYSICS if lambda_p is None else lambda_p
        self.ttf_weight = ttf_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss()

    def _alpha_factor(self, y_cls):
        if self.alpha_mode == "balanced":
            return self.alpha * y_cls + (1.0 - self.alpha) * (1.0 - y_cls)
        if self.alpha_mode == "none":
            return torch.ones_like(y_cls)
        return torch.full_like(y_cls, self.alpha)

    def forward(self, pred_cls, pred_ttf, y_cls, y_ttf, stress):
        bce = self.bce(pred_cls, y_cls)
        a_t = self._alpha_factor(y_cls)
        if self.gamma > 0:
            p = torch.sigmoid(pred_cls)
            p_t = p * y_cls + (1 - p) * (1 - y_cls)
            loss_cls = torch.mean(a_t * (1 - p_t) ** self.gamma * bce)
        else:
            loss_cls = torch.mean(a_t * bce)

        loss_ttf = self.mse(pred_ttf, y_ttf)

        if self.lambda_p > 0:
            physics = torch.mean(stress * torch.log1p(torch.abs(pred_ttf - y_ttf)))
        else:
            physics = torch.zeros((), device=pred_ttf.device)

        total = loss_cls + self.ttf_weight * loss_ttf + self.lambda_p * physics
        return total, loss_cls, loss_ttf, physics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_report():
    """세 모델의 파라미터 수와 두 방향의 상대비를 함께 반환.

    [검토 D-6] 논문의 "파라미터 15.5% 적게" 는 분모가 반대였다.
    (CTPF - noattn)/CTPF = 13.4% 가 "단순 결합이 적게 쓴다" 의 값이고,
    15.5% 는 (CTPF - noattn)/noattn, 즉 "CTPF 가 더 쓴다" 의 값이다.
    """
    out = {}
    for name, cls in _ALL_CLASSES.items():
        try:
            out[name] = count_parameters(cls())
        except Exception as e:      # pragma: no cover
            out[name] = f"unavailable ({e})"
    if isinstance(out.get(CROSS_ATTN), int) and isinstance(out.get(NO_ATTN), int):
        a, b = out[CROSS_ATTN], out[NO_ATTN]
        out["_reduction_vs_proposed_pct"] = round(100.0 * (a - b) / a, 2)
        out["_increase_vs_ablation_pct"] = round(100.0 * (a - b) / b, 2)
    return out


if __name__ == "__main__":
    rep = parameter_report()
    print("파라미터 수")
    for k, v in rep.items():
        if not k.startswith("_"):
            print(f"  {k:24s} {v:>10,}")
    if "_reduction_vs_proposed_pct" in rep:
        print(f"\n  단순 결합이 제안 모델보다 적게 쓰는 비율 : "
              f"{rep['_reduction_vs_proposed_pct']:.2f}%   <- 논문에 쓸 값")
        print(f"  제안 모델이 단순 결합보다 더 쓰는 비율   : "
              f"{rep['_increase_vs_ablation_pct']:.2f}%   <- 기존 원고의 15.5%")
