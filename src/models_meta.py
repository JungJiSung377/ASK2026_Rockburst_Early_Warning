"""모델 식별자와 표기, 색상 매핑."""

import config

# 내부 키
SINGLE_AE = "Single AE"
NO_ATTN = "Ours w/o Attention"
CROSS_ATTN = "Cross-Timescale Attn"
LOGREG = "LogReg/Ridge"
GBM = "GradientBoosting"

PROPOSED_MODEL = CROSS_ATTN

# 표기
DISPLAY_NAME = {
    SINGLE_AE: "Spectrogram-only",
    NO_ATTN: "CTPF w/o attention",
    CROSS_ATTN: "CTPF (proposed)",
    LOGREG: "Logistic / Ridge",
    GBM: "Gradient boosting",
}

SHORT_NAME = {
    SINGLE_AE: "Spec-only",
    NO_ATTN: "CTPF-concat",
    CROSS_ATTN: "CTPF",
    LOGREG: "LR/Ridge",
    GBM: "GBM",
}

COLOR_KEY = {
    SINGLE_AE: "spectrogram_only",
    NO_ATTN: "no_attention",
    CROSS_ATTN: "proposed",
    LOGREG: "logreg",
    GBM: "gbm",
}

DEEP_MODEL_NAMES = [SINGLE_AE, CROSS_ATTN]
if config.INCLUDE_FUSION_ABLATION:
    DEEP_MODEL_NAMES = [SINGLE_AE, NO_ATTN, CROSS_ATTN]

CLASSICAL_MODEL_NAMES = [LOGREG, GBM]
ALL_MODEL_NAMES = CLASSICAL_MODEL_NAMES + DEEP_MODEL_NAMES

# 주 비교에서는 융합 ablation 을 제외하고 별도로 보고한다.
MAIN_TABLE_MODELS = [SINGLE_AE, CROSS_ATTN]
ABLATION_TABLE_MODELS = [NO_ATTN] if config.INCLUDE_FUSION_ABLATION else []
BASELINE_MODELS = [m for m in ALL_MODEL_NAMES if m != PROPOSED_MODEL]

# [검토 C-12] 부트스트랩이 켜져 있으면 고전 추정기도 시드마다 예측이
# 달라진다. 이름만 보고 "결정론적"이라 보고하면 허위 서술이 된다.
# 실제 판정은 관측된 분산으로 하고, 이 집합은 부트스트랩이 꺼졌을 때만
# 참조한다.
_NOMINALLY_DETERMINISTIC = {LOGREG}


def deterministic_models():
    """현재 설정에서 실제로 시드 불변인 모델 집합."""
    if config.CLASSICAL_BOOTSTRAP:
        return set()
    return set(_NOMINALLY_DETERMINISTIC)


# 하위 호환 (기존 코드가 import 하던 이름)
DETERMINISTIC_MODELS = deterministic_models()


def is_classical(name):
    return name in CLASSICAL_MODEL_NAMES


def get_proposed_name():
    return PROPOSED_MODEL


def display(name):
    return DISPLAY_NAME.get(name, name)


def short(name):
    return SHORT_NAME.get(name, name)


def color(name):
    return config.PALETTE.get(COLOR_KEY.get(name, "neutral"), "#999999")


def is_fusion_ablation(name):
    return name == NO_ATTN
