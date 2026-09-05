"""파이프라인 전역 설정.

[검토 반영 v2]
검토에서 지적된 항목은 FIX_* / *_MODE 플래그로 노출한다. 기본값은 모두
"수정된 동작"이며, 기존 결과를 재현해야 할 때만 플래그를 되돌린다.
어떤 플래그가 어떤 지적에 대응하는지는 CHANGES.md 참조.
"""

import os

# ----------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------
RAW_CSV = "./data_raw/train.csv"
ARTIFACT_DIR = "./artifacts"
RESULT_DIR = "./results"
FIGURE_DIR = "./figures"

for _d in (ARTIFACT_DIR, RESULT_DIR, FIGURE_DIR):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------
# 신호 처리
# ----------------------------------------------------------------------
SEGMENT_SIZE = 150_000        # 4 MHz 기준 37.5 ms
SAMPLING_RATE = 4_000_000
NPERSEG = 3000                # 주파수 bin 1501개
NOVERLAP = 1500               # 50% 중첩, 시간 프레임 101개
FREQ_BINS = 1501
TIME_STEPS = 101
MAX_SEGMENTS = 4195

# [수정] 점검 실행(run_all.py --smoke)에 쓰는 세그먼트 수.
#   LANL 데이터는 스틱슬립 사이클 하나가 약 250 세그먼트이므로,
#   400개로는 완결된 사이클이 2개뿐이라 학습/검증/평가 3분할이 성립하지
#   않았다(검증·평가가 비어 predict 에서 죽음). 5개 이상을 확보한다.
SMOKE_SEGMENTS = 1800

# 주파수 격자 (표기용 파생값)
FREQ_RESOLUTION_HZ = SAMPLING_RATE / NPERSEG          # 1,333.33 Hz
NYQUIST_HZ = SAMPLING_RATE / 2                        # 2 MHz


def h5_path_for(n_segments):
    """파일명에 세그먼트 수를 포함시켜, 다른 조건으로 만든 데이터셋이
    의도치 않게 재사용되는 것을 막는다."""
    return f"./data_processed_n{n_segments}.h5"


H5_PATH = h5_path_for(MAX_SEGMENTS)

# ----------------------------------------------------------------------
# 합성 저주파 간섭
# ----------------------------------------------------------------------
# [심사 지적 05 / 검토 D-x] bin 인덱스는 0부터 시작하므로 8개 bin(0..7)의
# 실제 상한 경계는 7.5 x df = 10.0 kHz 이다. 기존 주석의 "0-10.66 kHz"는
# 8 x df = bin 8 의 중심으로, 9개 bin 을 써야 나오는 값이었다.
# 논문 표기는 아래 NOISE_BAND_LABEL 을 그대로 사용할 것.
NOISE_BAND_BINS = 8
NOISE_MEAN = 12.0
NOISE_STD = 1.5
NOISE_SEED = 20240101

# 실측 TBM 스펙트럼이 아니라 가상 시나리오이다. 논문에도 그렇게 서술할 것.
NOISE_IS_SYNTHETIC = True


def noise_band_bounds_hz(n_bins=None):
    """주입 대역의 (중심 최댓값, 상한 경계) 를 Hz 로 반환."""
    n = NOISE_BAND_BINS if n_bins is None else n_bins
    top_center = (n - 1) * FREQ_RESOLUTION_HZ
    top_edge = (n - 0.5) * FREQ_RESOLUTION_HZ
    return top_center, top_edge


NOISE_BAND_LABEL = "DC – {:.1f} kHz".format(noise_band_bounds_hz()[1] / 1000.0)

# ----------------------------------------------------------------------
# 라벨
# ----------------------------------------------------------------------
WARNING_TTF_THRESHOLD = 3.0   # TTF 3초 이하를 경보 클래스로 정의

# [검토 B-10] 정의만 되고 쓰이지 않던 민감도 목록. 이제 step3 의
# --label-threshold 로 실제 실행 가능하다. y_cls 는 y_ttf 에서 로딩 시점에
# 다시 계산하므로 데이터셋 재생성이 필요 없다.
THRESHOLD_SENSITIVITY = [2.0, 3.0, 5.0]

# ----------------------------------------------------------------------
# 전조 상태 벡터
# ----------------------------------------------------------------------
# 인덱스 0-3 은 단일 37.5 ms 세그먼트 내부에서 계산되고(intra),
# 인덱스 4-5 는 직전 TREND_WINDOW 개 세그먼트를 요약한다(inter).
_BASE_STATE_FEATURES = [
    "rms_energy",
    "kurtosis",
    "peak_amplitude",
    "crest_factor",
    "energy_trend_slope",
    "cumulative_energy",
]

# [검토 B-6 / 심사 지적 11] cumulative_energy 는 사이클 누적이 아니라
# 0.75초 이동합이다. 이 플래그를 켜면 "사이클 시작부터의 진짜 누적합"이
# 7번째 특징으로 추가되어, 국소 이력만으로 충분한지를 ablation 으로
# 검증할 수 있다. 켜면 데이터셋을 재생성해야 한다.
ADD_CYCLE_CUMULATIVE_FEATURE = False

STATE_FEATURE_NAMES = list(_BASE_STATE_FEATURES)
if ADD_CYCLE_CUMULATIVE_FEATURE:
    STATE_FEATURE_NAMES.append("cycle_cumulative_energy")

STATE_DIM = len(STATE_FEATURE_NAMES)
INTRA_SEGMENT_IDX = [0, 1, 2, 3]
INTER_SEGMENT_IDX = [i for i in range(4, STATE_DIM)]
TREND_WINDOW = 20

TREND_WINDOW_SEC = TREND_WINDOW * SEGMENT_SIZE / SAMPLING_RATE   # 0.75 s

# [검토 C-3] 사이클 초반에는 이력이 TREND_WINDOW 보다 짧아 단순 합이
# 기계적으로 작아진다. 사이클 초반 = TTF 가 큰 구간이므로 라벨과 정렬된
# 인공 상관이 생긴다.
#   "scaled"  부분합을 전체 창 길이로 환산 (기본, 정보 손실 없음)
#   "exclude" 워밍업 구간을 아예 제외 (민감도 확인용, 약 7.7% 손실)
#   "raw"     기존 동작
INTER_WARMUP_POLICY = "scaled"

# 그림과 표에 사용할 표기
STATE_FEATURE_LABELS = {
    "rms_energy": "RMS amplitude",
    "kurtosis": "Kurtosis",
    "peak_amplitude": "Peak amplitude",
    "crest_factor": "Crest factor",
    "energy_trend_slope": "Energy trend slope",
    "cumulative_energy": "Windowed RMS sum",
    "cycle_cumulative_energy": "Cycle cumulative sum",
}
ABLATION_LABELS = {
    "full": "Full state vector",
    "drop_inter_segment": "w/o inter-segment (both)",
    "drop_intra_segment": "w/o intra-segment (all four)",
    "drop_all": "w/o state vector",
}

# [심사 지적 09] RMS 는 진폭 차원이고 에너지는 진폭의 제곱에 비례한다.
# 논문에서 "에너지"라 부를 경우 대리 지표임을 명시해야 한다.
# True 로 두면 RMS 대신 평균제곱(mean(x^2), 에너지 비례량)을 특징으로 쓴다.
# 켜면 데이터셋을 재생성해야 한다.
USE_TRUE_ENERGY_FEATURE = False

# ----------------------------------------------------------------------
# 데이터 분할
# ----------------------------------------------------------------------
CYCLE_JUMP_THRESHOLD = 1.0    # TTF 불연속으로 새 스틱슬립 사이클을 판별
EXCLUDE_INCOMPLETE_CYCLES = True

# [검토 C-2] 파괴가 세그먼트 내부에서 일어나면 그 세그먼트의 마지막 샘플
# TTF 는 이미 다음 사이클의 큰 값이다. 결과적으로 가장 에너지가 큰
# 세그먼트가 "정상 / 큰 TTF" 로 잘못 라벨된다. 사이클당 1개.
DROP_CYCLE_BOUNDARY_SEGMENT = True

TRAIN_RATIO = 0.60
VAL_RATIO = 0.15
MIN_TEST_POSITIVES = 30

USE_CROSS_VALIDATION = False
CV_N_FOLDS = 4
CV_TEST_CYCLES = 2
CV_VAL_CYCLES = 2

# ----------------------------------------------------------------------
# 타깃 정규화
# ----------------------------------------------------------------------
NORMALIZE_TTF = True
HP_SELECTION_METRIC = "composite"   # "composite"(검증 PR-AUC + R2) 또는 "val_loss"

# ----------------------------------------------------------------------
# 학습
# ----------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 15
TUNE_EPOCHS = 8
NUM_WORKERS = 2
PATIENCE = 5
GRAD_CLIP = 5.0

HP_GRID = {
    "lr": [1e-3, 3e-4],
    "ttf_weight": [0.25, 1.0, 4.0],
}

# [검토 A-8] DataLoader 의 generator 는 __iter__ 마다 상태가 전진하므로,
# 한 시드 안에서 여러 모델이 같은 로더를 재사용하면 배치 순서가 달라진다.
# 대응 t-검정의 전제("셔플링 잡음 공유")가 깨지고, 앞 모델의 조기 종료
# 시점이 뒤 모델 결과를 바꾸며, 재개 시 숫자가 달라진다.
FIX_LOADER_SEED_PER_MODEL = True

# [검토 B-16] 조기 종료를 검증 손실로 하면 w_ttf 가 큰 설정에서 사실상
# 회귀 성능만으로 에폭이 결정된다. 주 지표(PR-AUC)가 한 실행 안의 모델
# 선택에 전혀 개입하지 못한다.
#   "composite" 검증 PR-AUC + R2 (기본)
#   "val_loss"  기존 동작
EARLY_STOP_METRIC = "composite"

# [검토 B-17] 손실 세 항의 실제 비중을 기록한다.
LOG_LOSS_COMPONENTS = True

# ----------------------------------------------------------------------
# 손실 함수
# ----------------------------------------------------------------------
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# [검토 B-13] 기존 구현은 양성·음성에 같은 상수 alpha 를 곱해 클래스
# 재가중이 일어나지 않았고, 분류항 전체가 1/4 로 축소되는 부작용이 있었다.
#   "balanced" Lin et al. 원식의 alpha_t (양성 alpha, 음성 1-alpha)
#   "constant" 기존 동작 (상수 alpha)
#   "none"     alpha 를 쓰지 않음 (focusing 항만)
# 주의: 원식의 alpha=0.25 는 "희소 클래스가 전경"인 검출 문제 기준이라
# 소수 클래스를 낮춘다. 본 문제(양성 28%)에서는 alpha 를 0.5 이상으로
# 두어야 양성이 강조된다. 아래 기본값은 그에 맞춰 조정했다.
FOCAL_ALPHA_MODE = "balanced"
FOCAL_ALPHA_BALANCED = 0.6    # balanced 모드에서 양성에 주는 가중

LAMBDA_PHYSICS = 0.15
PURE_AE_BASELINE = True       # 스펙트로그램 전용 베이스라인은 에너지 가중항을 제외

# [검토 B-18] drop_all 은 "상태 입력 전체 제외" 인데, stress(=정규화 RMS)가
# 손실 가중치로 남아 상태 정보가 완전히 제거되지 않았다.
DROP_ALL_ALSO_DISABLES_PHYSICS = True

# ----------------------------------------------------------------------
# 모델 구조
# ----------------------------------------------------------------------
# [검토 B-14] 교차 어텐션 출력에 질의를 더하지 않으면 상태 벡터가
# 어텐션 가중치를 통해서만 출력에 도달한다. 단순 결합(concat)은 상태
# 벡터를 헤드에 직접 넘기므로, 잔차 없이는 동일 조건 비교가 아니다.
#   "add"    fusion = attn_out + query   (기본)
#   "concat" fusion = [attn_out ; query] (헤드 입력 2배)
#   "none"   기존 동작
ATTENTION_FUSION = "add"

# [검토 B-19] AdaptiveAvgPool 이 375 bin 을 4밴드로 평균한다(밴드당 약 94 bin).
# 논문의 "시간-주파수 구조를 정밀하게 반영"과 어긋나며, 스펙트로그램 전용
# 베이스라인이 이 병목 때문에 약해졌을 수 있다.
CNN_FREQ_BANDS = 4
ATTENTION_HEADS = 4
D_MODEL = 128
LSTM_LAYERS = 2

# 결합 방식 변형은 경쟁 베이스라인이 아니라 제안 모델의 융합 ablation 이다.
INCLUDE_FUSION_ABLATION = True

# ----------------------------------------------------------------------
# 시드
# ----------------------------------------------------------------------
TUNE_SEED = 2024              # 평가 시드와 분리
EVAL_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 202, 303, 404, 505, 606]
ABLATION_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]

CLASSICAL_BOOTSTRAP = True

# [검토 B-2] 심층 모델의 시드는 초기화 시드이고 고전 모델의 시드는 학습셋
# 부트스트랩 시드다. 두 계열 사이에 공유 잡음원이 없으므로 대응 검정의
# 전제가 성립하지 않는다.
#   "welch"  고전 베이스라인 비교는 비대응 Welch 검정 (기본)
#   "paired" 기존 동작
CLASSICAL_TEST = "welch"

# ----------------------------------------------------------------------
# 운영점
# ----------------------------------------------------------------------
TARGET_FAR_PCT = 15.0
FAR_SAFETY_FACTOR = 0.6
THRESHOLD_BOOTSTRAP_N = 300

# [검토 A-6 / 심사 지적 19] 기존 [5,10,15,20] 격자로는 "12-13% 교차점"을
# 특정할 수 없었다. 캐시된 예측만으로 재계산되므로 비용이 거의 없다.
MATCHED_FAR_GRID = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]

PRIMARY_CLS_METRIC = "pr_auc"
BOOTSTRAP_N = 1000

# [검토 00-B] bootstrap_ci 가 구현되어 있으나 어디서도 호출되지 않았다.
REPORT_BOOTSTRAP_CI = True

# [검토 C-14] 사이클 간 계통 편향 판정 기준이 metrics 와 step8 에서 서로
# 달랐다. 하나로 통일한다.
SYSTEMATIC_BIAS_RULE = 0.5    # bias 표준편차 > RULE x 통합 RMSE 이면 계통적

# ----------------------------------------------------------------------
# 그림 서식
# ----------------------------------------------------------------------
FIG_DPI = 600
FIG_FONT = ["Arial", "Helvetica", "DejaVu Sans"]
FIG_WARN_ON_FONT_FALLBACK = True     # [검토 C-13] 무음 대체 방지
FIG_BASE_FONTSIZE = 8
FIG_SINGLE_COL_IN = 3.5
FIG_DOUBLE_COL_IN = 7.2

# [검토 C-9 / 심사 지적 06] 스펙트로그램 그림.
# 기존: cmap="magma", vmin=0, vmax=최댓값  ->  대부분의 화소가 검게 눌림.
# 수정: 백색 배경 계열 컬러맵 + 백분위 스케일링.
#   "Greys"     흰 배경 · 검은 신호 (인쇄 안전, 기본)
#   "magma_r"   크림 배경 · 어두운 신호 (지각 균등, 색맹 안전)
#   "YlGnBu"    흰-노랑 배경 · 청색 신호
#   "cividis_r" 색맹 안전 대안
FIG_SPEC_CMAP = "Greys"
FIG_SPEC_PCT_LO = 2.0        # 하위 백분위를 배경(흰색)으로
FIG_SPEC_PCT_HI = 99.5       # 상위 백분위를 최대 농도로
FIG_SPEC_DIFF_CMAP = "RdBu_r"
FIG_SPEC_FREQ_BINS_SHOWN = 120   # 1501개 중 하위 120개(≈160 kHz)만 표시

# Okabe-Ito 색맹 안전 팔레트
PALETTE = {
    "proposed": "#0072B2",
    "spectrogram_only": "#D55E00",
    "no_attention": "#009E73",
    "gbm": "#CC79A7",
    "logreg": "#666666",
    "neutral": "#999999",
    "highlight": "#B22222",
}
