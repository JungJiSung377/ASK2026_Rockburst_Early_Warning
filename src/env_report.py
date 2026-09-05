"""재현성 보고를 위한 실행 환경 기록.

패키지 버전, 가속기 정보, 실험 설정을 기록한다.

여기서 사용하는 일부 CUDA 커널, 특히 adaptive average pooling의 역전파는
결정론적 구현이 없어 시드를 고정해도 비트 단위 재현이 보장되지 않는다.
반복 시드로 그 변동을 정량화한다.
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import config

DETERMINISM_NOTE = (
    "adaptive average pooling의 역전파는 결정론적 CUDA 구현이 없어 시드를 "
    "고정해도 비트 단위 재현이 보장되지 않는다. 반복 시드의 표준편차로 그 "
    "변동을 정량화한다."
)


def _safe(fn, default="unavailable"):
    try:
        return fn()
    except Exception:
        return default


def collect_environment():
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "determinism_note": DETERMINISM_NOTE,
    }

    pkgs = {}
    for name in ["torch", "numpy", "scipy", "pandas", "h5py", "sklearn",
                 "matplotlib", "seaborn"]:
        def _v(n=name):
            mod = __import__(n)
            return getattr(mod, "__version__", "unknown")
        pkgs[name] = _safe(_v)
    info["packages"] = pkgs

    gpu = {}
    try:
        import torch
        gpu["cuda_available"] = torch.cuda.is_available()
        gpu["torch_cuda_version"] = torch.version.cuda
        gpu["cudnn_version"] = _safe(lambda: torch.backends.cudnn.version())
        if torch.cuda.is_available():
            gpu["device_count"] = torch.cuda.device_count()
            gpu["device_names"] = [torch.cuda.get_device_name(i)
                                   for i in range(torch.cuda.device_count())]
            props = torch.cuda.get_device_properties(0)
            gpu["device0_total_memory_GB"] = round(props.total_memory / 1024 ** 3, 2)
            gpu["device0_capability"] = f"{props.major}.{props.minor}"
    except Exception as e:
        gpu["error"] = str(e)
    info["gpu"] = gpu

    info["nvidia_smi"] = _safe(
        lambda: subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL
        ).decode().strip()
    )

    info["experiment_config"] = {
        k: getattr(config, k) for k in [
            "SEGMENT_SIZE", "SAMPLING_RATE", "NPERSEG", "NOVERLAP", "MAX_SEGMENTS",
            "WARNING_TTF_THRESHOLD", "STATE_FEATURE_NAMES", "TREND_WINDOW",
            "TRAIN_RATIO", "VAL_RATIO", "EXCLUDE_INCOMPLETE_CYCLES",
            "BATCH_SIZE", "EPOCHS", "HP_GRID", "TUNE_SEED", "EVAL_SEEDS",
            "NOISE_MEAN", "NOISE_STD", "NOISE_BAND_BINS", "NOISE_SEED",
            "FOCAL_ALPHA", "FOCAL_GAMMA", "LAMBDA_PHYSICS", "PURE_AE_BASELINE",
            "TARGET_FAR_PCT", "FAR_SAFETY_FACTOR", "MATCHED_FAR_GRID",
            # [검토 반영 v2] 동작을 바꾸는 플래그는 반드시 기록에 남긴다
            "FIX_LOADER_SEED_PER_MODEL", "EARLY_STOP_METRIC",
            "FOCAL_ALPHA_MODE", "FOCAL_ALPHA_BALANCED", "ATTENTION_FUSION",
            "CNN_FREQ_BANDS", "DROP_CYCLE_BOUNDARY_SEGMENT",
            "INTER_WARMUP_POLICY", "ADD_CYCLE_CUMULATIVE_FEATURE",
            "USE_TRUE_ENERGY_FEATURE", "DROP_ALL_ALSO_DISABLES_PHYSICS",
            "CLASSICAL_TEST", "CLASSICAL_BOOTSTRAP", "USE_CROSS_VALIDATION",
        ]
    }
    return info


def set_deterministic(seed):
    """시드를 고정하고 가능한 범위에서 결정론적 커널을 활성화."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass
    except Exception:
        pass


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_sec = time.perf_counter() - self._start
        return False


def save_environment(path=None, extra=None):
    path = path or os.path.join(config.RESULT_DIR, "environment.json")
    info = collect_environment()
    if extra:
        info["run_notes"] = extra
    # 기존 기록을 덮어쓰지 않고 이력으로 누적한다.
    if os.path.exists(path):
        try:
            with open(path) as f:
                old = json.load(f)
            history = old.get("history", [])
            history.append({"timestamp": old.get("timestamp_utc"),
                            "run_notes": old.get("run_notes")})
            info["history"] = history[-20:]
        except Exception:
            pass
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    return path, info


if __name__ == "__main__":
    p, info = save_environment()
    print(json.dumps(info, indent=2, default=str)[:2000])
    print(f"\n환경 정보 저장: {p}")
