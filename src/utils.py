"""직렬화 보조 함수.

numpy 스칼라는 JSON으로 직렬화되지 않으므로, 저장 전에 변환해야 재개용
진행 파일을 다시 읽을 때 시드 키가 정수로 유지된다.
"""

import json
import os

import numpy as np


def to_native(obj):
    """numpy 타입을 JSON 직렬화 가능한 파이썬 타입으로 재귀 변환."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_native(data), f, indent=2)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def restore_nan(rows, numeric_keys):
    """null로 저장된 NaN 값을 복원."""
    out = []
    for r in rows:
        r = dict(r)
        for k in numeric_keys:
            if k in r and r[k] is None:
                r[k] = np.nan
        out.append(r)
    return out


def _self_test():
    d = {"seed": np.int64(11), "pr_auc": np.float64(0.7234),
         "nan_val": np.float64(np.nan), "flag": np.bool_(True),
         "arr": np.array([1, 2, 3])}
    back = json.loads(json.dumps(to_native(d)))
    assert isinstance(back["seed"], int)
    assert isinstance(back["pr_auc"], float)
    assert back["nan_val"] is None
    assert back["arr"] == [1, 2, 3]

    import pandas as pd
    df = pd.DataFrame(restore_nan([back], ["nan_val", "pr_auc"]))
    assert df["seed"].dtype.kind in "iu"
    assert np.isnan(df["nan_val"].iloc[0])
    print("utils: 정상 (시드 dtype %s 유지)" % df["seed"].dtype)


if __name__ == "__main__":
    _self_test()
