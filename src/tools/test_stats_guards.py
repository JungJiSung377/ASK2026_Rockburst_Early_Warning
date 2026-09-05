"""통계 보고 경로의 회귀 시험.

점검 실행(run_all.py --smoke)에서 드러난 결함을 다시 잡기 위한 시험이다.
시드가 1개일 때 유의성 검정이 성립하지 않는데도 "유의함"으로 출력되던
문제가 핵심이다.

    python tools/test_stats_guards.py

torch/h5py 가 없는 환경에서도 통계 부분만 확인할 수 있도록, 필요한 경우에만
무거운 의존성을 최소 스텁으로 대체한다. 코랩처럼 실물이 설치된 환경에서는
스텁이 끼어들지 않고 실제 모듈을 그대로 시험한다.
"""

import os
import sys
import types
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def _stub_if_missing():
    """실제 모듈이 없을 때만 최소 스텁을 심는다."""
    stubbed = []
    try:
        import torch  # noqa: F401
    except ImportError:
        for name in ("torch", "torch.nn", "torch.optim",
                     "torch.utils", "torch.utils.data"):
            sys.modules[name] = types.ModuleType(name)
        t = sys.modules["torch"]
        t.nn, t.optim = sys.modules["torch.nn"], sys.modules["torch.optim"]
        t.utils = sys.modules["torch.utils"]
        t.utils.data = sys.modules["torch.utils.data"]
        t.Tensor = object
        t.no_grad = lambda: (lambda f: f)
        sys.modules["torch.nn"].Module = object
        for cls in ("DataLoader", "Dataset", "Subset"):
            setattr(sys.modules["torch.utils.data"], cls, object)
        stubbed.append("torch")
    try:
        import h5py  # noqa: F401
    except ImportError:
        sys.modules["h5py"] = types.ModuleType("h5py")
        sys.modules["h5py"].File = object
        stubbed.append("h5py")
    return stubbed


# --------------------------------------------------------------------------
def test_holm_nan():
    """NaN p값이 '유의함'으로 뒤집히지 않아야 한다."""
    from step4_stats import holm_bonferroni as hb

    # (1) 전부 NaN — 수정 전에는 보정 p 가 전부 0.0, 전부 기각이었다
    adj, rej = hb(np.full(12, np.nan))
    assert np.isnan(adj).all(), adj
    assert not rej.any(), rej

    # (2) NaN 하나가 섞여도 유효 비교의 보정을 오염시키지 않아야 한다
    #     수정 전에는 p=1.0 하나가 running 을 끌어올려 나머지가 전부 1.0 이 되었다
    adj, rej = hb(np.array([0.001, 0.02, np.nan, 0.9]))
    assert np.isnan(adj[2]) and not rej[2]
    assert np.allclose(adj[[0, 1, 3]], [0.003, 0.04, 0.9]), adj

    # (3) NaN 이 없을 때의 기존 동작은 그대로여야 한다
    adj, rej = hb(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06]), adj
    assert list(rej) == [True, False, False], rej

    # (4) 단조성: 보정 p 는 원 p 의 순서를 따라 비감소
    rng = np.random.default_rng(0)
    for _ in range(2000):
        q = rng.random(8)
        q[rng.random(8) < 0.3] = np.nan
        a, r = hb(q)
        v = ~np.isnan(q)
        assert not r[~v].any()
        if v.sum() > 1:
            o = np.argsort(q[v])
            assert np.all(np.diff(a[v][o]) >= -1e-12)

    print("  holm_bonferroni  : NaN 격리 · 단조성 · 기존 동작 보존 확인")


def test_ablation_single_seed():
    """시드 1개 손실 ablation 이 유의 판정을 내지 않아야 한다."""
    from step5_ablation import _test_vs_full

    # 실제 점검 실행에서 나온 값 (시드 11 하나)
    df = pd.DataFrame([
        dict(loss_setting="full",        seed=11, pr_auc=0.9880, rmse=2.7033, r2=-0.7828),
        dict(loss_setting="no_physics",  seed=11, pr_auc=0.9695, rmse=2.8037, r2=-0.9176),
        dict(loss_setting="no_focusing", seed=11, pr_auc=0.9874, rmse=1.2964, r2=+0.5900),
        dict(loss_setting="plain_bce",   seed=11, pr_auc=0.9878, rmse=1.8177, r2=+0.1940),
        dict(loss_setting="neither",     seed=11, pr_auc=0.9879, rmse=1.5383, r2=+0.4227),
    ])
    sig = _test_vs_full(df, "loss_setting", metrics=("pr_auc", "rmse", "r2"))
    assert len(sig) == 12, len(sig)
    assert not sig["significant_holm"].any(), \
        "시드 1개인데 유의 판정이 남아 있습니다 (수정 전 12건 전부 True)"
    assert sig["p_ttest_holm"].isna().all()
    assert not sig["testable"].any()

    # 시드가 2개면 정상적으로 검정이 이뤄져야 한다
    df2 = pd.concat([df, df.assign(seed=22, pr_auc=df.pr_auc + 0.002,
                                   rmse=df.rmse + 0.05, r2=df.r2 - 0.01)])
    sig2 = _test_vs_full(df2, "loss_setting", metrics=("pr_auc", "rmse", "r2"))
    assert sig2["testable"].all()
    assert sig2["p_ttest_holm"].notna().all()

    print("  손실 ablation    : 시드 1개 -> 유의 0건, 시드 2개 -> 검정 수행")


def test_split_guards():
    """분할이 빈 파티션을 만들지 않아야 한다."""
    from splitting import cycle_aware_split, InsufficientCyclesError

    try:
        cycle_aware_split(np.repeat([0, 1], 160))
    except InsufficientCyclesError:
        pass
    else:
        raise AssertionError("사이클 2개인데 예외가 나지 않았습니다")

    rng = np.random.default_rng(0)
    for _ in range(5000):
        n = int(rng.integers(3, 14))
        cid = np.repeat(np.arange(n), rng.integers(1, 1000, size=n))
        c = cycle_aware_split(cid, 0.70, 0.15)["cycles"]
        order = c["train"] + c["val"] + c["test"]
        assert c["train"] and c["val"] and c["test"], c
        assert order == sorted(order) and len(set(order)) == n, c

    print("  사이클 분할      : 5,000개 무작위 조건에서 빈 파티션·순서 위반 0건")


def main():
    stubbed = _stub_if_missing()
    print("=" * 68)
    print("  통계 보고 경로 회귀 시험")
    if stubbed:
        print(f"  (스텁 사용: {', '.join(stubbed)} — 통계 부분만 검사)")
    print("=" * 68)
    test_holm_nan()
    test_ablation_single_seed()
    test_split_guards()
    print("=" * 68)
    print("  전부 통과")


if __name__ == "__main__":
    main()
