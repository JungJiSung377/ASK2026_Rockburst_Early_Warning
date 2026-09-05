"""학술지 제출용 matplotlib 공통 서식.

[검토 반영 v2]
  C-13 Arial 이 없는 환경에서 조용히 DejaVu Sans 로 대체되던 문제.
       실제 사용된 서체를 확인해 경고한다.
  C-9  스펙트로그램용 백색 배경 정규화 헬퍼를 제공한다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from models_meta import display, color  # noqa: F401

_FONT_CHECKED = False


def _check_font():
    """[C-13] 지정한 서체가 실제로 존재하는지 확인하고, 대체되면 알린다."""
    global _FONT_CHECKED
    if _FONT_CHECKED or not config.FIG_WARN_ON_FONT_FALLBACK:
        return
    _FONT_CHECKED = True
    try:
        from matplotlib import font_manager
        wanted = config.FIG_FONT[0]
        path = font_manager.findfont(wanted, fallback_to_default=True)
        actual = font_manager.FontProperties(fname=path).get_name()
        if actual.lower() != wanted.lower():
            print(f"    [그림] 서체 '{wanted}' 없음 -> '{actual}' 로 대체됨. "
                  f"제출본 서체 요구사항을 확인하세요.")
    except Exception:
        pass


def apply_style():
    _check_font()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": config.FIG_FONT,
        "font.size": config.FIG_BASE_FONTSIZE,
        "axes.labelsize": config.FIG_BASE_FONTSIZE,
        "axes.titlesize": config.FIG_BASE_FONTSIZE + 1,
        "xtick.labelsize": config.FIG_BASE_FONTSIZE - 1,
        "ytick.labelsize": config.FIG_BASE_FONTSIZE - 1,
        "legend.fontsize": config.FIG_BASE_FONTSIZE - 1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # 백색 배경을 명시적으로 고정 (투명 배경으로 저장되면 인쇄 시
        # 배경이 어떻게 합성될지 알 수 없다)
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def spectrogram_norm(data, pct_lo=None, pct_hi=None):
    """[C-9] 스펙트로그램 표시 범위를 백분위로 정한다.

    기존 코드는 vmin=0, vmax=최댓값 이라 화소의 대부분이 컬러맵의 최하단
    (magma 기준 검정)에 눌려 신호가 보이지 않았다. 하위 백분위를 배경으로
    잘라내면 실제 신호가 살아난다.
    """
    lo = config.FIG_SPEC_PCT_LO if pct_lo is None else pct_lo
    hi = config.FIG_SPEC_PCT_HI if pct_hi is None else pct_hi
    v_lo = float(np.percentile(data, lo))
    v_hi = float(np.percentile(data, hi))
    if not np.isfinite(v_hi) or v_hi <= v_lo:
        v_lo, v_hi = float(np.min(data)), float(np.max(data)) or 1.0
    return v_lo, v_hi


LINESTYLE = {
    "Cross-Timescale Attn": "-",
    "Ours w/o Attention": (0, (4, 1.5)),
    "Single AE": (0, (1.5, 1.5)),
    "GradientBoosting": (0, (5, 1.5, 1, 1.5)),
    "LogReg/Ridge": (0, (3, 1, 1, 1, 1, 1)),
}


def style_of(name):
    return LINESTYLE.get(name, "-")


def panel_label(ax, text, dx=-0.16, dy=1.04):
    """관례적 위치에 굵은 패널 문자를 추가."""
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=config.FIG_BASE_FONTSIZE + 2, fontweight="bold",
            va="top", ha="left")


def save(fig, name, dpi=None):
    import os
    path = os.path.join(config.FIGURE_DIR, name)
    fig.savefig(path, format="pdf", dpi=dpi or config.FIG_DPI)
    # 미리보기용 PNG 도 함께 남긴다 (원고 삽입 전 확인용)
    png = path.replace(".pdf", ".png")
    try:
        fig.savefig(png, format="png", dpi=200)
    except Exception:
        png = None
    plt.close(fig)
    print(f"    {path}" + (f"  (+ {png})" if png else ""))
    return path
