"""점검 실행 로그를 읽어, 수정이 실제로 작동했는지 확인한다.

"고쳤다"는 말 대신 로그에서 증거를 찾는다. 각 항목은 v2.1 이전 로그에서는
실패하고 v2.2 로그에서는 통과하도록 짜여 있다.

    python run_all.py --smoke > smoke.log 2>&1
    python tools/verify_fixes.py smoke.log

종료 코드: 0 전부 통과 / 1 실패 항목 있음
"""

import argparse
import re
import sys

OK, FAIL, SKIP = "통과", "실패", "해당없음"


def _find(txt, pat, flags=re.M):
    m = re.search(pat, txt, flags)
    return m


def check_split(txt):
    """검증·평가 파티션이 비어 있지 않아야 한다 (v2.1 크래시의 원인)."""
    rows = re.findall(r"^\s+(train|val|test)\s+\|\s+세그먼트\s+(\d+)", txt, re.M)
    if not rows:
        return SKIP, "분할 요약을 찾지 못했습니다"
    sizes = {}
    for name, n in rows:
        sizes.setdefault(name, int(n))
    empty = [k for k, v in sizes.items() if v == 0]
    if empty:
        return FAIL, f"{', '.join(empty)} 파티션이 비었습니다 {sizes}"
    return OK, f"train {sizes.get('train')} / val {sizes.get('val')} / test {sizes.get('test')}"


def check_single_seed_guard(txt):
    """시드 1개 ablation 이 유의 판정을 내면 안 된다."""
    if "표 6b" not in txt:
        return SKIP, "손실 ablation 표가 없습니다"
    block = txt.split("표 6b", 1)[1][:2500]
    n_seed1 = _find(txt, r"설정 \d+개 x 시드 1개")
    lie = "해당 항을 제거하면 성능이 유의하게 저하되므로"
    if n_seed1:
        if lie in txt:
            return FAIL, "시드 1개인데 '유의하게 저하' 결론 문장이 출력되었습니다"
        if "검정 불가" not in block:
            return FAIL, "시드 1개인데 '검정 불가' 표기가 없습니다"
        n_true = block.count(" True")
        if n_true:
            return FAIL, f"시드 1개인데 significant=True 가 {n_true}건 있습니다"
        return OK, "시드 1개 -> '검정 불가', 유의 판정 0건"
    return SKIP, "시드가 2개 이상인 실행입니다"


def check_zero_variance(txt):
    """양쪽 분산이 0인 비교에 p=0.0000 이 나오면 안 된다."""
    if "분산 0 (검정 불가)" in txt:
        return OK, "분산 0 비교를 검정 불가로 처리했습니다"
    bad = re.findall(r"^\s*\S+\s+\S.*?\s+inf\s+\w+\s+\d/\d\s+True", txt, re.M)
    if bad:
        return FAIL, f"효과크기 inf 인데 유의 판정된 행이 {len(bad)}건 있습니다"
    return SKIP, "분산 0 사례가 없었습니다"


def check_attention_wording(txt):
    """어텐션 해석 문구가 실제 p 값·방향과 맞아야 한다."""
    m = _find(txt, r"경보 ([\d.]+) 대 정상 ([\d.]+) \(Welch t검정 p = ([\d.eE+-]+)\)")
    if not m:
        return SKIP, "어텐션 엔트로피 출력이 없습니다"
    ep, en, p = float(m.group(1)), float(m.group(2)), float(m.group(3))
    narrower, signif = ep < en, p < 0.05
    lie = "유의하게 좁아지지만 여전히 넓게 분포한다"
    if signif and narrower:
        want, label = lie, "유의하게 좁아짐"
    elif signif and not narrower:
        want, label = "오히려 유의하게 넓어집니다", "유의하게 넓어짐"
    else:
        want, label = "차이가 유의하지 않습니다", "유의차 없음"
    if lie in txt and not (signif and narrower):
        return FAIL, (f"p={p:g}, 경보 {ep:.4f} vs 정상 {en:.4f} 인데 "
                      f"'유의하게 좁아진다'고 서술했습니다")
    if want not in txt:
        return FAIL, f"기대 문구({label})가 보이지 않습니다"
    return OK, f"p={p:g} -> '{label}' 로 서술"


def check_realtime_factor(txt):
    """실시간 배수의 분모가 실측 경과시간이어야 한다."""
    m = _find(txt, r"실시간 배수 ([\d.]+)배\s*\[(\S+?)\]")
    if not m:
        if _find(txt, r"실시간 배수"):
            return FAIL, "실측/공칭 표시가 없습니다 (구버전 출력)"
        return SKIP, "지연시간 측정이 없습니다"
    return OK, f"{m.group(1)}배 [{m.group(2)} 기준]"


def check_cycle_mean(txt):
    """사이클 평균 길이가 한 값으로만 나와야 한다."""
    old = _find(txt, r"^\s+사이클 평균\s+=\s+[\d.]+ 초", re.M)
    new = _find(txt, r"논문에 쓸 사이클 평균 길이 = ([\d.]+) \+/- ([\d.]+)초 "
                     r"\(사이클별 TTF 구간, (\d+)개")
    if old:
        return FAIL, "경과시간 블록이 아직 별도의 사이클 평균을 출력합니다"
    if not new:
        return SKIP, "클래스 균형 출력이 없습니다"
    return OK, (f"{new.group(1)} +/- {new.group(2)}초 "
                f"(TTF 구간, {new.group(3)}개 사이클)")


def check_seed_warning(txt):
    """축소 실행에서 시드 수 오경보가 없어야 한다."""
    false_alarm = re.findall(r"경고: .*시드가 \d+개입니다 \(설정 \d+개\)", txt)
    if false_alarm:
        return FAIL, f"구버전 시드 경고가 {len(false_alarm)}건 있습니다"
    if _find(txt, r"모델마다 시드 수가 다릅니다"):
        return FAIL, "모델 간 시드 수가 실제로 어긋납니다 (캐시 확인 필요)"
    return OK, "오경보 없음"


def check_completed(txt):
    m = _find(txt, r"파이프라인 완료(?:, 총 ([\d.]+)분)?")
    if not m:
        return FAIL, "완료 표시가 없습니다 (중단되었거나 실행 중)"
    return OK, f"총 {m.group(1)}분" if m.group(1) else "완료"


CHECKS = [
    ("분할이 3개 파티션을 만들었는가", check_split),
    ("시드 1개 유의 판정을 막았는가", check_single_seed_guard),
    ("분산 0 비교를 걸렀는가", check_zero_variance),
    ("어텐션 문구가 검정과 맞는가", check_attention_wording),
    ("실시간 배수가 실측 기준인가", check_realtime_factor),
    ("사이클 평균이 한 값인가", check_cycle_mean),
    ("시드 수 오경보가 없는가", check_seed_warning),
    ("파이프라인이 완주했는가", check_completed),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="smoke.log")
    args = ap.parse_args()

    try:
        txt = open(args.log, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        print(f"  {args.log} 가 없습니다.")
        print("  먼저:  python run_all.py --smoke > smoke.log 2>&1")
        return 1

    print("=" * 72)
    print(f"  수정 검증 — {args.log}")
    print("=" * 72)
    n_fail = 0
    for name, fn in CHECKS:
        try:
            status, detail = fn(txt)
        except Exception as e:                       # 검사 자체가 깨져도 계속
            status, detail = FAIL, f"검사 중 예외: {e}"
        mark = {OK: "O", FAIL: "X", SKIP: "-"}[status]
        print(f"  {mark}  {name:32s} {detail}")
        n_fail += status == FAIL

    print("=" * 72)
    if n_fail:
        print(f"  {n_fail}건 실패. 위 항목을 확인하세요.")
    else:
        print("  전부 통과. 본 실행으로 넘어가도 됩니다.")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
