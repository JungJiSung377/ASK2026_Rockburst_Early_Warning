"""실행 로그를 읽어 진행률과 남은 시간을 보여 준다.

백그라운드로 띄운 run_all.py 가 지금 어느 단계에 있고 언제 끝날지를
로그만 보고는 알기 어렵다. 이 도구는 로그에서 완료된 학습 횟수를 세어
진행률과 예상 완료 시각을 계산하고, 끝나면 알려 준다.

    python tools/watch_run.py                # 30초마다 갱신, 끝나면 종료
    python tools/watch_run.py --once         # 지금 상태만 한 번 출력
    python tools/watch_run.py --log 다른.log --every 60

종료 코드: 0 정상 완료 / 1 오류로 중단 / 2 프로세스 없음(미완료)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

# 학습 1회가 끝날 때 찍히는 줄 (단계별로 형식이 다르다)
PAT_TUNE = re.compile(r"->\s*composite\s")                 # 2단계 탐색
PAT_MAIN = re.compile(r"->\s*PR-AUC\s")                    # 3단계 본실험
PAT_ABL = re.compile(r"^\s+seed\s+\d+:\s*PR-AUC", re.M)    # 5단계 ablation

PAT_STAGE = re.compile(r"^#\s+(\d)단계\s+(.+?)\s*$", re.M)
PAT_BUILD = re.compile(r"-\s+(\d+)/(\d+)\s+\(cycle")       # 1단계 진행
PAT_DONE = re.compile(r"파이프라인 완료(?:, 총 ([\d.]+)분)?")
PAT_ERR = re.compile(r"^(?:Traceback \(most recent call last\)|"
                     r"\s*(?:ValueError|RuntimeError|KeyError|FileNotFoundError|"
                     r"MemoryError|InsufficientCyclesError|OSError):)", re.M)

PAT_TOTAL = re.compile(r"예상 학습 횟수\s+(\d+)회")
PAT_PLAN = re.compile(r"=\s*학습\s*(\d+)회")
PAT_MODE = re.compile(r"모드\s+(\S+)")

# 학습 1회에 걸린 초 (ETA 계산용)
PAT_SEC_MAIN = re.compile(r"\((\d+)s\)")
PAT_SEC_TUNE = re.compile(r"\|\s*(\d+)s\s*$", re.M)


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def process_alive(pattern="run_all.py"):
    """run_all.py 프로세스가 살아 있는지."""
    try:
        r = subprocess.run(["pgrep", "-f", pattern],
                           capture_output=True, text=True)
        return r.returncode == 0 and bool(r.stdout.strip())
    except FileNotFoundError:
        return None          # pgrep 이 없는 환경


def parse(txt):
    st = {}
    st["mode"] = (PAT_MODE.search(txt).group(1) if PAT_MODE.search(txt) else "?")

    st["n_tune"] = len(PAT_TUNE.findall(txt))
    st["n_main"] = len(PAT_MAIN.findall(txt))
    st["n_abl"] = len(PAT_ABL.findall(txt))
    st["n_done"] = st["n_tune"] + st["n_main"] + st["n_abl"]

    m = PAT_TOTAL.search(txt)
    st["n_total"] = int(m.group(1)) if m else None

    # 각 단계가 스스로 밝힌 계획 학습 횟수 (탐색, 특징 ablation, 손실 ablation)
    st["plan"] = [int(x) for x in PAT_PLAN.findall(txt)]

    stages = PAT_STAGE.findall(txt)
    st["stage"] = f"{stages[-1][0]}단계 {stages[-1][1]}" if stages else "시작 대기"

    b = PAT_BUILD.findall(txt)
    st["build"] = (int(b[-1][0]), int(b[-1][1])) if b else None

    d = PAT_DONE.search(txt)
    st["finished"] = bool(d)
    st["total_min"] = float(d.group(1)) if d and d.group(1) else None

    st["error"] = bool(PAT_ERR.search(txt))

    secs = [int(x) for x in PAT_SEC_MAIN.findall(txt)]
    secs += [int(x) for x in PAT_SEC_TUNE.findall(txt)]
    secs = [s for s in secs if 0 < s < 3600]
    st["sec_per_run"] = (sum(secs[-30:]) / len(secs[-30:])) if secs else None
    return st


def bar(frac, width=34):
    frac = max(0.0, min(1.0, frac))
    k = int(round(frac * width))
    return "#" * k + "." * (width - k)


def render(st, log_path, started_at, seen):
    lines = []
    age = time.time() - os.path.getmtime(log_path)
    lines.append(f"  모드 {st['mode']} | 현재 {st['stage']}")

    if st["build"] and st["n_done"] == 0:
        cur, tot = st["build"]
        lines.append(f"  데이터셋 구축  [{bar(cur/tot)}] {cur}/{tot} "
                     f"({100*cur/tot:.0f}%)")

    total = st["n_total"]
    if total:
        frac = st["n_done"] / total
        lines.append(f"  학습 진행      [{bar(frac)}] {st['n_done']}/{total} "
                     f"({100*frac:.1f}%)")
        lines.append(f"                 탐색 {st['n_tune']} · "
                     f"본실험 {st['n_main']} · ablation {st['n_abl']}")

        remain = total - st["n_done"]
        rate = None
        # 이 도구를 켜 둔 동안 관측한 속도가 가장 정확하다
        el = time.time() - started_at
        if seen is not None and st["n_done"] > seen and el > 60:
            rate = (st["n_done"] - seen) / el                 # 회/초
        elif st["sec_per_run"]:
            rate = 1.0 / st["sec_per_run"]
        if rate and rate > 0 and remain > 0:
            eta_s = remain / rate
            eta = datetime.now() + timedelta(seconds=eta_s)
            lines.append(f"  남은 학습 {remain}회 · 예상 {eta_s/3600:.1f}시간 "
                         f"(완료 예정 {eta:%m/%d %H:%M})")
    else:
        lines.append(f"  학습 완료 {st['n_done']}회 (총 횟수 아직 미확인)")

    lines.append(f"  로그 갱신 {age:.0f}초 전")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="full_run.log")
    ap.add_argument("--every", type=int, default=30, help="갱신 주기(초)")
    ap.add_argument("--once", action="store_true", help="한 번만 출력")
    ap.add_argument("--tail", type=int, default=3, help="함께 보여줄 로그 줄 수")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"  {args.log} 가 없습니다. 실행을 아직 시작하지 않았거나 "
              f"경로가 다릅니다.")
        return 2

    started_at, seen = time.time(), None
    while True:
        txt = read(args.log)
        st = parse(txt)
        alive = process_alive()

        print("=" * 62)
        print(f"  {datetime.now():%m/%d %H:%M:%S}   {args.log}")
        print("=" * 62)
        if not st["finished"]:
            print(render(st, args.log, started_at, seen))
        else:
            print(f"  모드 {st['mode']} | 학습 {st['n_done']}회 완료 "
                  f"(탐색 {st['n_tune']} · 본실험 {st['n_main']} · "
                  f"ablation {st['n_abl']})")
        if args.tail:
            body = [l for l in txt.splitlines() if l.strip()][-args.tail:]
            print("  " + "-" * 58)
            for l in body:
                print(f"  | {l[:120]}")

        if st["finished"]:
            t = f" (총 {st['total_min']:.1f}분)" if st["total_min"] else ""
            print("=" * 62)
            print(f"  완료{t}")
            print("  다음: python tools/check_integrity.py")
            print("=" * 62)
            return 0
        if st["error"] and alive is False:
            print("=" * 62)
            print("  오류로 중단된 것으로 보입니다. 아래로 원인을 확인하세요.")
            print(f"    tail -40 {args.log}")
            print("=" * 62)
            return 1
        if alive is False:
            print("=" * 62)
            print("  run_all.py 프로세스가 없는데 완료 표시도 없습니다.")
            print("  세션이 끊겼거나 백그라운드 프로세스가 정리되었습니다.")
            print("  같은 명령을 다시 실행하면 완료된 학습은 건너뛰고 이어집니다.")
            print("=" * 62)
            return 2

        if args.once:
            return 0
        if seen is None:
            seen = st["n_done"]
        try:
            time.sleep(args.every)
        except KeyboardInterrupt:
            print("\n  보기를 멈춥니다 (학습은 계속 돕니다).")
            return 0
        print()


if __name__ == "__main__":
    sys.exit(main())
