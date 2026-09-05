"""대회 아카이브에서 데이터셋을 내려받는다.

토큰을 ~/.kaggle/access_token에 기록하고 KAGGLE_API_TOKEN 환경변수로도
설정한다. 다운로드 전에 가벼운 요청으로 인증을 먼저 확인하여, 인증 실패와
대회 규정 미동의를 구분할 수 있게 한다.
"""

import os
import subprocess
import sys

import config

# ----------------------------------------------------------------------
KAGGLE_API_TOKEN = ""
# ----------------------------------------------------------------------

COMPETITION = "LANL-Earthquake-Prediction"
RAW_DIR = "./data_raw"


def setup_credentials():
    token = KAGGLE_API_TOKEN.strip()
    if not token:
        print("  KAGGLE_API_TOKEN이 비어 있습니다. 파일 상단에 값을 설정하세요.")
        return False

    os.environ["KAGGLE_API_TOKEN"] = token

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    path = os.path.join(kaggle_dir, "access_token")
    with open(path, "w") as f:
        f.write(token + "\n")
    os.chmod(path, 0o600)

    legacy = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(legacy):
        print(f"  구형 인증 파일 감지: {legacy} (access_token이 우선 적용됨)")

    print(f"  토큰 설정 완료 (길이 {len(token)}, 접두사 {token[:5]})")
    return True


def verify_auth():
    print("\n  인증 확인")
    r = subprocess.run(["kaggle", "competitions", "list", "-s", "LANL"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("  인증 실패:")
        print(r.stderr or r.stdout)
        print("\n  토큰이 만료되었거나 폐기되었을 수 있습니다.")
        print("  kaggle 패키지가 오래되면 이 토큰 방식을 지원하지 않을 수 있습니다:")
        print("    pip install -U kaggle")
        return False
    print("  인증 성공")
    print(r.stdout[:300])
    return True


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"명령 실패: {' '.join(cmd)}")


def main():
    if not setup_credentials():
        return 1
    if not verify_auth():
        return 1

    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(config.RAW_CSV):
        size_gb = os.path.getsize(config.RAW_CSV) / 1024 ** 3
        print(f"\n  {config.RAW_CSV} 이미 존재 ({size_gb:.2f} GB), 다운로드 생략")
        return 0

    print(f"\n  {COMPETITION} 다운로드")
    print("  대회 규정에 먼저 동의해야 하며, 그렇지 않으면 403이 반환됩니다.")
    try:
        run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", RAW_DIR])
    except RuntimeError:
        print("\n  인증은 성공했으나 다운로드에 실패했습니다.")
        print("  해당 계정이 대회 규정에 동의했는지 확인하세요.")
        return 1

    print("\n  압축 해제")
    zips = [f for f in os.listdir(RAW_DIR) if f.endswith(".zip")]
    if not zips:
        print("  압축 파일을 찾을 수 없습니다")
        return 1
    for z in zips:
        run(["unzip", "-o", os.path.join(RAW_DIR, z), "-d", RAW_DIR])

    if os.path.exists(config.RAW_CSV):
        size_gb = os.path.getsize(config.RAW_CSV) / 1024 ** 3
        print(f"\n  완료: {config.RAW_CSV} ({size_gb:.2f} GB)")
        return 0

    print(f"\n  {config.RAW_CSV}를 찾을 수 없습니다")
    return 1


if __name__ == "__main__":
    sys.exit(main())
