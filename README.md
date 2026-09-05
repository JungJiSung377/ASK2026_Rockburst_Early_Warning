## :link: :1st_place_medal: ASK2026 : Rockburst Early Warning Using Cumulative Energy History and Acoustic Emission Signals

주제 : 에너지 누적 이력과 음향 방출 신호를 활용한 Rockburst 조기경보 모델

**저자 : 정지성, 설재훈, 김가빈, 이규원, 오준석, 김영균**
**ACK 2026 한국정보처리학회 학술대회논문집 33권 2호 ooo-ooo(0pages)**
<br>
## Abstract

구간 단위 스펙트로그램은 구간 경계를 넘는 이력을 담을 수 없다. 본 연구는 개별 구간의 스펙트로그램과 직전 20개 구간의 RMS 이동 이력을 서로 다른 두 시간척도로 처리한 뒤 결합하는 조기경보 모델 CTPF(Cross-Timescale Precursor state Fusion) 를 제안하고, 제거 실험으로 성능 향상을 유도한 정보의 종류를 규명한다.

<br>

## 1. Problem

암반 파괴는 미세 균열이 누적되어 발생한다. 따라서 순간의 신호가 동일해도 축적된 에너지가 다르면 파괴까지 남은 시간(TTF)은 달라진다.

기존 연구는 신호를 일정 길이로 나눈 각 구간의 특성만으로 위험을 판별해 왔고, 다중 정보를 결합하는 접근 역시 결합된 정보가 실제로 상호 보완적인지 검증하지 않은 채 성능의 근거를 모델 복잡성에서 찾아왔다.

연구 질문 — 구간 경계를 넘는 이력은 스펙트로그램으로 대체할 수 없는 정보를 담고 있는가?

<br>
## 2. Dataset

LANL 실험실 stick–slip 마찰 실험 AE 데이터.
