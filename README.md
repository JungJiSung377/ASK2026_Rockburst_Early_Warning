<div align="center">
:link: :1st_place_medal: ASK 2026 · Rockburst Early Warning
에너지 누적 이력과 음향 방출 신호를 활용한 Rockburst 조기경보 모델

Rockburst Early Warning Using Cumulative Energy History and Acoustic Emission Signals

<br>

정지성¹ · 설재훈² · 김가빈³ · 이규원⁴ · 오준석⁵ · 김영균⁶

<sub>¹ 강원대학교 문화예술·공과대학 에너지자원공학과 · ² 강원대학교 문화예술·공과대학 기계의용·메카트로닉스공학과 · ³ 강원대학교 AI융합학과<br> ⁴ 서울대학교 공과대학 기계공학과 · ⁵ 연세대학교 일반대학원 배터리공학과 · ⁶ 융합소프트웨어랩</sub>

<br>

이미지 표시 이미지 표시 이미지 표시 이미지 표시 이미지 표시

ACK 2026 한국정보처리학회 학술대회논문집 33권 2호 ooo-ooo (0 pages)

</div>
Abstract (EN)

Deep underground development has increased the risk of rockburst, creating demand for real-time early-warning systems. Existing acoustic-emission (AE) approaches classify risk from the characteristics of individual fixed-length segments, and therefore cannot reflect the history accumulated across segment boundaries on the path to failure. This work proposes CTPF (Cross-Timescale Precursor state Fusion), which encodes the within-segment spectrogram and the RMS history of the preceding 20 segments on two separate timescales and then fuses them.

The contribution is not only the performance gain but the empirical identification of which kind of information produced it. Removing the two inter-segment history features collapses PR-AUC by 19.3 % — back to the spectrogram-only level — while removing all four intra-segment statistics changes nothing (0.6 %, n.s.). Fusion mechanism (cross-attention vs. simple concatenation) makes no significant difference, so the gain comes from combining two timescales at all, not from the sophistication of the fusion. Attention weights carry no performance benefit but were causally validated as decision evidence through a deletion test.

Key numbers — PR-AUC 0.736 vs. 0.588 (spectrogram-only) · Recall 80.98 % vs. 50.48 % · Recall at matched FAR 15 %: 76.7 % vs. 68.8 % (Logistic/Ridge) · End-to-end latency 7.37 ms against a 39.30 ms segment period (5.33× headroom) · 15 independent seeds, Holm–Bonferroni corrected.

요약 (KR)

심부 지하 공간 개발의 확대로 Rockburst 재해 위험이 증가하면서 재해 예방 실시간 조기경보 체계가 요구되고 있다. 음향 방출 신호 기반의 기존 연구는 신호를 일정 길이로 나눈 각 구간 특성을 주로 활용하여 파괴에 이르기까지 누적된 직전 구간들의 이력 정보를 반영하지 못하는 한계가 있다. 이에 본 연구는 개별 구간의 스펙트로그램과 직전 20개 구간의 RMS 이동 이력을 서로 다른 두 시간척도로 처리한 뒤 결합하는 조기경보 모델을 제안한다. 이는 현장의 선제적 대피 판단을 지원함과 동시에 조기경보 모델 설계에 실질적으로 필요한 정보의 근거를 제시할 것으로 기대된다.

:pushpin: 한눈에 보는 핵심 결과
#	발견	근거
1	이력 정보가 성능 향상의 실체다	구간 경계를 넘는 Inter 특징 2종을 제거하면 PR-AUC −0.142 (−19.3 %)로 스펙트로그램 단독 수준으로 회귀. 반면 구간 내부 Intra 4종을 전부 제거해도 −0.004로 유의하지 않음
2	향상의 원인은 '결합' 그 자체였다	교차 어텐션 대 단순 결합의 PR-AUC 차이 +0.003, 8개 지표 전부 유의하지 않음 (Holm p = 1.00). 파라미터를 13.4 % 적게 쓰는 단순 결합으로도 동등 성능
3	어텐션의 가치는 성능이 아니라 '검증 가능한 근거'다	삭제 실험에서 상위 어텐션 프레임 마스킹의 경보 확률 변화가 무작위 대비 약 3배, k = 5·10·20 전부 p < 0.001
4	우위는 조기경보의 운영 구간에서 발현된다	FAR ≤ 12.5 %에서는 고전 baseline과 유의차 없음. FAR 15 % 이상에서만 유의하게 앞섬 (목표 운영점과 일치)
5	실시간 동작이 가능하다	전처리 포함 전체 지연 7.37 ms vs. 구간 실제 발생 주기 39.30 ms → 5.33배 여유
:page_with_curl: 목차
1. 연구 배경과 문제 정의
2. 데이터셋 구축 및 전처리
3. 제안 방법 — CTPF
4. 손실 함수와 평가 프로토콜
5. 실험 결과
6. 결론
7. 한계와 향후 과제
8. 용어 정리
참고문헌
인용
1. 연구 배경과 문제 정의
1.1 왜 조기경보인가

지상 공간 포화와 자원 고갈로 심부 지하 공간 개발이 확대되면서, 높은 토피압과 지각 운동에 의한 고압 굴착 환경에서 탄성 변형 에너지가 폭발적으로 방출되는 Rockburst 재해 위험이 증가하고 있다 [1]. 이 재해는 순간적으로 발생하며 복구가 불가능하므로, 파괴 이전에 내리는 경보가 사실상 유일한 대응 수단이다.

1.2 기존 접근의 구조적 한계

음향 방출(AE) 신호 기반 실시간 조기경보가 모색되어 왔으나, 기존 연구 [2]는 신호를 일정 길이로 나눈 각 구간의 특성만으로 위험을 판별해 왔다 [3][4]. 그러나 암반 파괴는 미세 균열이 점차 누적되어 발생하는 현상이므로,

순간의 신호가 동일하더라도 축적된 에너지의 양이 다르면 파괴까지 남은 시간(TTF)은 달라진다 [5].

구간 단위 관측 방식만으로는 단절된 신호 간의 장기적 역학 관계와 에너지 축적 양상을 통합하기 어렵다. 이를 보완하려는 다중 정보 결합 접근 역시 결합된 정보가 실제로 상호 보완적인지 검증하지 않은 채 성능 향상의 근거를 모델의 복잡성에서 찾아왔다 [6][7].

1.3 연구 질문

[!IMPORTANT] 구간 경계를 넘는 이력은, 스펙트로그램으로 대체할 수 없는 정보를 실제로 담고 있는가?

본 연구는 이 질문에 답하기 위해 개별 구간의 스펙트로그램과 직전 20개 구간의 RMS 이동 이력을 서로 다른 두 시간척도로 처리한 뒤 결합하는 모델 CTPF (Cross-Timescale Precursor state Fusion) 를 제안하고, 구성 요소별 제거 실험을 통해 성능 향상을 유도한 정보의 종류를 정량적으로 규명하였다.

2. 데이터셋 구축 및 전처리
2.1 원자료와 구간 분할

실험실 직접전단 마찰 실험에서 stick–slip 파괴 사이클을 반복 재현한 LANL 음향 방출(AE) 데이터 [8]를 활용하였다.

항목	값	비고
샘플링 주파수	4 MHz	
구간 분할 단위	150,000 샘플	
공칭 구간 길이	37.5 ms	
실측 구간 길이 (중앙값)	39.30 ms	공칭 대비 +4.8 %
STFT 창 / 중첩	3,000 샘플 / 50 %	
스펙트로그램 크기	1,501 × 101	(주파수 × 시간 프레임)

[!NOTE] LANL 원자료는 블록 단위로 기록되어 시간 축이 연속이 아니다. 따라서 원자료의 time-to-failure 채널에서 측정한 실제 경과시간(39.30 ms)이 공칭값(37.5 ms)보다 4.8 % 길다. 이 문서의 모든 시간 관련 수치는 실측값 기준이다.

2.2 데이터셋 구성과 분할
항목	값
유효 구간 수	4,131개 (원본 4,194개 − 사이클 경계 16개 − 미완결 사이클)
완전 파괴 사이클	16개
분할 방식	사이클 단위 시간순 학습 : 검증 : 평가 = 10 : 3 : 3
평균 사이클 길이	10.66 ± 2.65 s
경보 임계 시간	TTF ≤ 3.0 s (평균 사이클의 약 28 %)
클래스 구성	경보 : 비경보 = 1 : 2.47 (경보 구간 28.9 %)
test 분할 경보 비율	0.276 ← PR-AUC 기준선
원시 TTF 범위	0.006 – 16.103 s

경보 라벨의 정의

사이클 시작 ──────────────────────────────▶ 파괴
       [        정상  TTF > 3.0 s        ][ 경보 TTF ≤ 3.0 s ]

[!WARNING] 3.0 s는 평균 사이클 길이를 기준으로 설정한 값이며, 현장의 실제 대피 소요 시간에 대한 별도 검증은 수행하지 않았다. 향후 과제이다.

인접 구간은 같은 사이클의 정보를 공유하므로, 무작위 분할은 정보 유입을 일으킨다. 이를 막기 위해 사이클 단위로 시간 순서를 지켜 분할하였고, 이로써 평가를 "미관측 사이클로의 일반화" 로 정의하였다.

2.3 현장 잡음 모사

실험실 데이터는 현장 대비 잡음이 지나치게 낮다. 이 한계를 보완하고자, 회전 기계에 기인하여 저주파 대역에 집중되는 현장 잡음 특성을 기반으로 최저 8개 주파수 bin(DC – 10.0 kHz, bin 폭 1.33 kHz)에 가우시안 잡음을 가산하였다. 또한 TTF가 파괴 순간 급증하는 특성을 이용해 파괴 사이클 경계를 검출하였다.

<div align="center"> <img src="assets/fig1_noise_injected_spectrogram.png" alt="잡음이 주입된 음향 방출 신호 스펙트로그램" width="90%"> <br> <sub><b>(그림 1)</b> 잡음이 주입된 음향 방출 신호 스펙트로그램 — 좌: 원 실험실 신호, 중: 간섭 주입 신호, 우: 주입 성분</sub> </div>

[!WARNING] 주입된 잡음은 실측 TBM 스펙트럼이 아니라 가상 시나리오이다. 실측 현장 잡음 환경에서의 재평가는 향후 과제로 남는다.

3. 제안 방법 — CTPF
3.1 설계 원리: 두 시간척도의 분리

STFT 스펙트로그램은 분할 구간 단위의 닫힌 연산의 결과이므로, 내부의 시간–주파수 구조는 정밀하게 반영하지만 구간의 경계를 넘는 이력 정보는 원리적으로 포함하기 어렵다. 이에 누적 이력을 저차원으로 요약한 6차원 전조 상태 벡터를 별도의 느린 시간척도로 구성하였다.

	빠른 척도 (Fast)	느린 척도 (Slow)
관측 창	현재 구간 내 39.3 ms	직전 20구간 ≈ 0.786 s
표현	STFT 스펙트로그램 1,501 × 101	전조 상태 벡터 ℝ⁶
담는 정보	구간 내부의 시간–주파수 구조	구간 경계를 넘는 에너지 누적 이력
역할	근거 (Key · Value)	질의 (Query)
3.2 전조 상태 벡터의 구성
<div align="center">

〈표 1〉 전조 상태 벡터 구성

Group	Feature	Description	Window
Intra	RMS amplitude	Emission level	39.3 ms
Intra	Kurtosis, Peak, Crest	Waveform impulsiveness	39.3 ms
Inter	Energy trend slope	Slope of past RMS history	0.786 s
Inter	Windowed RMS sum	Sum of past RMS history	0.786 s
</div>
Intra 특징 4종은 현재 구간만으로 산출된다. 즉 원리적으로 스펙트로그램에서도 복원 가능한 정보이다.
Inter 특징 2종은 직전 20개 구간의 RMS 이력을 봐야만 얻어진다. 이것이 스펙트로그램이 담을 수 없는 정보이며, 본 연구가 검증하려는 대상이다.
Inter 특징은 현재 값을 이력에 반영하기 전에 계산하며, 사이클 경계마다 초기화된다.

[!NOTE] 용어에 대한 정확한 서술 — windowed RMS sum은 직전 20구간의 이동합이지 사이클 전체의 누적합이 아니다. 0.786 s는 평균 사이클 10.66 s의 약 7 %에 불과하므로, 본 문서에서는 이를 "단기 누적 이력" 으로 표현한다.

3.3 모델 구조
<div align="center"> <img src="assets/fig2_ctpf_architecture.png" alt="CTPF 아키텍처" width="78%"> <br> <sub><b>(그림 2)</b> 모델(CTPF) 아키텍처</sub> </div> <br>
경로	구성	출력
느린 척도	전조 상태 벡터 ℝ⁶ → MLP 인코더	128-d 임베딩 → Q
빠른 척도	스펙트로그램 1,501 × 101 → 2-D CNN	101 × 256
	→ LSTM	101 × 128 → K, V
융합	비대칭 교차 어텐션 Attn(Q_slow, K_fast, V_fast) + Q_slow (잔차 연결)	
출력	분류 헤드 / 회귀 헤드	경보 확률 / 잔여 시간 TTF

설계상의 세 가지 선택

2-D CNN은 주파수 축만 축약하고 시간 프레임 101개를 그대로 보존한다. 그래야 이후 프레임별 어텐션 가중치를 해석할 수 있다.
CNN 뒤에 LSTM을 둔 이유 — CNN의 시간 축 수용영역은 국소적인 반면, LSTM은 구간 전체에 걸친 순서 의존성을 누적해 프레임별 표현을 만든다. 어텐션이 참조할 대상을 만들기 위한 선택이다.
융합은 비대칭이다. 느린 척도가 "무엇을 볼지" 를 질의하고 빠른 척도가 근거를 제공한다. 이를 통해 시간 프레임별 해석 가능한 가중치를 제공 [9]하는 동시에, 누적 상태가 각 구간 내 주목 구간을 결정할 수 있다. Q 잔차 연결을 두어 어텐션을 거치지 않고도 느린 척도 정보가 출력에 도달할 수 있게 하였다.
4. 손실 함수와 평가 프로토콜
4.1 통합 손실 함수 (식 1)

분류와 회귀를 동시에 최적화한다.

𝐿
  
=
  
𝐿
cls
  
+
  
𝜆
𝑟
 
𝐿
reg
  
+
  
𝜆
𝑝
 
𝐿
energy
L=L
cls
	​

+λ
r
	​

L
reg
	​

+λ
p
	​

L
energy
	​

항	역할	설정

𝐿
cls
L
cls
	​

	경보 여부 판별 — 다수 클래스 편향 완화를 위해 focal loss [10]	γ = 2.0, class-balanced α

𝐿
reg
L
reg
	​

	파괴까지 남은 시간 예측 오차	

𝐿
energy
L
energy
	​

	음향 에너지가 높아 파괴 활동이 활발한 구간에 더 큰 비용 부과	λ_p = 0.15

TTF 정규화 — 원시 TTF는 0.006–16.103 s 범위를 가져 정규화 없이는 회귀 항이 손실을 지배한다. 따라서 학습 분할의 최댓값으로 나누어 [0, 1]로 정규화해 학습하고, 예측값은 다시 초(s) 단위로 역변환해 보고하였다.

[!IMPORTANT] 손실 항 제거 실험에서 어느 항도 유의한 차이를 만들지 않았다 (8 seed, 4개 설정 전부 Holm 보정 p = 1.00). 따라서 본 연구는 성능의 근거로 "물리 정보 기반"을 주장하지 않는다.

4.2 하이퍼파라미터 탐색

학습률과 회귀 가중치는 비교 모델과 제안 모델에 동일한 그리드로 탐색하였다. 다만 검증 손실이 λ에 비례해 커져 항상 최솟값이 선택되는 문제를 피하고자, 선택 기준을 검증 PR-AUC와 R²의 합으로 두었다.

4.3 운영 임계치 결정 (식 2)

조기경보에서 미탐지는 인명 손실로 회복이 불가능한 비용인 반면, 오경보는 작업 중단이라는 회복 가능한 비용이다. 따라서 허용 오경보율 제약 아래 재현율을 최대화하는 임계값을 운영점으로 정의하였다.

𝑡
∗
  
=
  
arg
⁡
max
⁡
𝑡
  
R
e
c
a
l
l
val
(
𝑡
)
s.t.
F
A
R
val
(
𝑡
)
  
≤
  
𝜅
⋅
F
A
R
target
t
∗
=arg
t
max
	​

Recall
val
	​

(t)s.t.FAR
val
	​

(t)≤κ⋅FAR
target
	​

기호	의미	값

𝑡
t	경보 판정 임계값	—

F
A
R
target
FAR
target
	​

	목표 오경보율 (재현율 확보 우선)	15 %

𝜅
κ	안전계수 — validation↔test 분포 이동에 따른 FAR 상승 대비	0.6
⇒ 검증에서 겨냥하는 FAR	
𝜅
⋅
F
A
R
target
κ⋅FAR
target
	​

	9 %

[!IMPORTANT] 정보 유입 차단 — 정규화 통계와 운영 임계치 
𝑡
∗
t
∗
 는 학습·검증 분할에서만 결정하여 test에 불변으로 적용하였다. 적용 시점에 존재하지 않는 정보의 사용을 원천적으로 막기 위함이다.

4.4 통계적 검정 설계
원칙	내용
반복	15개 독립 시드 (특징·손실 제거 실험은 8 seed)
주 지표	PR-AUC (기준선 = test 경보 비율 0.276)
심층 모델 간 비교	초기화·배치 순서의 공통 변동을 상쇄하도록 동일 시드로 학습된 모델 쌍을 짝지어 대응 t-검정
다중비교 보정	Holm–Bonferroni (모든 비교에 적용)
운영점 정렬	모든 모델을 8개 공통 FAR (2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20 %)에서 재평가
5. 실험 결과
5.1 분류·회귀 성능 〈표 2〉
<div align="center">

〈표 2〉 비교 모델별 분류·회귀 성능 비교 (15개 독립 시드의 평균 ± 표준편차)

Model	Input	PR-AUC	Recall (%)	FAR (%)	RMSE (s)	R²
CTPF (proposed)	Spec + State	0.736 ± 0.006	80.98 ± 0.93	16.29 ± 0.55	2.627 ± 0.022	0.502
CTPF w/o attention	Spec + State	0.733 ± 0.005	80.26 ± 1.42	16.14 ± 0.49	2.637 ± 0.023	0.498
Logistic / Ridge	State	0.726 ± 0.004	75.96 ± 1.41	18.78 ± 0.53	2.884 ± 0.010	0.400
Gradient boosting	State	0.691 ± 0.028	61.93 ± 4.04	13.07 ± 1.25	2.915 ± 0.056	0.387
Spectrogram-only	Spec	0.588 ± 0.027	50.48 ± 4.55	13.63 ± 0.95	2.979 ± 0.059	0.360
</div>

스펙트로그램 단독 대비 — PR-AUC +0.148 (+25.3 %) · 재현율 +30.5 %p · RMSE −0.352 s. 24개 주요 비교 전부 Holm 보정 후 유의 (p < 0.05).

[!WARNING] 두 가지를 정직하게 짚어야 한다.

FAR은 낮을수록 좋은 지표인데 제안 모델의 FAR이 오히려 높다. 따라서 "모든 지표에서 우위" 라고 말할 수 없다.
상태 벡터 6차원만 사용한 Logistic/Ridge가 PR-AUC 0.726을 기록해 스펙트로그램 단독 딥러닝 모델(0.588)을 크게 상회하였다. 이는 이진 판별력의 상당 부분이 저차원 상태 특징에 담겨 있음을 시사하며, 오히려 본 연구의 핵심 주장을 뒷받침한다.

또한 모델마다 임계치가 만드는 운영점이 다르므로 재현율을 그대로 비교하면 운영점 차이가 혼재된다. 이를 분리하기 위해 다음 절에서 동일 FAR로 재평가한다.

5.2 동일 오경보율에서의 재평가
<div align="center"> <img src="assets/fig3_recall_at_common_far.png" alt="동일한 오경보율(FAR) 조건에서의 재현율" width="80%"> <br> <sub><b>(그림 3)</b> 동일한 오경보율(FAR) 조건에서의 재현율 — 15시드 평균 ± 표준편차</sub> </div> <br> <div align="center">

목표 운영점 FAR 15 %에서의 재현율

Model	Recall @ FAR 15 %	CTPF 대비
CTPF (proposed)	76.7 %	—
Logistic / Ridge	68.8 %	+7.9 %p
Gradient boosting	66.8 %	+9.9 %p
Spectrogram-only	54.5 %	+22.2 %p
</div>
FAR 12.5 % 이하 구간에서는 제안 모델과 고전 baseline 사이에 Holm 보정 후 유의한 차이가 확인되지 않았다. 이 구간에서는 상태 벡터만 쓰는 단순한 모델로도 충분하다는 뜻이다.
다만 스펙트로그램 단독 모델과의 차이는 전 구간에서 유의하였다.
FAR 12.5–15 % 구간을 기점으로 격차가 벌어지며, FAR 15 %에서 고전 baseline 전부를 Holm 보정 후에도 유의하게 앞선다. FAR 20 %에서도 Gradient boosting 대비 +10.8 %p, Logistic/Ridge 대비 +9.1 %p로 격차가 유지된다.
어텐션 유무의 차이는 모든 FAR에서 유의하지 않았다.

해석 — 저차원 상태 특징만으로는 낮은 오경보 조건에서의 판별에 그치지만, 스펙트로그램과의 결합이 "추가 오경보를 재현율로 전환하는 능력" 을 제공한다. 미탐지 비용이 오경보 비용보다 큰 조기경보의 특성상, 제안 모델의 이점은 본 문제의 목적에 부합하는 지점에서 정확히 발현된다.

[!WARNING] 이 값들은 평가 곡선에서 읽어낸 상한이며 실제 배포 시 달성되는 성능이 아니다. 실제 배포에서는 임계치를 사전에 고정해야 하므로, 실제 달성 성능은 〈표 2〉의 재현율 80.98 % / FAR 16.29 % 이다.

5.3 정보 종류별 기여도 검증 (핵심 결과)

제안 구조의 주된 가설은 "상태 벡터 중 각 구간 사이의 이력만이 스펙트로그램으로 대체 불가능한 정보를 담는다" 는 것이다. 이를 검증하기 위해 개별 성분 6종 · 군 단위 2종 · 전체 무력화 1종, 총 9개 설정에 대해 특징 제거 실험을 수행하였다. 제거는 0으로 채우는 대신 학습 분할의 평균값으로 대체하여, 차원 수 변화가 아닌 정보 제거의 효과만 분리하였다.

<div align="center"> <img src="assets/fig4_ablation_prauc.png" alt="상태 벡터 구성 요소 제거에 따른 PR-AUC 변화" width="85%"> <br> <sub><b>(그림 4)</b> 상태 벡터 구성 요소 제거에 따른 PR-AUC 변화 — 8시드, * = Holm 보정 후 유의</sub> </div> <br> <div align="center">
제거 설정	군	Δ PR-AUC	Holm 보정 후 유의
w/o inter-segment (both)	Inter	−0.142	✓
w/o state vector (전체)	—	−0.141	✓
w/o windowed RMS sum	Inter	−0.052	✓
w/o kurtosis	Intra	−0.004	n.s.
w/o intra-segment (all four)	Intra	−0.004	n.s.
w/o peak amplitude	Intra	+0.000	n.s.
w/o crest factor	Intra	+0.002	n.s.
w/o RMS amplitude	Intra	+0.003	n.s.
w/o energy trend slope	Inter	+0.008	✓
</div>
(1) 성능 변화의 원인은 차원 수가 아니라 정보의 종류다
Inter 2종 제거 → PR-AUC 0.735 → 0.593, −19.30 % — 스펙트로그램 단독 모델 수준으로 회귀
Intra 4종 전부 제거 → −0.60 %, 유의하지 않음

더 많은 차원을 제거했을 때 영향이 없고, 더 적은 차원을 제거했을 때 성능이 붕괴한다. 이 대조가 성능 변화가 입력 차원 수가 아니라 정보의 종류에 기인함을 보여 준다.

(2) 상태 벡터의 이득은 사실상 전부 Inter에서 온다

Inter 제거(0.5929)와 상태 벡터 전체 무력화(0.5942)의 결과가 통계적으로 구별되지 않았다. 즉 제안 모델이 스펙트로그램 단독 베이스라인 대비 획득한 이득의 약 90 %가 Inter 특징 2종과 관련된다 (동일 8 seed 기준).

(3) 두 Inter 특징의 역할은 대칭적이지 않다
	단독 제거	유의성
Windowed RMS sum	−0.052	Holm p < 0.001
Energy trend slope	+0.008 (소폭 상승)	Holm p = 0.049
동시 제거	−0.142	개별 효과 합(−0.044)의 3.24배

실제 정보를 담은 것은 windowed RMS sum이며, 추세 기울기는 단독으로는 오히려 잉여에 가깝다. 그럼에도 동시 제거 시 손실이 개별 합보다 3.24배 크게 나타나는 것은, 한쪽이 남아 있으면 다른 쪽의 정보를 상당 부분 보완할 수 있음을 뜻한다. 두 특징이 같은 물리량을 수준과 속도로 부호화한다고 단정하기보다, 서로를 부분적으로 대체하는 초가법적(super-additive) 관계로 해석하는 것이 결과에 충실하다.

(4) 손실 함수와 융합 방식은 유의한 차이를 만들지 않았다
제거 대상	결과	조건
융합 방식 (교차 어텐션 → 단순 결합)	PR-AUC 차이 +0.003, 8개 지표 전부 유의하지 않음	15 seed, Holm p = 1.00
손실 함수 (에너지 가중항·focal 항을 각각/함께 제거, 4개 설정)	PR-AUC·R² 전부 유의한 차이 없음	8 seed, 전부 Holm p = 1.00

[!IMPORTANT] 성능 향상을 견인한 요인은 결합 메커니즘의 정교함이 아니라, 두 시간척도의 정보를 결합한다는 사실 그 자체이다. 실용적으로는 오히려 좋은 소식인데, 파라미터를 13.4 % 적게 사용하는(374,210 대 432,066) 단순 결합 구조만으로도 동등한 성능을 얻을 수 있다는 뜻이기 때문이다.

5.4 판단 근거의 해석과 인과적 검증

어텐션 융합은 성능 이점을 제공하지 않았다. 그러나 이 구조는 세그먼트 내 101개 시간 프레임 각각에 부여된 가중치를 그대로 출력한다. 조기경보는 최종적으로 사람이 대피 여부를 결정하는 체계이므로, 경보의 근거를 제시할 수 있는지가 성능 못지않게 중요하다. 이에 어텐션 가중치가 실제 판단 근거로 기능하는지를 세 단계로 검증하였다.

<div align="center">
(a) 프레임별 가중치 차이	(b) 정규화 엔트로피	(c) 삭제 실험
<img src="assets/fig5a_attention_difference.png" width="290">	<img src="assets/fig5b_attention_entropy.png" width="185">	<img src="assets/fig5c_deletion_test.png" width="290">

<sub><b>(그림 5)</b> 교차 어텐션의 판단 근거 해석과 인과적 검증</sub>

</div>

(a) 주목 위치가 달라지는가 — 경보 상태와 평시 상태의 프레임별 가중치를 비교한 결과, 경보 상태에서 구간 양 끝의 비중이 줄고 내부로 옮겨갔으며, 일부 프레임에서 그 차이가 유의하였다 (95 % CI가 0을 배제).

(b) 주의가 좁아지는가 — 경보 구간의 정규화 엔트로피가 평시보다 유의하게 낮았다 (p < 0.001).

[!NOTE] 다만 유효 프레임 수로 환산하면 경보 구간의 주의도 여전히 101개 프레임의 절반 이상에 분포한다. 따라서 "특정 구간으로 집중된다" 가 아니라 "유의하게 좁아지되 여전히 넓게 분포한다" 가 정확한 서술이다.

(c) 그 가중치가 실제로 예측에 쓰였는가 — 삭제 실험 — (a)와 (b)는 가중치가 어떻게 분포하는지를 보여 줄 뿐, 그것이 실제 예측에 쓰였음을 보장하지 않는다. 이를 직접 확인하기 위해 어텐션 상위 k 개 프레임을 마스킹했을 때와 무작위 k 개를 마스킹했을 때의 경보 확률 변화를 대응 비교하였다.

k = 5, 10, 20 전부에서 상위 프레임 쪽 변화량이 약 3배 컸고, 모두 p < 0.001이었다. 가중치가 예측에 실제로 관여함을 보이는 직접적·인과적 증거이며, 엔트로피가 주지 못하는 근거이다.

종합 — 어텐션 융합의 가치는 성능이 아니라 검증 가능한 판단 근거에 있다. 동등한 성능을 내는 단순 결합 구조는 경보의 시간적 근거를 제시하지 못하는 반면, 제안 구조는 경보와 함께 모델이 참조한 구간을 제시할 수 있어 현장에서 경보의 수용 여부를 판단하는 데 활용될 수 있다.

5.5 실시간 처리 가능성

원신호 취득부터 추론까지 전 구간을 측정하였다.

<div align="center">
단계	소요 시간
STFT 전처리	3.347 ms
전조 상태 벡터 계산	1.718 ms
텐서 전송	0.398 ms
모델 추론	1.907 ms
전체 지연시간	7.370 ms
구간의 실제 발생 주기	39.30 ms
확보된 실시간 여유	5.33 배
</div>
구간 발생 주기 39.30 ms
├──────── 7.37 ms 처리 ────────┤├─────────── 남은 여유 31.93 ms ───────────┤

전처리 비용을 포함하고도 데이터 유입 속도를 크게 상회하여, 연속 모니터링 환경에서 실시간 조기경보 체계로 동작할 수 있음을 뒷받침한다.

[!NOTE] 측정 조건은 NVIDIA A100-SXM4-40GB 기준이며 하드웨어에 따라 값이 달라진다. 현장 엣지 장비에서는 재측정이 필요하다. 전조 상태 벡터 계산(대부분 kurtosis 비용)은 "원신호 취득부터 추론까지" 를 주장하려면 반드시 포함되어야 하는 항목이므로 명시적으로 계상하였다.

6. 결론

본 연구는 구간 단위 스펙트로그램이 원리적으로 담을 수 없는 직전 20개 구간의 RMS 이동 이력을 별도 시간척도로 처리하여 결합하는 Rockburst 조기경보 모델을 제안하였다.

이력 정보의 기여를 실증하였다. 제안 모델은 스펙트로그램 단독 베이스라인 대비 PR-AUC 0.7359 대 0.5875, 재현율 80.98 % 대 50.48 %로 우위를 보였다. 특징 제거 실험 결과 구간 사이의 이력 2종을 제거하면 PR-AUC가 19.30 % 하락하지만 구간 내부 통계 4종을 전부 제거해도 0.60 % 에 그쳐, 성능 향상의 약 90 %가 스펙트로그램으로 대체 불가능한 단기 누적 이력에서 비롯됨을 확인하였다.
향상의 원인은 결합 그 자체였다. 융합 방식(교차 어텐션 대 단순 결합)의 차이는 유의하지 않았으며, 파라미터를 13.4 % 적게 쓰는 단순 결합으로도 동등한 성능을 얻는다.
판단 근거를 인과적으로 검증하였다. 삭제 실험에서 상위 어텐션 프레임의 기여가 무작위 대비 약 3배로 유의하였다.
실시간 동작이 가능하다. 동일 오경보율 재평가를 통해 제안 모델의 이점이 미탐지 비용을 우선하는 본 연구의 설정 운영점(FAR 15 % 이상) 부근에서 발현됨을 확인하였으며, 전체 지연시간 7.37 ms로 실시간 처리 가능성을 검증하였다.

궁극적으로 본 연구는 현장의 선제적인 대피 의사 판단을 지원하고, 대형 재해 예방과 심부 지하 굴착공사의 안전성 향상에 이바지할 것으로 기대된다.

7. 한계와 향후 과제
구분	한계	향후 과제
경보 정의	3.0 s 임계값이 현장의 실제 대피 소요 시간과 일치하는지 미검증	현장 대피 소요 시간 기반 임계값 재정의
잡음 환경	주입 잡음은 실측 TBM 스펙트럼이 아닌 가상 시나리오	실측 현장 잡음 환경에서의 재평가
데이터 규모	완전 파괴 사이클 16개, 사이클 단위 편향 존재 가능	사이클 수를 늘린 일반화 검증, 사이클 단위 편향 보정
운영점	안전계수 κ = 0.6은 분포 이동을 가정한 값	오경보 안전계수의 재보정
성능 서술	제안 모델의 FAR이 baseline보다 높아 "전 지표 우위" 는 성립하지 않음	저 FAR 영역에서의 개선
손실 설계	물리 기반 손실 항이 유의한 기여를 보이지 않음	물리 정보를 실제로 반영하는 손실·구조 재설계
이력 창	windowed RMS sum은 0.786 s 이동합으로 사이클 길이의 약 7 %	더 긴 시간척도의 이력 표현 탐색
8. 용어 정리
용어	설명
Rockburst	고응력 암반에 저장된 탄성 변형 에너지가 굴착 등을 계기로 폭발적으로 방출되는 재해
AE (Acoustic Emission)	미세 균열 발생 시 방출되는 탄성파. 파괴 전조 탐지에 사용
TTF (Time to Failure)	파괴까지 남은 시간. 본 연구의 회귀 목표이자 경보 라벨의 기준
CTPF	Cross-Timescale Precursor state Fusion — 본 연구의 제안 모델
Intra / Inter 특징	현재 구간 내부에서만 계산되는 특징 / 구간 경계를 넘어야 얻어지는 특징
PR-AUC	Precision–Recall 곡선 아래 면적. 클래스 불균형에 강건한 주 지표 (기준선 = 양성 비율 0.276)
FAR (False Alarm Rate)	오경보율. 낮을수록 좋다
Matched FAR 평가	모델별 임계치가 만드는 운영점 차이를 제거하기 위해 모든 모델을 동일 FAR로 정렬해 재현율을 비교하는 방식
삭제 실험 (Deletion test)	어텐션이 높은 프레임을 가렸을 때의 출력 변화를 무작위 마스킹과 비교해, 가중치의 인과적 기여를 확인하는 검증
Holm–Bonferroni	다중비교 상황에서 1종 오류를 통제하는 보정 절차
참고문헌
Zhou, J., Li, X., Mitri, H. S., Classification of Rockburst in Underground Projects: Comparison of Ten Supervised Learning Methods, J. Comput. Civ. Eng., Vol. 30, No. 5, 04016003, 2016.
Rouet-Leduc, B., Hulbert, C., Lubbers, N., Barros, K., Humphreys, C. J., Johnson, P. A., Machine Learning Predicts Laboratory Earthquakes, Geophys. Res. Lett., Vol. 44, No. 18, pp. 9276–9282, 2017.
Lu, C.-P., Liu, G.-J., Liu, Y., Zhang, N., Xue, J.-H., Zhang, L., Microseismic Multi-parameter Characteristics of Rockburst Hazard Induced by Hard Roof Fall and High Stress Concentration, Int. J. Rock Mech. Min. Sci., Vol. 76, pp. 18–32, 2015.
Di, Y., Wang, E., Rock Burst Precursor Electromagnetic Radiation Signal Recognition Method and Early Warning Application Based on Recurrent Neural Networks, Rock Mech. Rock Eng., Vol. 54, No. 3, pp. 1449–1461, 2021.
Bolton, D. C., Shreedharan, S., Rivière, J., Marone, C., Acoustic Energy Release During the Laboratory Seismic Cycle: Insights on Laboratory Earthquake Precursors and Prediction, J. Geophys. Res. Solid Earth, Vol. 125, No. 8, e2019JB018975, 2020.
Wu, N., Jastrzębski, S., Cho, K., Geras, K. J., Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks, ICML, Baltimore, USA, 2022, pp. 24043–24055.
Grinsztajn, L., Oyallon, E., Varoquaux, G., Why Do Tree-based Models Still Outperform Deep Learning on Tabular Data?, NeurIPS, New Orleans, USA, 2022, pp. 507–520.
Johnson, P. A., Rouet-Leduc, B., Pyrak-Nolte, L. J., et al., Laboratory Earthquake Forecasting: A Machine Learning Competition, Proc. Natl. Acad. Sci. U.S.A., Vol. 118, No. 5, e2011362118, 2021.
Wiegreffe, S., Pinter, Y., Attention Is Not Not Explanation, EMNLP-IJCNLP, Hong Kong, China, 2019, pp. 11–20.
Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P., Focal Loss for Dense Object Detection, ICCV, Venice, Italy, 2017, pp. 2999–3007.
인용
bibtex
@inproceedings{jung2026rockburst,
  title     = {Rockburst Early Warning Using Cumulative Energy History and Acoustic Emission Signals},
  author    = {Jung, Jisung and Seol, Jaehoon and Kim, Gabeen and Lee, Quwon and Oh, JunSeok and Kim, Younggyun},
  booktitle = {Proceedings of the Annual Symposium of the Korea Information Processing Society (ASK 2026)},
  volume    = {33},
  number    = {2},
  pages     = {ooo--ooo},
  year      = {2026}
}
<div align="center">

Contact · 정지성 (교신) <a href="mailto:Jisung3911@gmail.com">Jisung3911@gmail.com</a>

<sub>본 저장소의 그림과 수치는 ASK 2026 발표 원고를 기준으로 하며, 모든 성능 수치는 15개 독립 시드(특징·손실 제거 실험은 8시드)의 평균이고 다중비교는 Holm–Bonferroni로 보정되었습니다.</sub>

</div>
