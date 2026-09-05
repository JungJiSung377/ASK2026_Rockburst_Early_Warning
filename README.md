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
현장 잡음 모사를 위해 최저 8개 주파수 bin(DC–10.0 kHz, bin 폭 1.33 kHz)에 가우시안 잡음을 가산했다. 단, 실측 TBM 스펙트럼이 아닌 가상 시나리오다.

<br>

## 3. Precursor State Vector

Intra 4종은 현재 구간만으로 산출되므로 원리적으로 스펙트로그램에서 복원 가능하다. Inter 2종은 직전 20구간의 이력을 봐야만 얻어지며, 이것이 검증 대상이다. Inter 특징은 현재 값을 이력에 반영하기 전에 계산하고 사이클 경계마다 초기화한다.

windowed RMS sum은 사이클 전체 누적합이 아닌 직전 20구간의 이동합이다(평균 사이클의 약 7 %).

<br>

## 4. Architecture

2-D CNN은 주파수 축만 축약하고 시간 프레임 101개를 보존한다 → 프레임별 어텐션 해석이 가능해진다.
LSTM은 구간 전체의 순서 의존성을 누적해 어텐션이 참조할 프레임별 표현을 만든다.
융합은 비대칭 — 느린 척도가 무엇을 볼지 질의(Q)하고, 빠른 척도가 근거(K, V)를 제공한다. Q 잔차 연결로 어텐션을 거치지 않는 경로도 확보한다.

<br>

## 5. Loss & Protocol

통합 손실 L = L_cls + λ_r·L_reg + λ_p·L_energy

L_cls : focal loss (γ = 2.0, class-balanced α)
L_reg : 잔여 시간 예측 오차 — TTF(0.006–16.103 s)는 학습 분할 최댓값으로 [0,1] 정규화 후 초 단위 역변환
L_energy : 방출이 활발한 구간에 더 큰 비용 (λ_p = 0.15)

운영 임계치 t* = argmax Recall_val(t)  s.t.  FAR_val(t) ≤ κ · FAR_target

FAR_target = 15 %, κ = 0.6 → 검증에서 9 % 를 겨냥
미탐지는 회복 불가능한 비용, 오경보는 회복 가능한 비용이라는 전제

평가 원칙

사이클 단위 시간순 분할 → 평가를 "미관측 사이클로의 일반화"로 정의
정규화 통계·운영 임계치를 학습·검증에서만 결정해 test에 불변 적용 (정보 유입 차단)
15개 독립 시드, 동일 시드 쌍 대응 t-검정, Holm–Bonferroni 보정, 주 지표 PR-AUC
모델별 운영점 차이를 제거하기 위해 8개 공통 FAR(2.5–20 %)에서 재평가

<br>

## 6. Results — Performance

15개 독립 시드 평균 ± 표준편차.

스펙트로그램 단독 대비 PR-AUC +25.3 %, 재현율 +30.5 %p, RMSE −0.352 s. 24개 주요 비교 전부 Holm 보정 후 유의.

FAR은 낮을수록 좋은 지표이며 제안 모델의 FAR이 더 높으므로 "전 지표 우위"는 성립하지 않는다. 또한 상태 벡터 6차원만 쓰는 Logistic/Ridge가 0.726으로 스펙트로그램 단독 딥러닝(0.588)을 크게 상회했다 — 판별력의 상당 부분이 저차원 상태 특징에 있다는 뜻이다.

<br>

## 7. Results — Matched FAR

모델별 임계치가 만드는 운영점 차이를 제거하고 동일 FAR에서 재현율을 비교했다.

FAR 12.5 % 이하에서는 고전 baseline과 유의차 없음 (단, 스펙트로그램 단독과의 차이는 전 구간 유의)
FAR 12.5–15 % 를 기점으로 격차가 벌어지며, FAR 20 %에서도 GB 대비 +10.8 %p, LR 대비 +9.1 %p 유지
어텐션 유무의 차이는 모든 FAR에서 유의하지 않음

저차원 상태 특징만으로는 낮은 오경보 조건의 판별에 그치지만, 스펙트로그램과의 결합이 추가 오경보를 재현율로 전환하는 능력을 제공한다. 미탐지 비용이 큰 조기경보 특성상 제안 모델의 이점은 목적에 부합하는 운영 구간에서 발현된다.

위 값은 평가 곡선에서 읽은 상한이며 배포 시 달성 성능이 아니다. 실제 달성치는 6절의 Recall 80.98 % / FAR 16.29 %다.

<br>

## 8. Results — Ablation

개별 성분 6종 · 군 단위 2종 · 전체 무력화 1종, 총 9개 설정. 제거는 0이 아닌 학습 분할 평균값으로 대체해 차원 수 변화 효과를 배제했다. (8 seed)

(1) 원인은 차원 수가 아니라 정보의 종류다. Inter 2종 제거 시 0.735 → 0.593(−19.30 %)로 스펙트로그램 단독 수준에 회귀하지만, Intra 4종을 전부 제거해도 −0.60 %로 유의하지 않다. 더 많이 빼도 영향이 없고 더 적게 뺐을 때 붕괴한다.

(2) 상태 벡터의 이득은 사실상 전부 Inter에서 온다. Inter 제거(0.5929)와 상태 벡터 전체 무력화(0.5942)가 통계적으로 구별되지 않아, 스펙트로그램 단독 대비 획득한 이득의 약 90 %가 Inter 2종과 관련된다.

(3) 두 Inter 특징은 대칭적이지 않다. 실제 정보를 담은 것은 windowed RMS sum(단독 제거 −0.052, p < 0.001)이고 추세 기울기는 단독으로는 잉여에 가깝다(+0.008, p = 0.049). 그럼에도 동시 제거 시 −0.142로 개별 합(−0.044)의 3.24배 — 한쪽이 남으면 다른 쪽을 부분적으로 대체하는 초가법적 관계다.

(4) 손실 함수와 융합 방식은 유의한 차이를 만들지 않았다. 융합 방식 차이 +0.003 (15 seed, 8개 지표 전부 Holm p = 1.00), 손실 항 제거 4개 설정 전부 n.s. (8 seed, Holm p = 1.00). 즉 성능을 견인한 것은 융합 메커니즘의 정교함이 아니라 두 시간척도를 결합한다는 사실 자체이며, 파라미터를 13.4 % 적게 쓰는(374,210 대 432,066) 단순 결합으로도 동등한 성능을 얻는다. 따라서 성능 근거로 "물리 정보 기반"을 주장하지 않는다.

<br>

## 9. Results — Attention as Evidence

어텐션은 성능을 올리지 않았다. 그러나 조기경보는 최종적으로 사람이 대피를 결정하는 체계이므로, 근거 제시 가능성을 3단계로 검증했다.

분포 — 경보 상태에서 구간 양 끝의 비중이 줄고 내부로 이동, 일부 프레임에서 유의.
집중도 — 경보 구간의 정규화 엔트로피가 평시보다 유의하게 낮음(p < 0.001). 단, 유효 프레임 수로는 여전히 101개의 절반 이상에 분포 → "집중된다"가 아니라 "유의하게 좁아지되 넓게 분포한다" 가 정확하다.
삭제 실험(인과) — 상위 k 개 어텐션 프레임 마스킹의 경보 확률 변화가 무작위 대비 약 3배, k = 5·10·20 전부 p < 0.001. 가중치가 예측에 실제로 관여한다는 직접 증거다.

어텐션의 가치는 성능이 아니라 검증 가능한 판단 근거에 있다. 동등 성능의 단순 결합은 경보의 시간적 근거를 제시하지 못한다.

<br>

## 10. Results — Latency

구간의 실제 발생 주기 39.30 ms 대비 5.33배 여유. NVIDIA A100-SXM4-40GB 기준이며 엣지 장비에서는 재측정이 필요하다.

<br>

## 11. Conclusion

이력 정보의 기여를 실증했다. 성능 향상의 약 90 %가 스펙트로그램으로 대체 불가능한 직전 20구간의 RMS 이동 이력에서 비롯된다.
향상의 원인은 결합 그 자체였다. 융합 방식의 차이는 유의하지 않으며, 파라미터를 13.4 % 적게 쓰는 단순 결합으로도 동등하다.
판단 근거를 인과적으로 검증했다. 삭제 실험에서 상위 어텐션 프레임의 기여가 무작위 대비 약 3배.
실시간 동작이 가능하다. 전처리 포함 7.37 ms.

<br>

## 12. Limitations

<br>

## References

1. Zhou, J., Li, X., Mitri, H. S., Classification of Rockburst in Underground Projects: Comparison of Ten Supervised Learning Methods, J. Comput. Civ. Eng., 30(5), 04016003, 2016.
2. Rouet-Leduc, B., et al., Machine Learning Predicts Laboratory Earthquakes, Geophys. Res. Lett., 44(18), 9276–9282, 2017.
3. Lu, C.-P., et al., Microseismic Multi-parameter Characteristics of Rockburst Hazard Induced by Hard Roof Fall and High Stress Concentration, Int. J. Rock Mech. Min. Sci., 76, 18–32, 2015.
4. Di, Y., Wang, E., Rock Burst Precursor Electromagnetic Radiation Signal Recognition Method and Early Warning Application Based on Recurrent Neural Networks, Rock Mech. Rock Eng., 54(3), 1449–1461, 2021.
5. Bolton, D. C., et al., Acoustic Energy Release During the Laboratory Seismic Cycle, J. Geophys. Res. Solid Earth, 125(8), e2019JB018975, 2020.
6. Wu, N., et al., Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks, ICML, 24043–24055, 2022.
7. Grinsztajn, L., et al., Why Do Tree-based Models Still Outperform Deep Learning on Tabular Data?, NeurIPS, 507–520, 2022.
8. Johnson, P. A., et al., Laboratory Earthquake Forecasting: A Machine Learning Competition, PNAS, 118(5), e2011362118, 2021.
9. Wiegreffe, S., Pinter, Y., Attention Is Not Not Explanation, EMNLP-IJCNLP, 11–20, 2019.
10. Lin, T.-Y., et al., Focal Loss for Dense Object Detection, ICCV, 2999–3007, 2017.

<br>

## Citation

@inproceedings{jung2026rockburst,
  title     = {Rockburst Early Warning Using Cumulative Energy History and Acoustic Emission Signals},
  author    = {Jung, Jisung and Seol, Jaehoon and Kim, Gabeen and Lee, Quwon and Oh, JunSeok and Kim, Younggyun},
  booktitle = {Proceedings of the Annual Symposium of the Korea Information Processing Society (ASK 2026)},
  volume    = {33},
  number    = {2},
  pages     = {ooo--ooo},
  year      = {2026}
}
