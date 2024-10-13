# LORA: 대규모 언어 모델의 저랭크 적응
Edward Hu∗ Yelong Shen∗ Phillip Wallis Zeyuan Allen-Zhu Yuanzhi Li Shean Wang Lu Wang Weizhu Chen
Microsoft Corporation
{edwardhu, yeshe, phwallis, zeyuana, yuanzhil, swang, luw, wzchen}@microsoft.com
yuanzhil@andrew.cmu.edu
(버전 2)

## 요약
자연어 처리의 중요한 패러다임은 일반 도메인 데이터에 대한 대규모 사전 학습과 특정 작업이나 도메인에 대한 적응을 포함한다. 우리가 더 큰 모델을 사전 학습할수록, 모든 모델 매개변수를 재학습하는 전체 미세 조정이 덜 실행 가능해진다. GPT-3 175B를 예로 들면, 각각 175B 매개변수를 가진 미세 조정된 모델의 독립적인 인스턴스를 배포하는 것은 비용이 매우 많이 든다. 우리는 저랭크 적응, 또는 LoRA를 제안한다. 이는 사전 학습된 모델 가중치를 고정하고 각 변환기 아키텍처 계층에 학습 가능한 랭크 분해 행렬을 주입한다. 이는 하류 작업에 대한 학습 가능한 매개변수의 수를 크게 줄인다. GPT-3 175B가 Adam으로 미세 조정된 것에 비해, LoRA는 학습 가능한 매개변수의 수를 10,000배, GPU 메모리 요구량을 3배 줄일 수 있다. LoRA는 RoBERTa, DeBERTa, GPT-2, 그리고 GPT-3에서 미세 조정과 동등하거나 더 나은 모델 품질을 보여준다. 이는 더 적은 학습 가능한 매개변수, 더 높은 학습 처리량을 가지며, 어댑터와 달리 추가적인 추론 지연이 없다. 우리는 또한 언어 모델 적응에서의 랭크 결핍에 대한 경험적인 조사를 제공하며, 이는 LoRA의 효과성에 대한 통찰력을 제공한다. 우리는 LoRA를 PyTorch 모델과 통합을 용이하게 하는 패키지를 제공하며, RoBERTa, DeBERTa, 그리고 GPT-2에 대한 우리의 구현과 모델 체크포인트를 제공한다. https://github.com/microsoft/LoRA 에서 확인할 수 있다.

## 1. 서론
자연어 처리에서 많은 응용 프로그램은 하나의 대규모 사전 학습된 언어 모델을 여러 하류 응용 프로그램에 적응하는 데 의존한다. 이러한 적응은 일반적으로 미세 조정을 통해 이루어지며, 이는 사전 학습된 모델의 모든 매개변수를 업데이트한다. 미세 조정의 주요 단점은 새 모델이 원래 모델과 동일한 많은 매개변수를 포함한다는 것이다. 더 큰 모델이 몇 개월마다 학습되면, 이는 GPT-2 (Radford et al., b) 또는 RoBERTa large (Liu et al., 2019)에 대한 단순한 "불편함"에서 GPT-3 (Brown et al., 2020)의 1750억 개의 학습 가능한 매개변수에 대한 중요한 배포 도전 과제로 바뀐다. 많은 사람들이 이를 완화하기 위해 일부 매개변수만 적응하거나 새로운 작업에 대한 외부 모듈을 학습하려고 했다. 이렇게 하면, 우리는 각 작업에 대해 사전 학습된 모델 외에 작업 특정 매개변수를 소량만 저장하고 로드해야 하므로, 배포 시 운영 효율성이 크게 향상된다. 그러나, 기존 기술은 여전히 한계가 있다.

모델의 깊이를 확장하거나 모델의 사용 가능한 시퀀스 길이를 줄이는 방식으로 추론 지연을 종종 도입합니다(Houlsby et al., 2019; Rebuffi et al., 2017)(Li & Liang, 2021; Lester et al., 2021; Hambardzumyan et al., 2020; Liu et al., 2021)(3장 참조). 더 중요한 것은, 이러한 방법들은 종종 미세 조정 기준선을 충족시키지 못하며, 효율성과 모델 품질 사이에 트레이드오프를 초래합니다.

우리는 Li et al. (2018a); Aghajanyan et al. (2020)의 연구에서 영감을 얻었습니다. 이 연구들은 학습된 과다 매개변수 모델이 실제로는 낮은 본질적 차원에 위치한다는 것을 보여줍니다. 우리는 모델 적응 과정에서 가중치의 변화도 낮은 "본질적 순위"를 가지고 있다는 가설을 세우고, 이를 바탕으로 저순위 적응(Low-Rank Adaptation, LoRA) 방법을 제안합니다. LoRA는 우리가 신경망의 일부 밀집 계층을 간접적으로 훈련시킬 수 있게 해주며, 이는 적응 과정에서 밀집 계층의 변화에 대한 순위 분해 행렬을 최적화함으로써 이루어집니다. 이는 동시에 사전 훈련된 가중치를 고정시키며, 그림 1에서 보여주는 것처럼 이루어집니다. GPT-3 175B를 예로 들면, 매우 낮은 순위(즉, 그림 1의 r)가 전체 순위(즉, d)가 12,288에 이르는 경우에도 충분하다는 것을 보여줍니다. 이는 LoRA가 저장 및 계산 효율성을 모두 갖추게 합니다. LoRA는 여러 가지 주요한 장점을 가지고 있습니다.

• 사전 훈련된 모델은 공유되어 다양한 작업에 대한 많은 작은 LoRA 모듈을 구축하는 데 사용될 수 있습니다. 우리는 공유 모델을 고정시키고, 그림 1의 행렬 A와 B를 교체함으로써 효율적으로 작업을 전환할 수 있습니다. 이는 저장 요구 사항과 작업 전환 오버헤드를 크게 줄입니다.
• LoRA는 훈련을 더 효율적으로 만들고, 적응형 최적화기를 사용할 때 하드웨어 진입 장벽을 최대 3배까지 낮춥니다. 이는 대부분의 매개변수에 대해 기울기를 계산하거나 최적화기 상태를 유지할 필요가 없기 때문입니다. 대신, 우리는 주입된, 훨씬 작은 저순위 행렬만 최적화합니다.
• 우리의 단순한 선형 설계는 고정된 가중치와 함께 훈련 가능한 행렬을 얻을 수 있게 하며, 이는 완전히 미세 조정된 모델에 비해 추론 지연을 도입하지 않습니다.
• LoRA는 많은 이전 방법과 직교하며, 많은 방법들과 결합될 수 있습니다. 예를 들어, 접두사 튜닝과 같은 것입니다. 우리는 부록 E에서 예를 제공합니다.

용어와 관례: 우리는 Transformer 아키텍처에 대한 자주 참조하며, 그 차원에 대한 전통적인 용어를 사용합니다. 우리는 Transformer 계층의 입력 및 출력 차원 크기를 d라고 부릅니다. 우리는 W, W, W, W를 사용하여 self-attention 모듈에서의 query/key/value/output 투영 행렬을 가리킵니다. W 또는 W는 사전 훈련된 가중치 행렬을 가리키며, ∆W는 적응 과정에서의 누적된 그래디언트 업데이트를 가리킵니다. 우리는 r을 사용하여 LoRA 모듈의 순위를 나타냅니다. 우리는 (Vaswani et al., 2017; Brown et al., 2020)에 의해 설정된 관례를 따르며, 모델 최적화를 위해 Adam (Loshchilov & Hutter, 2019; Kingma & Ba, 2017)을 사용하고, Transformer MLP feedforward 차원 $$d_{ffn}$$ 모델을 사용합니다.

2 문제 정의

우리의 제안은 훈련 목표에 대해 중립적이지만, 우리는 언어 모델링을 우리의 동기 부여 사례로 집중합니다. 아래에는 언어 모델링 문제와 특히 작업 특정 프롬프트가 주어진 조건부 확률의 최대화에 대한 간략한 설명이 있습니다.

Φ로 매개변수화된 사전 훈련된 자동 회귀 언어 모델 P(y|x)가 주어진다고 가정합니다. 예를 들어, P(y|x)는 Transformer 아키텍처(Radford et al., b; Brown et al., 2020)를 기반으로 하는 일반적인 다중 작업 학습자인 GPT가 될 수 있습니다. 이 사전 훈련된 모델을 요약, 기계 독해(MRC), 자연어를 SQL(NL2SQL)로 변환하는 등의 다운스트림 조건부 텍스트 생성 작업에 적응시키는 것을 고려해봅시다. 각 다운스트림 작업은 컨텍스트-타겟 쌍의 훈련 데이터셋으로 표현됩니다: Z = {(x_i, y_i)}_{i=1,..,N}, 여기서 x와 y는 모두 토큰의 시퀀스입니다. 예를 들어, NL2SQL에서, x_i는 자연어 쿼리이고 y_i는 해당 SQL 명령입니다. 요약의 경우, x_i는 기사의 내용이고 y_i는 그 요약입니다.

전체적인 미세조정 과정에서, 모델은 사전 학습된 가중치 Φ로 초기화되고, 조건부 언어 모델링 목표를 최대화하기 위해 경사를 반복적으로 따라 Φ + ∆Φ로 업데이트됩니다:

$$
\max_{\Phi} \sum_{(x,y)\in Z} \sum_{t=1}^{|y|} \log(P(y_t | x, y_{<t}; \Phi))
$$

전체 미세조정의 주요 단점 중 하나는 하류 작업마다 다른 파라미터 집합 ∆Φ를 학습하며, 이 파라미터 집합의 차원 |∆Φ|은 |Φ|와 같다는 것입니다. 따라서 사전 학습된 모델이 큰 경우 (예: |Φ| ≈ 175Billion인 GPT-3와 같은 경우), 많은 독립적인 미세조정 모델을 저장하고 배포하는 것은 까다롭거나 실현 가능하지 않을 수 있습니다.

이 논문에서는 더 많은 파라미터 효율적인 접근법을 채택하며, 작업 특정 파라미터 증가 ∆Φ = ∆Φ(Θ)는 훨씬 작은 크기의 파라미터 집합 Θ에 의해 추가로 인코딩됩니다. 여기서 |Θ| << |Φ|입니다. ∆Φ를 찾는 작업은 Θ에 대해 최적화하는 것이 됩니다:

$$
\max_{\Theta} \sum_{(x,y)\in Z} \sum_{t=1}^{|y|} \log(p(y_t | x, y_{<t}; \Phi_0 + \Delta\Phi(\Theta)))
$$

이후 섹션에서는 ∆Φ를 인코딩하기 위해 계산 및 메모리 효율적인 저랭크 표현을 사용하는 것을 제안합니다. 사전 학습된 모델이 GPT-3 175B인 경우, 학습 가능한 파라미터 |Θ|는 |Φ|의 0.01%에 불과할 수 있습니다.

3. 기존 솔루션은 충분하지 않은가?

우리가 해결하려는 문제는 결코 새로운 것이 아닙니다. 전이 학습의 시작 이후 수십 개의 작업들이 모델 적응을 더욱 파라미터 및 계산 효율적으로 만들려고 노력했습니다. 잘 알려진 작업들에 대한 설문조사는 6장에서 확인할 수 있습니다. 언어 모델링을 예로 들면, 효율적인 적응에 대한 두 가지 주요 전략이 있습니다: 어댑터 레이어 추가(Houlsby et al., 2019; Rebuffi et al., 2017; Pfeiffer et al., 2021; Ru¨ckle´ et al., 2020) 또는 입력 레이어 활성화의 일부 형태 최적화(Li & Liang, 2021; Lester et al., 2021; Hambardzumyan et al., 2020; Liu et al., 2021). 그러나 두 전략 모두 대규모 및 지연 시간에 민감한 생산 시나리오에서 제한 사항이 있습니다.

어댑터 레이어는 추론 지연을 도입합니다. 어댑터에는 많은 변형이 있습니다. 우리는 Transformer 블록당 두 개의 어댑터 레이어를 가진 Houlsby et al. (2019)의 원래 디자인과 블록당 하나만 가지지만 추가적인 LayerNorm(Ba et al., 2016)을 가진 Lin et al. (2020)의 최근 디자인에 초점을 맞춥니다. 레이어를 가지치기하거나 다중 작업 설정을 활용함으로써 전체 지연을 줄일 수 있지만, 어댑터 레이어에서 추가 계산을 우회하는 직접적인 방법은 없습니다. 어댑터 레이어는 작은 병목 차원을 가지고 있어 원래 모델의 1% 미만의 파라미터를 가지도록 설계되었기 때문에 이것은 문제가 아닌 것처럼 보입니다. 그러나 대규모 신경망은 하드웨어 병렬성에 의존하여 지연 시간을 줄이며, 어댑터 레이어는 순차적으로 처리되어야 합니다. 이는 일반적으로 배치 크기가 하나 정도로 작은 온라인 추론 설정에서 차이를 만듭니다. 단일 GPU에서 GPT-2(Radford et al., b) 중간을 실행하는 일반적인 시나리오에서는, 어댑터를 사용할 때 지연 시간이 눈에 띄게 증가하는 것을 볼 수 있습니다. 이는 매우 작은 병목 차원을 가지고 있음에도 불구하고입니다.

이 문제는 Shoeybi et al. (2020); Lepikhin et al. (2020)에서와 같이 모델을 샤딩해야 할 때 더욱 악화됩니다. 추가적인 깊이가 더 많은 동기식 GPU 작업을 필요로 하기 때문입니다. 이는 AllReduce와 Broadcast와 같은 작업이 필요하며, 이를 해결하기 위해 어댑터 파라미터를 여러 번 중복해서 저장하지 않는 한, 이 문제를 해결할 수 없습니다.

프롬프트를 직접 최적화하는 것은 어렵습니다. 다른 방향으로, 접두사 튜닝(Li & Liang, 2021)을 예로 들면, 다른 도전을 직면하게 됩니다. 접두사 튜닝은 최적화하기 어렵고, 학습 가능한 파라미터에서 성능이 비모노토닉하게 변화하는 것을 관찰했습니다. 이는 원래 논문에서의 유사한 관찰을 확인합니다. 더 근본적으로, 적응을 위해 시퀀스 길이의 일부를 예약하는 것은 하류 작업을 처리할 수 있는 시퀀스 길이를 필연적으로 줄입니다. 이로 인해 프롬프트 튜닝이 다른 방법에 비해 성능이 떨어지는 것으로 의심됩니다. 작업 성능에 대한 연구는 5장으로 미룹니다.

배치 크기 32 16 1
시퀀스 길이 512 256 128
|Θ| 0.5M 11M 11M
Fine-Tune/LoRA 1449.4±0.8 338.0±0.6 19.8±2.7
AdapterL 1482.0±1.0(+2.2%) 354.8±0.5(+5.0%) 23.9±2.1(+20.7%)
AdapterH 1492.2±1.0(+3.0%) 366.3±0.5(+8.4%) 25.8±2.2(+30.3%)
표1: GPT-2 중간 모델에서 단일 전방향 패스의 추론 지연 시간을 밀리초 단위로 측정하였으며, 100회 시행에 대한 평균값입니다. NVIDIA Quadro RTX 8000을 사용하였습니다. “|Θ|”는 어댑터 계층에서 학습 가능한 매개변수의 수를 나타냅니다. AdapterL과 AdapterH는 어댑터 튜닝의 두 가지 변형으로, 5.1절에서 설명하였습니다. 어댑터 계층에 의해 도입된 추론 지연 시간은 온라인, 짧은 시퀀스 길이 시나리오에서 상당히 중요할 수 있습니다. 전체 연구는 부록 B에서 확인할 수 있습니다.

4. 우리의 방법
우리는 LoRA의 간단한 설계와 그 실용적인 이점을 설명합니다. 여기서 제시된 원칙들은 딥러닝 모델의 밀집 계층에 적용할 수 있지만, 우리는 Transformer 언어 모델의 특정 가중치에만 초점을 맞추었습니다.

4.1 저랭크 매개변수화 업데이트 행렬
신경망은 행렬 곱셈을 수행하는 많은 밀집 계층을 포함하고 있습니다. 이러한 계층의 가중치 행렬은 일반적으로 전랭크를 가집니다. 특정 작업에 적응할 때, Aghajanyan 등(2020)은 사전 학습된 언어 모델이 낮은 "내재 차원"을 가지고 있으며, 더 작은 부분 공간으로의 무작위 투영에도 불구하고 효율적으로 학습할 수 있다는 것을 보여줍니다. 이에 영감을 받아, 우리는 가중치에 대한 업데이트도 적응 과정에서 낮은 "내재 랭크"를 가질 것이라고 가설을 세웠습니다. 사전 학습된 가중치 행렬 $$W \in R^{d \times k}$$에 대해, 우리는 그 업데이트를 저랭크 분해를 사용하여 표현함으로써 제한합니다. 즉, $$W + \Delta W = W + BA$$, 여기서 $$B \in R^{d \times r}, A \in R^{r \times k}$$이고 랭크 $$r \leq min(d, k)$$입니다.

학습 동안, $$W$$는 고정되어 있고 경사 업데이트를 받지 않으며, $$A$$와 $$B$$는 학습 가능한 매개변수를 포함합니다. $$W$$와 $$\Delta W = BA$$는 동일한 입력과 곱해지며, 그들의 각각의 출력 벡터는 좌표별로 합산됩니다. $$h = Wx$$에 대해, 우리의 수정된 전방향 패스는 다음과 같습니다: $$h = Wx + \Delta Wx = Wx + BAx$$.

우리는 그림 1에서 우리의 재매개변수화를 보여줍니다. 우리는 $$A$$에 대해 무작위 가우시안 초기화를 사용하고 $$B$$에 대해 0을 사용하므로, $$\Delta W = BA$$는 학습의 시작 시점에서 0입니다. 그런 다음 우리는 $$\Delta Wx$$를 $$\alpha$$로 스케일링합니다. 여기서 $$\alpha$$는 $$r$$에 대한 상수입니다. Adam을 사용하여 최적화할 때, 초기화를 적절히 스케일링하면 $$\alpha$$를 조정하는 것은 학습률을 조정하는 것과 대략적으로 동일합니다. 결과적으로, 우리는 단순히 $$\alpha$$를 우리가 시도하는 첫 번째 $$r$$로 설정하고 조정하지 않습니다. 이 스케일링은 $$r$$을 변화시킬 때 하이퍼파라미터를 다시 조정할 필요를 줄여줍니다(Yang & Hu, 2021).

전체 Fine-tuning의 일반화. 더 일반적인 형태의 fine-tuning은 사전 학습된 매개변수의 부분 집합을 학습하는 것을 허용합니다. LoRA는 한 걸음 더 나아가서, 가중치 행렬에 대한 누적된 경사 업데이트가 적응 과정에서 전랭크를 가질 필요가 없습니다. 이는 LoRA를 모든 가중치 행렬에 적용하고 모든 바이어스를 학습할 때, 우리는 사전 학습된 가중치 행렬의 랭크를 LoRA 랭크 $$r$$로 설정함으로써 전체 fine-tuning의 표현력을 대략적으로 회복한다는 것을 의미합니다. 다시 말해, 학습 가능한 매개변수의 수를 늘릴수록, LoRA 학습은 원래 모델을 학습하는 것으로 수렴하며, 어댑터 기반 방법은 MLP로 수렴하고, 접두사 기반 방법은 긴 입력 시퀀스를 처리할 수 없는 모델로 수렴합니다.

추가적인 추론 지연 없음. 제품에 배포될 때, 우리는 $$W = W + BA$$를 명시적으로 계산하고 저장하고, 평소처럼 추론을 수행할 수 있습니다. $$W$$와 $$BA$$ 모두 $$R^{d \times k}$$에 있음에 주목하세요. 다른 하위 작업으로 전환해야 할 때, 우리는 $$BA$$를 빼서 $$W$$를 복구하고, 다른 $$B'A'$$를 더하는 빠른 작업을 수행할 수 있습니다. 이는 매우 적은 메모리 오버헤드를 가집니다. 중요한 것은, 이 방법은 추론 시간에 추가적인 지연을 발생시키지 않습니다.

우리는 추론 과정에서 세밀하게 조정된 모델에 비해 추가적인 지연을 도입하지 않는다는 것을 보장합니다.
4.2 LoRA를 Transformer에 적용하기
원칙적으로, 우리는 신경망의 가중치 행렬의 어떤 부분집합에도 LoRA를 적용하여 학습 가능한 매개변수의 수를 줄일 수 있습니다. Transformer 아키텍처에서는 self-attention 모듈에 네 개의 가중치 행렬(W_q, W_k, W_v, W_o)이 있고 MLP 모듈에 두 개가 있습니다. 우리는 W_q (또는 W_k, W_v)를 출력 차원이 일반적으로 attention heads로 분할되더라도 d × d 차원의 단일 행렬로 취급합니다. 우리는 단순성과 매개변수 효율성을 위해 downstream 작업에 대해 attention 가중치만 조정하고 MLP 모듈을 고정(즉, downstream 작업에서 학습되지 않음)하는 것으로 연구를 제한합니다. 우리는 또한 Transformer에서 다른 유형의 attention 가중치 행렬을 조정하는 효과를 7.1절에서 더 자세히 연구합니다. MLP 레이어, LayerNorm 레이어, 그리고 편향에 대한 적응을 실증적으로 조사하는 것은 향후의 작업으로 남겨둡니다.

실용적인 이점과 한계. 가장 중요한 이점은 메모리와 저장 공간 사용의 감소에서 비롯됩니다. Adam으로 학습된 큰 Transformer의 경우, 우리는 고정된 모델 매개변수에 대한 최적화 상태를 저장할 필요가 없기 때문에 VRAM 사용량을 최대 2/3까지 줄일 수 있습니다. GPT-3 175B에서는 학습 중 VRAM 소비를 1.2TB에서 350GB로 줄입니다. r = 4이고 query와 value projection 행렬만이 조정되는 경우, 체크포인트 크기는 대략 10,000배(350GB에서 35MB로) 줄어듭니다. 이는 우리가 훨씬 적은 GPU로 학습하고 I/O 병목 현상을 피할 수 있게 해줍니다. 또 다른 이점은 모든 매개변수 대신 LoRA 가중치만 교체함으로써 훨씬 낮은 비용으로 작업을 전환할 수 있다는 것입니다. 이는 많은 맞춤형 모델을 생성하고 VRAM에 사전 학습된 가중치를 저장하는 기계에서 실시간으로 교체할 수 있게 합니다. 우리는 또한 GPT-3 175B에서 전체 fine-tuning에 비해 학습 중에 25%의 속도 향상을 관찰했습니다. 이는 대부분의 매개변수에 대한 기울기를 계산할 필요가 없기 때문입니다.

그러나 LoRA에도 한계가 있습니다. 예를 들어, 추가적인 추론 지연을 제거하기 위해 A와 B를 W에 흡수시키려는 경우, 다른 작업에 대한 입력을 단일 전달로 배치하는 것은 간단하지 않습니다. 그러나 가중치를 병합하지 않고 동적으로 배치에서 샘플에 사용할 LoRA 모듈을 선택하는 것은 지연이 중요하지 않은 시나리오에서 가능합니다.

5 실증적 실험
우리는 RoBERTa (Liu et al., 2019), DeBERTa (He et al., 2021), 그리고 GPT-2 (Radford et al., b)에서 LoRA의 downstream 작업 성능을 평가하고, GPT-3 175B (Brown et al., 2020)로 확장하기 전에 평가합니다. 우리의 실험은 자연어 이해(NLU)에서 생성(NLG)까지의 광범위한 작업을 다룹니다. 특히, 우리는 GLUE (Wang et al., 2019) 벤치마크에서 RoBERTa와 DeBERTa를 평가합니다. 우리는 Li & Liang (2021)의 GPT-2 설정을 따르고, 직접 비교를 위해 WikiSQL (Zhong et al., 2017) (NL to SQL queries)와 SAMSum (Gliwa et al., 2019) (대화 요약)을 GPT-3에서의 대규모 실험에 추가합니다. 우리가 사용하는 데이터셋에 대한 자세한 내용은 부록 C를 참조하십시오. 모든 실험에는 NVIDIA Tesla V100을 사용했습니다.

5.1 기준선
우리는 다른 기준선과 넓게 비교하기 위해 이전 작업에서 사용된 설정을 복제하고 가능한 경우에는 그들이 보고한 숫자를 재사용합니다. 그러나 이는 일부 기준선이 특정 실험에서만 나타날 수 있다는 것을 의미합니다.
Fine-Tuning(FT)은 적응을 위한 일반적인 접근법입니다. fine-tuning 동안, 모델은 사전 학습된 가중치와 편향으로 초기화되고, 모든 모델 매개변수는 기울기 업데이트를 받습니다. 간단한 변형은 일부 레이어만 업데이트하고 다른 레이어는 고정하는 것입니다. 우리는 이전 작업(Li & Liang, 2021)에서 보고된 GPT-2에 대한 이러한 기준선을 포함시킵니다. 이는 마지막 두 레이어만을 조정합니다(FTTop2).

모델&방법 #훈련 가능한
매개변수 MNLI SST-2 MRPC CoLA QNLI QQP RTE STS-B 평균
RoB (FT)* 125.0M 87.6 94.8 90.2 63.6 92.8 91.9 78.7 91.2 86.4
기본
RoB (BitFit)* 0.1M 84.7 93.7 92.7 62.0 91.8 84.0 81.5 90.8 85.2
기본
RoB (AdptD)* 0.3M 87.1 94.2 88.5 60.8 93.1 90.2 71.5 89.7 84.4
기본 ±.0 ±.1 ±1.1 ±.4 ±.1 ±.0 ±2.7 ±.3
RoB (AdptD)* 0.9M 87.3 94.7 88.4 62.6 93.0 90.6 75.9 90.3 85.4
기본 ±.1 ±.3 ±.1 ±.9 ±.2 ±.0 ±2.2 ±.1
RoB (LoRA) 0.3M 87.5 95.1 89.7 63.4 93.3 90.8 86.6 91.5 87.2
기본 ±.3 ±.2 ±.7 ±1.2 ±.3 ±.1 ±.7 ±.2
RoB (FT)* 355.0M 90.2 96.4 90.9 68.0 94.7 92.2 86.6 92.4 88.9
큰
RoB (LoRA) 0.8M 90.6 96.2 90.9 68.2 94.9 91.6 87.4 92.6 89.0
큰 ±.2 ±.5 ±1.2 ±1.9 ±.3 ±.1 ±2.5 ±.2
RoB (AdptP)† 3.0M 90.2 96.1 90.2 68.3 94.8 91.9 83.8 92.1 88.4
큰 ±.3 ±.3 ±.7 ±1.0 ±.2 ±.1 ±2.9 ±.7
RoB (AdptP)† 0.8M 90.5 96.6 89.7 67.8 94.8 91.7 80.1 91.9 87.9
큰 ±.3 ±.2 ±1.2 ±2.5 ±.3 ±.2 ±2.9 ±.4
RoB (AdptH)† 6.0M 89.9 96.2 88.7 66.5 94.7 92.1 83.4 91.0 87.8
큰 ±.5 ±.3 ±2.9 ±4.4 ±.2 ±.1 ±1.1 ±1.7
RoB (AdptH)† 0.8M 90.3 96.3 87.7 66.3 94.7 91.5 72.9 91.5 86.4
큰 ±.3 ±.5 ±1.7 ±2.0 ±.2 ±.1 ±2.9 ±.5
RoB (LoRA)† 0.8M 90.6 96.2 90.2 68.2 94.8 91.6 85.2 92.3 88.6
큰 ±.2 ±.5 ±1.0 ±1.9 ±.3 ±.2 ±1.1 ±.5
DeB (FT)* 1500.0M 91.8 97.2 92.0 72.0 96.0 92.7 93.9 92.9 91.1
XXL
DeB (LoRA) 4.7M 91.9 96.9 92.6 72.4 96.0 92.9 94.9 93.0 91.3
XXL ±.2 ±.2 ±.6 ±1.1 ±.1 ±.1 ±.4 ±.2
표 2: RoBERTa, RoBERTa, 그리고 DeBERTa는 다른 적응 방법들을 GLUE 벤치마크의 기본, 큰, XXL 크기에 적용하였습니다. MNLI의 경우 전체(일치 및 불일치) 정확도를, CoLA의 경우 Matthew의 상관관계를, STS-B의 경우 Pearson 상관관계를, 그리고 다른 작업들의 경우 정확도를 보고합니다. 모든 지표에서 높은 값이 더 좋습니다. *는 이전 작업에서 발표된 숫자를 나타냅니다. †는 Houlsby 등(2019)의 설정과 유사하게 구성된 실행을 나타냅니다.

Bias-only 또는 BitFit은 우리가 편향 벡터만 훈련하고 나머지는 모두 고정하는 기본선입니다. 동시에, 이 기본선은 BitFit(Zaken 등, 2021)에 의해 연구되었습니다.

Prefix-embedding tuning(PreEmbed)은 입력 토큰 사이에 특수 토큰을 삽입합니다. 이 특수 토큰들은 훈련 가능한 단어 임베딩을 가지며, 일반적으로 모델의 어휘에는 포함되어 있지 않습니다. 이러한 토큰을 어디에 배치할지는 성능에 영향을 줄 수 있습니다. 우리는 "prefixing"에 집중하며, 이는 프롬프트 앞에 이러한 토큰을 추가하는 것을 의미합니다. 그리고 "infixing"은 프롬프트에 추가하는 것을 의미하며, 이 두 가지는 모두 Li & Liang(2021)에서 논의되었습니다. 우리는 $$l_p$$ (resp. $$l_i$$)를 prefix(resp. infix) 토큰의 수로 표시합니다. 훈련 가능한 매개변수의 수는 $$|\Theta|=d \times (l_p + l_i)$$입니다.

Prefix-layer tuning(PreLayer)은 prefix-embedding tuning의 확장입니다. 특수 토큰에 대한 단어 임베딩(또는 동등하게, 임베딩 레이어 이후의 활성화)만 학습하는 대신, 모든 Transformer 레이어 이후의 활성화를 학습합니다. 이전 레이어에서 계산된 활성화는 단순히 훈련 가능한 것으로 대체됩니다. 결과적으로 훈련 가능한 매개변수의 수는 $$|\Theta|=L \times d \times (l_p + l_i)$$이며, 여기서 L은 Transformer 레이어의 수입니다.

Houlsby 등(2019)이 제안한 Adapter tuning은 자기 주의 모듈(그리고 MLP 모듈)과 그 후의 잔차 연결 사이에 adapter 레이어를 삽입합니다. adapter 레이어에는 두 개의 완전히 연결된 레이어와 그 사이의 비선형성이 있습니다. 이 원래 디자인을 AdapterH라고 부릅니다. 최근에, Lin 등(2020)은 MLP 모듈 이후에만 adapter 레이어를 적용하고 LayerNorm 이후에 적용하는 더 효율적인 디자인을 제안했습니다. 이를 AdapterL이라고 부릅니다. 이는 Pfeiffer 등(2021)에서 제안한 또 다른 디자인과 매우 유사하며, 이를 AdapterP라고 부릅니다. 또한, 우리는 일부 adapter 레이어를 삭제하여 더 큰 효율성을 얻는 또 다른 기본선인 AdapterDrop(Ru¨ckle´ 등, 2020)을 포함합니다. 가능한 한 많은 기본선과 비교하기 위해 이전 작업에서의 숫자를 인용합니다. 이들은 첫 번째 열에 별표(*)가 있는 행에 있습니다. 모든 경우에, 우리는 $$|\Theta|=Lˆ_{Adpt} \times (2 \times d_{model} \times r + r + d_{LN}) + 2 \times Lˆ_{LN} \times d_{model}$$를 가지며, 여기서 $$Lˆ_{Adpt}$$는 adapter 레이어의 수이고 $$Lˆ_{LN}$$는 훈련 가능한 LayerNorm의 수입니다(예: AdapterL).

LoRA는 기존 가중치 행렬에 병렬로 훈련 가능한 순위 분해 행렬 쌍을 추가합니다. 4.2절에서 언급했듯이, 우리는 대부분의 실험에서 간단하게 LoRA를 $$W_q$$와 $$W_v$$에만 적용합니다. 훈련 가능한 매개변수의 수는 순위 r과 원래 가중치의 모양에 의해 결정되며, $$|\Theta|=2 \times Lˆ_{LoRA} \times d_{model} \times r$$이며, 여기서 $$Lˆ_{LoRA}$$는 LoRA를 적용하는 가중치 행렬의 수입니다.

모델 및 방법 #훈련 가능한 E2ENLGChallenge
매개변수 BLEU NIST MET ROUGE-L CIDEr
GPT-2M(FT)* 354.92M 68.2 8.62 46.2 71.0 2.47
GPT-2M(AdapterL)* 0.37M 66.3 8.41 45.0 69.8 2.40
GPT-2M(AdapterL)* 11.09M 68.9 8.71 46.1 71.3 2.47
GPT-2M(AdapterH) 11.09M 67.3 8.50 46.0 70.7 2.44
±.6 ±.07 ±.2 ±.2 ±.01
GPT-2M(FTTop2)* 25.19M 68.1 8.59 46.0 70.8 2.41
GPT-2M(PreLayer)* 0.35M 69.7 8.81 46.1 71.4 2.49
GPT-2M(LoRA) 0.35M 70.4 8.85 46.8 71.8 2.53
±.1 ±.02 ±.2 ±.1 ±.02
GPT-2L(FT)* 774.03M 68.5 8.78 46.0 69.9 2.45
GPT-2L(AdapterL) 0.88M 69.1 8.68 46.3 71.4 2.49
±.1 ±.03 ±.0 ±.2 ±.0
GPT-2L(AdapterL) 23.00M 68.9 8.70 46.1 71.3 2.45
±.3 ±.04 ±.1 ±.2 ±.02
GPT-2L(PreLayer)* 0.77M 70.3 8.85 46.2 71.7 2.47
GPT-2L(LoRA) 0.77M 70.4 8.89 46.8 72.0 2.47
±.1 ±.02 ±.2 ±.2 ±.02
표 3: E2E NLG Challenge에서 다양한 적응 방법을 사용한 GPT-2 중간(M) 및 대형(L) 모델. 모든 메트릭에서 높은 값이 더 좋습니다. LoRA는 비교 가능하거나 더 적은 훈련 가능한 매개변수를 가진 여러 기준선을 능가합니다. 실험을 진행한 경우 신뢰 구간이 표시됩니다. *는 이전 작업에서 발표된 숫자를 나타냅니다.

5.2 ROBERTABASE/LARGE
RoBERTa(Liu et al., 2019)는 BERT(Devlin et al., 2019a)에서 처음 제안된 사전 훈련 레시피를 최적화하고, 훈련 가능한 매개변수를 크게 늘리지 않고 후자의 작업 성능을 향상시켰습니다. RoBERTa는 최근 몇 년 동안 GLUE 벤치마크(Wang et al., 2019)와 같은 NLP 리더보드에서 훨씬 큰 모델들에게 밀려났지만, 그 크기에 비해 경쟁력이 있고 실무자들 사이에서 인기 있는 사전 훈련 모델로 남아 있습니다. 우리는 HuggingFace Transformers 라이브러리(Wolf et al., 2020)에서 사전 훈련된 RoBERTa base(125M)와 RoBERTa large(355M)를 가져와서 GLUE 벤치마크에서 다양한 효율적인 적응 접근법의 성능을 평가합니다. 또한 Houlsby et al. (2019)와 Pfeiffer et al. (2021)를 그들의 설정에 따라 복제합니다. 공정한 비교를 위해, 우리는 어댑터와 비교할 때 LoRA를 평가하는 방법에 두 가지 중요한 변경 사항을 적용합니다. 첫째, 모든 작업에 대해 동일한 배치 크기를 사용하고 어댑터 기준선과 일치하도록 시퀀스 길이를 128로 설정합니다. 둘째, MRPC, RTE, STS-B에 대해 모델을 사전 훈련된 모델로 초기화하고, MNLI에 이미 적응된 모델이 아닌 fine-tuning 기준선과 같이 합니다. Houlsby et al. (2019)에서 더 제한적인 설정을 따르는 실행은 †로 표시됩니다. 결과는 표 2(Top Three Sections)에 제시되어 있습니다. 사용된 하이퍼파라미터에 대한 자세한 내용은 섹션 D.1을 참조하십시오.

5.3 DEBERTAXXL
DeBERTa (He et al., 2021)는 BERT의 최근 변형으로, 훨씬 더 큰 규모에서 훈련되었으며 GLUE (Wang et al., 2019) 및 SuperGLUE (Wang et al., 2020)와 같은 벤치마크에서 매우 경쟁력있게 수행됩니다. 우리는 LoRA가 완전히 fine-tuned된 DeBERTa XXL (1.5B)의 성능을 GLUE에서 여전히 따라잡을 수 있는지 평가합니다. 결과는 표 2 (Bottom Section)에 제시되어 있습니다. 사용된 하이퍼파라미터에 대한 자세한 내용은 섹션 D.2를 참조하십시오.

5.4 GPT-2MEDIUM/LARGE
NLU에서 LoRA가 전체 fine-tuning에 대한 경쟁력 있는 대안이 될 수 있다는 것을 보여준 후, 우리는 LoRA가 NLG 모델, 예를 들어 GPT-2 중간 및 대형(Radford et al., b)에서도 여전히 우세한지 확인하고자 합니다. 우리는 직접 비교를 위해 가능한 한 Li & Liang (2021)의 설정에 가깝게 유지합니다. 공간 제약으로 인해, 이 섹션에서는 E2E NLG Challenge (표 3)에 대한 결과만 제시합니다. WebNLG (Gardent et al., 2017) 및 DART (Nan et al., 2020)에 대한 결과는 섹션 F.1을 참조하십시오. 사용된 하이퍼파라미터의 목록은 섹션 D.3에 포함되어 있습니다.

#Trainable WikiSQL MNLI-m SAMSum
모델&방법
파라미터 정확도 (%) 정확도 (%) R1/R2/RL
GPT-3(FT) 175,255.8M 73.8 89.5 52.0/28.0/44.5
GPT-3(BitFit) 14.2M 71.3 91.0 51.3/27.4/43.5
GPT-3(PreEmbed) 3.2M 63.1 88.6 48.3/24.2/40.5
GPT-3(PreLayer) 20.2M 70.1 89.5 50.8/27.3/43.5
GPT-3(AdapterH) 7.1M 71.9 89.8 53.0/28.9/44.8
GPT-3(AdapterH) 40.1M 73.2 91.5 53.2/29.0/45.1
GPT-3(LoRA) 4.7M 73.4 91.7 53.8/29.8/45.9
GPT-3(LoRA) 37.7M 74.0 91.6 53.4/29.2/45.1
표4: GPT-3175B에 대한 다양한 적응 방법의 성능. WikiSQL에서의 논리적 형식 검증 정확도, MultiNLI-matched에서의 검증 정확도, 그리고 SAMSum에서의 Rouge-1/2/L을 보고합니다. LoRA는 완전한 미세 조정을 포함한 이전 접근법보다 더 나은 성능을 보여줍니다. WikiSQL에서는 ±0.5%, MNLI-m에서는 ±0.1%, SAMSum에서는 ±0.2/±0.2/±0.1의 변동이 있습니다.
5.5 GPT-3175B로 스케일링
LoRA에 대한 최종 스트레스 테스트로, 우리는 GPT-3를 1750억 개의 파라미터로 확장합니다. 높은 훈련 비용으로 인해, 우리는 무작위 시드에 대한 주어진 작업의 전형적인 표준 편차만을 보고하며, 모든 항목에 대해 하나를 제공하는 것은 아닙니다. 사용된 하이퍼파라미터에 대한 자세한 내용은 섹션 D.4를 참조하십시오.
표4에서 보여지는 것처럼, LoRA는 세 가지 데이터셋에서 모두 미세 조정 기준을 맞추거나 초과합니다. 모든 방법이 더 많은 훈련 가능한 파라미터를 가지는 것으로부터 단조롭게 이익을 얻는 것은 아니라는 점에 유의하십시오. 우리는 접두사 임베딩 튜닝을 위해 256개 이상의 특수 토큰을 사용하거나, 접두사 레이어 튜닝을 위해 32개 이상의 특수 토큰을 사용할 때 성능이 크게 떨어지는 것을 관찰했습니다. 이는 Li & Liang (2021)에서의 유사한 관찰을 뒷받침합니다. 이 현상에 대한 철저한 조사는 이 작업의 범위를 벗어나지만, 더 많은 특수 토큰을 가지는 것이 사전 훈련 데이터 분포에서 입력 분포를 더 멀리 이동시키는 원인이 될 수 있다고 우리는 의심합니다. 별도로, 우리는 섹션 F.3에서 저 데이터 영역에서 다양한 적응 접근법의 성능을 조사합니다.
6 관련 작업
Transformer 언어 모델. Transformer (Vaswani et al., 2017)는 자기 주의를 많이 사용하는 시퀀스-투-시퀀스 아키텍처입니다. Radford et al.(a)는 이를 자동 회귀 언어 모델링에 적용하여 Transformer 디코더의 스택을 사용했습니다. 이후로, Transformer 기반 언어 모델은 NLP를 지배하며, 많은 작업에서 최첨단을 달성했습니다. BERT (Devlin et al., 2019b)와 GPT-2 (Radford et al., b)와 함께 새로운 패러다임이 등장했습니다 - 둘 다 큰 Transformer 언어 모델입니다.

일반 도메인 데이터에 대한 사전 학습 후, 특정 작업에 대한 데이터를 미세 조정하는 것은 특정 작업 데이터에 대해 직접 학습하는 것에 비해 상당한 성능 향상을 제공합니다. 더 큰 Transformer를 학습시키는 것은 일반적으로 더 나은 성능을 가져오며, 이는 활발한 연구 방향입니다. GPT-3(Brown et al., 2020)는 현재까지 학습된 가장 큰 단일 Transformer 언어 모델로, 1750억 개의 파라미터를 가지고 있습니다.

프롬프트 엔지니어링과 미세 조정: GPT-3 175B는 몇 가지 추가 학습 예제만으로도 그 행동을 적응시킬 수 있지만, 결과는 입력 프롬프트에 크게 의존합니다(Brown et al., 2020). 이는 모델의 성능을 최대화하기 위해 프롬프트를 구성하고 형식화하는 경험적인 예술을 필요로 합니다. 이를 프롬프트 엔지니어링 또는 프롬프트 해킹이라고 합니다. 미세 조정은 일반 도메인에서 사전 학습된 모델을 특정 작업에 재학습시킵니다(Devlin et al., 2019b; Radford et al., a). 이의 변형은 파라미터의 일부만 학습하는 것을 포함합니다(Devlin et al., 2019b; Collobert & Weston, 2008), 그러나 실무자들은 종종 하류 성능을 최대화하기 위해 모든 파라미터를 재학습합니다. 그러나, GPT-3 175B의 거대함은 큰 체크포인트를 생성하고 높은 하드웨어 진입 장벽을 가지고 있기 때문에 일반적인 방식으로 미세 조정을 수행하는 것이 어렵습니다.

파라미터 효율적인 적응: 많은 사람들이 기존의 뉴럴 네트워크 계층 사이에 어댑터 계층을 삽입하는 것을 제안했습니다(Houlsby et al., 2019; Rebuffi et al., 2017; Lin et al., 2020). 우리의 방법은 가중치 업데이트에 대한 저랭크 제약을 부과하기 위해 유사한 병목 구조를 사용합니다. 주요한 기능적 차이점은 우리의 학습된 가중치가 추론 중에 주요 가중치와 병합될 수 있으며, 이는 어댑터 계층의 경우가 아닙니다(섹션 3). 어댑터의 현대적인 확장은 COMPACTER(Mahabadi et al., 2021)로, 기본적으로 어댑터 계층을 Kronecker 곱과 일부 사전 결정된 가중치 공유 체계를 사용하여 매개변수화합니다. 마찬가지로, LoRA와 다른 텐서 곱 기반 방법을 결합하면 그것의 파라미터 효율성을 향상시킬 수 있을 것이며, 이는 우리가 미래의 작업으로 남겨둡니다. 최근에는 많은 사람들이 미세 조정 대신 입력 단어 임베딩을 최적화하는 것을 제안했습니다. 이는 프롬프트 엔지니어링의 연속적이고 미분 가능한 일반화와 유사합니다(Li & Liang, 2021; Lester et al., 2021; Hambardzumyan et al., 2020; Liu et al., 2021). 우리는 실험 섹션에서 Li & Liang(2021)과의 비교를 포함합니다. 그러나, 이러한 작업들은 프롬프트에서 더 많은 특수 토큰을 사용함으로써만 확장할 수 있으며, 이는 위치 임베딩이 학습될 때 작업 토큰에 대한 사용 가능한 시퀀스 길이를 차지합니다.

딥러닝에서의 저랭크 구조: 저랭크 구조는 기계 학습에서 매우 흔합니다. 많은 기계 학습 문제들은 특정한 본질적인 저랭크 구조를 가지고 있습니다(Li et al., 2016; Cai et al., 2010; Li et al., 2018b; Grasedyck et al., 2013). 또한, 많은 딥러닝 작업들, 특히 과도하게 매개변수화된 뉴럴 네트워크를 가진 작업들은, 학습된 뉴럴 네트워크가 학습 후에 저랭크 속성을 가질 것이라는 것이 알려져 있습니다(Oymak et al., 2019). 일부 이전 작업들은 원래의 뉴럴 네트워크를 학습시킬 때 명시적으로 저랭크 제약을 부과하기도 했습니다(Sainath et al., 2013; Povey et al., 2018; Zhang et al., 2014; Jaderberg et al., 2014; Zhao et al., 2016; Khodak et al., 2021; Denil et al., 2014); 그러나, 우리가 알기로는, 이러한 작업들 중 어떤 것도 하류 작업에 적응하기 위해 고정된 모델에 대한 저랭크 업데이트를 고려하지 않았습니다. 이론적인 문헌에서는, 뉴럴 네트워크가 기본 개념 클래스가 특정 저랭크 구조를 가질 때, 다른 고전적인 학습 방법들, 포함하여 해당하는(유한 너비의) 뉴럴 탄젠트 커널(Allen-Zhu et al., 2019; Li & Liang, 2018)보다 더 우수하다는 것이 알려져 있습니다(Ghorbani et al., 2020; Allen-Zhu & Li, 2019; Allen-Zhu & Li, 2020a). 또 다른 이론적인 결과인 Allen-Zhu & Li(2020b)는 저랭크 적응이 적대적인 학습에 유용할 수 있다는 것을 제안합니다. 결론적으로, 우리는 우리가 제안한 저랭크 적응 업데이트가 문헌에 의해 잘 동기 부여되었다고 믿습니다.

7. 저랭크 업데이트 이해하기
LoRA의 경험적인 이점을 고려할 때, 우리는 하류 작업에서 학습된 저랭크 적응의 속성을 더 자세히 설명하고자 합니다. 저랭크 구조는 여러 실험을 동시에 실행할 수 있게 해주는 하드웨어 진입 장벽을 낮추는 것뿐만 아니라, 업데이트 가중치가 사전 학습된 가중치와 어떻게 연관되는지에 대한 더 나은 해석 가능성을 제공합니다. 우리는 우리가 가장 큰 훈련 가능한 파라미터의 감소(최대 10,000배)를 달성하면서 작업 성능에 부정적인 영향을 미치지 않은 GPT-3 175B에 대한 연구에 집중합니다.

우리는 다음 질문에 대답하기 위해 일련의 경험적 연구를 수행합니다: 1) 파라미터 예산 제약이 주어진 경우, 사전 학습된 Transformer에서 어떤 가중치 행렬의 부분 집합을 우리가 적응시켜야 하는가?

하류 작업의 성능을 극대화하기 위해 2) "최적"의 적응 행렬 ∆W는 실제로 순위 결함이 있는가? 만약 그렇다면, 실제로 사용하기에 좋은 순위는 무엇인가? 3) ∆W와 W 사이의 연결은 무엇인가? ∆W는 W와 높은 상관관계를 가지고 있는가? ∆W는 W에 비해 얼마나 큰가? 
우리는 질문 2)와 3)에 대한 답변이 하류 작업을 위한 사전 훈련된 언어 모델을 사용하는 기본 원칙에 대한 이해를 높이는 데 도움이 될 것이라고 믿습니다. 이는 NLP에서 중요한 주제입니다.

7.1 트랜스포머에서 어떤 가중치 행렬에 LoRA를 적용해야 하는가?
제한된 매개변수 예산이 주어졌을 때, 어떤 유형의 가중치를 LoRA로 조정하여 하류 작업에서 최상의 성능을 얻을 수 있을까요? 섹션 4.2에서 언급했듯이, 우리는 자기 주의 모듈에서의 가중치 행렬만을 고려합니다. 우리는 GPT-3 175B에 대해 18M(대략 FP16으로 저장될 경우 35MB)의 매개변수 예산을 설정하였고, 이는 우리가 한 종류의 주의 가중치를 조정하면 r = 8, 두 종류를 조정하면 r = 4로 해당됩니다. 결과는 표 5에 제시되어 있습니다.

W와 W를 모두 조정하면 전반적으로 최상의 성능을 얻을 수 있습니다. 우리는 무작위 시드 간의 표준 편차가 주어진 데이터셋에 대해 일관되게 나타나는 것을 발견했습니다. 이는 첫 번째 열에 보고되어 있습니다.

모든 매개변수를 ∆W 또는 ∆W에 넣으면 성능이 크게 저하되는 반면, W와 W를 모두 조정하면 최상의 결과를 얻을 수 있습니다. 이는 4의 순위조차도 ∆W에서 충분한 정보를 포착하여, 더 큰 순위를 가진 단일 유형의 가중치를 조정하는 것보다 더 많은 가중치 행렬을 조정하는 것이 바람직하다는 것을 제안합니다.

7.2 LoRA에 대한 최적의 순위 r은 무엇인가?
우리는 순위 r이 모델 성능에 미치는 영향에 주목합니다. 우리는 {W ,W }, {W ,W ,W ,W }, 그리고 단지 W를 비교하기 위해 조정합니다.

표 6은 놀랍게도, LoRA가 이미 매우 작은 r(특히 {W ,W }보다는 W)로 경쟁력 있는 성능을 보여준다는 것을 보여줍니다. 이는 업데이트 행렬 ∆W가 매우 작은 "내재적 순위"를 가질 수 있음을 제안합니다. 이러한 발견을 더 지지하기 위해, 우리는 다른 r의 선택과 다른 무작위 시드에 의해 학습된 부분 공간의 겹침을 확인합니다. 우리는 r을 증가시키는 것이 더 의미 있는 부분 공간을 커버하지 않는다고 주장하며, 이는 낮은 순위의 적응 행렬이 충분하다는 것을 제안합니다.

그러나, 우리는 작은 r이 모든 작업이나 데이터셋에 대해 작동한다고 기대하지 않습니다. 사전 훈련에 사용된 언어와 다른 언어에서의 하류 작업을 고려하면, 전체 모델을 재훈련하는 것(LoRA와 r=d의 유사한 LoRA)이 작은 r을 가진 LoRA보다 확실히 성능이 뛰어날 수 있습니다.

다른 r에 대한 부분공간 유사성. A와 A는 각각 r = 8과 64의 순위를 가진 동일한 사전 훈련된 모델을 사용하여 학습된 적응 행렬입니다. 우리는 특이값 분해를 수행하고 오른쪽 특이 유니터리 행렬 U와 U를 얻습니다. 우리는 다음 질문에 대답하고자 합니다: U의 상위 i 특이 벡터에 의해 생성된 부분 공간의 얼마나 많은 부분이 U의 상위 j 특이 벡터에 의해 생성된 부분 공간에 포함되어 있는가? 이런 양을 우리는 Grassmann 거리를 기반으로 한 정규화된 부분공간 유사성으로 측정합니다(보다 공식적인 논의는 부록 G 참조).

$$φ(A_{r=8},A_{r=64},i,j)= \frac{||U_{i}^{A_{r=8}} U_{j}^{A_{r=64}}||_{F}^{2}}{min(i,j)} ∈[0,1] (4)$$

여기서 $U_{i}^{A_{r=8}}$는 상위 i 특이 벡터에 해당하는 U의 열을 나타냅니다.

φ(·)의 범위는 [0,1]이며, 1은 부분 공간의 완전한 중첩을, 0은 완전한 분리를 나타냅니다. i와 j를 변화시키면서 φ가 어떻게 변하는지는 그림 3을 참조하십시오. 공간 제약으로 인해 96개의 레이어 중 48번째 레이어만 살펴보지만, 다른 레이어에 대해서도 동일한 결론이 유지됩니다. 이는 섹션 H.1에서 보여줍니다.

그림 3: A와 A의 열 벡터 사이의 부분 공간 유사성, ∆W와 ∆W에 대해.
세 번째와 네 번째 그림은 첫 두 그림의 왼쪽 하단 삼각형을 확대한 것입니다. r = 8의 상위 방향은 r = 64에 포함되어 있고, 그 반대도 마찬가지입니다.

그림 3에서 중요한 관찰을 합니다.
A와 A의 상위 특이 벡터에 해당하는 방향은 상당히 중첩되는 반면, 다른 것들은 그렇지 않습니다. 구체적으로, A의 ∆W(또는 ∆W)와 A의 ∆W(또는 ∆W)는 정규화된 유사성이 > 0.5인 차원 1의 부분 공간을 공유하며, 이는 r = 1이 GPT-3의 다운스트림 작업에서 상당히 잘 수행되는 이유를 설명합니다.

A와 A는 동일한 사전 훈련된 모델을 사용하여 학습되므로, 그림 3은 A와 A의 상위 특이 벡터 방향이 가장 유용하며, 다른 방향은 훈련 중에 축적된 대부분의 무작위 잡음을 포함할 수 있다는 것을 나타냅니다. 따라서 적응 행렬은 실제로 매우 낮은 순위를 가질 수 있습니다.

다른 랜덤 시드 간의 부분 공간 유사성. 우리는 두 개의 무작위로 시드된 실행 사이의 정규화된 부분 공간 유사성을 그림 4에 표시함으로써 이를 더 확인합니다. ∆W는 ∆W보다 더 높은 "내재적 순위"를 가지는 것으로 보이며, 이는 두 실행 모두에서 더 많은 공통 특이값 방향이 ∆W에 의해 학습되었음을 나타냅니다. 이는 표 6에서의 경험적 관찰과 일치합니다.
비교를 위해, 우리는 두 개의 무작위 가우시안 행렬을 그립니다. 이들은 서로 어떤 공통 특이값 방향도 공유하지 않습니다.

7.3 적응 행렬 ∆W는 W와 어떻게 비교되는가?
우리는 ∆W와 W 사이의 관계를 더 조사합니다. 특히, ∆W는 W와 높게 상관되는가? (또는 수학적으로, ∆W는 대부분 W의 상위 특이 방향에 포함되는가?) 또한, 비슷한 분석은 B와 왼쪽 특이 유니터리 행렬에 대해 수행될 수 있습니다. 우리는 우리의 실험을 위해 A를 사용합니다.

그림 4: 왼쪽과 가운데: 두 랜덤 시드에서 Ar=64의 열 벡터 사이의 정규화된 부분 공간 유사성, 48번째 레이어에서 ∆W와 ∆Wq, ∆Wv에 대해. 오른쪽: 두 랜덤 가우시안 행렬의 열 벡터 사이의 동일한 히트맵. 다른 레이어에 대해서는 섹션 H.1을 참조하십시오.

W와 그에 해당하는 방향에 비해 ∆W의 "크기"가 얼마나 큰지는 사전 훈련된 언어 모델을 적응하는 기본 메커니즘에 대한 이해를 돕습니다.

이 질문에 답하기 위해, ∆W의 r-차원 부분 공간에 W를 투영하여 U(cid:62)WV(cid:62)를 계산합니다. 여기서 U/V는 ∆W의 왼쪽/오른쪽 특이 벡터 행렬입니다. 그런 다음, (cid:107)U(cid:62)WV(cid:62)(cid:107)와 (cid:107)W(cid:107) 사이의 프로베니우스 노름을 비교합니다. 비교를 위해, 우리는 또한 W의 상위 r 특이 벡터 또는 랜덤 행렬로 U, V를 대체하여 (cid:107)U(cid:62)WV(cid:62)(cid:107)를 계산합니다.

표 7: U(cid:62)WV(cid:62)의 프로베니우스 노름, 여기서 U와 V는 (1) ∆W, (2) W, 또는 (3) 랜덤 행렬의 왼쪽/오른쪽 상위 r 특이 벡터 방향입니다. 가중치 행렬은 GPT-3의 48번째 레이어에서 가져옵니다.

표 7에서 몇 가지 결론을 내릴 수 있습니다. 첫째, ∆W는 랜덤 행렬에 비해 W와 더 강한 상관관계를 가지며, 이는 ∆W가 이미 W에 있는 일부 특징을 증폭한다는 것을 나타냅니다. 둘째, W의 상위 특이 방향을 반복하는 대신, ∆W는 W에서 강조되지 않은 방향만을 증폭합니다. 셋째, 증폭 요소는 상당히 크며, r = 4일 때 21.5 ≈ 6.91/0.32입니다. 왜 r = 64가 더 작은 증폭 요소를 가지는지에 대해서는 섹션 H.4를 참조하십시오. 또한, W에서 더 많은 상위 특이 방향을 포함하면서 상관관계가 어떻게 변하는지에 대한 시각화를 섹션 H.3에서 제공합니다.

이는 저랭크 적응 행렬이 일반적인 사전 훈련 모델에서 강조되지 않았던 특정 하위 스트림 작업에 중요한 특징을 증폭할 수 있음을 나타냅니다.

8 결론 및 향후 작업

거대한 언어 모델을 미세 조정하는 것은 필요한 하드웨어와 다른 작업에 대한 독립적인 인스턴스를 호스팅하는 저장/스위칭 비용 면에서 금지적으로 비싸다. 우리는 LoRA를 제안합니다. 이는 추론 지연을 도입하지 않고 입력 시퀀스 길이를 줄이지 않으면서 높은 모델 품질을 유지하는 효율적인 적응 전략입니다. 중요한 것은, 모델 매개변수의 대부분을 공유함으로써 서비스로 배포될 때 빠른 작업 전환을 가능하게 합니다. 우리는 Transformer 언어 모델에 초점을 맞추었지만, 제안된 원칙은 밀집 레이어가 있는 모든 신경망에 일반적으로 적용 가능합니다.

향후 작업 방향은 많습니다. 1) LoRA는 다른 효율적인 적응 방법과 결합될 수 있으며, 이는 직교적인 개선을 제공할 수 있습니다. 2) 미세 조정 또는 LoRA 뒤에 있는 메커니즘은 아직 명확하지 않습니다 - 사전 훈련 중에 학습된 특징은 하위 스트림 작업에서 어떻게 변환되어 잘 수행되는가? 우리는 LoRA가 이를 답하는 데 더 적합하다고 믿습니다.

3) 우리는 주로 LoRA를 적용할 가중치 행렬을 선택하기 위해 휴리스틱에 의존합니다. 이를 수행하는 더 원칙적인 방법이 있을까요? 4) 마지막으로, ∆W의 랭크 결핍은 W도 랭크 결핍일 수 있음을 제안하는데, 이는 또한 미래의 연구에 대한 영감의 원천이 될 수 있습니다.

참고문헌
Armen Aghajanyan, Luke Zettlemoyer, 그리고 Sonal Gupta. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. arXiv:2012.13255 [cs], 2020년 12월. URL http://arxiv.org/abs/2012.13255.
Zeyuan Allen-Zhu와 Yuanzhi Li. What Can ResNet Learn Efficiently, Going Beyond Kernels? NeurIPS, 2019. 전체 버전은 http://arxiv.org/abs/1905.10337에서 확인 가능.
Zeyuan Allen-Zhu와 Yuanzhi Li. Backward feature correction: How deep learning performs deep learning. arXiv preprint arXiv:2001.04413, 2020a.
Zeyuan Allen-Zhu와 Yuanzhi Li. Feature purification: How adversarial training performs robust deep learning. arXiv preprint arXiv:2005.10190, 2020b.
Zeyuan Allen-Zhu, Yuanzhi Li, 그리고 Zhao Song. A convergence theory for deep learning via over-parameterization. ICML, 2019. 전체 버전은 http://arxiv.org/abs/1811.03962에서 확인 가능.
Jimmy Lei Ba, Jamie Ryan Kiros, 그리고 Geoffrey E. Hinton. Layer normalization, 2016.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, 그리고 Dario Amodei. Language Models are Few-Shot Learners. arXiv:2005.14165 [cs], 2020년 7월. URL http://arxiv.org/abs/2005.14165.
Jian-Feng Cai, Emmanuel J Cande`s, 그리고 Zuowei Shen. A singular value thresholding algorithm for matrix completion. SIAM Journal on optimization, 20(4):1956–1982, 2010.
Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, 그리고 Lucia Specia. Semeval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), 2017. doi: 10.18653/v1/s17-2001. URL http://dx.doi.org/10.18653/v1/S17-2001.
Ronan Collobert 그리고 Jason Weston. A unified architecture for natural language processing: deep neural networks with multitask learning. Proceedings of the 25th international conference on Machine learning, ICML ’08, pp. 160–167, New York, NY, USA, 2008년 7월. Association for Computing Machinery. ISBN 978-1-60558-205-4. doi: 10.1145/1390156.1390177. URL https://doi.org/10.1145/1390156.1390177.
Misha Denil, Babak Shakibi, Laurent Dinh, Marc’Aurelio Ranzato, 그리고 Nando de Freitas. Predicting parameters in deep learning, 2014.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, 그리고 Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019a.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, 그리고 Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805 [cs], 2019년 5월. URL http://arxiv.org/abs/1810.04805. arXiv: 1810.04805.
William B. Dolan 그리고 Chris Brockett. Automatically constructing a corpus of sentential paraphrases. Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005. URL https://aclanthology.org/I05-5002.
Claire Gardent, Anastasia Shimorina, Shashi Narayan, 그리고 Laura Perez-Beltrachini. The webnlg challenge: Generating text from rdf data. Proceedings of the 10th International Conference on Natural Language Generation, pp. 124–133, 2017.

Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, 그리고 Andrea Montanari. 신경망이 커널 방법을 능가하는 시점은 언제인가? arXiv 사전 인쇄 arXiv:2006.13409, 2020.
Bogdan Gliwa, Iwona Mochol, Maciej Biesek, 그리고 Aleksander Wawer. Samsum corpus: 추상적 요약을 위한 인간 주석이 달린 대화 데이터셋. CoRR, abs/1911.12237, 2019. URL http://arxiv.org/abs/1911.12237.
Lars Grasedyck, Daniel Kressner, 그리고 Christine Tobler. 저랭크 텐서 근사 기법에 대한 문헌 조사. GAMM-Mitteilungen, 36(1):53–78, 2013.
Jihun Ham 그리고 Daniel D. Lee. Grassmann 판별 분석: 부분 공간 기반 학습에 대한 통합적인 시각. ICML에서, pp. 376–383, 2008. URL https://doi.org/10.1145/1390156.1390204.
Karen Hambardzumyan, Hrant Khachatrian, 그리고 Jonathan May. WARP: Word-level Adversarial ReProgramming. arXiv:2101.00121[cs], 2020년 12월. URL http://arxiv.org/abs/2101.00121. arXiv: 2101.00121.
Pengcheng He, Xiaodong Liu, Jianfeng Gao, 그리고 Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention, 2021.
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, 그리고 Sylvain Gelly. Parameter-Efficient Transfer Learning for NLP. arXiv:1902.00751 [cs, stat], 2019년 6월. URL http://arxiv.org/abs/1902.00751.
Max Jaderberg, Andrea Vedaldi, 그리고 Andrew Zisserman. 저랭크 확장을 이용한 합성곱 신경망 가속화. arXiv 사전 인쇄 arXiv:1405.3866, 2014.
Mikhail Khodak, Neil Tenenholtz, Lester Mackey, 그리고 Nicolo` Fusi. 팩터화된 신경 층의 초기화와 정규화, 2021.
Diederik P. Kingma 그리고 Jimmy Ba. Adam: 확률적 최적화를 위한 방법, 2017.
Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, 그리고 Zhifeng Chen. Gshard: 조건부 계산과 자동 샤딩을 이용한 거대 모델 확장, 2020.
Brian Lester, Rami Al-Rfou, 그리고 Noah Constant. The Power of Scale for Parameter-Efficient Prompt Tuning. arXiv:2104.08691[cs], 2021년 4월. URL http://arxiv.org/abs/2104.08691. arXiv: 2104.08691.
Chunyuan Li, Heerad Farkhoor, Rosanne Liu, 그리고 Jason Yosinski. 목표 지형의 본질적 차원 측정. arXiv:1804.08838 [cs, stat], 2018년 4월. URL http://arxiv.org/abs/1804.08838. arXiv: 1804.08838.
Xiang Lisa Li 그리고 Percy Liang. Prefix-Tuning: 생성을 위한 연속적인 프롬프트 최적화. arXiv:2101.00190[cs], 2021년 1월. URL http://arxiv.org/abs/2101.00190.
Yuanzhi Li 그리고 Yingyu Liang. 구조화된 데이터를 통한 과매개변수화된 신경망 학습. Advances in Neural Information Processing Systems에서, 2018.
Yuanzhi Li, Yingyu Liang, 그리고 Andrej Risteski. 교대 최소화를 통한 가중치 저랭크 근사의 복구 보장. International Conference on Machine Learning에서, pp. 2358–2367. PMLR, 2016.
Yuanzhi Li, Tengyu Ma, 그리고 Hongyang Zhang. 과매개변수화된 행렬 감지와 이차 활성화를 가진 신경망에서의 알고리즘 정규화. Conference On Learning Theory에서, pp. 2–47. PMLR, 2018b.
Zhaojiang Lin, Andrea Madotto, 그리고 Pascale Fung. 매개변수 효율적인 전송 학습을 통한 다재다능한 생성 언어 모델 탐색. Association for Computational Linguistics: EMNLP 2020의 발견에서, pp. 441–459, 온라인, 2020년 11월. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.41. URL https://aclanthology.org/2020.findings-emnlp.41.

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, 그리고 Jie Tang. GPT도 이해한다. arXiv:2103.10385 [cs], 2021년 3월. URL http://arxiv.org/abs/2103.10385. arXiv: 2103.10385.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, 그리고 Veselin Stoyanov. Roberta: 견고하게 최적화된 BERT 사전 학습 접근법, 2019.

Ilya Loshchilov 그리고 Frank Hutter. 분리된 가중치 감소 정규화. arXiv 사전 인쇄 arXiv:1711.05101, 2017.

Ilya Loshchilov 그리고 Frank Hutter. 분리된 가중치 감소 정규화, 2019.

Rabeeh Karimi Mahabadi, James Henderson, 그리고 Sebastian Ruder. Compacter: 효율적인 저랭크 초복소 어댑터 레이어, 2021.

Linyong Nan, Dragomir Radev, Rui Zhang, Amrit Rau, Abhinand Sivaprasad, Chiachun Hsieh, Xiangru Tang, Aadit Vyas, Neha Verma, Pranav Krishna 등. Dart: 오픈 도메인 구조화된 데이터 레코드 텍스트 생성. arXiv 사전 인쇄 arXiv:2007.02871, 2020.

Jekaterina Novikova, Ondˇrej Dušek, 그리고 Verena Rieser. e2e 데이터셋: 종단 간 생성을 위한 새로운 도전 과제. arXiv 사전 인쇄 arXiv:1706.09254, 2017.

Samet Oymak, Zalan Fabian, Mingchen Li, 그리고 Mahdi Soltanolkotabi. 자코비안의 저랭크 구조를 활용한 신경망을 위한 일반화 보장. arXiv 사전 인쇄 arXiv:1906.05392, 2019.

Jonas Pfeiffer, Aishwarya Kamath, Andreas Ru¨ckle´, Kyunghyun Cho, 그리고 Iryna Gurevych. 어댑터 퓨전: 전송 학습을 위한 비파괴적 작업 구성, 2021.

Daniel Povey, Gaofeng Cheng, Yiming Wang, Ke Li, Hainan Xu, Mahsa Yarmohammadi, 그리고 Sanjeev Khudanpur. 딥 뉴럴 네트워크를 위한 반직교 저랭크 행렬 인수분해. Interspeech에서, pp.3743–3747, 2018.

Alec Radford, Karthik Narasimhan, Tim Salimans, 그리고 Ilya Sutskever. 생성적 사전 학습을 통한 언어 이해 향상. pp. 12, a.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, 그리고 Ilya Sutskever. 언어 모델은 비지도 다중 작업 학습자다. pp. 24, b.

Pranav Rajpurkar, Robin Jia, 그리고 Percy Liang. 당신이 모르는 것을 알아라: SQuAD를 위한 불가능한 질문들. CoRR, abs/1806.03822, 2018. URL http://arxiv.org/abs/1806.03822.

Sylvestre-Alvise Rebuffi, Hakan Bilen, 그리고 Andrea Vedaldi. 잔여 어댑터를 이용한 다중 시각 도메인 학습. arXiv:1705.08045[cs,stat], 2017년 11월. URL http://arxiv.org/abs/1705.08045. arXiv: 1705.08045.

Andreas Ru¨ckle´, Gregor Geigle, Max Glockner, Tilman Beck, Jonas Pfeiffer, Nils Reimers, 그리고 Iryna Gurevych. 어댑터 드롭: 트랜스포머에서 어댑터의 효율성, 2020.

Tara N Sainath, Brian Kingsbury, Vikas Sindhwani, Ebru Arisoy, 그리고 Bhuvana Ramabhadran. 고차원 출력 대상을 가진 딥 뉴럴 네트워크 학습을 위한 저랭크 행렬 인수분해. 2013 IEEE 국제 음향, 음성 및 신호 처리 회의에서, pp. 6655–6659. IEEE, 2013.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, 그리고 Bryan Catanzaro. Megatron-lm: 모델 병렬화를 사용한 수십억 파라미터 언어 모델 학습, 2020.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, 그리고 Christopher Potts. 감정 트리뱅크에 대한 의미 구성성을 위한 재귀적 딥 모델. 2013년 자연어 처리 실증 방법에 대한 회의에서, pp.1631–1642, 시애틀, 워싱턴, 미국, 2013년 10월. 계산 언어학회. URL https://aclanthology.org/D13-1170. 15

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, 그리고 Illia Polosukhin. "Attention is all you need". 31회 신경정보처리시스템 국제회의 논문집, pp.6000-6010, 2017.
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, 그리고 Samuel R. Bowman. "Glue: 자연어 이해를 위한 다중 작업 벤치마크 및 분석 플랫폼", 2019.
Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, 그리고 Samuel R. Bowman. "Superglue: 일반 목적 언어 이해 시스템을 위한 더욱 끈적한 벤치마크", 2020.
Alex Warstadt, Amanpreet Singh, 그리고 Samuel R Bowman. "신경망 수용성 판단". arXiv 사전 인쇄 arXiv:1805.12471, 2018.
Adina Williams, Nikita Nangia, 그리고 Samuel Bowman. "추론을 통한 문장 이해를 위한 광범위한 도전 말뭉치". 2018년 북미 컴퓨터 언어학회: 인간 언어 기술 학회, 제1권 (롱 페이퍼), pp.1112-1122, 루이지애나, 뉴올리언스, 2018년 6월. 컴퓨터 언어학회. doi: 10.18653/v1/N18-1101.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Re´mi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, 그리고 Alexander M. Rush. "Transformers: 최첨단 자연어 처리". 2020년 경험적 방법을 이용한 자연어 처리 학회: 시스템 데모, pp. 38-45, 온라인, 2020년 10월. 컴퓨터 언어학회.
Greg Yang 그리고 Edward J. Hu. "무한폭 신경망에서의 특징 학습". arXiv:2011.14522[cond-mat], 2021년 5월.
Elad Ben Zaken, Shauli Ravfogel, 그리고 Yoav Goldberg. "Bitfit: Transformer 기반 마스크 언어 모델을 위한 간단하고 효율적인 파인 튜닝", 2021.
Yu Zhang, Ekapol Chuangsuwanich, 그리고 James Glass. "저랭크 행렬 분해를 이용한 딥 신경망 병목 특징 추출". 2014년 IEEE 국제 음향, 음성 및 신호 처리 학회 (ICASSP), pp.185-189. IEEE, 2014.
Yong Zhao, Jinyu Li, 그리고 Yifan Gong. "딥 신경망을 위한 저랭크 플러스 대각선 적응". 2016년 IEEE 국제 음향, 음성 및 신호 처리 학회 (ICASSP), pp.5005-5009. IEEE, 2016.
Victor Zhong, Caiming Xiong, 그리고 Richard Socher. "Seq2sql: 강화 학습을 이용한 구조화된 쿼리 생성". CoRR, abs/1709.00103, 2017.

큰 언어 모델들은 여전히 매개변수 업데이트가 필요합니다
훈련 샘플이 소수일 때, 피-shot 학습 또는 프롬프트 엔지니어링은 매우 유리합니다. 그러나 실제로는, 성능에 민감한 애플리케이션에 대해 수천 개 이상의 훈련 예제를 정리할 수 있습니다. 표 8에서 보여지는 것처럼, fine-tuning은 대규모 및 소규모 데이터셋에서 피-shot 학습에 비해 모델 성능을 크게 향상시킵니다. 우리는 GPT-3 페이퍼(Brown et al., 2020)에서 RTE에 대한 GPT-3 피-shot 결과를 가져왔습니다. MNLI-matched의 경우, 우리는 클래스당 두 개의 데모와 총 여섯 개의 in-context 예제를 사용했습니다.

방법 MNLI-m(검증 정확도/%) RTE(검증 정확도/%)
GPT-3Few-Shot 40.6 69.0
GPT-3Fine-Tuned 89.5 85.4
표8: GPT-3(Brown 등., 2020)에서는 Fine-tuning이 Few-shot 학습을 크게 앞섭니다.

B 추론 지연 시간: 어댑터 레이어에 의한
어댑터 레이어는 사전 훈련된 모델에 순차적으로 추가되는 외부 모듈이며, 반면에 우리의 제안인 LoRA는 병렬적으로 추가되는 외부 모듈로 볼 수 있습니다. 따라서 어댑터 레이어는 기본 모델 외에도 계산되어야 하므로, 추가적인 지연 시간을 불가피하게 도입합니다. Ru¨ckle´ 등.(2020)에서 지적한 것처럼, 모델 배치 크기와/또는 시퀀스 길이가 하드웨어 병렬성을 충분히 활용할 수 있을 정도로 크면 어댑터 레이어에 의해 도입된 지연 시간을 완화할 수 있습니다. 우리는 GPT-2 중간 크기 모델에서 비슷한 지연 시간 연구를 통해 이러한 관찰을 확인하고, 배치 크기가 작은 온라인 추론과 같이 추가된 지연 시간이 상당히 큰 시나리오가 있다는 점을 지적합니다.
우리는 NVIDIA Quadro RTX8000에서 단일 포워드 패스의 지연 시간을 100회 시행의 평균으로 측정합니다. 우리는 입력 배치 크기, 시퀀스 길이, 그리고 어댑터 병목 차원 r을 변화시킵니다. 우리는 두 가지 어댑터 디자인을 테스트합니다: Houlsby 등.(2019)에 의한 원래의 것을 AdapterH라고 부르고, Lin 등.(2020)에 의한 최근의 더 효율적인 변형을 AdapterL이라고 부릅니다. 디자인에 대한 자세한 내용은 5.1절을 참조하십시오. 우리는 어댑터가 없는 기준에 비해 느려진 비율을 그림 5에 표시합니다.

그림5: 어댑터가 없는 (r = 0) 기준에 비해 추론 지연 시간의 백분율 느려짐. 상단 행은 AdapterH의 결과를 보여주고, 하단 행은 AdapterL을 보여줍니다. 더 큰 배치 크기와 시퀀스 길이는 지연 시간을 완화하는 데 도움이 되지만, 온라인, 짧은 시퀀스 길이 시나리오에서는 느려짐이 30% 이상이 될 수 있습니다. 우리는 더 나은 가시성을 위해 컬러맵을 조정합니다.

C 데이터셋 세부 사항
GLUE 벤치마크는 자연어 이해 작업의 광범위한 컬렉션입니다. 이에는 MNLI(추론, Williams 등.(2018)), SST-2(감성 분석, Socher 등.(2013)), MRPC(패러프레이즈 감지, Dolan & Brockett (2005)), CoLA(언어적 수용성, Warstadt 등.(2018)), QNLI(추론, Rajpurkar 등.(2018)), QQP8(질문-답변), RTE(추론) 등이 포함됩니다.

그리고 STS-B (텍스트 유사성, Cer 등 (2017)). 이러한 폭넓은 커버리지로 인해 GLUE 벤치마크는 RoBERTa와 DeBERTa와 같은 NLU 모델을 평가하는 표준 지표가 되었습니다. 개별 데이터셋은 각기 다른 허용 라이선스 하에 공개되었습니다.

WikiSQL은 Zhong 등 (2017)에서 소개되었으며, 56,355개의 학습 예제와 8,421개의 검증 예제를 포함하고 있습니다. 이 작업의 목표는 자연어 질문과 테이블 스키마로부터 SQL 쿼리를 생성하는 것입니다. 우리는 컨텍스트를 x = {테이블 스키마, 쿼리}로 인코딩하고, 타겟을 y = {SQL}로 인코딩합니다. 이 데이터셋은 BSD 3-Clause License 하에 공개되었습니다.

SAMSum은 Gliwa 등 (2019)에서 소개되었으며, 14,732개의 학습 예제와 819개의 테스트 예제를 포함하고 있습니다. 이는 두 사람 사이의 연출된 채팅 대화와 언어학자가 작성한 해당 요약문으로 구성되어 있습니다. 우리는 컨텍스트를 "\n"으로 연결된 발화문에 이어 "\n\n"을 추가하여 인코딩하고, 타겟을 y = {요약}으로 인코딩합니다. 이 데이터셋은 비상업적 라이선스인 Creative Commons BY-NC-ND 4.0 하에 공개되었습니다.

E2E NLG Challenge는 처음으로 Novikova 등 (2017)에서 소개되었으며, end-to-end, 데이터 기반의 자연어 생성 시스템 학습을 위한 데이터셋으로 사용되며, 데이터-텍스트 평가에 일반적으로 사용됩니다. E2E 데이터셋은 대략 42,000개의 학습, 4,600개의 검증, 그리고 4,600개의 테스트 예제로 구성되어 있으며, 이는 모두 레스토랑 도메인에서 추출되었습니다. 각 입력 소스 테이블은 여러 참조를 가질 수 있습니다. 각 샘플 입력 (x,y)은 슬롯-값 쌍의 시퀀스와 함께 해당하는 자연어 참조 텍스트로 구성됩니다. 이 데이터셋은 Creative Commons BY-NC-SA 4.0 하에 공개되었습니다.

DART는 Nan 등 (2020)에서 설명된 오픈 도메인 데이터-텍스트 데이터셋입니다. DART 입력은 ENTITY - RELATION - ENTITY 삼중체의 시퀀스로 구성됩니다. 총 82K개의 예제로, DART는 E2E에 비해 훨씬 크고 복잡한 데이터-텍스트 작업입니다. 이 데이터셋은 MIT 라이선스 하에 공개되었습니다.

WebNLG는 또 다른 데이터-텍스트 평가에 일반적으로 사용되는 데이터셋입니다(Gardent 등, 2017). 총 22K개의 예제로, WebNLG는 14개의 고유한 카테고리를 포함하며, 이 중 9개는 학습 중에 볼 수 있습니다. 총 14개의 카테고리 중 5개는 학습 중에는 볼 수 없지만 테스트 세트에 표시되므로, 평가는 일반적으로 "보인" 카테고리(S), "보이지 않은" 카테고리(U) 그리고 "모든" 카테고리(A)로 나누어 진행됩니다. 각 입력 예제는 SUBJECT - PROPERTY - OBJECT 삼중체의 시퀀스로 표현됩니다. 이 데이터셋은 Creative Commons BY-NC-SA 4.0 하에 공개되었습니다.

실험에서 사용된 하이퍼파라미터

D.1 ROBERTA

우리는 AdamW를 사용하여 선형 학습률 감소 스케줄로 학습합니다. 우리는 LoRA의 학습률, 학습 에폭 수, 그리고 배치 크기를 조정합니다. Liu 등 (2019)을 따라, 우리는 MRPC, RTE, 그리고 STS-B에 적응할 때 LoRA 모듈을 MNLI 체크포인트로 초기화합니다, 이는 일반적인 초기화 방법이 아닙니다; 사전 학습된 모델은 모든 작업에 대해 고정되어 있습니다. 우리는 5개의 랜덤 시드에 대한 중앙값을 보고합니다; 각 실행의 결과는 최적의 에폭에서 얻어집니다. Houlsby 등 (2019)과 Pfeiffer 등 (2021)의 설정과 공정한 비교를 위해, 우리는 모델의 시퀀스 길이를 128로 제한하고 모든 작업에 대해 고정된 배치 크기를 사용합니다. 중요한 점은, 우리가 MRPC, RTE, 그리고 STS-B에 적응할 때, MNLI에 이미 적응된 모델이 아닌 사전 학습된 RoBERTa large 모델로 시작한다는 것입니다. 이 제한된 설정으로 실행된 실행은 †로 표시됩니다. 우리의 실행에서 사용된 하이퍼파라미터는 표 9에서 확인할 수 있습니다.

D.2 DEBERTA

우리는 다시 AdamW를 사용하여 선형 학습률 감소 스케줄로 학습합니다. He 등 (2021)을 따라, 우리는 학습률, 드롭아웃 확률, 워밍업 스텝, 그리고 배치 크기를 조정합니다. 우리는 He 등 (2021)이 사용한 것과 동일한 모델 시퀀스 길이를 사용하여 비교가 공정하게 이루어지도록 합니다. He 등 (2021)을 따라, 우리는 MRPC, RTE, 그리고 STS-B에 적응할 때 LoRA 모듈을 MNLI 체크포인트로 초기화합니다, 이는 일반적인 초기화 방법이 아닙니다; 사전 학습된 모델은 모든 작업에 대해 고정되어 있습니다. 우리는 5개의 랜덤 시드에 대한 중앙값을 보고합니다; 각 실행의 결과는 최적의 에폭에서 얻어집니다. 우리의 실행에서 사용된 하이퍼파라미터는 표 10에서 확인할 수 있습니다.

메소드 데이터셋 MNLI SST-2 MRPC CoLA QNLI QQP RTE STS-B
최적화기 AdamW
WarmupRatio 0.06
LRSchedule 선형
BatchSize 16 16 16 32 32 16 32 16
#Epochs 30 60 30 80 25 25 80 40
RoBERTabase
학습률 5E-04 5E-04 4E-04 4E-04 4E-04 5E-04 5E-04 4E-04
LoRA
LoRAConfig. r =r =8
q v
LoRAα 8
MaxSeq.Len. 512
BatchSize 4 4 4 4 4 4 8 8
#Epochs 10 10 20 20 10 20 20 30
RoBERTalarge
학습률 3E-04 4E-04 3E-04 2E-04 2E-04 3E-04 4E-04 2E-04
LoRA
LoRAConfig. r =r =8
q v
LoRAα 16
MaxSeq.Len. 128 128 512 128 512 512 512 512
BatchSize 4
#Epochs 10 10 20 20 10 20 20 10
RoBERTalarge
학습률 3E-04 4E-04 3E-04 2E-04 2E-04 3E-04 4E-04 2E-04
LoRA†
LoRAConfig. r =r =8
q v
LoRAα 16
MaxSeq.Len. 128
BatchSize 32
RoBERTalarge #Epochs 10 20 20 20 10 20 20 20
AdptP(3M)† 학습률 3E-05 3E-05 3E-04 3E-04 3E-04 3E-04 3E-04 3E-04
Bottleneckr 64
MaxSeq.Len. 128
BatchSize 32
RoBERTalarge #Epochs 5 20 20 20 10 20 20 20
AdptP(0.8M)† 학습률 3E-04 3E-04 3E-04 3E-04 3E-04 3E-04 3E-04 3E-04
Bottleneckr 16
MaxSeq.Len. 128
BatchSize 32
RoBERTalarge #Epochs 10 5 10 10 5 20 20 10
AdptH(6M)† 학습률 3E-05 3E-04 3E-04 3E-04 3E-04 3E-04 3E-04 3E-04
Bottleneckr 64
MaxSeq.Len. 128
BatchSize 32
RoBERTalarge #Epochs 10 5 10 10 5 20 20 10
AdptH(0.8M)† 학습률 3E-04 3E-04 3E-04 3E-04 3E-04 3E-04 3E-04 3E-04
Bottleneckr 8
MaxSeq.Len. 128
표9: GLUE 벤치마크에서 RoBERTa에 사용한 하이퍼파라미터.

D.3 GPT-2
우리는 모든 GPT-2 모델을 AdamW(Loshchilov&Hutter,2017)를 사용하여 선형 학습률 스케줄로 5 에폭 동안 훈련시켰습니다. 우리는 Li&Liang(2021)에서 설명한 배치 크기, 학습률, 그리고 빔 서치 빔 크기를 사용했습니다. 따라서, 우리는 LoRA에 대한 위의 하이퍼파라미터도 조정했습니다. 우리는 3개의 랜덤 시드에 대한 평균을 보고하며, 각 실행의 결과는 최적의 에폭에서 가져왔습니다. GPT-2에서 LoRA에 사용된 하이퍼파라미터는 표11에 나열되어 있습니다. 다른 기준선에 대해 사용된 것은 Li&Liang(2021)을 참조하십시오.

D.4 GPT-3
모든 GPT-3 실험에서, 우리는 AdamW(Loshchilov&Hutter,2017)를 사용하여 2 에폭 동안 훈련시켰으며, 배치 크기는 128 샘플이고 가중치 감소 요인은 0.1입니다. 우리는 384의 시퀀스 길이를 사용했습니다.

메소드 데이터셋 MNLI SST-2 MRPC CoLA QNLI QQP RTE STS-B
최적화기 AdamW
웜업비율 0.1
학습률 스케줄 선형
배치 크기 8 8 32 4 6 8 4 4
에폭 수 5 16 30 10 8 11 11 10
DeBERTaXXL
학습률 1E-04 6E-05 2E-04 1E-04 1E-04 1E-04 2E-04 2E-04
LoRA
가중치 감소 0 0.01 0.01 0 0.01 0.01 0.01 0.1
CLS 드롭아웃 0.15 0 0 0.1 0.1 0.2 0.2 0.2
LoRAConfig. r =r =8
q v
LoRAα 8
최대 시퀀스 길이 256 128 128 64 512 320 320 128
표10: GLUE 벤치마크에 포함된 작업들에 대한 DeBERTaXXL의 하이퍼파라미터.

데이터셋 E2E WebNLG DART
훈련
최적화기 AdamW
가중치 감소 0.01 0.01 0.0
드롭아웃 확률 0.1 0.1 0.0
배치 크기 8
에폭 수 5
웜업 스텝 500
학습률 스케줄 선형
레이블 스무딩 0.1 0.1 0.0
학습률 0.0002
적응 r =r =4
q v
LoRAα 32
추론
빔 크기 10
길이 패널티 0.9 0.8 0.8
norepeatngramsize 4
표11: E2E, WebNLG 및 DART에 대한 GPT-2 LoRA의 하이퍼파라미터.

WikiSQL(Zhongetal.,2017), MNLI(Williamsetal.,2018)에 대해 768, SAMSum(Gliwa etal.,2019)에 대해 2048을 사용합니다. 모든 방법-데이터셋 조합에 대해 학습률을 조정합니다. 접두사 임베딩 튜닝에 대한 자세한 내용은 섹션 D.4를 참조하십시오. 접두사 임베딩 튜닝에 대해 최적의 l과 l을 각각 256과 8로 찾아 총 3.2M의 훈련 가능한 파라미터를 얻습니다. 전체적으로 최상의 성능을 얻기 위해 l = 8 및 l = 8을 사용하여 20.2M의 훈련 가능한 파라미터를 가진 접두사 레이어 튜닝을 사용합니다. LoRA에 대해 두 가지 파라미터 예산을 제시합니다: 4.7M (r = r = 1 또는 r = 2) 및 37.7M (r = r = 8 또는 r =r =r =r =2). 각 실행에서 가장 좋은 검증 성능을 보고합니다. GPT-3 실험에서 사용한 훈련 하이퍼파라미터는 표 12에 나열되어 있습니다.

E LoRA와 접두사 튜닝 결합하기

LoRA는 기존의 접두사 기반 접근법과 자연스럽게 결합할 수 있습니다. 이 섹션에서는 WikiSQL과 MNLI에서 LoRA와 접두사 튜닝의 변형을 결합한 두 가지를 평가합니다.

LoRA+PrefixEmbed(LoRA+PE)는 LoRA와 접두사 임베딩 튜닝을 결합하며, 여기서 우리는 임베딩이 훈련 가능한 파라미터로 취급되는 l +l 특수 토큰을 삽입합니다. 접두사 임베딩 튜닝에 대한 자세한 내용은 섹션 5.1을 참조하십시오.

LoRA+PrefixLayer(LoRA+PL)는 LoRA와 접두사 레이어 튜닝을 결합합니다. 우리는 또한 l +l 특수 토큰을 삽입합니다. 그러나 이 토큰들의 숨겨진 표현이 자연스럽게 진화하는 대신, 이들 토큰의 숨겨진 표현을 훈련 가능한 파라미터로 취급합니다.

하이퍼파라미터 Fine-Tune PreEmbed PreLayer BitFit AdapterH LoRA
최적화기 AdamW
BatchSize 128
#Epoch 2
WarmupTokens 250,000
LRSchedule Linear
LearningRate 5.00E-06 5.00E-04 1.00E-04 1.6E-03 1.00E-04 2.00E-04
표 12: 다양한 GPT-3 적응 방법에 사용된 훈련 하이퍼파라미터. 학습률을 조정한 후 모든 데이터셋에 대해 동일한 하이퍼파라미터를 사용합니다.
일반적으로, 우리는 모든 Transformer 블록 이후에 입력에 무관한 벡터로 대체합니다. 따라서 임베딩과 이후의 Transformer 블록 활성화는 모두 훈련 가능한 파라미터로 취급됩니다. prefix-layer 튜닝에 대한 자세한 내용은 섹션 5.1을 참조하십시오.
표 15에서는 LoRA+PE와 LoRA+PL의 WikiSQL과 MultiNLI에 대한 평가 결과를 보여줍니다.
먼저, LoRA+PE는 WikiSQL에서 LoRA와 prefix-embedding 튜닝을 모두 크게 능가하는 것을 보여줍니다. 이는 LoRA가 어느 정도로 prefix-embedding 튜닝과 직교하다는 것을 나타냅니다. MultiNLI에서는 LoRA+PE의 조합이 LoRA보다 더 나은 성능을 내지 못하는데, 이는 LoRA 자체가 이미 인간 기준선에 준하는 성능을 달성하기 때문일 수 있습니다. 둘째로, 우리는 LoRA+PL이 더 많은 훈련 가능한 파라미터를 가지고도 LoRA보다 약간 더 나쁜 성능을 내는 것을 알 수 있습니다. 우리는 이를 prefix-layer 튜닝이 학습률 선택에 매우 민감하며, 따라서 LoRA+PL에서 LoRA 가중치의 최적화를 더 어렵게 만든다는 사실에 기인한다고 생각합니다.
F 추가적인 경험적 실험
F.1 GPT-2에 대한 추가 실험
우리는 또한 DART (Nan et al., 2020)와 WebNLG (Gardent et al., 2017)에 대한 실험을 Li & Liang (2021)의 설정을 따라 반복합니다. 결과는 표 13에 나타나 있습니다. 섹션 5에서 보고된 E2E NLG Challenge에 대한 결과와 유사하게, LoRA는 동일한 수의 훈련 가능한 파라미터를 가진 prefix-based 접근법보다 더 나은 성능을 내거나 적어도 동등한 성능을 보입니다.
표 13: DART에서 다양한 적응 방법을 사용한 GPT-2. MET와 TER의 분산은 모든 적응 접근법에 대해 0.01보다 작습니다.
21

웹NLG 방법론
표 14: 웹NLG 테스트 세트에서 다양한 적응 방법론을 사용한 GPT-2. MET와 TER의 분산은 모든 실험에서 0.01 미만입니다. "U"는 보이지 않는 카테고리를, "S"는 보이는 카테고리를, "A"는 웹NLG 테스트 세트의 모든 카테고리를 나타냅니다.

F.2 GPT-3에서의 추가 실험
표 15에서는 GPT-3에서 다양한 적응 방법론을 사용한 추가 실험을 제시합니다. 이는 성능과 훈련 가능한 매개변수의 수 사이의 트레이드오프를 확인하는 데 초점을 맞추고 있습니다.

F.3 저데이터 상황
저데이터 상황에서 다양한 적응 접근법의 성능을 평가하기 위해, 우리는 MNLI의 전체 훈련 세트에서 무작위로 100개, 1k개, 10k개의 훈련 예제를 샘플링하여 저데이터 MNLI-n 작업을 형성합니다. 표 16에서는 MNLI-n에서 다양한 적응 접근법의 성능을 보여줍니다. 놀랍게도, PrefixEmbed와 PrefixLayer는 MNLI-100 데이터셋에서 매우 나쁜 성능을 보이며, PrefixEmbed는 무작위 선택(37.6% 대 33.3%)보다 약간 더 나은 성능을 보입니다. PrefixLayer는 PrefixEmbed보다는 더 나은 성능을 보이지만, MNLI-100에서 Fine-Tune이나 LoRA보다는 훨씬 나쁩니다. 훈련 예제의 수를 늘릴수록 prefix 기반 접근법과 LoRA/Fine-tuning 간의 차이는 줄어들며, 이는 prefix 기반 접근법이 GPT-3에서 저데이터 작업에 적합하지 않을 수 있음을 시사할 수 있습니다. LoRA는 MNLI-100과 MNLI-Full에서 Fine-tuning보다 더 나은 성능을 보이며, 무작위 시드로 인한 (±0.3)의 분산을 고려할 때 MNLI-1k와 MNLI-10K에서 비슷한 결과를 보입니다.

MNLI-n에서 다양한 적응 접근법의 훈련 하이퍼파라미터는 표 17에 보고되어 있습니다. 우리는 MNLI-100 세트에서 PrefixLayer에 대해 더 작은 학습률을 사용하였는데, 이는 더 큰 학습률로는 훈련 손실이 감소하지 않기 때문입니다.

G. 부분 공간 간의 유사성 측정
이 논문에서는 우리는 두 개의 열 정규 직교 행렬 $$Ui \in Rd \times i$$와 $$Uj \in Rd \times j$$ 사이의 부분 공간 유사성을 측정하기 위해 $$\phi(A,B,i,j)=\psi(Ui,Uj)= ||U Ai^T UB||^2 F$$를 사용합니다. 이는 A와 B의 왼쪽 특이 행렬의 열을 취함으로써 얻어집니다. 우리는 이 유사성이 단순히 부분 공간 간의 거리를 측정하는 표준 투영 지표의 역임을 지적합니다(Ham & Lee, 2008).

**표 15: WikiSQL과 MNLI에서 다양한 적응 방법의 하이퍼파라미터 분석.** 
접두사 임베딩 튜닝(PrefixEmbed)과 접두사 레이어 튜닝(PrefixLayer)은 훈련 가능한 매개변수의 수를 늘릴수록 성능이 떨어지지만, LoRA의 성능은 안정화됩니다. 성능은 검증 정확도로 측정됩니다.

**표 16: GPT-3 175B를 사용하여 MNLI의 하위 집합에서 다양한 방법의 검증 정확도.** 
MNLI-n은 n개의 훈련 예제를 가진 하위 집합을 설명합니다. 우리는 전체 검증 세트로 평가합니다. LoRA는 다른 방법들, 포함하여 Fine-tuning에 비해 샘플 효율성이 우수하게 나타납니다.

구체적으로, Ui와 Uj의 단일 값이 σA, σB, ..., σp이라고 하자. 여기서 p는 min{i,j}입니다. 우리는 Projection Metric Ham & Lee(2008)이 다음과 같이 정의되어 있다는 것을 알고 있습니다:

$$d(Ui,Uj)=\sqrt{\sum_{i=1}^{p} (σ_{A}^{2}-σ_{B}^{2})} \in [0,\sqrt{p}]$$

하이퍼파라미터 적응 MNLI-100 MNLI-1k MNLI-10K MNLI-392K
최적화기 - AdamW
WarmupTokens - 250,000
LRSchedule - 선형
BatchSize - 20 20 100 128
#Epoch - 40 40 4 2
FineTune 5.00E-6
PrefixEmbed 2.00E-04 2.00E-04 4.00E-04 5.00E-04
학습률
PrefixLayer 5.00E-05 5.00E-05 5.00E-05 1.00E-04
LoRA 2.00E-4
PrefixEmbedl 16 32 64 256
p
적응- PrefixEmbedl 8
i
특정 PrefixTune l =l =8
p i
LoRA r =r =8
q v
표17: MNLI(m)-n에 대한 다양한 GPT-3 적응 방법에 사용된 하이퍼파라미터.
우리의 유사성은 다음과 같이 정의됩니다:
$$\phi(A,B,i,j)=\psi(U_i,U_j)=\sum_{i=1}^{p} \sigma_{i}^{2} (1-d(U_i,U_j)^{2})$$
이 유사성은 만약 $U_i$와 $U_j$가 같은 열 범위를 공유하면, $\phi(A,B,i,j) = 1$이라는 것을 만족합니다. 만약 그들이 완전히 직교하면, $\phi(A,B,i,j)=0$입니다. 그렇지 않으면, $\phi(A,B,i,j)\in(0,1)$입니다.

H 저랭크 행렬에 대한 추가 실험
우리는 저랭크 업데이트 행렬에 대한 조사에서 추가 결과를 제시합니다.

H.1 LoRA 모듈 간의 상관관계
그림 6과 그림 7을 참조하여 그림 3과 그림 4에서 제시된 결과가 다른 레이어로 어떻게 일반화되는지 확인하세요.

H.2 r의 GPT-2에 대한 효과
우리는 r의 효과에 대한 실험을 GPT-2에서 반복합니다. E2ENLG Challenge 데이터셋을 예로 들어, 26,000 단계 학습 후에 다른 r 선택에 의해 달성된 검증 손실과 테스트 메트릭을 보고합니다. 우리의 결과는 표 18에 제시되어 있습니다. GPT-2 Medium에 대한 최적의 랭크는 사용된 메트릭에 따라 4와 16 사이입니다, 이는 GPT-3 175B에 대한 것과 유사합니다. 모델 크기와 적응을 위한 최적 랭크 간의 관계는 여전히 미해결된 질문입니다.

H.3 W와 ∆W 사이의 상관관계
r의 변화에 따른 W와 ∆W 사이의 정규화된 부분공간 유사성에 대해 그림 8을 참조하세요.
다시 한번, ∆W는 W의 최상위 특이 방향을 포함하지 않는다는 것을 주목하세요, 이는 ∆W의 상위 4 방향과 W의 상위 10% 사이의 유사성이 거의 0.2를 초과하지 않기 때문입니다. 이는 ∆W가 W에서 그렇게 강조되지 않는 "작업 특정" 방향을 포함한다는 증거를 제공합니다.
다음으로 답해야 할 흥미로운 질문은, 모델 적응이 잘 작동하도록 이러한 작업 특정 방향을 얼마나 "강하게" 증폭해야 하는지에 대한 것입니다.

**그림 6: 96층 트랜스포머에서 1번째, 32번째, 64번째, 96번째 층의 ∆Wq와 ∆Wv에 대해 A와 A의 열 벡터 사이의 정규화된 부분 공간 유사성**

H.4 증폭 요소

특성 증폭 요소를 ∆W의 SVD 분해의 왼쪽과 오른쪽 특이 행렬인 U와 V에 대한 비율인 ∆W / F로 자연스럽게 고려할 수 있습니다. (UUWVV는 ∆W에 의해 생성된 부분 공간으로 W의 "투영"을 제공합니다.)

직관적으로, ∆W가 주로 작업 특정 방향을 포함하면, 이 수량은 ∆W에 의해 얼마나 증폭되는지를 측정합니다. 7.3절에서 보여준 것처럼, r = 4의 경우, 이 증폭 요소는 최대 20까지입니다. 다시 말해, 사전 훈련된 모델 W에서 각 층의 특성 방향(전체 특성 공간 중) 네 개가 매우 큰 요소 20에 의해 증폭되어야 하며, 이는 하류 특정 작업에 대한 우리의 보고된 정확도를 달성하기 위함입니다. 그리고, 각각의 다른 하류 작업에 대해 매우 다른 특성 방향 집합이 증폭되어야 할 것으로 예상해야 합니다.

그러나, r = 64의 경우, 이 증폭 요소는 대략 2 정도로, r = 64로 학습된 ∆W의 대부분의 방향이 크게 증폭되지 않는다는 것을 알 수 있습니다. 이는 놀라운 일이 아니며, 실제로는 "작업 특정 방향"(따라서 모델 적응)을 표현하는 데 필요한 본질적인 순위가 낮다는 증거를 (다시 한번) 제공합니다. 반대로, ∆W의 순위 4 버전(즉, r = 4에 해당)의 방향은 훨씬 더 큰 요소 20에 의해 증폭됩니다.

그림 7: 두 개의 무작위로 생성된 실행에서 A의 열 벡터 사이의 정규화된 부분 공간 유사성, 96층 트랜스포머에서 1번째, 32번째, 64번째, 96번째 층의 ∆Wq와 ∆Wv에 대해.

표 18: LoRA가 GPT-2 Medium에서 다양한 순위 r을 사용하여 E2E NLG Challenge에서 달성한 검증 손실 및 테스트 세트 메트릭. GPT-3에서 r = 1이 많은 작업에 충분한 반면, 여기서는 검증 손실에 대해 r = 16에서, BLEU에 대해 r = 4에서 성능이 최고를 찍음으로써, GPT-2 Medium이 GPT-3175B와 비슷한 본질적인 순위를 가지고 적응한다는 것을 제안합니다. 우리의 일부 하이퍼파라미터는 r = 4에서 조정되며, 이는 다른 기준선의 매개변수 수와 일치하므로, 다른 r의 선택에 대해 최적이지 않을 수 있습니다.

그림 8: Wq와 ∆Wq의 단일 방향 사이의 정규화된 부분 공간 유사성, 변화하는 r과 무작위 기준. ∆Wq는 중요하지만 W에서 강조되지 않는 방향을 증폭합니다. 더 큰 r을 가진 ∆Wq는 이미 W에서 강조된 더 많은 방향을 선택하는 경향이 있습니다.

